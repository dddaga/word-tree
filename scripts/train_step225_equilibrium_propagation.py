"""Step 225: Equilibrium propagation pilot — local learning without backprop.

MOTIVATION
==========
SGNNET's iterative routing converges to a steady state (equilibrium).
Equilibrium Propagation (Scellier & Bengio 2017) trains such systems using only
LOCAL information: compare equilibrium states under free vs nudged conditions.

Free phase:   run K_iter routing steps → Z_free (standard forward pass)
Nudged phase: clamp output toward target with strength β, run K_iter more → Z_nudged
Update rule:  ΔW ∝ (1/β) * (∂E_nudged/∂W - ∂E_free/∂W)

Energy function: E = -Σ_i Σ_k Z[i] · Z[conn_hh[i,k]]
  — sum of activation alignments along edges (negative = want neurons to agree with neighbors)

This is a PILOT — just test whether EP can make any progress vs backprop baseline.
Actual EP convergence is slow and noisy at Tier-0; the question is: does it learn at all?

Configs:
  Ref : standard backprop (AntiHebbian, step199 config)
  A   : equilibrium propagation β=0.1 (gentle nudge)
  B   : equilibrium propagation β=0.5 (stronger nudge)

EP configs use manual training loop (no Trainer class).

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep — Tier-0)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (e.g. Ref,A). Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step225_equilibrium_propagation.json"

# EP hyperparams
EP_LR     = 1e-3   # EP update step size
EP_NUDGE_ITER = K_ITER  # same number of iterations for nudged phase


# ---------------------------------------------------------------------------
# SGNNET core routing (extracted for EP manual loop)
# ---------------------------------------------------------------------------

class SGNNETCore(nn.Module):
    """Bare SGNNET routing without Trainer wrapper.

    Returns (Z_final, logits). Z_final is the last routing state.
    Used by EP to access intermediate states.
    """

    def __init__(self, base_sw, resonant, alpha_ahebb=1.0, beta=0.0, n_out=N_OUT):
        """
        beta: nudge strength (0 = free phase, >0 = nudged phase)
        """
        super().__init__()
        self.base_sw    = base_sw
        self.resonant   = resonant
        self.alpha_ahebb = alpha_ahebb
        self.beta       = beta
        self.n_out      = n_out

        N_h = base_sw.N_hidden
        # Precompute AH suppression weights (static, W_pos-based)
        W_n = F.normalize(base_sw.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[base_sw.conn_hh]).sum(-1)
        self._supp_w = (1.0 - alpha_ahebb * pos_sim.clamp(min=0)
                        ).unsqueeze(0).unsqueeze(-1)  # [1, N, K_hh, 1]

    @property
    def W_pos(self):   return self.base_sw.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase

    def _energy(self, Z):
        """E = -Σ_i Σ_k Z[i] · Z[conn_hh[i,k]].
        Returns scalar energy per sample: [B].
        """
        conn_hh = self.base_sw.conn_hh
        Z_nb = Z[:, conn_hh, :]           # [B, N, K_hh, D]
        align = (Z.unsqueeze(2) * Z_nb).sum(-1)  # [B, N, K_hh]
        return -align.sum(dim=(1, 2))      # [B]

    def _route(self, Z, n_iter, target_logits=None, beta=0.0):
        """Run routing for n_iter steps with optional output clamping.

        target_logits: [B, N_out] — target class logits for nudging.
        beta: nudge strength. If beta > 0, adds -beta * CE_loss to energy gradient.
        """
        conn_hh = self.base_sw.conn_hh
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
        supp_w = self._supp_w.to(Z.device)

        Z_reflected = torch.zeros_like(Z)

        for _ in range(n_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]  # [B, N, K_hh, D]
            Z_nb  = Z_nb * supp_w

            Z_struct = Z_nb.sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.resonant.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

            # Nudge toward target: add a small gradient step in the direction of
            # decreasing cross-entropy. This is the EP "weakly clamped" version.
            if beta > 0.0 and target_logits is not None:
                logits = self.base_sw._readout(Z)   # [B, N_out]
                # CE gradient w.r.t. Z: approximate via softmax difference
                p_pred = F.softmax(logits, dim=-1)   # [B, N_out]
                p_tgt  = F.softmax(target_logits / 1.0, dim=-1)  # [B, N_out]
                # Nudge: subtract beta * grad_Z(CE). We use finite-difference approximation:
                # grad_Z(CE) ≈ (p_pred - p_tgt) @ W_out (backprop through readout only)
                W_out  = self.base_sw.W_pos[self.base_sw.N_hidden:]  # [N_out, D]
                delta  = (p_pred - p_tgt)  # [B, N_out]
                nudge  = delta @ W_out     # [B, D] — gradient signal to output neurons
                # Apply only to output-connected hidden neurons (approximate)
                # For simplicity: broadcast nudge to all hidden neurons (crude EP)
                Z_nudge = Z.clone()
                Z_nudge[:, :self.n_out, :] = (
                    Z[:, :self.n_out, :] - beta * nudge.unsqueeze(1).expand_as(Z[:, :self.n_out, :])
                )
                Z = F.normalize(Z_nudge.clamp(-10, 10), dim=-1)

        return Z

    def forward(self, x):
        Z = self.base_sw._seed(x)
        Z_eq = self._route(Z, self.base_sw.K_iter, beta=0.0)
        return self.base_sw._readout(Z_eq)


# ---------------------------------------------------------------------------
# EP training loop
# ---------------------------------------------------------------------------

def train_ep(model: SGNNETCore, tr, va, n_epochs: int, lr: float, beta: float,
             device: torch.device):
    """Manual EP training loop.

    EP update rule:
        ΔW_pos ∝ (1/β) * [∂E_nudged/∂W - ∂E_free/∂W]

    We approximate this via:
        1. Free phase: forward with beta=0 → Z_free, E_free
        2. Nudged phase: forward with beta=β → Z_nudged, E_nudged
        3. Update: W_pos -= lr * (1/β) * (dE_nudged/dW - dE_free/dW)

    For eval: use standard forward pass (beta=0) + argmax.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(n_epochs):
        model.train()
        for batch in tr:
            feats, soft_labels, labels = batch
            feats  = feats.to(device)
            labels = labels.to(device)
            soft_labels = soft_labels.to(device)

            # --- Free phase ---
            Z_seed = model.base_sw._seed(feats)
            opt.zero_grad()
            Z_free = model._route(Z_seed.detach().requires_grad_(False),
                                   model.base_sw.K_iter, beta=0.0)
            E_free = model._energy(Z_free)   # [B]

            # --- Nudged phase ---
            # Use soft_labels as target
            target_logits = soft_labels  # already log-space soft; use as direction
            Z_nudged = model._route(Z_seed.detach().requires_grad_(False),
                                     model.base_sw.K_iter,
                                     target_logits=target_logits, beta=beta)
            E_nudged = model._energy(Z_nudged)  # [B]

            # EP loss = (1/β) * (E_nudged - E_free).mean()
            # This approximates the EP contrastive gradient
            ep_loss = (1.0 / beta) * (E_nudged - E_free).mean()

            ep_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Clear MPS cache periodically
            if str(device) == "mps":
                torch.mps.empty_cache()

        # Eval
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for feats, _, labels in va:
                feats  = feats.to(device)
                labels = labels.to(device)
                logits = model(feats)
                correct += (logits.argmax(1) == labels).sum().item()
                total   += labels.size(0)
        val_top1 = correct / total
        history.append({"epoch": epoch, "val_top1": val_top1})
        if (epoch + 1) % 5 == 0:
            print(f"    EP ep{epoch+1:3d}  val={val_top1:.4f}", flush=True)

    return history


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _make_base_resonant():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return base, resonant


def main():
    all_keys = ["Ref", "A", "B"]
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]
    else:
        run_keys = all_keys

    labels = {
        "Ref": "Ref: standard backprop (step199 AntiHebbian)",
        "A":   "A: equilibrium propagation β=0.1 (gentle nudge)",
        "B":   "B: equilibrium propagation β=0.5 (stronger nudge)",
    }
    beta_map = {"A": 0.1, "B": 0.5}

    print(f"\n{'='*70}")
    print(f"Step 225 — Equilibrium Propagation pilot (Tier-0 scouts)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"Question: can SGNNET learn without backprop via EP?")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        print(f"\n{'─'*60}")
        print(f"Config {labels.get(key, key)}")
        print(f"{'─'*60}")

        base, resonant = _make_base_resonant()
        t0 = time.time()

        if key == "Ref":
            model = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
            model = model.to(DEVICE)
            n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  params={n_p:,}  (standard backprop)")

            kw = trainer_kwargs(N, n_epochs=EPOCHS)
            trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                               device=DEVICE, **kw)
            history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: None)

        else:
            beta = beta_map[key]
            ep_model = SGNNETCore(base, resonant, alpha_ahebb=ALPHA_AHEBB, beta=beta)
            ep_model = ep_model.to(DEVICE)
            n_p = sum(p.numel() for p in ep_model.parameters() if p.requires_grad)
            print(f"  params={n_p:,}  (EP β={beta})")
            history = train_ep(ep_model, tr, va, n_epochs=EPOCHS,
                                lr=EP_LR, beta=beta, device=DEVICE)

        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1

        results[key] = {
            "label": labels.get(key, key),
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "epochs_run": len(history), "top1_history": top1h,
            "elapsed_s": round(elapsed, 1), "n_params": n_p,
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "beta": beta_map.get(key, 0.0),
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print("STEP 225 SUMMARY — Equilibrium propagation pilot")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        verdict = ""
        if key != "Ref":
            if r["top1_best"] > 0.15:    verdict = " → EP LEARNS (interesting!)"
            elif r["top1_best"] > 0.11:  verdict = " → EP better than random"
            else:                        verdict = " → EP failed (chance level)"
        print(f"  {key}: {r['top1_best']:.4f}{delta}{verdict}")

    print(f"\n  Note: EP at 20ep is expected to be weaker than backprop.")
    print(f"  Success criterion: EP > chance (>10%) and shows learning trend.")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
