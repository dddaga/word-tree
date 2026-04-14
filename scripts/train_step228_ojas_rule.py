"""Step 228: Oja's Rule for W_pos — replace AdamW W_pos update with local Hebbian rule.

MOTIVATION
==========
Oja's rule: Δw_i = η * (x_i * y_i - y_i² * w_i)
where y_i = w_i · x_i (projection, the "activation")
      x_i = aggregated neighbor input to neuron i

This naturally keeps W_pos on S^{D-1} (converges to unit sphere by construction).
2024 paper: "overcomes challenges of training neural networks under biological constraints" —
preserves activation subspaces, mitigates exploding/vanishing signals.

Key question: can local Hebbian learning match or replace gradient-based W_pos learning?
Secondary question: does Oja make AH redundant (both decorrelate W_pos)?

Implementation: after each routing step, apply Oja update to W_pos[i] using the
aggregated neighbor input as x_i and the current W_pos[i]·x_i as y_i.
Backprop is DISABLED for W_pos in Configs A/B/C — only fc_out uses gradient.

Configs:
  Ref: standard backprop + AH (control, identical to step199 base)
  A:   Oja for W_pos + backprop for fc_out only (with AH)
  B:   Oja for W_pos + backprop for fc_out only (WITHOUT AH — test if Oja replaces AH)
  C:   Hybrid — Oja during routing, then backprop refines W_pos in outer loop

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
                    help="Comma-separated config keys (e.g. Ref,A,B). Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

# Oja hyperparameters
OJA_LR = 1e-3   # Oja learning rate (same order as AdamW lr_wpos=2.364e-3)

FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step228_ojas_rule.json"


# ──────────────────────────────────────────────────────────────────────────────
# Oja-updated model
# ──────────────────────────────────────────────────────────────────────────────

class SGNNET_Oja(nn.Module):
    """SGNNET where W_pos is updated by Oja's local Hebbian rule.

    Routing loop is identical to SGNNET_AntiHebbian (AH suppression goes inside
    the routing loop, not as a loss). After the loop, Oja update is applied
    to W_pos in-place using the accumulated neighbor inputs.

    use_ah:  if True, apply AH suppression during routing (Config A)
             if False, no AH suppression (Config B)
    hybrid:  if True, record gradients for W_pos (outer backprop loop can refine)
             if False, W_pos is updated ONLY by Oja (Configs A, B)
    """

    def __init__(self, base, resonant, alpha_ahebb=1.0,
                 use_ah=True, hybrid=False, oja_lr=OJA_LR):
        super().__init__()
        self.base        = base
        self.resonant    = resonant
        self.alpha_ahebb = alpha_ahebb
        self.use_ah      = use_ah
        self.hybrid      = hybrid
        self.oja_lr      = oja_lr
        # Accumulate neighbor inputs over the batch for Oja update
        # shape: [N_hidden, D] — reset each forward pass
        self._oja_acc    = None
        self._oja_count  = 0

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase

    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()

    def _apply_oja(self):
        """Apply accumulated Oja update to W_pos[:N_hidden].

        Oja's rule (vectorized over all N hidden neurons simultaneously):
          x_i = mean aggregated neighbor input to neuron i  (from _oja_acc)
          y_i = W_pos[i] · x_i                             (projection)
          Δw_i = η * (x_i * y_i - y_i² * W_pos[i])

        This is a rank-1 Hebbian update with weight decay that keeps norms bounded.
        After update, W_pos stays close to S^{D-1} (re-normalize for numerical stability).
        """
        if self._oja_acc is None or self._oja_count == 0:
            return

        N_h = self.base.N_hidden
        with torch.no_grad():
            # x: average aggregated input per neuron [N_h, D]
            x = self._oja_acc / max(self._oja_count, 1)   # [N_h, D]
            w = self.W_pos[:N_h]                           # [N_h, D]

            # y = dot(w, x) per neuron: [N_h]
            y = (w * x).sum(dim=-1)                        # [N_h]

            # Δw = η * (x * y[:, None] - y²[:, None] * w)
            y2 = (y * y).unsqueeze(-1)                     # [N_h, 1]
            delta = self.oja_lr * (x * y.unsqueeze(-1) - y2 * w)  # [N_h, D]

            self.W_pos[:N_h].add_(delta)
            # Re-normalize to keep W_pos near S^{D-1}
            self.W_pos[:N_h].data = F.normalize(self.W_pos[:N_h].data, dim=-1)
            # Clamp back to [0, 1] box after normalize (positions live in [0,1]^D)
            self.W_pos.data.clamp_(0, 1.0)

        # Reset accumulator
        self._oja_acc   = None
        self._oja_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.base._seed(x)
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.base.conn_hh
        N_h       = self.base.N_hidden

        # AH suppression weights (pre-computed outside loop if use_ah=True)
        if self.use_ah:
            W_n     = F.normalize(self.W_pos[:N_h], dim=-1)
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)        # [N, K_hh]
            supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                      ).unsqueeze(0).unsqueeze(-1)                       # [1,N,K_hh,1]

        Z_reflected = torch.zeros_like(Z)

        # Accumulator for Oja update (sum over routing steps and batch)
        # We use the MEAN neighbor input across K_iter steps as the Oja x_i
        if self.training:
            oja_batch_acc = torch.zeros(N_h, D, device=Z.device, dtype=Z.dtype)
            oja_step_count = 0

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]                               # [B,N,K_hh,D]

            if self.use_ah:
                Z_nb = Z_nb * supp_w

            Z_struct = Z_nb.sum(dim=2)                                  # [B,N,D]

            # Accumulate Z_struct as Oja x_i (mean over batch)
            if self.training:
                # Detach so Oja accumulation doesn't interfere with autograd graph
                oja_batch_acc  = oja_batch_acc + Z_struct.detach().mean(dim=0)
                oja_step_count += 1

            Z_remainder = Z_fwd - Z
            Z_reflected = self.resonant.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected
            Z     = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        # Accumulate for cross-batch Oja update (called externally per batch)
        if self.training:
            if self._oja_acc is None:
                self._oja_acc = oja_batch_acc
            else:
                self._oja_acc = self._oja_acc + oja_batch_acc
            self._oja_count += oja_step_count

        return self.base._readout(Z)


# ──────────────────────────────────────────────────────────────────────────────
# Oja trainer: handles W_pos via Oja, fc_out via backprop
# ──────────────────────────────────────────────────────────────────────────────

class OjaTrainer:
    """Training loop for Oja configurations.

    W_pos: updated by Oja's rule (called after each forward pass).
    fc_out (output parameters): updated by AdamW + backprop.
    Hybrid mode: also runs a backprop step on W_pos after Oja.
    """

    def __init__(self, model, train_loader, val_loader, device,
                 n_epochs=20, hybrid=False, lr_out=2.364e-3):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.n_epochs     = n_epochs
        self.hybrid       = hybrid

        # Optimizer for fc_out (and W_phase, theta from resonant)
        # Exclude W_pos from backprop params in non-hybrid mode
        if not hybrid:
            out_params = [p for name, p in model.named_parameters()
                         if "W_pos" not in name]
        else:
            # Hybrid: include W_pos in backprop too (Oja runs first, backprop refines)
            out_params = [{"params": [model.W_pos], "lr": lr_out * 0.1, "weight_decay": 0.0}]
            other = [p for name, p in model.named_parameters() if "W_pos" not in name]
            out_params = out_params + [{"params": other, "lr": lr_out}]

        self.optimizer = torch.optim.AdamW(
            out_params if not hybrid else out_params,
            lr=lr_out, weight_decay=0.0
        ) if not hybrid else torch.optim.AdamW(out_params)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7,
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0; n = 0

        for features, soft_labels, hard_labels in self.train_loader:
            features    = features.to(self.device)
            soft_labels = soft_labels.to(self.device)

            self.optimizer.zero_grad()
            scores   = self.model(features)
            loss     = F.kl_div(F.log_softmax(scores, dim=-1),
                                 soft_labels, reduction="batchmean")
            loss.backward()
            self.optimizer.step()

            # Oja update (applied per batch using accumulated neighbor inputs)
            self.model._apply_oja()

            # Clamp W_pos to [0, 1] box
            with torch.no_grad():
                self.model.W_pos.clamp_(0, 1.0)

            total_loss += loss.item(); n += 1

        return {"train_loss": total_loss / max(n, 1)}

    def evaluate(self):
        self.model.eval()
        all_scores = []; all_labels = []
        val_loss = 0.0; n = 0

        with torch.no_grad():
            for features, soft_labels, labels in self.val_loader:
                features    = features.to(self.device)
                soft_labels = soft_labels.to(self.device)
                scores      = self.model(features)
                val_loss   += F.kl_div(F.log_softmax(scores, dim=-1),
                                        soft_labels, reduction="batchmean").item()
                all_scores.append(scores.cpu())
                all_labels.append(labels)
                n += 1

        all_s = torch.cat(all_scores, dim=0)
        all_l = torch.cat(all_labels, dim=0)
        top1  = (all_s.argmax(dim=-1) == all_l).float().mean().item()
        return {"val_loss": val_loss / max(n, 1), "val_top1": top1}

    def train(self, log_fn=None):
        history = []
        for epoch in range(self.n_epochs):
            train_m = self.train_epoch()
            val_m   = self.evaluate()
            if hasattr(self.model, "tick_epoch"):
                self.model.tick_epoch()
            self.scheduler.step(train_m["train_loss"])
            combined = {"epoch": epoch, **train_m, **val_m}
            history.append(combined)
            if log_fn:
                log_fn(combined)
            if (epoch + 1) % 5 == 0:
                print(f"  e{epoch+1:3d}  loss={train_m['train_loss']:.4f}  "
                      f"top1={val_m['val_top1']:.4f}", flush=True)
        return history


# ──────────────────────────────────────────────────────────────────────────────
# Build helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_base():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return base, resonant


def build_ref():
    base, resonant = build_base()
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def build_oja(use_ah=True, hybrid=False):
    base, resonant = build_base()
    return SGNNET_Oja(base, resonant, alpha_ahebb=ALPHA_AHEBB,
                       use_ah=use_ah, hybrid=hybrid, oja_lr=OJA_LR)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    all_keys = ["Ref", "A", "B", "C"]
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else all_keys

    labels = {
        "Ref": "Ref: standard backprop + AH (control)",
        "A":   "A: Oja for W_pos + backprop fc_out only (with AH)",
        "B":   "B: Oja for W_pos + backprop fc_out only (WITHOUT AH — Oja replaces AH?)",
        "C":   "C: Hybrid — Oja during routing + backprop refines W_pos",
    }

    print(f"\n{'='*70}")
    print(f"Step 228 — Oja's Rule for W_pos (Tier-0 scouts)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M) | Oja LR={OJA_LR}")
    print(f"Question: can local Hebbian W_pos learning match backprop?")
    print(f"Question: does Oja make AH redundant?")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n   = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}

    for key in run_keys:
        print(f"\n{'─'*60}")
        print(f"Config {labels.get(key, key)}")
        print(f"{'─'*60}")
        t0 = time.time()

        if key == "Ref":
            model   = build_ref().to(DEVICE)
            n_p     = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  params={n_p:,}  (standard backprop)")
            kw      = trainer_kwargs(N, n_epochs=EPOCHS)
            trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                              device=DEVICE, **kw)
            history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: None)

        elif key == "A":
            model   = build_oja(use_ah=True, hybrid=False).to(DEVICE)
            n_p     = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  params={n_p:,}  (Oja + AH)")
            oja_t   = OjaTrainer(model=model, train_loader=tr, val_loader=va,
                                 device=DEVICE, n_epochs=EPOCHS, hybrid=False)
            history = oja_t.train()

        elif key == "B":
            model   = build_oja(use_ah=False, hybrid=False).to(DEVICE)
            n_p     = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  params={n_p:,}  (Oja only, no AH)")
            oja_t   = OjaTrainer(model=model, train_loader=tr, val_loader=va,
                                 device=DEVICE, n_epochs=EPOCHS, hybrid=False)
            history = oja_t.train()

        elif key == "C":
            model   = build_oja(use_ah=True, hybrid=True).to(DEVICE)
            n_p     = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  params={n_p:,}  (Oja + backprop hybrid)")
            oja_t   = OjaTrainer(model=model, train_loader=tr, val_loader=va,
                                 device=DEVICE, n_epochs=EPOCHS, hybrid=True)
            history = oja_t.train()

        elapsed = time.time() - t0
        top1h   = [round(h.get("val_top1", 0.), 4) for h in history]
        best    = max(top1h); bep = int(np.argmax(top1h)) + 1

        results[key] = {
            "label":       labels.get(key, key),
            "top1_best":   best,
            "top1_last":   top1h[-1],
            "best_epoch":  bep,
            "epochs_run":  len(history),
            "top1_history": top1h,
            "elapsed_s":   round(elapsed, 1),
            "n_params":    n_p,
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "oja_lr":      OJA_LR,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print("STEP 228 SUMMARY — Oja's Rule vs Backprop for W_pos")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for key in run_keys:
        r     = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        verdict = ""
        if key != "Ref":
            if   r["top1_best"] - ref_best > 0.005:  verdict = " → ADVANCE"
            elif r["top1_best"] - ref_best > -0.010: verdict = " → NEUTRAL"
            else:                                     verdict = " → KILL"
        print(f"  {key}: {r['top1_best']:.4f}{delta}{verdict}")

    # Specific diagnostic: does B (no AH) match A (with AH)?
    if "A" in results and "B" in results:
        diff = results["B"]["top1_best"] - results["A"]["top1_best"]
        if abs(diff) < 0.005:
            print(f"\n  NOTE: B ≈ A (Δ={diff:+.4f}) — Oja likely replaces AH")
        elif diff < 0:
            print(f"\n  NOTE: B < A (Δ={diff:+.4f}) — AH still needed even with Oja")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
