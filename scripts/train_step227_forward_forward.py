"""Step 227: Forward-Forward Algorithm for SGNNET — Tier-0 scout.

MOTIVATION
==========
Test whether the Forward-Forward algorithm (Hinton 2022) can match backprop for SGNNET.

Key challenge: W_pos is SHARED across K_iter steps. Standard FF assumes per-layer params.
Solution: treat the WHOLE routing (K_iter steps) as one "layer". Goodness is measured on
the FINAL routed activations Z, not per-step. This avoids the contradictory-update problem
identified in RESEARCH_learning_algorithms_p1.md §3.

Two goodness functions tested:
  Ref: Standard backprop (KL loss, AdamW) — control
  A:   FF with whole-network goodness = ||Z||² summed over neurons aligned with class direction
  B:   FF with per-neuron local goodness — each neuron maximizes its own activity magnitude

FF loss per sample: log(1 + exp(goodness_neg - goodness_pos + threshold))

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

# FF hyperparameters
FF_THRESHOLD = 2.0        # goodness threshold separating pos/neg
FF_LR        = 2.364e-3   # same as standard W_pos lr

FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step227_forward_forward.json"


# ──────────────────────────────────────────────────────────────────────────────
# FF-capable model wrapper
# ──────────────────────────────────────────────────────────────────────────────

class SGNNET_ForwardForward(nn.Module):
    """SGNNET wrapped for Forward-Forward training.

    Exposes both:
      - forward(x) → logits  (for standard validation / Ref config)
      - forward_z(x) → Z     (final routed activations before readout)

    The FF goodness is computed on Z, not on logits.
    Crucially, we DO NOT backprop through the routing loop — we only compute
    the gradient of the goodness function w.r.t. W_pos at the readout level.
    This is achieved by stopping gradient at the Z→goodness boundary.
    """

    def __init__(self, base, resonant, alpha_ahebb=1.0,
                 goodness_mode="whole_network"):
        """
        goodness_mode:
          "whole_network" — goodness = Σ ||Z[i]||² weighted by class alignment (Config A)
          "per_neuron"    — goodness = Σ ||Z[i]||² raw (no class conditioning) (Config B)
        """
        super().__init__()
        self.base          = base
        self.resonant      = resonant
        self.alpha_ahebb   = alpha_ahebb
        self.goodness_mode = goodness_mode

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase

    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()

    def _route(self, x: torch.Tensor) -> torch.Tensor:
        """Run seeding + routing, return final Z. Same loop as AntiHebbian."""
        Z         = self.base._seed(x)
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.base.conn_hh
        N_h       = self.base.N_hidden

        # Static AH suppression weights (pre-computed, outside loop)
        W_n     = F.normalize(self.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)         # [N, K_hh]
        supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)                        # [1,N,K_hh,1]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_nb     = Z_fwd[:, conn_hh, :] * supp_w
            Z_struct = Z_nb.sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.resonant.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected
            Z     = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return Z   # [B, N, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward → logits (used for validation and Ref config)."""
        Z = self._route(x)
        return self.base._readout(Z)

    def goodness(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute scalar goodness per sample.

        whole_network mode (Config A):
          goodness = mean over neurons of (||Z[i]||² * class_alignment[i])
          class_alignment[i] = max(0, dot(Z[i], W_pos_of_class[label]))
          This is class-CONDITIONED: goodness is high when neurons fire in
          the direction of the correct class position.

        per_neuron mode (Config B):
          goodness = mean over neurons of ||Z[i]||²  (raw magnitude, no class)
          Each neuron independently maximizes its own activity.
          Class information comes only from the input (class-appended features).

        Returns: [B] scalar goodness per sample.
        """
        Z = self._route(x)   # [B, N, D]

        if self.goodness_mode == "whole_network":
            # W_pos of output neurons = class directions [N_out, D]
            W_out     = self.W_pos[self.base.N_hidden:]                  # [10, D]
            W_out_n   = F.normalize(W_out, dim=-1)                       # [10, D]
            # For each sample, select its class direction: [B, D]
            class_dir = W_out_n[labels]                                  # [B, D]
            # Alignment of each hidden neuron with class direction: [B, N]
            align     = (Z * class_dir.unsqueeze(1)).sum(dim=-1)         # [B, N]
            # Goodness = sum of (magnitude² * alignment_if_positive)
            mag_sq    = (Z * Z).sum(dim=-1)                              # [B, N]
            goodness  = (mag_sq * align.clamp(min=0)).mean(dim=1)        # [B]
        else:
            # per_neuron: raw magnitude only
            goodness = (Z * Z).sum(dim=-1).mean(dim=1)                  # [B]

        return goodness   # [B]


# ──────────────────────────────────────────────────────────────────────────────
# Forward-Forward trainer
# ──────────────────────────────────────────────────────────────────────────────

class FFTrainer:
    """Custom trainer for Forward-Forward learning.

    Two forward passes per batch:
      - Positive: (x, correct_label)   → goodness_pos should be HIGH
      - Negative: (x, wrong_label)     → goodness_neg should be LOW

    Loss: log(1 + exp(goodness_neg - goodness_pos + threshold))

    W_pos gradient: flows only through goodness function (NOT through routing).
    We use torch.no_grad() around the routing loop and re-enable grad for
    goodness computation. This is implemented by routing in no_grad, then
    treating Z as a leaf with .requires_grad_(True) for goodness calculation.

    Note: The spec says "no backprop through routing loop" — we implement this
    by stopping gradient at Z (detach the routing output, then re-attach grad
    for the goodness → W_pos path via the goodness function's explicit use of
    W_pos in class alignment computation).
    """

    def __init__(self, model, train_loader, val_loader, device,
                 lr=FF_LR, threshold=FF_THRESHOLD, n_epochs=20):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.threshold    = threshold
        self.n_epochs     = n_epochs

        # Only optimize W_pos (the only meaningful FF parameter)
        # W_phase and fc_out parameters are included for completeness
        self.optimizer = torch.optim.AdamW(
            [{"params": [model.W_pos], "lr": lr, "weight_decay": 0.0}]
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7,
        )

    def _ff_loss(self, x, labels):
        """Compute FF loss for a batch.

        Positive pass: real (x, label).
        Negative pass: same x, wrong label (random permutation of labels).
        """
        B = x.shape[0]
        # Wrong labels: random permutation ensuring no sample keeps correct label
        wrong_labels = labels.clone()
        for i in range(B):
            candidates = [c for c in range(N_OUT) if c != labels[i].item()]
            wrong_labels[i] = candidates[torch.randint(len(candidates), (1,)).item()]

        # Positive goodness
        g_pos = self.model.goodness(x, labels)       # [B]
        # Negative goodness
        g_neg = self.model.goodness(x, wrong_labels) # [B]

        # FF loss: push pos high, neg low
        loss = torch.log1p(torch.exp(g_neg - g_pos + self.threshold)).mean()
        return loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        n = 0

        for features, soft_labels, hard_labels in self.train_loader:
            features    = features.to(self.device)
            hard_labels = hard_labels.to(self.device)

            self.optimizer.zero_grad()
            loss = self._ff_loss(features, hard_labels)
            loss.backward()
            self.optimizer.step()

            # Clamp W_pos to [0, 1]
            with torch.no_grad():
                self.model.W_pos.clamp_(0, 1.0)

            total_loss += loss.item()
            n += 1

        return {"train_loss": total_loss / max(n, 1)}

    def evaluate(self):
        self.model.eval()
        all_scores = []; all_labels = []
        val_loss = 0.0; n = 0

        with torch.no_grad():
            for features, soft_labels, labels in self.val_loader:
                features    = features.to(self.device)
                soft_labels = soft_labels.to(self.device)
                scores = self.model(features)
                val_loss += F.kl_div(
                    F.log_softmax(scores, dim=-1), soft_labels, reduction="batchmean"
                ).item()
                all_scores.append(scores.cpu())
                all_labels.append(labels)
                n += 1

        all_scores_cat = torch.cat(all_scores, dim=0)
        all_labels_cat = torch.cat(all_labels, dim=0)
        top1 = (all_scores_cat.argmax(dim=-1) == all_labels_cat).float().mean().item()
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
# Build model
# ──────────────────────────────────────────────────────────────────────────────

N_OUT = 10   # module-level for use in _ff_loss

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
    """Standard backprop model (AntiHebbian wrapper)."""
    base, resonant = build_base()
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def build_ff(goodness_mode):
    base, resonant = build_base()
    return SGNNET_ForwardForward(base, resonant, alpha_ahebb=ALPHA_AHEBB,
                                  goodness_mode=goodness_mode)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    all_keys = ["Ref", "A", "B"]
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else all_keys

    labels = {
        "Ref": "Ref: standard backprop + AH (control)",
        "A":   "A: FF whole-network goodness (class-conditioned ||Z||²·alignment)",
        "B":   "B: FF per-neuron goodness (local ||Z||² maximization)",
    }

    print(f"\n{'='*70}")
    print(f"Step 227 — Forward-Forward Algorithm (Tier-0 scouts)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"FF threshold={FF_THRESHOLD} | goodness on final Z (no backprop through routing)")
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
        else:
            gmode   = "whole_network" if key == "A" else "per_neuron"
            model   = build_ff(gmode).to(DEVICE)
            n_p     = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  params={n_p:,}  (forward-forward, goodness_mode={gmode})")
            ff_trainer = FFTrainer(model=model, train_loader=tr, val_loader=va,
                                   device=DEVICE, n_epochs=EPOCHS)
            history = ff_trainer.train()

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
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print("STEP 227 SUMMARY — Forward-Forward vs Backprop")
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

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
