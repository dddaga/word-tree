"""Step 79: Mechanism-aware auxiliary losses on patched arch at N=4096.

MOTIVATION
==========
step39 tested phase coherence, inhibition sparsity, and routing diversity aux
losses — but on buggy arch (input coverage gap + alpha_reflect silenced) at N=1024.
ALL configs killed: A-F = 47.44% (−4.41pp vs Ref=51.85%).

We don't know if the failure was:
  (a) aux losses fundamentally conflict with task signal, OR
  (b) aux signals were being computed on a broken forward pass

This experiment re-tests on patched arch + Gen4+ params + N=4096.
Comparison baseline: step70 Config B = 97.32% (100%/150ep full run).
Ablation protocol: 50%/75ep — Ref here establishes the ablation baseline.

CONFIGS (N=4096, D=64, K_iter=8, Gen4+ params, 50%/75ep)
=========================================================
  Ref  no aux loss (establishes N=4096 50%/75ep baseline)
  A    phase coherence    λ=0.01   (step39 scale)
  B    phase coherence    λ=0.001  (10× lower — step39 may have been too aggressive)
  C    inhibition sparsity λ=0.01
  D    inhibition sparsity λ=0.001
  E    routing diversity   λ=0.01
  F    routing diversity   λ=0.001

KEY QUESTION: does any aux signal improve over Ref on the real architecture?
If yes: which direction (phase alignment / sparsity / diversity)?
If still negative: aux losses fundamentally conflict with task loss — close the question.

To reproduce:
    python -u scripts/train_step79_aux_loss_patched.py --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld       import SGNNET_SmallWorld
from src.sgnnet.model_resonant         import SGNNET_Resonant
from src.training.trainer              import Trainer
from src.training.experiment_config    import (
    trainer_kwargs, topology_kwargs, run_metadata, GA_BEST,
)
from src.training.dataset              import make_loaders
from src.sgnnet.losses                 import load_balance_loss, safety_valve_loss

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS    = 75
BATCH     = 128
SEED      = 42
DATA      = "data/store.h5"
N         = 4096
D         = 64

# Gen4+ params (step70 Config B winner)
K_PHASE        = 8
BEAM_SIZE      = 16
GEO_GAMMA      = 0.5
ALPHA_REFLECT  = 0.5
ALPHA_TURING   = 0.0
ALPHA_AHEBB    = 1.0

# Full-run reference (100%/150ep — for context only; ablation Ref established here)
STEP70_BEST = 0.9732

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n   = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(
            subset, batch_size=BATCH, shuffle=True, num_workers=0
        )
        _loaders = (tr, va)
    return _loaders


# ── Model: AH with aux loss threaded into routing loop ───────────────────────

class SGNNET_AHMechLoss(nn.Module):
    """SGNNET_AntiHebbian (wpos) with mechanism-aware aux loss tracked per routing step.

    Replicates the full AH routing loop from SGNNET_AntiHebbian.forward() and
    computes aux loss inside the loop — single forward pass, no duplication.

    Aux types:
      'phase'   — phase coherence: -mean(cosine_sim(Z[j], Z[conn_hh[j]]))
      'sparse'  — inhibition sparsity: mean(|Z_fwd|) — L1 on post-threshold activations
      'div'     — routing diversity: -entropy(softmax(|Z_nb|.sum(-1) + ε))
    """

    def __init__(
        self,
        resonant: SGNNET_Resonant,
        alpha_ahebb: float = 1.0,
        aux_type: str | None = None,
    ):
        super().__init__()
        self.m           = resonant
        self.alpha_ahebb = alpha_ahebb
        self.aux_type    = aux_type       # None → no aux loss (Ref config)
        self.last_aux_loss = torch.tensor(0.0)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                                   # [B, N, D]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)          # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)                    # [N, D]
        conn_hh   = self.m.base.conn_hh                                    # [N, K_hh]

        # Pre-compute static AH suppression weights (wpos variant)
        N_h    = self.m.base.N_hidden
        W_n    = F.normalize(self.m.W_pos[:N_h], dim=-1)                   # [N_h, D]
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)               # [N_h, K_hh]
        supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                   ).unsqueeze(0).unsqueeze(-1)                            # [1,N,K_hh,1]

        Z_reflected = torch.zeros_like(Z)
        aux_accum   = []

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)                               # [B, N, D]
            Z_nb     = Z_fwd[:, conn_hh, :]                                # [B, N, K_hh, D]
            Z_struct = (Z_nb * supp_w).sum(dim=2)                          # [B, N, D]

            # Reflection accumulator (alpha_reflect fix)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            # Phase inhibition (skipped when alpha_turing=0)
            if self.m.alpha_turing != 0.0:
                Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
                Z_new = Z_struct + Z_reflected + self.m.alpha_turing * Z_inh
            else:
                Z_new = Z_struct + Z_reflected

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

            # ── Aux loss computation (inside routing loop) ─────────────────
            if self.aux_type is not None:
                if self.aux_type == "phase":
                    # Maximize cosine alignment between Z[j] and its neighbors
                    Z_nb_norm = F.normalize(Z_nb, dim=-1)                  # [B,N,K_hh,D]
                    cos_sim   = (Z.unsqueeze(2) * Z_nb_norm).sum(-1)       # [B,N,K_hh]
                    aux_accum.append(-cos_sim.mean())

                elif self.aux_type == "sparse":
                    # Penalize large post-threshold activations (want sparse firing)
                    aux_accum.append(Z_fwd.abs().mean())

                elif self.aux_type == "div":
                    # Reward even routing distribution across neighbors
                    Z_nb_mag = Z_fwd[:, conn_hh, :].norm(dim=-1)          # [B,N,K_hh]
                    log_prob = F.log_softmax(Z_nb_mag + 1e-8, dim=-1)
                    prob     = log_prob.exp()
                    entropy  = -(prob * log_prob).sum(-1).mean()
                    aux_accum.append(-entropy)   # maximize entropy

        if aux_accum:
            self.last_aux_loss = torch.stack(aux_accum).mean()
        else:
            self.last_aux_loss = torch.tensor(0.0, device=x.device)

        return self.m.base._readout(Z)


# ── AuxTrainer: adds model.last_aux_loss to total loss ───────────────────────

class AuxTrainer(Trainer):

    def __init__(self, lambda_aux: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_aux = lambda_aux

    def train_epoch(self) -> dict:
        self.model.train()
        total_sum = task_sum = safety_sum = lb_sum = aux_sum = 0.0
        n_batches = 0

        for features, soft_labels, _labels in self.train_loader:
            features    = features.to(self.device)
            soft_labels = soft_labels.to(self.device)
            self.optimizer.zero_grad()

            _amp_ctx = (
                torch.autocast(str(self.device).split(":")[0], dtype=torch.float16)
                if self.use_amp
                else torch.autocast("cpu", enabled=False)
            )
            with _amp_ctx:
                scores    = self.model(features)
                task_loss = F.kl_div(F.log_softmax(scores, dim=-1),
                                     soft_labels, reduction="batchmean")
                safety    = safety_valve_loss(self.model.W_pos, self.box_size,
                                              task_loss=task_loss)
                lb_loss   = load_balance_loss(scores.abs().sum(dim=0))
                aux_loss  = (getattr(self.model, "last_aux_loss", torch.tensor(0.0))
                             .to(self.device))
                loss      = (task_loss
                             + self.lambda_safety * safety
                             + self.lambda_lb * lb_loss
                             + self.lambda_aux * aux_loss)

            _opt_params = [p for g in self.optimizer.param_groups for p in g["params"]]
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(_opt_params, self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_opt_params, self.grad_clip_norm)
                self.optimizer.step()

            if hasattr(self.model, "tick_step"):
                self.model.tick_step()
            with torch.no_grad():
                self.model.W_pos.clamp_(0, self.box_size)

            total_sum   += loss.item()
            task_sum    += task_loss.item()
            safety_sum  += safety.item()
            lb_sum      += lb_loss.item()
            aux_sum     += aux_loss.item()
            n_batches   += 1

        n = max(n_batches, 1)
        return {
            "train_loss":  total_sum / n,
            "task_loss":   task_sum / n,
            "safety_loss": safety_sum / n,
            "lb_loss":     lb_sum / n,
            "aux_loss":    aux_sum / n,
        }


def make_model(aux_type: str | None, seed_offset: int = 0) -> SGNNET_AHMechLoss:
    torch.manual_seed(SEED + seed_offset)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=8, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base,
        K_phase=K_PHASE,
        alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING,
        beam_size=BEAM_SIZE,
        geo_gamma=GEO_GAMMA,
        mode="dynamic_z_geo",
        resonance_threshold=0.0,
    )
    return SGNNET_AHMechLoss(resonant, alpha_ahebb=ALPHA_AHEBB, aux_type=aux_type)


def run(label: str, model: nn.Module, lambda_aux: float, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = AuxTrainer(
        lambda_aux=lambda_aux,
        model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk,
    )
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best      = max(top1_hist)
    best_ep   = int(np.argmax(top1_hist)) + 1
    frac      = best_ep / len(history)

    result = {
        "label":           label,
        "top1_best":       best,
        "top1_last":       history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in history[-5:]])),
        "final_aux_loss":  float(np.mean([h.get("aux_loss",  0.0) for h in history[-5:]])),
        "best_epoch":      best_ep,
        "epochs_run":      len(history),
        "elapsed_s":       round(elapsed, 1),
        "best_epoch_frac": round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history":    top1_hist,
        "_meta":           run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  t={elapsed:.0f}s"
    )
    return result


# ── Configs ────────────────────────────────────────────────────────────────────
# (key, label, aux_type, lambda_aux, seed_offset)
CONFIGS = [
    ("Ref", "Ref   no aux loss (N=4096 50%/75ep baseline)",          None,     0.0,   0),
    ("A",   "A     phase coherence  λ=0.01  (step39 scale)",         "phase",  0.01,  1),
    ("B",   "B     phase coherence  λ=0.001 (10× lower)",            "phase",  0.001, 2),
    ("C",   "C     inhibition sparsity λ=0.01",                      "sparse", 0.01,  3),
    ("D",   "D     inhibition sparsity λ=0.001",                     "sparse", 0.001, 4),
    ("E",   "E     routing diversity λ=0.01",                        "div",    0.01,  5),
    ("F",   "F     routing diversity λ=0.001",                       "div",    0.001, 6),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 79: Mechanism-aware aux losses on patched arch N={N}")
    print(f"step39 comparison: all configs = 47.44% (−4.41pp) on buggy arch N=1024")
    print(f"Full-run reference: step70-B = {STEP70_BEST:.4f} (100%/150ep)")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step79_aux_loss_patched.json"
    ref_val  = None

    for key, label, aux_type, lambda_aux, seed_off in CONFIGS:
        model = make_model(aux_type=aux_type, seed_offset=seed_off).to(DEVICE)
        meta  = {
            "N": N, "D": D, "K_iter": 8,
            "alpha_ahebb": ALPHA_AHEBB, "alpha_reflect": ALPHA_REFLECT,
            "alpha_turing": ALPHA_TURING, "beam_size": BEAM_SIZE,
            "aux_type": aux_type, "lambda_aux": lambda_aux,
            "data_frac": 0.5,
        }
        results[key] = run(label, model, lambda_aux, meta)
        if key == "Ref":
            ref_val = results["Ref"]["top1_best"]
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    # ── Final summary ─────────────────────────────────────────────────────────
    ref_val = ref_val or results.get("Ref", {}).get("top1_best", 0.0)
    print(f"\n{'='*70}")
    print(f"STEP 79 COMPLETE — Aux losses on patched arch N={N}")
    print(f"step39 baseline (buggy arch N=1024): all = 47.44% (−4.41pp)")
    print(f"Ablation Ref (patched arch N={N} 50%/75ep): {ref_val:.4f}")
    print()
    print(f"  {'Key':4s}  {'top1':>8s}  {'vs_Ref':>8s}  {'aux_type':>8s}  {'λ':>8s}")
    for key, label, aux_type, lambda_aux, _ in CONFIGS:
        if key not in results:
            continue
        r     = results[key]
        delta = r["top1_best"] - ref_val
        print(f"  {key:4s}  {r['top1_best']:.4f}    {delta:+.4f}    "
              f"{str(aux_type):>8s}  {lambda_aux:>8.3f}")
    print()
    print("  Interpretation:")
    print("  A/B > Ref  → phase coherence aux helps on patched arch — include in Gen4+")
    print("  C/D > Ref  → sparsity reward helps — inhibitory threshold well-calibrated")
    print("  E/F > Ref  → routing diversity helps — neighbors are over-correlated")
    print("  all ≤ Ref  → aux losses fundamentally conflict with task; close question")
    print("  lower λ wins → step39 was killed by λ scale, not signal direction")
