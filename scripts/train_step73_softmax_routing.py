"""Step 73: Softmax redistribution routing — gate-death theorem bypass.

MOTIVATION
==========
Wave-1 experiments (steps 58-66) all failed via gate-death:
    signal ∝ Π g_k ≈ 0.004 at K_iter=8 → 250× gradient collapse.

Root fix: replace signal-destroying gates with signal-conserving redistribution.
softmax(score, dim=K_hh): Σ_j w_j = 1 over K_hh neighbors.
No per-step attenuation. Gradient flows cleanly through all K_iter steps.

Current AH routing (weighted sum, NOT normalized):
    Z_struct[h] = Σ_j supp_w[h,j] × Z_nb[h,j,:]

Softmax redistribution (conservative, Σw=1):
    logit[h,j] = score(Z_h, Z_nb_j) / τ + logit_AH[h,j]
    w[h,j]     = softmax(logit, dim=j)            # Σ_j w = 1
    Z_struct[h] = Σ_j w[h,j] × Z_nb[h,j,:]

AH suppression is folded into the softmax logit (NOT as a separate multiplier)
so it influences routing without attenuating signal:
    logit_AH[h,j] = -alpha_ahebb × clamp(pos_sim[h,j], min=0)

CONFIGS (N=1024, D=64, K_iter=8, Gen4+ params, 50%/75ep)
==========================================================
  Ref : current AH sum routing (control — reproduces step69 Ref = 83.36%)
  A   : softmax(dot(Z_h, Z_nb) / τ=1.0)            — Z-state score only, no AH
  B   : softmax(dot(Z_h, Z_nb) / τ=1.0 + logit_AH) — Z-state + AH folded in
  C   : softmax(logit_AH)                            — AH-only softmax (static, no Z score)
  D   : same as B, τ=0.3                             — sharper routing decisions

EXPECTED DIRECTION
==================
  B vs Ref: main hypothesis — conservative routing + AH diversity pressure
  C vs Ref: does AH benefit from Σ=1 normalization alone (no Z dynamics)?
  A vs B:   how much does AH contribute vs pure Z-state routing?
  D vs B:   sharper routing (lower τ) helps or hurts?
  Any gain over Ref = 83.36% confirms gate-death was the wave-1 failure mode.

To reproduce:
    python -u scripts/train_step73_softmax_routing.py --device mps
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
from src.sgnnet.mechanisms_inhibitory  import SGNNET_AntiHebbian
from src.training.trainer              import Trainer
from src.training.experiment_config    import (
    trainer_kwargs, topology_kwargs, run_metadata, GA_BEST,
)
from src.training.dataset              import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS    = 75
BATCH     = 128
SEED      = 42
DATA      = "data/store.h5"
N         = 1024
D         = 64

# Gen4+ params (step70 Config B: turing=0.0)
K_PHASE       = 8
BEAM_SIZE     = 16
GEO_GAMMA     = 0.5
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
ALPHA_AHEBB   = 1.0

STEP69_REF = 0.8336   # AH sum routing, same arch, 50%/75ep

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


# ── Softmax routing wrapper ────────────────────────────────────────────────────

class SGNNET_SoftmaxRouting(nn.Module):
    """AH routing with softmax redistribution instead of weighted sum.

    Replaces the non-normalised weighted sum in SGNNET_AntiHebbian with a
    softmax over K_hh neighbors (Σ_j w_j = 1 → conservative, no signal loss).
    AH diversity pressure is preserved by folding it into the softmax logit.

    Parameters
    ----------
    resonant    : SGNNET_Resonant backbone
    alpha_ahebb : AH suppression strength (folded into logit offset)
    score_mode  : 'z_dot'   — logit = dot(Z_h, Z_nb) / tau
                  'z_ah'    — logit = dot(Z_h, Z_nb) / tau - alpha * pos_sim
                  'ah_only' — logit = -alpha * pos_sim (static, no Z score)
    tau         : softmax temperature (smaller = sharper routing)
    """

    def __init__(
        self,
        resonant: SGNNET_Resonant,
        alpha_ahebb: float = 1.0,
        score_mode: str = "z_ah",
        tau: float = 1.0,
    ):
        super().__init__()
        self.m           = resonant
        self.alpha_ahebb = alpha_ahebb
        self.score_mode  = score_mode
        self.tau         = tau

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)             # [B, N, D]
        N_h       = self.m.base.N_hidden
        conn_hh   = self.m.base.conn_hh              # [N_h, K_hh]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)  # [1, N, 1]

        # Pre-compute static AH logit offset: -alpha × clamp(pos_sim, 0)
        # Shape: [N_h, K_hh]  — broadcast over batch dimension later
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)  # [N_h, K_hh]
        ah_logit = -self.alpha_ahebb * pos_sim.clamp(min=0)   # [N_h, K_hh]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)           # [B, N, D]
            Z_nb  = Z_fwd[:, conn_hh, :]            # [B, N, K_hh, D]

            # Compute routing logits
            if self.score_mode == "z_dot":
                # Dynamic Z-state score only (no AH)
                z_score = (Z_fwd.unsqueeze(2) * Z_nb).sum(-1)  # [B, N, K_hh]
                logit = z_score / self.tau

            elif self.score_mode == "z_ah":
                # Z-state score + AH logit offset (main hypothesis)
                z_score = (Z_fwd.unsqueeze(2) * Z_nb).sum(-1)  # [B, N, K_hh]
                logit = z_score / self.tau + ah_logit.unsqueeze(0)

            elif self.score_mode == "ah_only":
                # Static AH-only score (tests whether Σ=1 normalization alone helps)
                logit = ah_logit.unsqueeze(0).expand(Z_fwd.shape[0], -1, -1)

            # Softmax redistribution: Σ_j w_j = 1 — no signal attenuation
            w       = F.softmax(logit, dim=2)                # [B, N, K_hh]
            Z_struct = (w.unsqueeze(-1) * Z_nb).sum(dim=2)  # [B, N, D]

            # Reflection accumulator (identical to AH)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            # Phase inhibition (disabled when alpha_turing=0.0)
            if self.m.alpha_turing != 0.0:
                W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
                Z_inh     = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
                Z_new     = Z_struct + Z_reflected + self.m.alpha_turing * Z_inh
            else:
                Z_new = Z_struct + Z_reflected

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ── Model factory ──────────────────────────────────────────────────────────────

def make_base(seed_offset: int = 0) -> tuple[SGNNET_SmallWorld, SGNNET_Resonant]:
    """Build SmallWorld + Resonant backbone (shared structure for all configs)."""
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
    return base, resonant


def make_model_ref(seed_offset: int = 0) -> SGNNET_AntiHebbian:
    _, resonant = make_base(seed_offset)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def make_model_softmax(
    score_mode: str,
    tau: float,
    seed_offset: int = 0,
) -> SGNNET_SoftmaxRouting:
    _, resonant = make_base(seed_offset)
    return SGNNET_SoftmaxRouting(resonant, alpha_ahebb=ALPHA_AHEBB,
                                 score_mode=score_mode, tau=tau)


# ── Training loop ──────────────────────────────────────────────────────────────

def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best      = max(top1_hist)
    best_ep   = int(np.argmax(top1_hist)) + 1
    frac      = best_ep / len(history)
    delta     = best - STEP69_REF

    result = {
        "label":            label,
        "top1_best":        best,
        "top1_last":        history[-1].get("val_top1", 0.0),
        "final_task_loss":  float(np.mean([h.get("task_loss", 0.0) for h in history[-5:]])),
        "best_epoch":       best_ep,
        "epochs_run":       len(history),
        "elapsed_s":        round(elapsed, 1),
        "best_epoch_frac":  round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history":     top1_hist,
        "step69_ref":       STEP69_REF,
        "delta_vs_ref":     round(delta, 4),
        "_meta":            run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  vs_ref={delta:+.4f}  t={elapsed:.0f}s"
    )
    return result


# ── Configs ────────────────────────────────────────────────────────────────────

CONFIGS = [
    # (key, label, score_mode_or_ref, tau, seed_offset)
    ("Ref", "Ref  AH sum routing (control — step69 Ref)",
     "ref", 1.0, 0),
    ("A",   "A    softmax(Z_dot / τ=1.0) — Z-state only, no AH",
     "z_dot", 1.0, 1),
    ("B",   "B    softmax(Z_dot / τ=1.0 + AH_logit) — Z-state + AH (main hypothesis)",
     "z_ah", 1.0, 2),
    ("C",   "C    softmax(AH_logit) — AH-only softmax redistribution",
     "ah_only", 1.0, 3),
    ("D",   "D    softmax(Z_dot / τ=0.3 + AH_logit) — sharper redistribution",
     "z_ah", 0.3, 4),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 73: Softmax redistribution routing — gate-death bypass")
    print(f"Ref: step69 AH sum routing = {STEP69_REF:.4f}")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step73_softmax_routing.json"

    for key, label, score_mode, tau, seed_off in CONFIGS:
        if score_mode == "ref":
            model = make_model_ref(seed_offset=seed_off).to(DEVICE)
        else:
            model = make_model_softmax(score_mode, tau, seed_offset=seed_off).to(DEVICE)
        meta = {
            "N": N, "D": D, "K_iter": 8,
            "score_mode":    score_mode,
            "tau":           tau,
            "alpha_ahebb":   ALPHA_AHEBB,
            "alpha_reflect": ALPHA_REFLECT,
            "alpha_turing":  ALPHA_TURING,
            "data_frac":     0.5,
        }
        results[key] = run(label, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"STEP 73 COMPLETE — Softmax redistribution routing")
    print(f"Ref (AH sum routing): {STEP69_REF:.4f}")
    print()
    print(f"  {'Key':4s}  {'top1':>8s}  {'vs_ref':>8s}  {'score_mode':>12s}  {'tau':>5s}")
    for key, label, score_mode, tau, _ in CONFIGS:
        if key not in results:
            continue
        r = results[key]
        print(f"  {key:4s}  {r['top1_best']:.4f}    {r['delta_vs_ref']:+.4f}  "
              f"{score_mode:>12s}  {tau:>5.1f}")
    print()
    print("  Interpretation:")
    print("  B > Ref → softmax redistribution + AH beats sum routing")
    print("  B > A   → AH diversity pressure contributes even in softmax mode")
    print("  C > Ref → Σ=1 normalization alone is beneficial (AH sum ≠ redistribution)")
    print("  D > B   → sharper routing (lower τ) is beneficial")
