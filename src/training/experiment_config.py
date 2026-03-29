"""Shared experiment configuration and hyperparameter utilities.

Single source of truth for all learnings from Phase 4-5 experiments.
Import from here rather than hardcoding per script so improvements
propagate automatically.

Learnings baked in
------------------
- AdamW with weight_decay=0.0 on W_pos (weight decay collapses positions)
- lambda_safety must scale down with N — safety_valve uses O(N²) pairs
  in a fixed box_size, so density rises and each pair contributes more
- Safety valve disabled above N=5000 (cdist would OOM at ~10 GB)
- Gradient clipping DISABLED (float('inf')) — iter1 had no clipping; norm=1.0 capped
  W_pos movement to <20% of the box in 120ep, causing the persistent ~19-20% ceiling
- log1p softening in safety_valve_loss — smooth Coulomb cap
- Early stopping on train_loss (val loss unreliable on small GA subsets)
- python -u in tmux scripts for unbuffered stdout
"""

from __future__ import annotations


# ── GA best config from Phase 4 Exp 1 ───────────────────────────────────────

GA_BEST = {
    "K": 2,
    "N_hidden": 256,
    "lr_Wpos": 2.364e-3,
    "lambda_safety": 0.691,   # tuned for N=256, D=4, box_size=1.0
    "batch_size": 64,
}


# ── Lambda safety scaling ────────────────────────────────────────────────────

BASE_N = 256      # N for which GA_BEST["lambda_safety"] was tuned
BASE_D = 4        # geometric dimensionality
MAX_N_SAFETY = 5000   # above this, safety_valve_loss returns 0 (OOM guard)


def scaled_lambda_safety(
    n_hidden: int,
    base_lambda: float = GA_BEST["lambda_safety"],
    base_n: int = BASE_N,
    d: int = BASE_D,
) -> float:
    """Return lambda_safety scaled for a given N_hidden.

    As N grows, r* = 0.5/N^(1/D) shrinks and the Coulomb repulsion
    activates more aggressively. Scale lambda down proportionally so
    the safety loss contribution stays comparable to the task loss.

    Formula: lambda_eff = base_lambda * (base_n / n_hidden)^(1/D)

    At n_hidden > MAX_N_SAFETY: return 0.0 (safety loss is disabled).
    """
    if n_hidden > MAX_N_SAFETY:
        return 0.0
    return base_lambda * (base_n / n_hidden) ** (1.0 / d)


# ── Standard Trainer kwargs ──────────────────────────────────────────────────

def trainer_kwargs(
    n_hidden: int,
    lr_wpos: float | None = None,
    n_epochs: int = 150,
    sched_type: str = "plateau",
) -> dict:
    """Return Trainer constructor kwargs with all learnings applied.

    Parameters
    ----------
    n_hidden  : number of hidden neurons (for lambda_safety scaling)
    lr_wpos   : override learning rate (default: GA best)
    n_epochs  : total training epochs — used for cosine T_max
    sched_type: "plateau" (ReduceLROnPlateau) | "cosine" (CosineAnnealingLR) | "none" (constant LR)

    Usage:
        trainer = Trainer(model, train_loader, val_loader,
                          **trainer_kwargs(n_hidden, n_epochs=150), device=device)
    """
    return {
        "lr_wpos": lr_wpos if lr_wpos is not None else GA_BEST["lr_Wpos"],
        "lambda_safety": scaled_lambda_safety(n_hidden),
        "lambda_lb": 0.01,
        "use_amp": True,
        "grad_clip_norm": float("inf"),  # no clipping — matches iter1 (26.52%)
        "sched_type": sched_type,
        "sched_patience": 10,        # only used when sched_type="plateau"
        "sched_factor": 0.5,
        "sched_cosine_T": n_epochs,  # only used when sched_type="cosine"
        "min_lr": 1e-7,
        "early_stop_patience": 50,   # generous — let model converge fully
        "early_stop_delta": 5e-4,
    }


# ── SmallWorld / ProximityWave topology defaults ─────────────────────────────

def topology_kwargs(n_hidden: int) -> dict:
    """Return small-world topology parameters for a given N_hidden.

    K_random >= 2 is required for ~100% graph connectivity.
    K_local = 4 gives good local clustering without excess fan-in.
    norm_mode="l2": validated Step 1 — l2 per-neuron normalisation beats
    masked (10% random) and relu (8% due to D=4 near-zero vectors).
    """
    return {
        "K_local": 4,
        "K_random": 2,       # minimum for global connectivity (empirically verified)
        "K_in": 50,          # input fan-in per hidden neuron
        "K_iter": 3,
        "n_groups": max(8, n_hidden // 8),
        "norm_mode": "l2",   # VALIDATED: 23.9% vs masked=10.2% vs relu=8.2%
    }
