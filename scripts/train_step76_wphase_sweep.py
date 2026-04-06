"""Step 76: alpha_turing sweep on patched arch — W_phase utility at Gen4+ base.

MOTIVATION
==========
W_phase is an nn.Parameter in SGNNET_Resonant (shape [N, D]) that was never
added to the optimizer in Gen4+ experiments (Trainer only adds W_pos; W_phase
added only when lr_wphase is explicitly set). This means W_phase NEVER LEARNED
in any Gen4+ run — it stayed at its random initialisation.

Known results from step69:
  Ref (turing=0.0): 83.36%   — W_phase not used (alpha_turing=0 → _phase_inhibit skipped)
  A   (turing=0.3): 85.04%   — W_phase barely used (frozen random init) → +1.68pp somehow

Questions:
1. What is the full turing α curve at N=1024? (only 0.0 and 0.3 known)
2. What if W_phase actually LEARNS while turing > 0? (theta bug analogy)
3. Can W_phase contribute beyond the inhibition path (step46 reconnect concept)?

Note: at N=4096 (step70), turing=0.0 > turing=0.3 by 0.12pp — N-dependent.
Step 76 maps the full curve at N=1024 to understand the shape before scaling.

CONFIGS (N=1024, D=64, K_iter=8, Gen4+ params, 50%/75ep)
==========================================================
  Ref : alpha_turing=0.0  (Gen4+ baseline = 83.36% — control)
  A   : alpha_turing=0.1  (conservative phase inhibition)
  B   : alpha_turing=0.3  (step69 A repro — confirms consistency)
  C   : alpha_turing=0.5  (moderate)
  D   : alpha_turing=1.0  (strong — equal weight to structural excitation)

All configs A-D: W_phase is added to optimizer at lr_wpos (step77 analogy —
W_phase was frozen-at-random-init; should learn while turing is active).

EXPECTED DIRECTION
==================
  Ref vs B: must match step69 Ref vs A = +1.68pp (confirms reproducibility)
  A-D curve: is turing contribution monotone, or does it peak at 0.3?
  W_phase learning: configs A-D now have W_phase trained — may shift peak vs step69

To reproduce:
    python -u scripts/train_step76_wphase_sweep.py --device mps
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

K_PHASE       = 8
BEAM_SIZE     = 16
GEO_GAMMA     = 0.5
ALPHA_REFLECT = 0.5
ALPHA_AHEBB   = 1.0

LR_WPOS    = GA_BEST["lr_Wpos"]   # 2.364e-3
STEP69_REF = 0.8336
STEP69_A   = 0.8504   # turing=0.3, W_phase frozen-at-random

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


def make_model(alpha_turing: float, seed_offset: int = 0) -> SGNNET_AntiHebbian:
    torch.manual_seed(SEED + seed_offset)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=8, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=K_PHASE, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=alpha_turing, beam_size=BEAM_SIZE,
        geo_gamma=GEO_GAMMA, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def run(label: str, model: nn.Module, alpha_turing: float, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)

    # Inject W_phase into optimizer for configs where turing > 0
    # (W_phase was never trained in step69; this is the step77-analogy fix)
    if alpha_turing > 0.0:
        trainer.optimizer.add_param_group({
            "params": [model.m.W_phase],
            "lr": LR_WPOS,        # same LR as spatial positions
            "weight_decay": 0.0,  # let phase anchors explore freely
        })
        print(f"  W_phase: LEARNABLE at lr={LR_WPOS:.3e}  (was frozen-at-random in step69)")
        print(f"  alpha_turing={alpha_turing:.1f}  — phase inhibition ACTIVE")
    else:
        print(f"  W_phase: not used (alpha_turing=0.0)")

    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best      = max(top1_hist)
    best_ep   = int(np.argmax(top1_hist)) + 1
    frac      = best_ep / len(history)

    result = {
        "label":            label,
        "alpha_turing":     alpha_turing,
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
        "step69_A_turing03": STEP69_A,
        "delta_vs_ref":     round(best - STEP69_REF, 4),
        "delta_vs_step69A": round(best - STEP69_A, 4),
        "_meta":            run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  vs_ref={best-STEP69_REF:+.4f}  vs_step69A={best-STEP69_A:+.4f}  t={elapsed:.0f}s"
    )
    return result


CONFIGS = [
    # (key, label, alpha_turing, seed_offset)
    ("Ref", "Ref  alpha_turing=0.0  (Gen4+ baseline = 83.36%)",   0.0, 0),
    ("A",   "A    alpha_turing=0.1  (conservative)",              0.1, 1),
    ("B",   "B    alpha_turing=0.3  (step69 A repro — 85.04%?)",  0.3, 2),
    ("C",   "C    alpha_turing=0.5",                              0.5, 3),
    ("D",   "D    alpha_turing=1.0  (equal weight to structural)", 1.0, 4),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 76: alpha_turing sweep — W_phase now learned (theta-fix analogy)")
    print(f"step69 Ref (turing=0.0): {STEP69_REF:.4f}")
    print(f"step69 A   (turing=0.3, W_phase frozen-random): {STEP69_A:.4f}")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step76_wphase_sweep.json"

    for key, label, alpha_turing, seed_off in CONFIGS:
        model = make_model(alpha_turing=alpha_turing, seed_offset=seed_off).to(DEVICE)
        meta = {
            "N": N, "D": D, "K_iter": 8,
            "alpha_turing":  alpha_turing,
            "alpha_reflect": ALPHA_REFLECT,
            "alpha_ahebb":   ALPHA_AHEBB,
            "wphase_trained": alpha_turing > 0.0,
            "data_frac":     0.5,
        }
        results[key] = run(label, model, alpha_turing, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    print(f"\n{'='*70}")
    print(f"STEP 76 COMPLETE — alpha_turing sweep (W_phase now learned)")
    print(f"step69 Ref={STEP69_REF:.4f}  step69 A(turing=0.3 frozen)={STEP69_A:.4f}")
    print()
    print(f"  {'Key':4s}  {'top1':>8s}  {'vs_ref':>8s}  {'vs_step69A':>11s}  {'α_turing':>8s}")
    for key, label, alpha_turing, _ in CONFIGS:
        if key not in results:
            continue
        r = results[key]
        print(f"  {key:4s}  {r['top1_best']:.4f}    {r['delta_vs_ref']:+.4f}"
              f"    {r['delta_vs_step69A']:+.4f}    {alpha_turing:>8.1f}")
    print()
    print("  B > step69 A: learning W_phase improves on frozen-random init")
    print("  Monotone: α=0.3 is still optimal (N=1024 finding holds)")
    print("  Peak shifts: N-dependent turing — cross-ref with step70 N=4096 results")
