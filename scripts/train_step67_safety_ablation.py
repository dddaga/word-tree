"""Step 67: Safety Valve λ ablation with AntiHebb α=1.0.

MOTIVATION
==========
The safety_valve_loss is a quadratic W_pos position repulsion regularizer in
[0,1]^D designed to prevent neuron clustering (collapse). It contributes to
training loss as: total = task + λ_safety * safety + λ_lb * load_balance.

Question: Is safety valve load-bearing when AntiHebb is active?

AntiHebb already enforces W_pos diversity implicitly: it suppresses neurons
with similar W_pos cosine similarity, so gradient descent naturally spreads
W_pos to avoid suppression. The position-space repulsion in [0,1]^D may be
redundant when AH=1.0 is active.

NOTE: Current effective λ for N=1024 is scaled_lambda_safety(1024) ≈ 0.489
(formula: 0.691 × (256/1024)^(1/4)). BASE_D=4 is used in the scaling
denominator — not the actual Fourier encoding D=64. This is a potential
calibration bug that this experiment also illuminates.

CONFIGS (N=1024, D=64, K_iter=8, AH=1.0, 50% data, 75ep)
==========================================================
  Ref  : λ_safety = default (scaled_lambda_safety(1024) ≈ 0.49)  [current baseline]
  A    : λ_safety = 0.0   (safety valve completely off)
  B    : λ_safety = 0.05  (minimal — boundary hint only)
  C    : λ_safety = 0.1   (light repulsion)

KEY QUESTIONS
=============
  A vs Ref  → does removing safety valve break training with AH=1.0?
  B vs Ref  → does a 10x smaller lambda match or beat current default?
  C vs Ref  → intermediate — at what λ does safety become neutral?
  All vs REF_BASELINE (73.53%) → does safety valve affect final ceiling?

Hypothesis: AntiHebb enforces W_pos diversity via cosine suppression, making
position-space repulsion redundant. λ=0 should match λ≈0.49. If confirmed,
simplify all future experiments to λ=0 and remove the O(N²) cdist overhead.

To reproduce:
    python -u scripts/train_step67_safety_ablation.py --device mps
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
    trainer_kwargs, topology_kwargs, run_metadata, scaled_lambda_safety
)
from src.training.dataset              import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS  = 75
BATCH   = 128
SEED    = 42
DATA    = "data/store.h5"
N       = 1024
D       = 64
K_ITER  = 8

# Gen4 calibrated parameters
K_PHASE       = 8
ALPHA_REFLECT = 0.5
BEAM_SIZE     = 16
GEO_GAMMA     = 0.5

DEFAULT_LAMBDA = scaled_lambda_safety(N)   # ≈ 0.489 for N=1024

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


def make_model(seed_offset: int = 0) -> SGNNET_AntiHebbian:
    """Standard Gen4 AH=1.0 model."""
    torch.manual_seed(SEED + seed_offset)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_ITER, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base,
        K_phase=K_PHASE,
        alpha_reflect=ALPHA_REFLECT,
        alpha_turing=0.0,
        beam_size=BEAM_SIZE,
        geo_gamma=GEO_GAMMA,
        mode="dynamic_z_geo",
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=1.0, variant="wpos")


def run(label: str, model: nn.Module, lambda_safety: float, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va = get_loaders()
    tk     = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    # Override lambda_safety with the ablation value
    tk["lambda_safety"] = lambda_safety

    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best      = max(top1_hist)
    best_ep   = int(np.argmax(top1_hist)) + 1
    frac      = best_ep / len(history)
    last5     = history[-5:]

    result = {
        "label":           label,
        "top1_best":       best,
        "top1_last":       history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
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
        f"  diag={result['convergence_diag']}  t={elapsed:.0f}s"
    )
    return result


# ── Load REF_BASELINE ─────────────────────────────────────────────────────────
_ref_path = ROOT / "results" / "train_step57_benchmark_ablation.json"
try:
    REF_BASELINE = json.loads(_ref_path.read_text()).get("top1_best", 0.7353)
    print(f"Loaded REF_BASELINE from step57: {REF_BASELINE:.4f}")
except Exception:
    REF_BASELINE = 0.7353
    print(f"step57 not found — using fallback REF_BASELINE={REF_BASELINE:.4f}")


# ── Configs ───────────────────────────────────────────────────────────────────
# (key, label, lambda_safety, seed_offset)
CONFIGS = [
    (
        "Ref",
        f"Ref   AH=1.0, λ_safety={DEFAULT_LAMBDA:.3f} (default scaled)",
        DEFAULT_LAMBDA,
        0,
        {"lambda_safety": DEFAULT_LAMBDA},
    ),
    (
        "A",
        "A     AH=1.0, λ_safety=0.0  (safety valve OFF)",
        0.0,
        1,
        {"lambda_safety": 0.0},
    ),
    (
        "B",
        "B     AH=1.0, λ_safety=0.05 (minimal)",
        0.05,
        2,
        {"lambda_safety": 0.05},
    ),
    (
        "C",
        "C     AH=1.0, λ_safety=0.1  (light)",
        0.1,
        3,
        {"lambda_safety": 0.1},
    ),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 67: Safety Valve λ Ablation  |  REF_BASELINE={REF_BASELINE:.4f}")
    print(f"Default λ_safety for N={N}: {DEFAULT_LAMBDA:.4f}  (BASE_D=4 scaling)")
    print("Testing whether AH=1.0 makes W_pos repulsion redundant.")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step67_safety_ablation.json"

    for key, label, lam, seed_offset, meta_extra in CONFIGS:
        model = make_model(seed_offset=seed_offset).to(DEVICE)
        meta  = {"N": N, "D": D, "K_iter": K_ITER, "ah_alpha": 1.0,
                 "data_frac": 0.5, **meta_extra}
        results[key] = run(label, model, lam, meta)
        results[key].update(meta_extra)

        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    print(f"\n{'='*70}")
    print(f"STEP 67 COMPLETE  |  REF_BASELINE={REF_BASELINE:.4f}  default_λ={DEFAULT_LAMBDA:.3f}")
    for key, label, _, _, _ in CONFIGS:
        r = results[key]
        delta = r["top1_best"] - REF_BASELINE
        print(f"  {key:4s}  λ={results[key]['lambda_safety']:.3f}  "
              f"{r['top1_best']:.4f}  ({delta:+.4f} vs REF)  ep={r['best_epoch']}/{r['epochs_run']}")
    print()
    print("  Key: does A (λ=0) match Ref? → AH makes safety redundant.")
    print("  Key: does B (λ=0.05) match Ref? → current λ≈0.49 is excessive.")
