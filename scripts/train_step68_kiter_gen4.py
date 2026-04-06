"""Step 68: K_iter scaling at Gen4 (AH=1.0 base, 50% data, 75ep).

MOTIVATION
==========
step48 (K_iter sweep at D=64) was SLOT-KILLED: only Ref(K_iter=8) and A(K_iter=12)
ran. The critical question — does K_iter>8 WITH AH=1.0 help? — was never answered.

Key finding from step48: K_iter=12 WITHOUT AH = 55.08% (vs Ref=58.24%) — over-smoothing.
But AntiHebb inhibition suppresses similar-direction neighbors, which should prevent
the collapse that causes over-smoothing. At Gen4 (AH=1.0), deeper routing may be viable.

CONFIGS (N=1024, D=64, AH=1.0, 50% data, 75ep)
================================================
  Ref  : K_iter=8  AH=1.0  [replicate REF_BASELINE ~73.53%]
  A    : K_iter=10 AH=1.0  [shallow extension]
  B    : K_iter=12 AH=1.0  [where D=64 no-AH collapsed]
  C    : K_iter=16 AH=1.0  [moderate depth]
  D    : K_iter=24 AH=1.0  [deep]

KEY QUESTIONS
=============
  A vs Ref  → does K_iter=10 help with AH?
  B vs Ref  → does AH prevent the collapse seen at K_iter=12 no-AH (55.08%)?
  C vs B    → is there continued gain above K_iter=12 with AH?
  D vs C    → over-smoothing threshold with AH?

To reproduce:
    python -u scripts/train_step68_kiter_gen4.py --device mps
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

# Gen4 calibrated parameters (step22b + step29c)
K_PHASE       = 8
ALPHA_REFLECT = 0.5
BEAM_SIZE     = 16
GEO_GAMMA     = 0.5

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


def make_model(k_iter: int, seed_offset: int = 0) -> SGNNET_AntiHebbian:
    torch.manual_seed(SEED + seed_offset)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=k_iter, n_groups=tk["n_groups"],
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
        "_meta":            run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  t={elapsed:.0f}s"
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


# (key, label, k_iter, seed_offset)
CONFIGS = [
    ("Ref", "Ref   K_iter=8  AH=1.0 (REF_BASELINE)",  8,  0),
    ("A",   "A     K_iter=10 AH=1.0",                 10,  1),
    ("B",   "B     K_iter=12 AH=1.0",                 12,  2),
    ("C",   "C     K_iter=16 AH=1.0",                 16,  3),
    ("D",   "D     K_iter=24 AH=1.0",                 24,  4),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 68: K_iter scaling at Gen4  |  REF_BASELINE={REF_BASELINE:.4f}")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step68_kiter_gen4.json"

    for key, label, k_iter, seed_off in CONFIGS:
        model = make_model(k_iter=k_iter, seed_offset=seed_off).to(DEVICE)
        meta  = {"N": N, "D": D, "K_iter": k_iter, "ah_alpha": 1.0, "data_frac": 0.5}
        results[key] = run(label, model, meta)
        results[key]["K_iter"] = k_iter

        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    print(f"\n{'='*70}")
    print(f"STEP 68 COMPLETE  |  REF_BASELINE={REF_BASELINE:.4f}")
    for key, label, k_iter, _ in CONFIGS:
        r = results[key]
        delta = r["top1_best"] - REF_BASELINE
        print(f"  {key:4s}  K_iter={k_iter:2d}  {r['top1_best']:.4f}  ({delta:+.4f} vs REF)"
              f"  ep={r['best_epoch']}/{r['epochs_run']}")
    print()
    print("  Key: does B (K_iter=12+AH) beat Ref? → AH prevents K_iter=12 collapse.")
    print("  Key: does C (K_iter=16+AH) continue gain? → depth scaling law.")
