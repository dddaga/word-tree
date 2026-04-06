"""Step 88: AH alpha recalibration sweep at N=4096, patched arch.

MOTIVATION
==========
alpha_ahebb=1.0 was calibrated in step29c on buggy arch at N=1024.
The patched arch (+9.83pp) and full scale (N=4096) may shift the optimal alpha.
Step86 Ref confirmed n_groups=512 gives 96.03% — now find the best alpha at this scale.

DESIGN
======
4 configs, 40ep calibration (direction-finding only, not final).
Base: N=4096, D=64, K_iter=8, K_hh=6, turing=0.0, reflect=0.5,
      n_groups=512, 50% data.

  Ref : alpha=0.5
  A   : alpha=1.0  (current default — expected winner)
  B   : alpha=1.5
  C   : alpha=2.0

Decision rule: winner with highest top1_best at 40ep → adopt for all future
N=4096 experiments. If A (1.0) wins — confirmed, no change. If not — recalibrate.

FLOPs (K_hh=6, K_iter=8, D=64): same as step86 Ref = 47.2M FLOPs/sample.

To reproduce:
    python -u scripts/train_step88_alpha_sweep_n4096.py --device cpu
    python -u scripts/train_step88_alpha_sweep_n4096.py --device mps
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
from src.training.experiment_config    import trainer_kwargs, run_metadata
from src.training.dataset              import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS    = 40
BATCH     = 128
SEED      = 42
DATA      = "data/store.h5"
N         = 4096
N_IN      = 25088
N_OUT     = 10
D         = 64
K_IN      = 50
K_LOCAL   = 4
K_RANDOM  = 2
K_ITER    = 8
K_PHASE   = 8
BEAM_SIZE = 16
GEO_GAMMA = 0.5
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
N_GROUPS      = max(8, N // 8)   # = 512 — matches step71/step86 topology_kwargs(N)

# Reference: step86 Ref (N=4096, n_groups=512, 50%/75ep, correct run)
STEP86_REF = 0.9603

ALPHAS = {
    "Ref": 0.5,
    "A":   1.0,   # current default
    "B":   1.5,
    "C":   2.0,
}

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


def make_model(alpha: float, seed_offset: int = 0) -> SGNNET_AntiHebbian:
    torch.manual_seed(SEED + seed_offset)
    base = SGNNET_SmallWorld(
        N_in=N_IN, N_hidden=N, N_out=N_OUT,
        K_local=K_LOCAL, K_random=K_RANDOM,
        K_in=K_IN, K_iter=K_ITER,
        n_groups=N_GROUPS,
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=K_PHASE, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=BEAM_SIZE,
        geo_gamma=GEO_GAMMA, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=alpha, variant="wpos")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(key: str, alpha: float, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"{key}  alpha_ahebb={alpha}  (N={N}, D={D}, K_iter={K_ITER})")
    print(f"{'='*60}")

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
        "label":              f"{key}  alpha={alpha}",
        "alpha_ahebb":        alpha,
        "top1_best":          best,
        "top1_last":          history[-1].get("val_top1", 0.0),
        "final_task_loss":    float(np.mean([h.get("task_loss", 0.0) for h in history[-5:]])),
        "best_epoch":         best_ep,
        "epochs_run":         len(history),
        "elapsed_s":          round(elapsed, 1),
        "best_epoch_frac":    round(frac, 3),
        "convergence_diag":   "training_too_short" if frac < 0.7 else "converged",
        "top1_history":       top1_hist,
        "step86_ref":         STEP86_REF,
        "delta_vs_step86_ref":round(best - STEP86_REF, 4),
        "params":             count_params(model),
        "_meta":              run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  vs_step86_ref={best-STEP86_REF:+.4f}  t={elapsed:.0f}s"
    )
    return result


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  N={N}  Data: 50%")
    print(f"Step 88: AH alpha recalibration sweep at N=4096")
    print(f"n_groups={N_GROUPS}  K_iter={K_ITER}  D={D}  turing={ALPHA_TURING}")
    print(f"Step86 Ref (baseline): {STEP86_REF:.4f}")
    print()
    print(f"  {'Key':4s}  {'alpha':>6s}  Description")
    for k, a in ALPHAS.items():
        note = " ← current default" if k == "A" else ""
        print(f"  {k:4s}  {a:>6.2f}  alpha_ahebb={a}{note}")
    print()

    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step88_alpha_sweep_n4096.json"

    for i, (key, alpha) in enumerate(ALPHAS.items()):
        model  = make_model(alpha, seed_offset=i).to(DEVICE)
        meta   = {
            "N": N, "D": D, "K_iter": K_ITER,
            "K_hh": K_LOCAL + K_RANDOM, "n_groups": N_GROUPS,
            "alpha_ahebb": alpha, "alpha_turing": ALPHA_TURING,
            "data_frac": 0.5,
        }
        results[key] = run(key, alpha, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    print(f"\n{'='*60}")
    print(f"STEP 88 COMPLETE — AH Alpha Recalibration at N=4096")
    print(f"Step86 Ref: {STEP86_REF:.4f}")
    print()
    print(f"  {'Key':4s}  {'alpha':>6s}  {'top1':>8s}  {'vs_ref':>8s}  {'ep':>5s}")
    for key, alpha in ALPHAS.items():
        if key not in results:
            continue
        r = results[key]
        print(f"  {key:4s}  {alpha:>6.2f}  {r['top1_best']:.4f}    "
              f"{r['delta_vs_step86_ref']:+.4f}  {r['best_epoch']:>3d}/{r['epochs_run']}")
    winner = max(results, key=lambda k: results[k]["top1_best"])
    print(f"\n  Winner: {winner}  alpha={ALPHAS[winner]:.2f}")
    if winner == "A":
        print("  → alpha=1.0 confirmed optimal at N=4096 patched arch. No change needed.")
    else:
        print(f"  → RECALIBRATE: adopt alpha={ALPHAS[winner]:.2f} for all future N=4096 experiments.")
