"""Step 77: Learnable theta — fix frozen threshold bug.

MOTIVATION
==========
theta is an nn.Parameter in SGNNET_Resonant (shape: [N,], init=0.1).
It gates every routing step:  Z_fwd = relu(Z - |theta|)
But the Trainer only adds W_pos (and optionally W_phase) to the optimizer.
theta was NEVER added → it has been frozen at 0.1 through ALL experiments to date.

Fix: after Trainer construction, inject theta as an additional param_group:
    trainer.optimizer.add_param_group({"params": [model.m.theta], "lr": lr_theta})

This experiment sweeps theta LR to find the optimal scale.
Comparison baseline: step69 Ref = 83.36% (theta frozen, same arch + params).

CONFIGS (N=1024, D=64, K_iter=8, Gen4+ params, 50% data, 75ep)
===============================================================
  Ref  : theta frozen at 0.1 (reproduces step69 Ref — control)
  A    : theta learns at lr_wpos (2.364e-3) — same LR as spatial positions
  B    : theta learns at 0.1 × lr_wpos (2.364e-4) — conservative
  C    : theta learns at 10 × lr_wpos (2.364e-2) — aggressive

EXPECTED DIRECTION
==================
  A/B/C vs Ref: any gain confirms per-neuron thresholding is underutilised.
  LR sensitivity tells us how quickly theta wants to move.
  If C is best: theta adapts quickly → current 0.1 is far from optimal per-neuron.
  If B is best: theta should update slowly → smooth threshold landscape.

To reproduce:
    python -u scripts/train_step77_learnable_theta.py --device mps
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

# Gen4+ params (step70 Config B winner: turing=0.0)
K_PHASE   = 8
BEAM_SIZE = 16
GEO_GAMMA = 0.5
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0

LR_WPOS   = GA_BEST["lr_Wpos"]   # 2.364e-3

# Comparison baseline
STEP69_REF = 0.8336   # theta frozen, same arch, 50%/75ep

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
    torch.manual_seed(SEED + seed_offset)
    tk = topology_kwargs(N)
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
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=1.0, variant="wpos")


def run(label: str, model: nn.Module, lr_theta: float | None, meta: dict) -> dict:
    """Train model. If lr_theta is None, theta stays frozen (control)."""
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)

    # Inject theta into optimizer if this config uses learnable theta.
    # theta lives at model.m.theta (SGNNET_Resonant inside SGNNET_AntiHebbian).
    # weight_decay=0: threshold should explore freely, not shrink toward 0.
    if lr_theta is not None:
        trainer.optimizer.add_param_group({
            "params": [model.m.theta],
            "lr": lr_theta,
            "weight_decay": 0.0,
        })
        theta_init_mean = model.m.theta.mean().item()
        print(f"  theta: LEARNABLE at lr={lr_theta:.3e}  init_mean={theta_init_mean:.4f}")
    else:
        print(f"  theta: FROZEN at {model.m.theta[0].item():.4f} (control)")

    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best      = max(top1_hist)
    best_ep   = int(np.argmax(top1_hist)) + 1
    frac      = best_ep / len(history)

    # Record final theta statistics
    theta_final = model.m.theta.detach().cpu()
    theta_stats = {
        "theta_final_mean": round(float(theta_final.mean()), 4),
        "theta_final_std":  round(float(theta_final.std()), 4),
        "theta_final_min":  round(float(theta_final.min()), 4),
        "theta_final_max":  round(float(theta_final.max()), 4),
    }

    result = {
        "label":            label,
        "lr_theta":         lr_theta,
        "top1_best":        best,
        "top1_last":        history[-1].get("val_top1", 0.0),
        "final_task_loss":  float(np.mean([h.get("task_loss", 0.0) for h in history[-5:]])),
        "best_epoch":       best_ep,
        "epochs_run":       len(history),
        "elapsed_s":        round(elapsed, 1),
        "best_epoch_frac":  round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history":     top1_hist,
        **theta_stats,
        "_meta":            run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    delta = best - STEP69_REF
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  vs_step69_ref={delta:+.4f}  t={elapsed:.0f}s"
    )
    print(f"  theta: mean={theta_stats['theta_final_mean']:.4f}  "
          f"std={theta_stats['theta_final_std']:.4f}  "
          f"range=[{theta_stats['theta_final_min']:.4f}, {theta_stats['theta_final_max']:.4f}]")
    return result


# ── Configs ────────────────────────────────────────────────────────────────────
# (key, label, lr_theta, seed_offset)
CONFIGS = [
    (
        "Ref",
        "Ref   theta frozen=0.1 (control — reproduces step69 Ref)",
        None, 0,
    ),
    (
        "A",
        f"A     theta learns lr={LR_WPOS:.3e} (= lr_wpos)",
        LR_WPOS, 1,
    ),
    (
        "B",
        f"B     theta learns lr={LR_WPOS * 0.1:.3e} (0.1 × lr_wpos — conservative)",
        LR_WPOS * 0.1, 2,
    ),
    (
        "C",
        f"C     theta learns lr={LR_WPOS * 10:.3e} (10 × lr_wpos — aggressive)",
        LR_WPOS * 10.0, 3,
    ),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 77: Learnable theta — theta never updated in any prior experiment")
    print(f"Comparison: step69 Ref = {STEP69_REF:.4f} (theta frozen, same arch)")
    print(f"lr_wpos = {LR_WPOS:.3e}")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step77_learnable_theta.json"

    for key, label, lr_theta, seed_off in CONFIGS:
        model = make_model(seed_offset=seed_off).to(DEVICE)
        meta = {
            "N": N, "D": D, "K_iter": 8,
            "alpha_ahebb": 1.0, "variant": "wpos",
            "alpha_reflect": ALPHA_REFLECT,
            "alpha_turing":  ALPHA_TURING,
            "beam_size": BEAM_SIZE, "geo_gamma": GEO_GAMMA,
            "lr_theta": lr_theta,
            "theta_init": 0.1,
            "data_frac": 0.5,
        }
        results[key] = run(label, model, lr_theta, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"STEP 77 COMPLETE — Learnable theta sweep")
    print(f"Comparison: step69 Ref (frozen theta) = {STEP69_REF:.4f}")
    print()
    print(f"  {'Key':4s}  {'top1':>8s}  {'vs step69':>10s}  {'lr_theta':>12s}  "
          f"{'θ_mean':>8s}  {'θ_std':>7s}")
    for key, label, *_ in CONFIGS:
        if key not in results:
            continue
        r     = results[key]
        delta = r["top1_best"] - STEP69_REF
        lr_str = f"{r['lr_theta']:.2e}" if r["lr_theta"] else "frozen"
        print(f"  {key:4s}  {r['top1_best']:.4f}    {delta:+.4f}    {lr_str:>12s}  "
              f"{r['theta_final_mean']:>8.4f}  {r['theta_final_std']:>7.4f}")
