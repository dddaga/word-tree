"""Step 89 (G1): K_iter=12 + K_hh=4 full-scale validation at N=4096.

MOTIVATION
==========
Three confirmed wins from calibration runs combine:
  - K_iter=12: +0.79pp over K_iter=8 (step71 C, 50%/75ep)
  - K_hh=4:    +0.56pp over K_hh=6  (step86 A, 50%/75ep, −18% FLOPs)
  - turing=0.0: already default (step70 B confirmed)

This is the first full-scale (100%/150ep) run with K_hh=4 as default.
Compounding K_iter=12 + K_hh=4 is the primary hypothesis: independent gains
should compound (expected ~97.5%+ vs step70 B = 97.38%).

Also tests whether K_iter=12 advantage persists at K_hh=4 by including
a K_iter=8 + K_hh=4 control (same as step86 A at full scale).

CONFIGS (N=4096, D=64, 100%/150ep)
====================================
  Ref : K_iter=8,  K_hh=4  — step86 A at full scale (control)
  A   : K_iter=12, K_hh=4  — compound winner (expected best)
  B   : K_iter=12, K_hh=6  — isolate K_iter=12 effect at K_hh=6

All: turing=0.0, reflect=0.5, AH=1.0, n_groups=512, K_in=50, D=64.

FLOPs:
  Ref (K_iter=8,  K_hh=4): 38.8M  (32.4% of VGG16 FC)
  A   (K_iter=12, K_hh=4): 54.6M  (45.7% of VGG16 FC)
  B   (K_iter=12, K_hh=6): 70.6M  (59.0% of VGG16 FC)

To reproduce:
    python -u scripts/train_step89_g1_kiter12_khh4.py --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass

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

EPOCHS    = 150
BATCH     = 128
SEED      = 42
DATA      = "data/store.h5"
N         = 4096
N_IN      = 25088
N_OUT     = 10
D         = 64
K_IN      = 50
K_RANDOM  = 2
K_PHASE   = 8
BEAM_SIZE = 16
GEO_GAMMA = 0.5
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
ALPHA_AHEBB   = 1.0
N_GROUPS      = max(8, N // 8)   # = 512

# Baselines
STEP70B_FULL = 0.9738   # step70 Config B: N=4096, K_iter=8, K_hh=6, 100%/150ep
STEP86A_50   = 0.9659   # step86 A: K_hh=4, K_iter=8, 50%/75ep calibration


@dataclass
class Config:
    key:     str
    label:   str
    K_local: int
    K_iter:  int


CONFIGS = [
    Config("Ref", "Ref  K_iter=8  K_hh=4  (step86-A at full scale, control)", 2, 8),
    Config("A",   "A    K_iter=12 K_hh=4  (compound winner — primary hypothesis)", 2, 12),
    Config("B",   "B    K_iter=12 K_hh=6  (isolate K_iter=12 at K_hh=6)", 4, 12),
]


_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def compute_flops(K_local: int, K_iter: int) -> int:
    K_hh    = K_local + K_RANDOM
    seed    = N * K_IN * D
    step    = N * K_hh * D * 2 + N * D + N * D * 2
    routing = K_iter * step
    readout = N * N_OUT * D
    return seed + routing + readout


def make_model(cfg: Config, seed_offset: int = 0) -> SGNNET_AntiHebbian:
    torch.manual_seed(SEED + seed_offset)
    base = SGNNET_SmallWorld(
        N_in=N_IN, N_hidden=N, N_out=N_OUT,
        K_local=cfg.K_local, K_random=K_RANDOM,
        K_in=K_IN, K_iter=cfg.K_iter,
        n_groups=N_GROUPS,
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=K_PHASE, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=BEAM_SIZE,
        geo_gamma=GEO_GAMMA, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(cfg: Config, model: nn.Module, meta: dict) -> dict:
    K_hh  = cfg.K_local + K_RANDOM
    flops = compute_flops(cfg.K_local, cfg.K_iter)

    print(f"\n{'='*70}")
    print(f"{cfg.label}")
    print(f"  K_hh={K_hh}  K_iter={cfg.K_iter}  FLOPs={flops/1e6:.1f}M"
          f"  vs_VGG16={flops/119_578_624:.2%}")
    print(f"{'='*70}")

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
        "label":            cfg.label,
        "K_hh":             K_hh,
        "K_iter":           cfg.K_iter,
        "top1_best":        best,
        "top1_last":        history[-1].get("val_top1", 0.0),
        "final_task_loss":  float(np.mean([h.get("task_loss", 0.0) for h in history[-5:]])),
        "best_epoch":       best_ep,
        "epochs_run":       len(history),
        "elapsed_s":        round(elapsed, 1),
        "best_epoch_frac":  round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history":     top1_hist,
        "step70b_full":     STEP70B_FULL,
        "delta_vs_step70b": round(best - STEP70B_FULL, 4),
        "flops_per_sample": flops,
        "flops_M":          round(flops / 1e6, 2),
        "flops_vs_vgg16":   round(flops / 119_578_624, 3),
        "params":           count_params(model),
        "_meta":            run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  vs_step70b={best-STEP70B_FULL:+.4f}  t={elapsed:.0f}s"
    )
    return result


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  N={N}  Data: 100%")
    print(f"Step 89 (G1): K_iter=12 + K_hh=4 full-scale compound validation")
    print(f"n_groups={N_GROUPS}  D={D}  turing={ALPHA_TURING}  reflect={ALPHA_REFLECT}")
    print(f"step70 B baseline (100%/150ep): {STEP70B_FULL:.4f}")
    print(f"step86 A calibration (50%/75ep): {STEP86A_50:.4f}")
    print()
    for cfg in CONFIGS:
        K_hh  = cfg.K_local + K_RANDOM
        flops = compute_flops(cfg.K_local, cfg.K_iter)
        print(f"  {cfg.key:4s}  K_hh={K_hh}  K_iter={cfg.K_iter}  "
              f"FLOPs={flops/1e6:.1f}M  {cfg.label}")
    print()

    tr, va = get_loaders()
    print(f"Dataset: train={len(tr.dataset)}  val={len(va.dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step89_g1_kiter12_khh4.json"

    for i, cfg in enumerate(CONFIGS):
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        meta  = {
            "N": N, "D": D, "K_iter": cfg.K_iter,
            "K_hh": cfg.K_local + K_RANDOM, "K_local": cfg.K_local,
            "K_random": K_RANDOM, "K_in": K_IN,
            "n_groups": N_GROUPS, "data_frac": 1.0,
            "alpha_turing": ALPHA_TURING, "alpha_ahebb": ALPHA_AHEBB,
        }
        results[cfg.key] = run(cfg, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    print(f"\n{'='*70}")
    print(f"STEP 89 G1 COMPLETE — K_iter=12 + K_hh=4 Full-Scale Validation")
    print(f"step70 B baseline: {STEP70B_FULL:.4f}")
    print()
    print(f"  {'Key':4s}  {'K_hh':>5s}  {'K_iter':>6s}  {'top1':>8s}  {'vs_70b':>8s}  {'FLOPs':>8s}")
    for cfg in CONFIGS:
        if cfg.key not in results:
            continue
        r     = results[cfg.key]
        flops = compute_flops(cfg.K_local, cfg.K_iter)
        print(f"  {cfg.key:4s}  {cfg.K_local+K_RANDOM:>5d}  {cfg.K_iter:>6d}  "
              f"{r['top1_best']:.4f}    {r['delta_vs_step70b']:+.4f}  {flops/1e6:>7.1f}M")
    winner = max(results, key=lambda k: results[k]["top1_best"])
    print(f"\n  Winner: {winner}")
    if winner == "A":
        print("  → K_iter=12 + K_hh=4 compound gain confirmed. New project best candidate.")
    elif winner == "Ref":
        print("  → K_iter=12 gain does not compound with K_hh=4. K_hh=4 is still the FLOPs winner.")
