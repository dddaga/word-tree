"""Step 80: N-scaling survey on patched arch.

MOTIVATION
==========
step56 established the N-scaling curve (N=512→84.36% at N=4096) but ran on
buggy arch (input coverage gap + alpha_reflect silenced). That entire curve
is invalid as an absolute reference.

This experiment re-establishes the N-scaling curve on the patched arch with
Gen4+ params at the ablation protocol (50% data, 75ep).

REFERENCE POINTS (from other experiments — same arch/params, 50%/75ep):
  N=1024 : step69 Ref / step77 Ref = 83.36%
  N=4096 : step79 Ref = TBD (running)

This script adds N={512, 2048} to fill the curve. N=4096 handled by step79.

CONFIGS (D=64, K_iter=8, Gen4+ params, AH=1.0 wpos, 50%/75ep)
==============================================================
  N=512    — below peak; is accuracy still viable?
  N=2048   — interpolation between 1024 (83.36%) and 4096 (97.32% full run)

KEY QUESTION: does the patched arch preserve the N-scaling law?
step56 (buggy): 512→69.58%, 1024→80.92%, 2048→81.10%, 4096→84.36% (peak), 10000→82.37%
Patched arch expected to show higher absolute values and potentially
different crossover/peak point.

To reproduce:
    python -u scripts/train_step80_nscale_patched.py --device cpu
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

EPOCHS = 75
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
D      = 64

# Gen4+ params (step70 Config B winner: turing=0.0)
K_PHASE       = 8
BEAM_SIZE     = 16
GEO_GAMMA     = 0.5
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
ALPHA_AHEBB   = 1.0

# Reference points from other experiments (50%/75ep, same params)
STEP56_BUGGY = {512: 0.6958, 1024: 0.8092, 2048: 0.8110, 4096: 0.8436}
STEP69_REF   = 0.8336   # N=1024, patched arch

_loaders_cache: dict = {}


def get_loaders(n: int):
    if n not in _loaders_cache:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        total = len(tr_full.dataset)
        idx   = torch.randperm(total, generator=torch.Generator().manual_seed(SEED))[:total // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(
            subset, batch_size=BATCH, shuffle=True, num_workers=0
        )
        _loaders_cache[n] = (tr, va)
    return _loaders_cache[n]


def make_model(n: int, seed_offset: int = 0) -> SGNNET_AntiHebbian:
    torch.manual_seed(SEED + seed_offset)
    tk   = topology_kwargs(n)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=n, N_out=10,
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
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def run(label: str, model: nn.Module, n: int, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders(n)
    tk      = trainer_kwargs(n, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best      = max(top1_hist)
    best_ep   = int(np.argmax(top1_hist)) + 1
    frac      = best_ep / len(history)

    buggy_ref = STEP56_BUGGY.get(n, None)
    delta_buggy = (best - buggy_ref) if buggy_ref else None

    result = {
        "label":             label,
        "N":                 n,
        "top1_best":         best,
        "top1_last":         history[-1].get("val_top1", 0.0),
        "final_task_loss":   float(np.mean([h.get("task_loss", 0.0) for h in history[-5:]])),
        "best_epoch":        best_ep,
        "epochs_run":        len(history),
        "elapsed_s":         round(elapsed, 1),
        "best_epoch_frac":   round(frac, 3),
        "convergence_diag":  "training_too_short" if frac < 0.7 else "converged",
        "top1_history":      top1_hist,
        "step56_buggy_ref":  buggy_ref,
        "delta_vs_buggy":    round(delta_buggy, 4) if delta_buggy is not None else None,
        "_meta":             run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    delta_str = f"  vs_buggy_arch={delta_buggy:+.4f}" if delta_buggy is not None else ""
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  t={elapsed:.0f}s{delta_str}"
    )
    return result


# ── Configs ────────────────────────────────────────────────────────────────────
# (key, N, seed_offset)
CONFIGS = [
    ("N512",  512,  0),
    ("N2048", 2048, 1),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 80: N-scaling survey on patched arch")
    print(f"step56 curve (buggy arch): {STEP56_BUGGY}")
    print(f"step69 Ref N=1024 (patched): {STEP69_REF:.4f}")
    print(f"N=4096 point: see step79 Ref (running)")
    get_loaders(512)
    print(f"Dataset: train={len(_loaders_cache[512][0].dataset)}"
          f"  val={len(_loaders_cache[512][1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step80_nscale_patched.json"

    for key, n, seed_off in CONFIGS:
        label = f"{key}  N={n} patched arch Gen4+ 50%/75ep"
        model = make_model(n=n, seed_offset=seed_off).to(DEVICE)
        meta  = {
            "N": n, "D": D, "K_iter": 8,
            "alpha_ahebb": ALPHA_AHEBB, "alpha_reflect": ALPHA_REFLECT,
            "alpha_turing": ALPHA_TURING, "beam_size": BEAM_SIZE,
            "data_frac": 0.5,
        }
        results[key] = run(label, model, n, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"STEP 80 COMPLETE — N-scaling patched arch")
    print(f"step56 buggy arch reference: {STEP56_BUGGY}")
    print()
    print(f"  {'N':>6}  {'top1':>8}  {'vs_buggy':>10}  {'best_ep':>8}")
    for key, n, _ in CONFIGS:
        if key not in results:
            continue
        r = results[key]
        delta = r.get("delta_vs_buggy")
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        print(f"  {n:>6}  {r['top1_best']:.4f}    {delta_str:>10}  {r['best_epoch']:>8}")
    print()
    print("  Cross-reference with:")
    print(f"  N=1024: step69/step77 Ref = {STEP69_REF:.4f} (patched)")
    print(f"  N=4096: step79 Ref (see results/train_step79_aux_loss_patched.json)")
    print()
    print("  Interpretation:")
    print("  patched > buggy → arch bugs suppressed N-scaling gains")
    print("  monotone increase → N-scaling law holds on correct arch")
    print("  plateau/reversal → find optimal N before scaling to N>4096")
