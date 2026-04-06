"""Step 70: Full-scale Gen4+ corrected run — N=4096, 150ep, 100% data.

MOTIVATION
==========
step69 established REF_BASELINE_v2=83.36% and Gen4+ candidate=85.04%
(Config A: alpha_turing=0.3, reflect=0.5, AH=1.0) at N=1024, 50% data, 75ep.

This script answers: does 85.04% (50%/75ep) translate to a new project best
at full scale (N=4096, 100% data, 150ep)?

Previous full-scale best: 84.36% (step56, N=4096, buggy arch, Gen4 params:
  alpha_turing=0.0, reflect=0.0 (silently), AH=1.0).
Both bugs now fixed (input_coverage_guarantee + alpha_reflect accumulator).
alpha_turing=0.3 restored (masked by bugs previously — step69 A confirms +1.68pp).

CONFIGS (N=4096, D=64, K_iter=8, 100% data, 150ep)
====================================================
  Ref : Gen4+ params (alpha_turing=0.3, alpha_reflect=0.5, AH=1.0) — MAIN RUN
        → Expected: above 84.36% old project best on corrected architecture
  B   : Gen4 baseline (alpha_turing=0.0, alpha_reflect=0.5, AH=1.0)
        → step56-equivalent on patched arch: how much does turing=0.3 add at scale?
        → B vs Ref: alpha_turing contribution at N=4096 (was +1.68pp at N=1024)

KEY QUESTIONS
=============
  Ref vs 84.36%  → net gain from both code fixes at N=4096 full scale
  Ref vs B       → alpha_turing contribution at N=4096 (scale-dependent?)
  Ref vs 85.04%  → does 50%/75ep extrapolate to full scale?

EXPECTED DIRECTION
==================
  Ref > 84.36%: patched arch + turing=0.3 should exceed old project best
  Ref > B:      turing=0.3 adds ~1-2pp at N=1024; scale may amplify or reduce
  Ref ≈ 86-87%: optimistic extrapolation from N-scaling trend on patched arch

To reproduce:
    python -u scripts/train_step70_fullscale_gen4plus.py --device mps
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
    trainer_kwargs, topology_kwargs, run_metadata,
)
from src.training.dataset              import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 150
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
N      = 4096
D      = 64

# Gen4+ routing params — same as step69 Config A (winner)
K_PHASE    = 8
BEAM_SIZE  = 16
GEO_GAMMA  = 0.5

# Comparison points
OLD_PROJECT_BEST = 0.8436   # step56, N=4096, buggy arch
STEP69_A_50PCT   = 0.8504   # step69 Config A, N=1024, 50%/75ep

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_model(
    alpha_reflect: float,
    alpha_turing: float,
    seed_offset: int = 0,
) -> SGNNET_AntiHebbian:
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
        alpha_reflect=alpha_reflect,
        alpha_turing=alpha_turing,
        beam_size=BEAM_SIZE,
        geo_gamma=GEO_GAMMA,
        mode="dynamic_z_geo",
        resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=1.0, variant="wpos")


def run(label: str, model: nn.Module, meta: dict, results: dict, out_path: Path) -> dict:
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
    result.update(meta)

    delta_old  = best - OLD_PROJECT_BEST
    delta_s69a = best - STEP69_A_50PCT
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  vs_old_best={delta_old:+.4f}  vs_step69A={delta_s69a:+.4f}  t={elapsed:.0f}s"
    )

    # Save after each config so partial results survive interruption
    results[meta["key"]] = result
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  [saved {out_path.name}]")
    return result


# ── Configs ───────────────────────────────────────────────────────────────────
CONFIGS = [
    {
        "key": "Ref",
        "label": "Ref  Gen4+ patched N=4096 (turing=0.3 reflect=0.5 AH=1.0) — MAIN RUN",
        "alpha_reflect": 0.5,
        "alpha_turing":  0.3,
        "seed_offset":   0,
    },
    {
        "key": "B",
        "label": "B    Gen4 baseline patched N=4096 (turing=0.0 reflect=0.5 AH=1.0)",
        "alpha_reflect": 0.5,
        "alpha_turing":  0.0,
        "seed_offset":   1,
    },
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 100%  N: {N}")
    print(f"Step 70: Full-scale Gen4+ corrected run")
    print(f"Old project best (step56, buggy arch): {OLD_PROJECT_BEST:.4f}")
    print(f"step69 Config A (N=1024, 50%/75ep):    {STEP69_A_50PCT:.4f}")
    print(f"Architecture fixes: input_coverage_guarantee + alpha_reflect_in_antihebb")

    tr, va = get_loaders()
    print(f"Dataset: train={len(tr.dataset)}  val={len(va.dataset)}")

    tk = topology_kwargs(N)
    print(f"Topology: N={N} K_local={tk['K_local']} K_random={tk['K_random']} "
          f"K_in={tk['K_in']} n_groups={tk['n_groups']}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step70_fullscale_gen4plus.json"

    for cfg in CONFIGS:
        model = make_model(
            alpha_reflect=cfg["alpha_reflect"],
            alpha_turing=cfg["alpha_turing"],
            seed_offset=cfg["seed_offset"],
        ).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  Config {cfg['key']}: {n_params:,} trainable params")

        meta = {
            "key":           cfg["key"],
            "N":             N, "D": D, "K_iter": 8,
            "alpha_ahebb":   1.0, "variant": "wpos",
            "alpha_reflect": cfg["alpha_reflect"],
            "alpha_turing":  cfg["alpha_turing"],
            "beam_size":     BEAM_SIZE, "geo_gamma": GEO_GAMMA,
            "data_frac":     1.0,
            "fixes_applied": "input_coverage_guarantee + alpha_reflect_in_antihebb",
        }
        run(cfg["label"], model, meta, results, out_path)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"STEP 70 COMPLETE")
    print(f"Old project best (step56, buggy):    {OLD_PROJECT_BEST:.4f}")
    print(f"step69 Config A (N=1024, 50%/75ep):  {STEP69_A_50PCT:.4f}")
    print()

    print(f"  {'Key':4s}  {'top1':>8s}  {'vs old best':>12s}  {'vs step69A':>12s}  Description")
    for cfg in CONFIGS:
        key = cfg["key"]
        if key not in results:
            continue
        r          = results[key]
        d_old      = r["top1_best"] - OLD_PROJECT_BEST
        d_s69a     = r["top1_best"] - STEP69_A_50PCT
        short_desc = cfg["label"].split("  ")[0].strip()
        print(f"  {key:4s}  {r['top1_best']:.4f}    {d_old:+.4f}        {d_s69a:+.4f}      {short_desc}")

    if "Ref" in results and "B" in results:
        turing_gain = results["Ref"]["top1_best"] - results["B"]["top1_best"]
        print(f"\n  alpha_turing=0.3 contribution at N=4096 (Ref - B): {turing_gain:+.4f}")
        print(f"  (at N=1024 this was +1.68pp in step69)")

    if "Ref" in results:
        ref = results["Ref"]["top1_best"]
        verdict = "NEW PROJECT BEST" if ref > OLD_PROJECT_BEST else f"below old best by {OLD_PROJECT_BEST - ref:+.4f}"
        print(f"\n  VERDICT: {ref:.4f} → {verdict}")
