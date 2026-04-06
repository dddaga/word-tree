"""Step 69: Corrected REF_BASELINE on fully-patched architecture.

MOTIVATION
==========
The REF_BASELINE (73.53%, step57) was obtained on code with two bugs:
  1. Input coverage gap: _build_fanin_conn used random per-neuron sampling
     → ~13% of VGG16 features (25088 inputs) never reached any hidden neuron
     at N=1024. Fixed with group round-robin guarantee (100% coverage).
  2. alpha_reflect dropped in AntiHebbian: SGNNET_AntiHebbian.forward()
     reimplemented the routing loop without Z_reflected accumulator.
     alpha_reflect=0.5 was stored in self.m but never applied across K_iter.
     All Gen4 results (73.53% through 84.36%) were obtained without reflection.

Both fixes are now in the codebase. This script re-establishes a valid baseline
and ablates each fix's contribution.

CONFIGS (N=1024, D=64, K_iter=8, 50% data, 75ep)
==================================================
  Ref  : Gen4 params (alpha_turing=0.0, alpha_reflect=0.5, AH=1.0)
         → NEW REF_BASELINE_v2 for all future wave-2 comparisons
  A    : step57-exact params (alpha_turing=0.3, alpha_reflect=0.5, AH=1.0)
         → how much did both code fixes change step57-identical setup?
         compare against old REF_BASELINE=73.53%
  B    : Gen4 params but alpha_reflect=0.0
         → ablate reflection fix: how much does Z_reflected add?
         B vs Ref → reflection contribution on patched code

KEY QUESTIONS
=============
  Ref vs 73.53%  → net gain from both code fixes (Gen4 params)
  A   vs 73.53%  → net gain from both code fixes (step57 params)
  Ref vs B       → alpha_reflect contribution (isolated on patched code)

EXPECTED DIRECTION
==================
  Ref > 73.53%: input coverage fix adds previously-dropped signal;
                reflection fix enables leaky self-inhibition memory
  Ref > B:      reflection should add ~2-5pp (calibrated +5pp at step22b
                diagnostic, though that was D=16 K_iter=3)

To reproduce:
    python -u scripts/train_step69_corrected_baseline.py --device mps
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

EPOCHS = 75
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
N      = 1024
D      = 64

# Gen4 routing params (step22b + step29c)
K_PHASE    = 8
BEAM_SIZE  = 16
GEO_GAMMA  = 0.5

# Old step57 REF_BASELINE (on buggy code) — all configs compare against this
OLD_REF_BASELINE = 0.7353

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
    delta_old = best - OLD_REF_BASELINE
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  vs_old_ref={delta_old:+.4f}  t={elapsed:.0f}s"
    )
    return result


# ── Configs ───────────────────────────────────────────────────────────────────
# (key, label, alpha_reflect, alpha_turing, seed_offset)
CONFIGS = [
    (
        "Ref",
        "Ref   Gen4 patched (turing=0.0 reflect=0.5 AH=1.0) — NEW REF_BASELINE_v2",
        0.5, 0.0, 0,
    ),
    (
        "A",
        "A     step57-exact patched (turing=0.3 reflect=0.5 AH=1.0)",
        0.5, 0.3, 1,
    ),
    (
        "B",
        "B     Gen4 patched, reflect=0.0  [ablate reflection]",
        0.0, 0.0, 2,
    ),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 69: Corrected REF_BASELINE on fully-patched architecture")
    print(f"Old REF_BASELINE (step57, buggy code): {OLD_REF_BASELINE:.4f}")
    print(f"Fixes applied: input coverage guarantee + alpha_reflect in AntiHebbian")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step69_corrected_baseline.json"

    for key, label, alpha_reflect, alpha_turing, seed_off in CONFIGS:
        model = make_model(
            alpha_reflect=alpha_reflect,
            alpha_turing=alpha_turing,
            seed_offset=seed_off,
        ).to(DEVICE)
        meta = {
            "N": N, "D": D, "K_iter": 8,
            "alpha_ahebb": 1.0, "variant": "wpos",
            "alpha_reflect": alpha_reflect,
            "alpha_turing": alpha_turing,
            "beam_size": BEAM_SIZE, "geo_gamma": GEO_GAMMA,
            "data_frac": 0.5,
            "fixes_applied": "input_coverage_guarantee + alpha_reflect_in_antihebb",
        }
        results[key] = run(label, model, meta)
        results[key].update({
            "alpha_reflect": alpha_reflect,
            "alpha_turing":  alpha_turing,
        })

        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"STEP 69 COMPLETE")
    print(f"Old REF_BASELINE (step57, buggy): {OLD_REF_BASELINE:.4f}")
    print()

    ref_top1 = results["Ref"]["top1_best"]
    print(f"  {'Key':4s}  {'top1':>8s}  {'vs old REF':>12s}  Description")
    for key, label, *_ in CONFIGS:
        r     = results[key]
        delta = r["top1_best"] - OLD_REF_BASELINE
        print(f"  {key:4s}  {r['top1_best']:.4f}    {delta:+.4f}      {label.split('  ')[0].strip()}")

    if "B" in results:
        reflect_gain = ref_top1 - results["B"]["top1_best"]
        print(f"\n  Reflection contribution (Ref - B): {reflect_gain:+.4f}")

    if "A" in results:
        a_vs_old = results["A"]["top1_best"] - OLD_REF_BASELINE
        print(f"  Code-fix gain on step57 params (A - old): {a_vs_old:+.4f}")

    print(f"\n  NEW REF_BASELINE_v2 (use for all wave-2+ comparisons): {ref_top1:.4f}")
    print()
    print("  Key: Ref > 73.53% → code fixes improve performance")
    print("  Key: Ref > B      → reflection accumulator adds real gain")
    print("  Key: A  vs 73.53% → combined code-fix effect on step57-identical params")
