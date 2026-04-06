"""Step 33b: D=128 calibration — find viable LR/schedule before full D=128 run.

ROOT CAUSE OF STEP33 FAILURE
==============================
step33 Config A (D=128 N=512) stuck at ~10% (random baseline) through e40:
  - Fourier encoding at D=128 uses 21 freq pairs per axis → very high-frequency
    spatial features; cosine similarities between seeds concentrate near 0 on S^127
  - Plateau scheduler (patience=10) halves LR if no improvement in 10 epochs
  - If D=128 needs >10 epochs to find signal, LR collapses before learning starts:
      LR: 2.36e-3 → 1.18e-3 (e20) → 5.91e-4 (e40) → death spiral

FIX STRATEGY
=============
1. Use COSINE schedule (not plateau) — avoids premature LR collapse
2. Sweep LR: {5e-4, 1e-3, 2e-3, 5e-3} — find the right learning rate at D=128
3. Sweep N: {256, 512, 1024} at best LR — find minimum viable network size
4. Phase 2 full runs only if D=128 reaches >=20% at 40ep (viable threshold)
5. D=64 N=2048 always runs — provides N-scaling data independent of D=128 result

PHASE 1 (CALIB_EPOCHS=40, cosine schedule):
  LR sweep:  D=128 N=512, lr ∈ {5e-4, 1e-3, 2e-3, 5e-3}  →  4 runs
  N sweep:   D=128 K_iter=8, N ∈ {256, 512, 1024}          →  3 runs

PHASE 2 (FULL_EPOCHS=150, cosine schedule):
  Ref              D=64  N=1024 K_iter=8  [confirms ≈56.28% ceiling]
  D64_N2048        D=64  N=2048 K_iter=8  [N-scaling question from step33]
  D128_best        D=128 N=best_N, best_lr  [only if viable]
  D128_N1024       D=128 N=1024, best_lr   [only if viable and best_N != 1024]

To reproduce:
    python -u scripts/train_step33b_d128_calib.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

CALIB_EPOCHS     = 40
FULL_EPOCHS      = 150
BATCH            = 128
SEED             = 42
DATA             = "data/store.h5"
VIABLE_THRESHOLD = 0.20   # D=128 must reach 20% at 40ep to proceed to full runs

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_resonant(N: int, D: int, K_iter: int = 8) -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_iter,
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )


def run(label: str, model: nn.Module, meta: dict, n_epochs: int,
        lr: float | None = None) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    # Always use cosine schedule — plateau kills D=128 before it finds signal
    tk      = trainer_kwargs(meta["N"], n_epochs=n_epochs,
                             sched_type="cosine", lr_wpos=lr)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=n_epochs)
    elapsed = time.time() - t0
    best    = max(h.get("val_top1", 0.0) for h in history)
    last5   = history[-5:]
    best_ep = int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1
    frac    = best_ep / len(history)
    result  = {
        "label": label, "top1_best": best,
        "top1_last": history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "best_epoch_frac": round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history": [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta": run_metadata(__file__, {**meta, "epochs": n_epochs, "lr": lr}),
    }
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
          f"  diag={result['convergence_diag']}  t={elapsed:.0f}s")
    return result


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Calib: {CALIB_EPOCHS}ep  Full: {FULL_EPOCHS}ep")
    print("Step 33b: D=128 calibration — root cause: plateau LR collapse")
    print("Fix: cosine schedule + LR sweep to prevent death spiral")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {"calibration": {}, "full": {}, "best_params": {}}

    # ── Phase 1a: LR sweep — D=128 N=512, cosine, 40ep ───────────────────────
    print(f"\n{'#'*70}")
    print("PHASE 1a: LR sweep — D=128 N=512 K_iter=8, cosine, 40ep")
    print(f"{'#'*70}")

    LR_SWEEP = [5e-4, 1e-3, 2e-3, 5e-3]
    lr_results = {}
    for lr in LR_SWEEP:
        key   = f"lr_{lr:.0e}"
        label = f"calib  D=128 N=512  lr={lr:.0e}  cosine  [LR sweep]"
        model = make_resonant(N=512, D=128).to(DEVICE)
        meta  = {"N": 512, "D": 128, "K_iter": 8, "lr": lr, "sched": "cosine"}
        r = run(label, model, meta, n_epochs=CALIB_EPOCHS, lr=lr)
        r.update(meta)
        lr_results[key] = r
        results["calibration"][key] = r

    best_lr_key  = max(lr_results, key=lambda k: lr_results[k]["top1_best"])
    best_lr      = lr_results[best_lr_key]["lr"]
    best_lr_top1 = lr_results[best_lr_key]["top1_best"]
    print(f"\n  LR winner: lr={best_lr:.0e}  top1={best_lr_top1:.4f}")

    # ── Phase 1b: N sweep — D=128, best_lr, cosine, 40ep ─────────────────────
    print(f"\n{'#'*70}")
    print(f"PHASE 1b: N sweep — D=128 K_iter=8  lr={best_lr:.0e}  cosine  40ep")
    print(f"{'#'*70}")

    N_SWEEP = [256, 512, 1024]
    n_results = {}
    for N in N_SWEEP:
        key   = f"N_{N}"
        label = f"calib  D=128 N={N:<5}  lr={best_lr:.0e}  cosine  [N sweep]"
        model = make_resonant(N=N, D=128).to(DEVICE)
        meta  = {"N": N, "D": 128, "K_iter": 8, "lr": best_lr, "sched": "cosine"}
        r = run(label, model, meta, n_epochs=CALIB_EPOCHS, lr=best_lr)
        r.update(meta)
        n_results[key] = r
        results["calibration"][key] = r

    best_N_key  = max(n_results, key=lambda k: n_results[k]["top1_best"])
    best_N      = n_results[best_N_key]["N"]
    best_N_top1 = n_results[best_N_key]["top1_best"]
    print(f"\n  N winner: N={best_N}  top1={best_N_top1:.4f}")

    d128_viable = best_lr_top1 >= VIABLE_THRESHOLD
    results["best_params"] = {
        "D128_lr": best_lr, "D128_N": best_N,
        "D128_top1_calib": best_lr_top1, "D128_viable": d128_viable,
    }

    if not d128_viable:
        print(f"\n  WARNING: D=128 max calib top1={best_lr_top1:.4f} < {VIABLE_THRESHOLD:.0%}")
        print("  D=128 appears architecturally non-viable — skipping D=128 full runs.")
        print("  Possible causes: Fourier encoding frequency collapse at D=128,")
        print("  or fundamental concentration-of-measure problem on S^127.")

    # ── Phase 2: full 150ep runs ──────────────────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"PHASE 2: Full {FULL_EPOCHS}ep runs")
    print(f"{'#'*70}")

    full_configs = [
        ("Ref",
         "Ref        D=64  N=1024 K_iter=8  [step22E ceiling ≈56.28%]",
         1024, 64, None),
        ("D64_N2048",
         "D64_N2048  D=64  N=2048 K_iter=8  [does more N help at D=64?]",
         2048, 64, None),
    ]
    if d128_viable:
        full_configs.append((
            "D128_best",
            f"D128_best  D=128 N={best_N} K_iter=8 lr={best_lr:.0e} [D=128 winner]",
            best_N, 128, best_lr,
        ))
        if best_N != 1024:
            full_configs.append((
                "D128_N1024",
                f"D128_N1024 D=128 N=1024 K_iter=8 lr={best_lr:.0e}",
                1024, 128, best_lr,
            ))

    for key, label, N, D, lr in full_configs:
        model = make_resonant(N=N, D=D).to(DEVICE)
        meta  = {"N": N, "D": D, "K_iter": 8, "lr": lr, "sched": "cosine"}
        r = run(label, model, meta, n_epochs=FULL_EPOCHS, lr=lr)
        r.update(meta)
        results["full"][key] = r

    # ── Save ──────────────────────────────────────────────────────────────────
    out = ROOT / "results" / "train_step33b_d128_calib.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    ref_val = results["full"].get("Ref", {}).get("top1_best", 0.5628)

    print(f"\n{'='*70}")
    print("CALIBRATION SUMMARY (Phase 1)")
    print(f"{'='*70}")
    print(f"  {'Config':<50}  {'top1@40ep':>10}")
    print("  " + "-"*65)
    for k, r in results["calibration"].items():
        print(f"  {r['label'][:50]:<50}  {r['top1_best']:>10.4f}")

    print(f"\n{'='*70}")
    print(f"FULL RUN SUMMARY (ref={ref_val:.4f})")
    print(f"{'='*70}")
    print(f"  {'Config':<55}  {'top1':>6}  {'vs_Ref':>8}  {'t(s)':>6}")
    print("  " + "-"*80)
    for k, r in results["full"].items():
        d = r["top1_best"] - ref_val
        print(f"  {r['label'][:55]:<55}  {r['top1_best']:>6.4f}  {d:>+8.4f}  "
              f"{r['elapsed_s']:>6.0f}")

    print("\n  Interpretation:")
    if d128_viable:
        print(f"  D128_best vs Ref → does D trend continue beyond D=64?")
        print(f"  D64_N2048 vs D128_N1024 → N-scaling vs D-scaling at same param budget")
    else:
        print(f"  D=128 non-viable — D=64 is the encoding ceiling for this architecture")
        print(f"  D64_N2048 vs Ref → check if N=2048 recovers signal at D=64")
        print(f"  Next step: investigate Fourier encoding frequency cap at D=128")
