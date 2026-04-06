"""Step 22b: Calibrate base routing hyperparameters for D=64 N=1024.

MOTIVATION
==========
Step 22E found D=64 N=1024 K_iter=8 = 56.28% using D=16-calibrated routing params.
These params are NOT scale-independent:

  alpha_turing=0.3 : Turing inhibition weight — cosine similarities concentrate near
                     zero on S^63 vs S^15 (curse of dimensionality), so the inhibitory
                     signal magnitude changes dramatically with D.

  alpha_reflect=0.3: Reflection self-inhibition weight — same dimensionality issue.

  K_phase=8        : Phase neighbourhood — on S^63 with N=1024, angular structure is
                     very different from S^15 with N=512. More neighbours may be needed
                     to capture meaningful phase proximity.

  beam_size=32     : Was 6% of N=512, now only 3% of N=1024. Matching the ratio
                     (beam_size=64) might recover broadcast coverage.

  geo_gamma=1.0    : W_pos distance penalty — with D=64 the position space geometry
                     may require different scale.

STRUCTURE
=========
Phase 1 — Calibration (40 epochs): sweep each parameter independently to find
           the best value at D=64 scale.

Phase 2 — Full runs (150 epochs): run with each single best param, then ALL best
           combined, vs the D=16 defaults reference.

EXPECTED OUTCOME
================
If any param is miscalibrated, Phase 2 "F" (all best) should beat step22E 56.28%.
This gives the true D=64 performance ceiling.

To reproduce:
    python -u scripts/train_step22b_routing_calib_d64.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

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

CALIB_EPOCHS = 40
FULL_EPOCHS  = 150
BATCH        = 128
SEED         = 42
DATA         = "data/store.h5"
D            = 64
N            = 1024
K_ITER       = 8

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_resonant(
    K_phase: int = 8,
    beam_size: int = 32,
    alpha_reflect: float = 0.3,
    alpha_turing: float = 0.3,
    geo_gamma: float = 1.0,
) -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_ITER,
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=K_phase, beam_size=beam_size,
        theta_init=0.1, alpha_reflect=alpha_reflect, alpha_turing=alpha_turing,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=geo_gamma,
    )


def run(label: str, model: SGNNET_Resonant, meta: dict, n_epochs: int) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(N, n_epochs=n_epochs, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
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
        "_meta": run_metadata(__file__, {**meta, "epochs": n_epochs}),
    }
    result.update(meta)
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})  "
          f"diag={result['convergence_diag']}  t={elapsed:.0f}s")
    return result


# ── Phase 1 calibration configs ───────────────────────────────────────────────

# alpha_turing sweep (Turing inhibition weight)
CALIB_AT = {
    "calib_at_00": dict(alpha_turing=0.0),   # no Turing inhibition — ablation
    "calib_at_01": dict(alpha_turing=0.1),
    "calib_at_03": dict(alpha_turing=0.3),   # D=16 default
    "calib_at_05": dict(alpha_turing=0.5),
    "calib_at_10": dict(alpha_turing=1.0),
}

# alpha_reflect sweep
CALIB_AR = {
    "calib_ar_00": dict(alpha_reflect=0.0),  # no reflection — ablation
    "calib_ar_01": dict(alpha_reflect=0.1),
    "calib_ar_03": dict(alpha_reflect=0.3),  # D=16 default
    "calib_ar_05": dict(alpha_reflect=0.5),
}

# K_phase sweep (phase neighbourhood size)
CALIB_KP = {
    "calib_kp_04": dict(K_phase=4),
    "calib_kp_08": dict(K_phase=8),    # D=16 default
    "calib_kp_16": dict(K_phase=16),
    "calib_kp_32": dict(K_phase=32),
}

# beam_size sweep
CALIB_BS = {
    "calib_bs_16":  dict(beam_size=16),   # 1.6% of N
    "calib_bs_32":  dict(beam_size=32),   # 3.1%, D=16 default
    "calib_bs_64":  dict(beam_size=64),   # 6.3%, matches D=16 ratio
    "calib_bs_128": dict(beam_size=128),  # 12.5%
}

# geo_gamma sweep
CALIB_GG = {
    "calib_gg_00": dict(geo_gamma=0.0),  # no position penalty — ablation
    "calib_gg_05": dict(geo_gamma=0.5),
    "calib_gg_10": dict(geo_gamma=1.0),  # D=16 default
    "calib_gg_20": dict(geo_gamma=2.0),
}

# Default values for all params (D=16 calibrated)
DEFAULTS = dict(
    alpha_turing=0.3,
    alpha_reflect=0.3,
    K_phase=8,
    beam_size=32,
    geo_gamma=1.0,
)


def _label_from_key(key: str, val: dict) -> str:
    return f"{key}  {list(val.items())[0]}  [D=64 calib sweep]"


if __name__ == "__main__":
    print(f"Device: {DEVICE}  CALIB_EPOCHS: {CALIB_EPOCHS}  FULL_EPOCHS: {FULL_EPOCHS}")
    print(f"D={D}  N={N}  K_iter={K_ITER}")
    print("Goal: calibrate D=16 routing params for D=64 N=1024 geometry")
    print(f"step22E baseline: 56.28% (D=16 defaults, 150ep)")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    calib_results = {}

    # ── Phase 1: Calibration sweeps (40 epochs) ───────────────────────────────

    print(f"\n{'#'*70}")
    print("PHASE 1 — CALIBRATION SWEEPS (40 epochs each)")
    print(f"{'#'*70}")

    # alpha_turing sweep
    print("\n--- alpha_turing sweep ---")
    calib_at_results = {}
    for key, override in CALIB_AT.items():
        params = {**DEFAULTS, **override}
        label  = _label_from_key(key, override)
        model  = make_resonant(**params).to(DEVICE)
        meta   = {"D": D, "N": N, "K_iter": K_ITER, **params}
        r      = run(label, model, meta, CALIB_EPOCHS)
        calib_results[key] = r
        calib_at_results[key] = r["top1_best"]

    best_at_key = max(calib_at_results, key=lambda k: calib_at_results[k])
    best_at     = CALIB_AT[best_at_key]["alpha_turing"]
    print(f"\n  alpha_turing winner: {best_at_key} -> alpha_turing={best_at}  "
          f"({calib_at_results[best_at_key]:.4f})")

    # alpha_reflect sweep
    print("\n--- alpha_reflect sweep ---")
    calib_ar_results = {}
    for key, override in CALIB_AR.items():
        params = {**DEFAULTS, **override}
        label  = _label_from_key(key, override)
        model  = make_resonant(**params).to(DEVICE)
        meta   = {"D": D, "N": N, "K_iter": K_ITER, **params}
        r      = run(label, model, meta, CALIB_EPOCHS)
        calib_results[key] = r
        calib_ar_results[key] = r["top1_best"]

    best_ar_key = max(calib_ar_results, key=lambda k: calib_ar_results[k])
    best_ar     = CALIB_AR[best_ar_key]["alpha_reflect"]
    print(f"\n  alpha_reflect winner: {best_ar_key} -> alpha_reflect={best_ar}  "
          f"({calib_ar_results[best_ar_key]:.4f})")

    # K_phase sweep
    print("\n--- K_phase sweep ---")
    calib_kp_results = {}
    for key, override in CALIB_KP.items():
        params = {**DEFAULTS, **override}
        label  = _label_from_key(key, override)
        model  = make_resonant(**params).to(DEVICE)
        meta   = {"D": D, "N": N, "K_iter": K_ITER, **params}
        r      = run(label, model, meta, CALIB_EPOCHS)
        calib_results[key] = r
        calib_kp_results[key] = r["top1_best"]

    best_kp_key = max(calib_kp_results, key=lambda k: calib_kp_results[k])
    best_kp     = CALIB_KP[best_kp_key]["K_phase"]
    print(f"\n  K_phase winner: {best_kp_key} -> K_phase={best_kp}  "
          f"({calib_kp_results[best_kp_key]:.4f})")

    # beam_size sweep
    print("\n--- beam_size sweep ---")
    calib_bs_results = {}
    for key, override in CALIB_BS.items():
        params = {**DEFAULTS, **override}
        label  = _label_from_key(key, override)
        model  = make_resonant(**params).to(DEVICE)
        meta   = {"D": D, "N": N, "K_iter": K_ITER, **params}
        r      = run(label, model, meta, CALIB_EPOCHS)
        calib_results[key] = r
        calib_bs_results[key] = r["top1_best"]

    best_bs_key = max(calib_bs_results, key=lambda k: calib_bs_results[k])
    best_bs     = CALIB_BS[best_bs_key]["beam_size"]
    print(f"\n  beam_size winner: {best_bs_key} -> beam_size={best_bs}  "
          f"({calib_bs_results[best_bs_key]:.4f})")

    # geo_gamma sweep
    print("\n--- geo_gamma sweep ---")
    calib_gg_results = {}
    for key, override in CALIB_GG.items():
        params = {**DEFAULTS, **override}
        label  = _label_from_key(key, override)
        model  = make_resonant(**params).to(DEVICE)
        meta   = {"D": D, "N": N, "K_iter": K_ITER, **params}
        r      = run(label, model, meta, CALIB_EPOCHS)
        calib_results[key] = r
        calib_gg_results[key] = r["top1_best"]

    best_gg_key = max(calib_gg_results, key=lambda k: calib_gg_results[k])
    best_gg     = CALIB_GG[best_gg_key]["geo_gamma"]
    print(f"\n  geo_gamma winner: {best_gg_key} -> geo_gamma={best_gg}  "
          f"({calib_gg_results[best_gg_key]:.4f})")

    # ── Phase 1 summary table ─────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("PHASE 1 SUMMARY — best value per parameter")
    print(f"{'='*70}")

    all_sweep_groups = [
        ("alpha_turing",  calib_at_results, CALIB_AT,  "alpha_turing"),
        ("alpha_reflect", calib_ar_results, CALIB_AR,  "alpha_reflect"),
        ("K_phase",       calib_kp_results, CALIB_KP,  "K_phase"),
        ("beam_size",     calib_bs_results, CALIB_BS,  "beam_size"),
        ("geo_gamma",     calib_gg_results, CALIB_GG,  "geo_gamma"),
    ]
    print(f"  {'Parameter':<18}  {'Value':<10}  {'top1':>8}  {'Winner'}")
    print("  " + "-"*60)
    for param_name, sweep_res, sweep_cfg, param_key in all_sweep_groups:
        for key, top1 in sorted(sweep_res.items(), key=lambda x: x[1], reverse=True):
            val  = sweep_cfg[key][param_key]
            star = " <-- BEST" if top1 == max(sweep_res.values()) else ""
            print(f"  {param_name:<18}  {str(val):<10}  {top1:>8.4f}{star}")
        print()

    best_params = {
        "alpha_turing":  best_at,
        "alpha_reflect": best_ar,
        "K_phase":       best_kp,
        "beam_size":     best_bs,
        "geo_gamma":     best_gg,
    }
    print(f"  Best params dict: {best_params}")

    # ── Phase 2: Full runs (150 epochs) ───────────────────────────────────────

    print(f"\n{'#'*70}")
    print("PHASE 2 — FULL RUNS (150 epochs)")
    print(f"{'#'*70}")

    FULL_CONFIGS = {
        "Ref": {
            "label": "Ref.  all D=16 defaults  [reproduces step22E ~56.28%]",
            "params": DEFAULTS,
        },
        "A": {
            "label": f"A.   best alpha_turing={best_at}  only",
            "params": {**DEFAULTS, "alpha_turing": best_at},
        },
        "B": {
            "label": f"B.   best alpha_reflect={best_ar}  only",
            "params": {**DEFAULTS, "alpha_reflect": best_ar},
        },
        "C": {
            "label": f"C.   best K_phase={best_kp}  only",
            "params": {**DEFAULTS, "K_phase": best_kp},
        },
        "D": {
            "label": f"D.   best beam_size={best_bs}  only",
            "params": {**DEFAULTS, "beam_size": best_bs},
        },
        "E": {
            "label": f"E.   best geo_gamma={best_gg}  only",
            "params": {**DEFAULTS, "geo_gamma": best_gg},
        },
        "F": {
            "label": "F.   ALL best params combined  [true D=64 ceiling estimate]",
            "params": best_params,
        },
    }

    full_results = {}
    for key, cfg in FULL_CONFIGS.items():
        model = make_resonant(**cfg["params"]).to(DEVICE)
        meta  = {"D": D, "N": N, "K_iter": K_ITER, **cfg["params"]}
        r     = run(cfg["label"], model, meta, FULL_EPOCHS)
        full_results[key] = r

    # ── Final summary ─────────────────────────────────────────────────────────

    ref_top1 = full_results.get("Ref", {}).get("top1_best", 0.5628)
    step22e  = 0.5628

    print(f"\n{'='*70}")
    print("PHASE 2 SUMMARY — full runs vs Ref")
    print(f"{'='*70}")
    print(f"  {'Config':<65}  {'top1':>8}  {'vs_Ref':>8}  {'vs_22E':>8}  "
          f"{'best_ep%':>9}  {'t(s)':>6}")
    print("  " + "-"*115)
    for key, r in full_results.items():
        d_ref  = r["top1_best"] - ref_top1
        d_22e  = r["top1_best"] - step22e
        frac   = r.get("best_epoch_frac", 0) * 100
        print(f"  {r['label']:<65}  {r['top1_best']:>8.4f}  {d_ref:>+8.4f}  "
              f"{d_22e:>+8.4f}  {frac:>8.1f}%  {r['elapsed_s']:>6.0f}")

    f_top1 = full_results.get("F", {}).get("top1_best", 0.0)
    print(f"\nTrue D=64 ceiling (with calibrated base): F = {f_top1:.2%}")

    # ── Save results ──────────────────────────────────────────────────────────

    output = {
        "calibration": calib_results,
        "full":        full_results,
        "best_params": best_params,
    }
    out = Path("results/train_step22b_routing_calib_d64.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(f"\nSaved -> {out}")
