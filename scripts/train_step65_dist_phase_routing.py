"""Step 65: Distance-Phase Routing ablation.

Strips phase back to first principles — no W_phase, no learned phase, just geometry.
Distance determines signal attenuation: exp(-γ·d_norm) weights on conn_hh edges.

MOTIVATION
==========
Every wave-1 mechanism (steps 58-63) failed via gate death or additive excitation
saturation. Complex learned phase dynamics introduced instability. This experiment
asks the simplest possible question:

    Does using spatial distance to weight the static conn_hh edges improve over
    uniform routing — with NO additional mechanisms?

This is step 1 in a progressive phase-mechanism build-up:
    step65: geometry only  → does distance weighting help?
    step66: geometry + dynamics  (if step65 wins)
    step67: geometry + dynamics + stabilization

Architecture
============
  dist[i,j]      = ||W_pos[i] - W_pos[j]||₂  for each (i,j) in conn_hh
  dist_norm[i,j] = dist[i,j] / mean(dist)      ← scale-independent normalization
  raw_w[i,j]     = exp(-gamma * dist_norm[i,j]) ← Gaussian falloff

  weighting='softmax': w = row-softmax(raw_w)   [sum-to-1, redistributes excitation]
  weighting='raw':     w = raw_w                 [unnormalized, boosts near / dims far]

  Routing each step:
    Z_fwd   = relu(Z - theta)      ← per-neuron threshold gate
    Z_nb    = Z_fwd[:, conn_hh, :]
    Z_struct = Σ_j w[i,j] * Z_nb[:, i, j, :]
    Z       = normalize(Z_struct.clamp(-10, 10), dim=-1)

γ controls falloff relative to mean inter-neuron distance (scale-independent):
  γ=0.5 → soft; all edges within ~2σ contribute meaningfully
  γ=1.0 → moderate; mean-distance edge gets weight exp(-1)≈0.37
  γ=2.0 → sharp; only nearest edges matter (near-hard-KNN)

CONFIGS (N=1024, D=64, K_iter=8, 50% data, 75ep)
=================================================
  Ref   : bare SmallWorld, no-phase, no-AH  [internal reference]
  A     : γ=0.5 raw weights
  B     : γ=0.5 softmax
  C     : γ=1.0 raw weights
  D     : γ=1.0 softmax
  E     : γ=2.0 raw weights
  F     : γ=2.0 softmax

KEY QUESTIONS
=============
  A/B vs Ref → does distance weighting help over uniform at γ=0.5?
  C/D vs Ref → same at γ=1.0
  E/F vs Ref → same at γ=2.0
  A vs B, C vs D, E vs F → raw vs softmax: which normalization is more stable?
  Best config vs REF_BASELINE (73.53%) → how far from AH baseline without AH?

To reproduce:
    python -u scripts/train_step65_dist_phase_routing.py --device mps
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
from src.sgnnet.model_dist_phase    import SGNNET_DistPhase
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset           import make_loaders

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
K_ITER = 8

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n   = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


def make_base() -> SGNNET_SmallWorld:
    """Bare SmallWorld — no wrapper, no AH. Internal reference point."""
    torch.manual_seed(SEED)
    tk = topology_kwargs(N)
    return SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_ITER, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )


def make_dist_phase(gamma: float, weighting: str, seed_offset: int = 0) -> SGNNET_DistPhase:
    """Distance-phase wrapper with fixed γ and weighting mode."""
    torch.manual_seed(SEED + seed_offset)
    return SGNNET_DistPhase(make_base(), gamma=gamma, weighting=weighting)


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
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
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
          f"  diag={result['convergence_diag']}  t={elapsed:.0f}s")
    return result


# Load REF_BASELINE from step57
_ref_path = ROOT / "results" / "train_step57_benchmark_ablation.json"
try:
    REF_BASELINE = json.loads(_ref_path.read_text()).get("top1_best", 0.7353)
    print(f"Loaded REF_BASELINE from step57: {REF_BASELINE:.4f}")
except Exception:
    REF_BASELINE = 0.7353
    print(f"step57 result not found — using fallback REF_BASELINE={REF_BASELINE:.4f}")


# (key, label, model_fn, meta_extra)
CONFIGS = [
    ("Ref", "Ref   bare SmallWorld no-phase no-AH",
     lambda: make_base(),
     {"mechanism": "smallworld_bare"}),

    ("A",   "A     γ=0.5 raw",
     lambda: make_dist_phase(0.5, "raw",     seed_offset=0),
     {"mechanism": "dist_phase", "gamma": 0.5, "weighting": "raw"}),

    ("B",   "B     γ=0.5 softmax",
     lambda: make_dist_phase(0.5, "softmax", seed_offset=1),
     {"mechanism": "dist_phase", "gamma": 0.5, "weighting": "softmax"}),

    ("C",   "C     γ=1.0 raw",
     lambda: make_dist_phase(1.0, "raw",     seed_offset=2),
     {"mechanism": "dist_phase", "gamma": 1.0, "weighting": "raw"}),

    ("D",   "D     γ=1.0 softmax",
     lambda: make_dist_phase(1.0, "softmax", seed_offset=3),
     {"mechanism": "dist_phase", "gamma": 1.0, "weighting": "softmax"}),

    ("E",   "E     γ=2.0 raw",
     lambda: make_dist_phase(2.0, "raw",     seed_offset=4),
     {"mechanism": "dist_phase", "gamma": 2.0, "weighting": "raw"}),

    ("F",   "F     γ=2.0 softmax",
     lambda: make_dist_phase(2.0, "softmax", seed_offset=5),
     {"mechanism": "dist_phase", "gamma": 2.0, "weighting": "softmax"}),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 65: Distance-Phase Routing  |  REF_BASELINE={REF_BASELINE:.4f}")
    print("exp(-γ·d_norm) weighting on conn_hh edges. No AH. No W_phase.")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    for key, label, model_fn, meta_extra in CONFIGS:
        model = model_fn().to(DEVICE)
        meta  = {"N": N, "D": D, "K_iter": K_ITER, "data_frac": 0.5, **meta_extra}
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step65_dist_phase_routing.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    print(f"\n-- Step 65: Distance-Phase Routing (internal_ref={ref_val:.4f} / step57={REF_BASELINE:.4f}) --")
    print(f"  {'Config':<55}  {'top1':>6}  {'vs_Ref':>8}  {'vs_AH':>8}  {'best_ep':>8}  {'t(s)':>6}")
    print("  " + "-"*100)
    for k, r in results.items():
        d_ref = r["top1_best"] - ref_val
        d_ah  = r["top1_best"] - REF_BASELINE
        print(f"  {r['label'][:55]:<55}  {r['top1_best']:>6.4f}  {d_ref:>+8.4f}  {d_ah:>+8.4f}  "
              f"{r.get('best_epoch', 0):>7d}    {r['elapsed_s']:>6.0f}")

    print("\n  Key questions:")
    print("  A/B vs Ref → does distance weighting help over uniform at γ=0.5?")
    print("  C/D vs Ref → same at γ=1.0")
    print("  E/F vs Ref → same at γ=2.0")
    print("  A vs B, C vs D, E vs F → raw vs softmax: which normalization is more stable?")
    print(f"  Best config vs REF_BASELINE ({REF_BASELINE:.4f}) → how far from AH baseline without AH?")
