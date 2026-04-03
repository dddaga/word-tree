"""Exp 3: ProximityWave at N=1024 and N=4096 with D=64.

QUESTION
========
Does geometry-based k-NN topology with dynamic phasor routing outperform
SmallWorld's fixed-group approach at the same N values?

ProximityWave differences vs SmallWorld:
  1. conn_hh from W_pos k-NN (no index-based groups)
  2. Phasor routing: Z_re/Z_im with distance-based phase rotation
  3. Periodic reconnection: topology rebuilds from W_pos every 10 epochs
  4. Anti-Hebbian suppression: inline wpos-similarity decorrelation (alpha=0.7)

Configs
-------
  N1024  N=1024  D=64 K_iter=8 K_local=4 K_random=2 reconnect_every=10
  N4096  N=4096  D=64 K_iter=8 K_local=4 K_random=2 reconnect_every=10

Both with anti_hebb_alpha=0.7 (wpos variant, matching confirmed winner).
150 epochs, batch=128, plateau LR (patience=10, factor=0.5).
"""

import argparse
import json
import os
import time
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from src.training.experiment_config import trainer_kwargs, run_metadata
from src.training.trainer import Trainer
from src.training.dataset import make_loaders
from src.sgnnet.model_proximity_wave import SGNNET_ProximityWave


# -- CLI -------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="mps")
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--batch", type=int, default=128)
ARGS = parser.parse_args()

DEVICE = ARGS.device
EPOCHS = ARGS.epochs
BATCH  = ARGS.batch
SEED   = 42
DATA   = "data/store.h5"

_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


# -- Model factory ---------------------------------------------------------
def make_proxwave(N: int, D: int = 64, K_iter: int = 8) -> SGNNET_ProximityWave:
    torch.manual_seed(SEED)
    return SGNNET_ProximityWave(
        N_hidden=N,
        N_out=10,
        D=D,
        N_in=25088,
        K_local=4,
        K_random=2,
        reconnect_every=10,
        K_in=50,
        K_iter=K_iter,
        sparsity=0.90,
        box_size=1.0,
        encoding_mode="fourier",
        norm_mode="l2",
        anti_hebb_alpha=0.7,
    )


# -- Run helper ------------------------------------------------------------
def run(label: str, N: int, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")

    model = make_proxwave(N=N).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  N_hidden={N}  params={n_params:,}  D={model.D}  K_iter={model.K_iter}")
    print(f"  K_local={model.K_local}  K_random={model.K_random}"
          f"  reconnect_every={model.reconnect_every}")
    print(f"  anti_hebb_alpha={model.anti_hebb_alpha}  norm_mode={model.norm_mode}")

    tr, va = get_loaders()
    tk = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)

    # -- Time forward pass at this N for benchmarking --
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(BATCH, 25088, device=DEVICE)
        # Warm up
        _ = model(dummy)
        t_fwd = []
        for _ in range(5):
            t0 = time.time()
            _ = model(dummy)
            if DEVICE == "mps":
                torch.mps.synchronize()
            t_fwd.append(time.time() - t0)
    ms_fwd = 1000 * sum(t_fwd) / len(t_fwd)
    print(f"  forward pass: {ms_fwd:.1f} ms/batch (batch={BATCH})")

    # -- Train --
    model.train()
    t0 = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    best_top1 = max(h["val_top1"] for h in history)
    best_ep = max(range(len(history)), key=lambda i: history[i]["val_top1"]) + 1
    frac = best_ep / EPOCHS
    ms_per_epoch = 1000 * elapsed / len(history) if history else 0

    topology_rebuilds = model._topology_changes

    print(f"  top1_best={best_top1:.4f}  best_ep={best_ep}/{EPOCHS} ({frac:.0%})"
          f"  t={elapsed:.0f}s  ms/ep={ms_per_epoch:.0f}")
    print(f"  topology_rebuilds={topology_rebuilds}")

    return {
        "label": label,
        "N": N,
        "N_hidden": N,
        "D": model.D,
        "K_iter": model.K_iter,
        "K_local": model.K_local,
        "K_random": model.K_random,
        "reconnect_every": model.reconnect_every,
        "anti_hebb_alpha": model.anti_hebb_alpha,
        "norm_mode": model.norm_mode,
        "params": n_params,
        "top1": best_top1,
        "best_ep": best_ep,
        "ep_frac": frac,
        "ms_fwd_batch": ms_fwd,
        "ms_per_epoch": ms_per_epoch,
        "total_time_s": elapsed,
        "topology_rebuilds": topology_rebuilds,
        "epochs_run": len(history),
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }


# -- Configs ---------------------------------------------------------------
CONFIGS = [
    ("N1024", "ProximityWave N=1024 D=64 K_iter=8 AntiHebb(0.7)", 1024),
    ("N4096", "ProximityWave N=4096 D=64 K_iter=8 AntiHebb(0.7)", 4096),
]


# -- Main ------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print("Goal: test ProximityWave (k-NN topology + phasor routing) at two scales")
    print("Compare vs SmallWorld+Resonant+AntiHebb baseline")

    results = {}

    for key, label, N in CONFIGS:
        meta = {
            "N": N, "D": 64, "K_iter": 8,
            "experiment": "exp3_proxwave",
        }
        results[key] = run(label, N, meta)

    # -- Save results --
    out_path = ROOT / "results" / "exp3_proxwave.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # -- Summary table --
    print(f"\n-- ProximityWave Results -----------------------------------------")
    print(f"  {'Config':<55}  {'top1':>6}  {'params':>8}  {'ms/ep':>7}  {'rebuilds':>8}")
    print("  " + "-" * 90)
    for key, label, N in CONFIGS:
        r = results[key]
        print(f"  {label:<55}  {r['top1']:.4f}  {r['params']:>8,}  "
              f"{r['ms_per_epoch']:>7.0f}  {r['topology_rebuilds']:>8}")

    if "N4096" in results:
        ms = results["N4096"]["ms_fwd_batch"]
        print(f"\n  N=4096 forward pass: {ms:.1f} ms/batch"
              f"  ({'OK' if ms < 500 else 'SLOW'})")
