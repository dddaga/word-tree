"""Step 702: K_iter × K_hh latency-accuracy Pareto frontier.

MOTIVATION
==========
K_iter=5 is the dominant compute cost. step700/701 confirmed K_iter is
load-bearing (can't parallelize). But: can we trade some accuracy for
speed with fewer serial passes? And does K_hh interact with K_iter
(more neighbors compensate for fewer passes)?

This is a Tier-0 sweep to map the frontier. We measure:
  - Val accuracy @ 30ep (quick directional signal)
  - Training time per step (MPS wall-clock)

Grid: K_iter={2,3,4,5} × K_hh={1,2,4}
Base: N=2048, D=16, AH=1.0, alpha_reflect=0.5 (efficiency config)
Reference: K_iter=5, K_hh=2 (step199 winner, expected ~90% at 30ep)

PAPER RELEVANCE
===============
If K_iter=3 gives 88%+ at 60% fewer FLOPs, that's a strong
efficiency operating point. The Pareto curve belongs in the paper.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.experiment_config import trainer_kwargs
from src.training.trainer import Trainer
from src.training.dataset import H5Dataset

# ── Config ────────────────────────────────────────────────────────────────────
N_IN      = 25088
N_CLASSES = 10
N         = 2048
D         = 16
ALPHA_AHEBB   = 1.0
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
K_IN      = 25

EPOCHS    = 30    # Tier-0 scout
FRAC_DATA = 0.5
BATCH     = 128
SEED      = 42

# Grid to sweep
K_ITER_VALS = [2, 3, 4, 5]
K_HH_VALS   = [1, 2, 4]


def make_model(device, k_iter, k_hh):
    torch.manual_seed(SEED)
    n_groups = max(8, N // 8)
    K_local  = max(1, k_hh - max(1, k_hh // 4)) if k_hh > 1 else 1
    K_random = max(0, k_hh - K_local)
    sw = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_CLASSES, D=D, N_in=N_IN,
        K_in=K_IN, K_local=K_local, K_random=K_random,
        n_groups=n_groups, K_iter=k_iter,
        norm_mode="l2", encoding_mode="fourier",
    ).to(device)
    res = SGNNET_Resonant(
        base=sw, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, mode="dynamic_z_geo",
    ).to(device)
    return SGNNET_AntiHebbian(base=res, alpha_ahebb=ALPHA_AHEBB, variant="wpos").to(device)


def measure_step_time(model, device, n_trials=20):
    """Wall-clock time for one training step (forward+backward+optim)."""
    opt  = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    x    = torch.randn(BATCH, N_IN, device=device)
    y    = torch.randint(0, N_CLASSES, (BATCH,), device=device)
    model.train()
    # warmup
    for _ in range(3):
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
    if device.type == "mps": torch.mps.synchronize()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        if device.type == "mps": torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    return round(float(np.median(times)) * 1000, 2)  # ms


def run_config(label, k_iter, k_hh, device, tr, va):
    print(f"\n  [{label}]  K_iter={k_iter}  K_hh={k_hh}")
    model = make_model(device, k_iter, k_hh)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops = N * k_iter * k_hh * D * 2   # routing MACs only
    print(f"    params={n_params:,}  routing_MACs={flops/1e6:.3f}M")

    # Timing before training (fresh model, no compile)
    step_ms = measure_step_time(model, device)
    print(f"    step_time={step_ms:.2f}ms/step")

    # Full Tier-0 training
    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=str(device), **kw)

    history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
        print(f"    ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
        if (m['epoch']+1) % 10 == 0 else None
    ))

    top1h = [round(h.get("val_top1", 0.0), 4) for h in history]
    best  = max(top1h)
    bep   = int(np.argmax(top1h)) + 1
    print(f"    best={best:.4f} @ ep{bep}")

    return {
        "label":       label,
        "k_iter":      k_iter,
        "k_hh":        k_hh,
        "n_params":    n_params,
        "routing_macs": flops,
        "step_ms":     step_ms,
        "top1_best":   best,
        "best_epoch":  bep,
        "top1_last":   top1h[-1],
        "top1_history": top1h,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 702: K_iter × K_hh Pareto frontier")
    parser.add_argument("--device",   default="mps")
    parser.add_argument("--epochs",   type=int, default=EPOCHS)
    parser.add_argument("--output",   default=None)
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"\n{'='*70}")
    print(f"Step 702 — K_iter × K_hh Pareto Frontier (Tier-0 {args.epochs}ep)")
    print(f"N={N} D={D} AH={ALPHA_AHEBB}  Grid: K_iter={K_ITER_VALS} × K_hh={K_HH_VALS}")
    print(f"Device: {device}")
    print(f"{'='*70}")

    train_ds = H5Dataset(str(ROOT / "data/store.h5"), split="train")
    val_ds   = H5Dataset(str(ROOT / "data/store.h5"), split="val")
    n_train  = int(len(train_ds) * FRAC_DATA)
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(train_ds), generator=g)[:n_train].tolist()
    train_sub = torch.utils.data.Subset(train_ds, idx)
    g2 = torch.Generator().manual_seed(SEED)
    tr = torch.utils.data.DataLoader(train_sub, batch_size=BATCH, shuffle=True, generator=g2)
    va = torch.utils.data.DataLoader(val_ds,    batch_size=BATCH, shuffle=False)
    print(f"Data: {n_train}/{len(train_ds)} train samples, {len(val_ds)} val")

    results = {}
    for k_iter in K_ITER_VALS:
        for k_hh in K_HH_VALS:
            label = f"ki{k_iter}_kh{k_hh}"
            results[label] = run_config(label, k_iter, k_hh, device, tr, va)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Label':<12}  {'K_iter':>6}  {'K_hh':>5}  {'MACs':>8}  {'step_ms':>8}  {'best_val':>9}  {'best_ep':>7}")
    print(f"{'-'*70}")
    ref_acc = results.get("ki5_kh2", {}).get("top1_best", None)
    ref_ms  = results.get("ki5_kh2", {}).get("step_ms", None)
    for label, r in sorted(results.items(), key=lambda x: -x[1]["top1_best"]):
        delta = f"({r['top1_best']-ref_acc:+.4f})" if ref_acc else ""
        speedup = f"({ref_ms/r['step_ms']:.2f}x)" if ref_ms else ""
        print(f"  {label:<12}  {r['k_iter']:>6}  {r['k_hh']:>5}  "
              f"{r['routing_macs']/1e6:>7.3f}M  {r['step_ms']:>7.2f}ms{speedup}  "
              f"{r['top1_best']:>9.4f} {delta}  ep{r['best_epoch']:>3}")

    out_data = {
        "config": {"N": N, "D": D, "alpha_ahebb": ALPHA_AHEBB, "epochs": args.epochs,
                   "device": str(device)},
        "results": results,
    }
    out_path = args.output or str(ROOT / "results" / f"train_step702_kiter_khh_frontier.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
