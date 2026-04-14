"""Step 140: N×K Connectivity Tradeoff Sweep.

MOTIVATION
==========
At N=4096/K_hh=4, connectivity is 0.1% — extreme sparsity may be under-utilizing
neurons. Hypothesis: fewer neurons with richer connectivity (N=512/K_hh=32 at 6.3%)
could match or beat N=4096/K_hh=4 at fraction of FLOPs.

Budget: ≤5% VGG16 FC FLOPs (6.18M). All configs use D=16, K_in=25 to fit budget.

CONFIGS (D=16, AH=1.0, turing=0.0, reflect=0.5, 50%/75ep)
============================================================
  Ref_1024 : N=1024, K_hh=4, K_iter=12  (~4.4M FLOPs, 0.4% connectivity)
  A        : N=512,  K_hh=32, K_iter=8  (~4.7M FLOPs, 6.3% connectivity)
  B        : N=512,  K_hh=16, K_iter=8  (~2.6M FLOPs, 3.1% connectivity)
  C        : N=1024, K_hh=8,  K_iter=8  (~3.1M FLOPs, 0.8% connectivity)
  D        : N=256,  K_hh=32, K_iter=12 (~3.5M FLOPs, 12.5% connectivity)
  E        : N=1024, K_hh=16, K_iter=8  (~5.6M FLOPs, 1.6% connectivity)

Ablation axes:
  - N vs K_hh at similar FLOPs (A vs C: same edges, different N)
  - Connectivity at fixed N (B vs E: N=512 vs N=1024 at different K_hh)
  - Extreme density (D: N=256 at 12.5% connectivity)

To reproduce:
    python -u scripts/train_step140_nk_tradeoff.py --device mps
    python -u scripts/train_step140_nk_tradeoff.py --device mps --epochs 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs, topology_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75,
                    help="Training epochs (default 75; use 20 for scout)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (e.g. A,D). Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N_IN = 25088; N_OUT = 10
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0


@dataclass
class Config:
    key: str
    label: str
    N: int
    K_hh: int
    K_iter: int
    D: int
    K_in: int

CONFIGS = [
    Config("Ref", "Ref  N=1024 K=4 K_iter=12 D=16 (baseline)",  1024, 4,  12, 16, 25),
    Config("A",   "A    N=512  K=32 K_iter=8  D=16 (6.3% conn)", 512, 32,  8, 16, 25),
    Config("B",   "B    N=512  K=16 K_iter=8  D=16 (3.1% conn)", 512, 16,  8, 16, 25),
    Config("C",   "C    N=1024 K=8  K_iter=8  D=16 (0.8% conn)",1024,  8,  8, 16, 25),
    Config("D",   "D    N=256  K=32 K_iter=12 D=16 (12.5% conn)", 256, 32, 12, 16, 25),
    Config("E",   "E    N=1024 K=16 K_iter=8  D=16 (1.6% conn)",1024, 16,  8, 16, 25),
]


_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    N = cfg.N
    # Compute K_local and K_random from K_hh
    K_random = max(1, cfg.K_hh // 4)
    K_local  = cfg.K_hh - K_random
    n_groups = max(8, N // 8)

    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=cfg.D, N_in=N_IN,
        K_in=cfg.K_in, K_iter=cfg.K_iter,
        K_local=K_local, K_random=K_random, n_groups=n_groups,
        norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def compute_flops(cfg: Config) -> int:
    N, K, Ki, D, Kin = cfg.N, cfg.K_hh, cfg.K_iter, cfg.D, cfg.K_in
    seed     = N * Kin * D
    per_step = N * K * D * 2 + N * D + N * D * 2
    routing  = Ki * per_step
    readout  = N * N_OUT * D
    return seed + routing + readout


def main():
    print(f"\n{'='*70}")
    print(f"Step 140 — N×K Connectivity Tradeoff Sweep")
    print(f"Budget: ≤6.18M FLOPs (5% VGG16 FC)")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    print(f"{'Key':4s}  {'N':>5}  {'K':>3}  {'Ki':>3}  {'D':>3}  {'FLOPs':>8}  {'Conn%':>6}  Label")
    print(f"{'─'*75}")
    for c in CONFIGS:
        flops = compute_flops(c)
        conn = c.K_hh / c.N * 100
        print(f"{c.key:4s}  {c.N:>5}  {c.K_hh:>3}  {c.K_iter:>3}  {c.D:>3}  "
              f"{flops/1e6:>7.2f}M  {conn:>5.1f}%  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step140_nk_tradeoff.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)
        flops    = compute_flops(cfg)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  FLOPs={flops/1e6:.2f}M  conn={cfg.K_hh/cfg.N*100:.1f}%")
        print(f"{'─'*60}")

        t0 = time.time()
        kw = trainer_kwargs(cfg.N, n_epochs=EPOCHS)
        trainer = Trainer(
            model=model, train_loader=get_loaders()[0],
            val_loader=get_loaders()[1], device=DEVICE, **kw,
        )
        history = trainer.train(n_epochs=EPOCHS)
        elapsed = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1

        results[cfg.key] = {
            "N": cfg.N, "D": cfg.D, "K_hh": cfg.K_hh,
            "K_iter": cfg.K_iter, "K_in": cfg.K_in,
            "connectivity_pct": round(cfg.K_hh / cfg.N * 100, 2),
            "flops": flops, "flops_M": round(flops / 1e6, 2),
            "alpha_ahebb": ALPHA_AHEBB,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params,
            "label": cfg.label,
        }

        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs_ref = top1_best - ref_best if ref_best > 0 else 0
        print(f"\n  top1={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"FLOPs={flops/1e6:.2f}M  elapsed={elapsed/60:.1f}min")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 140 SUMMARY — N×K Tradeoff")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Key':4s}  {'N':>5}  {'K':>3}  {'conn%':>6}  {'FLOPs':>8}  "
          f"{'params':>8}  {'top1':>7}  {'vs_Ref':>8}")
    print(f"{'─'*65}")
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"{key:4s}  {r['N']:>5}  {r['K_hh']:>3}  {r['connectivity_pct']:>5.1f}%  "
              f"{r['flops_M']:>7.2f}M  {r['n_params']:>8}  {r['top1_best']:.4f}  {vs:>+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
