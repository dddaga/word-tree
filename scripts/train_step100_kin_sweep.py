"""Step 100: K_in Sweep — input fan-in as FLOPs lever.

MOTIVATION
==========
K_in=50 (input connections per neuron) is a key FLOPs lever that has NEVER been swept.
Total input coverage = N × K_in. At N=1024, K_in=50 means 51200 total connections for
25088 input features (~2× coverage). Reducing K_in cuts input FLOPs proportionally.
But does accuracy degrade gracefully?

FLOPs_input ~ N × K_in × D (gather + sum in _seed).
FLOPs_routing ~ N × K_hh × K_iter × D (unchanged across configs).

CONFIGS (N=1024, D=64, K_hh=4, K_iter=12, AH=1.0, turing=0.0, 50%/75ep)
=========================================================================
  Ref : K_in=50  (current default)
  A   : K_in=5   (10× reduction — minimal coverage)
  B   : K_in=10  (5× reduction)
  C   : K_in=15  (3.3× reduction)
  D   : K_in=25  (2× reduction — ~1× coverage)
  E   : K_in=75  (1.5× default — does more help?)

To reproduce:
    python -u scripts/train_step100_kin_sweep.py --device mps
    python -u scripts/train_step100_kin_sweep.py --device mps --epochs 20  # scout
    python -u scripts/train_step100_kin_sweep.py --device mps --configs A,D
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
                    help="Training epochs (default 75; use 20 for Tier-0 scout)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys to run (e.g. A,D). Empty = run all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 64; K_ITER = 12
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
STEP69_REF = 0.8336   # D=64 N=1024 reference


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    K_in: int


CONFIGS = [
    Config("Ref", "Ref  K_in=50  (default)",            K_in=50),
    Config("A",   "A    K_in=5   (10× reduction)",      K_in=5),
    Config("B",   "B    K_in=10  (5× reduction)",       K_in=10),
    Config("C",   "C    K_in=15  (3.3× reduction)",     K_in=15),
    Config("D",   "D    K_in=25  (2× reduction)",       K_in=25),
    Config("E",   "E    K_in=75  (1.5× default)",       K_in=75),
]


# ---------------------------------------------------------------------------
# FLOPs estimation
# ---------------------------------------------------------------------------

def estimate_flops(K_in: int) -> dict:
    """Estimate FLOPs for input seeding and routing stages."""
    K_hh = 6  # K_local=4 + K_random=2
    # Input: gather [B, N, K_in, D] + sum over K_in → N * K_in * D MACs
    flops_input = N * K_in * D
    # Routing: K_iter steps of gather [B, N, K_hh, D] + sum → N * K_hh * D per step
    flops_routing = K_ITER * N * K_hh * D
    # Readout: einsum bhd,ho->bod → N * N_OUT * D
    flops_readout = N * N_OUT * D
    flops_total = flops_input + flops_routing + flops_readout
    return {
        "flops_input": flops_input,
        "flops_routing": flops_routing,
        "flops_readout": flops_readout,
        "flops_total": flops_total,
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    topo = topology_kwargs(N)
    topo.pop("K_in", None)    # we supply our own
    topo.pop("K_iter", None)  # we supply our own
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=cfg.K_in, K_iter=K_ITER, encoding_mode="fourier",
        **topo,
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data loaders (cached, 50% data)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 100 — K_in Sweep (input fan-in as FLOPs lever)")
    print(f"N={N}  D={D}  K_iter={K_ITER}  K_hh=4+2  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    print("Configs:")
    for c in CONFIGS:
        fl = estimate_flops(c.K_in)
        coverage = N * c.K_in / N_IN
        print(f"  {c.key:4s}  K_in={c.K_in:<3d}  coverage={coverage:.2f}×  "
              f"FLOPs_input={fl['flops_input']:>10,}  FLOPs_total={fl['flops_total']:>10,}  "
              f"{c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step100_kin_sweep.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)
        fl       = estimate_flops(cfg.K_in)
        coverage = N * cfg.K_in / N_IN

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  K_in={cfg.K_in}  coverage={coverage:.2f}×  params={n_params:,}")
        print(f"  FLOPs: input={fl['flops_input']:,}  routing={fl['flops_routing']:,}  "
              f"total={fl['flops_total']:,}")
        print(f"{'─'*60}")

        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=EPOCHS)

        trainer = Trainer(
            model=model,
            train_loader=get_loaders()[0],
            val_loader=get_loaders()[1],
            device=DEVICE,
            **kw,
        )

        history   = trainer.train(n_epochs=EPOCHS)
        elapsed   = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1

        results[cfg.key] = {
            "N": N, "D": D, "K_iter": K_ITER, "K_in": cfg.K_in,
            "coverage": round(coverage, 3),
            "alpha_ahebb": ALPHA_AHEBB,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params,
            "label": cfg.label,
            **fl,
        }

        # vs Ref (compute once Ref is available)
        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs_ref = top1_best - ref_best if ref_best > 0 else 0

        print(f"\n  top1_best={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"vs_step69={top1_best - STEP69_REF:+.4f}  "
              f"elapsed={elapsed/60:.1f}min")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary table
    print(f"\n{'='*70}")
    print(f"STEP 100 SUMMARY — K_in Sweep")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Config':<6}  {'K_in':>4}  {'cover':>5}  {'FLOPs_in':>10}  "
          f"{'FLOPs_tot':>10}  {'params':>8}  {'top1':>7}  {'vs_Ref':>8}")
    print(f"{'─'*70}")
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"{key:<6}  {r['K_in']:>4}  {r['coverage']:>5.2f}  "
              f"{r['flops_input']:>10,}  {r['flops_total']:>10,}  "
              f"{r['n_params']:>8,}  {r['top1_best']:.4f}  {vs:>+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
