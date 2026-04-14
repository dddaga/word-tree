"""Step 133: Tier-1 confirmation of α=1.10 at N=4096.

MOTIVATION
==========
step125 showed α=1.10 gives +1.53pp over α=1.0 at N=4096 (20ep scout).
step88 had confirmed α=1.0 as optimal, but that was pre-patched arch.
This is a zero-cost improvement — if it holds at 75ep, it becomes the new default
and directly applies to the project best run.

CONFIGS (N=4096, D=64, K_hh=4, K_iter=12, turing=0.0, reflect=0.5, 50%/75ep)
==============================================================================
  Ref : α=1.0 (current default)
  A   : α=1.10 (step125 winner)
  B   : α=1.05 (runner-up, in case 1.10 regresses at 75ep)

To reproduce:
    python -u scripts/train_step133_alpha110_tier1.py --device mps
    python -u scripts/train_step133_alpha110_tier1.py --device mps --epochs 20
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
                    help="Training epochs (default 75 for Tier-1)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (e.g. A,B). Empty = run all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 4096; N_IN = 25088; N_OUT = 10; D = 64; K_ITER = 12; K_IN = 50
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0


@dataclass
class Config:
    key: str
    label: str
    alpha_ahebb: float

CONFIGS = [
    Config("Ref", "Ref  α=1.00 (current default)", 1.0),
    Config("A",   "A    α=1.10 (step125 winner)",   1.10),
    Config("B",   "B    α=1.05 (runner-up)",         1.05),
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
    topo = topology_kwargs(N)
    topo.pop("K_in", None); topo.pop("K_iter", None)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, encoding_mode="fourier",
        **topo,
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=cfg.alpha_ahebb, variant="wpos")

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    print(f"\n{'='*70}")
    print(f"Step 133 — Tier-1: α=1.10 at N=4096")
    print(f"N={N}  D={D}  K_iter={K_ITER}  Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    for c in CONFIGS:
        print(f"  {c.key:4s}  α={c.alpha_ahebb:.2f}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step133_alpha110_tier1.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}  params={n_params:,}")
        print(f"{'─'*60}")

        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
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
            "N": N, "D": D, "K_iter": K_ITER,
            "alpha_ahebb": cfg.alpha_ahebb,
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
        print(f"\n  top1_best={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 133 SUMMARY — α fine-tune Tier-1 at N=4096")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"  {key:4s}  α={r['alpha_ahebb']:.2f}  top1={r['top1_best']:.4f}  "
              f"vs_Ref={vs:+.4f}  ep={r['best_epoch']}/{r['epochs_run']}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")

if __name__ == "__main__":
    main()
