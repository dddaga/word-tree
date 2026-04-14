"""Step 176: N=2048 D=32 K_hh=4 K_iter=8 scratch Tier-2 — full data 150ep.

MOTIVATION
==========
step173-Ref (N=2048 D=32 K_hh=4 K_iter=8 scratch, 50%/75ep): 94.93% — extraordinary.
step173-B (warm+W_proj): 94.70% — UNDERPERFORMS Ref by -0.23pp (warm-start reversal at scale).

Key insight: at N=2048 D=32, scratch K=8 builds optimal topology for K=8 routing.
Warm-start from K=12 teacher creates topology mismatch → net negative.

Tier-2 projection: Ref 94.93% at 50%/75ep → full data 150ep expected ~95.5-96%.
This is the cleanest phase-exit confirmation path at ~6.1M FLOPs (within 6.18M budget).

Phase-exit criterion: ≥95% @ ≤6.18M FLOPs.
N=2048 D=32 K_hh=4 K_iter=8 FLOPs ≈ 6.1M ✓

No teacher needed — pure scratch run.

CONFIGS (N=2048, D=32, K_hh=4, K_iter=8, AH=1.0, 100% data, 150ep — Tier-2)
=============================================================================
  A   : scratch α=1.0 ← clean phase-exit confirmation
  B   : scratch α=1.05 ← calibration (confirmed +0.79pp at N=4096)

To reproduce:
    python -u scripts/train_step176_n2048_d32_scratch_tier2.py --device mps
    python -u scripts/train_step176_n2048_d32_scratch_tier2.py --device cpu
    python -u scripts/train_step176_n2048_d32_scratch_tier2.py --configs A --device mps
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

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--configs", default="", help="Comma-sep config keys. Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 32; K_HH = 4; K_IN = 25; K_ITER = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

OUT_PATH = ROOT / "results" / "train_step176_n2048_d32_scratch_tier2.json"


@dataclass
class Config:
    key:         str
    label:       str
    alpha_ahebb: float

CONFIGS = [
    Config("A", "A  scratch N=2048 D=32 K_hh=4 α=1.0  (phase-exit baseline)", 1.0),
    Config("B", "B  scratch N=2048 D=32 K_hh=4 α=1.05 (calibration)", 1.05),
]

_loaders_cache = None

def get_loaders():
    global _loaders_cache
    if _loaders_cache is None:
        _loaders_cache = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    return _loaders_cache


def _build(seed: int = SEED) -> nn.Module:
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER,
        K_local=K_l, K_random=K_r, n_groups=ng,
        norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=1.0, variant="wpos")


def main():
    print(f"\n{'='*70}")
    print(f"Step 176 — N=2048 D=32 K_hh=4 scratch Tier-2 (full data 150ep)")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  Data=100%  Epochs={EPOCHS}")
    print(f"step173-Ref reference (50%/75ep scratch): 94.93%")
    print(f"step173-B (warm+W_proj): 94.70% — UNDERPERFORMS Ref (warm-start reversal)")
    print(f"Tier-2 expected: ~95.5-96% (clean phase exit at ~6.1M FLOPs)")
    print(f"Phase-exit target: ≥95% @ FLOPs ~6.1M\n{'='*70}")

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active = [(i, c) for i, c in enumerate(CONFIGS) if not cfg_filter or c.key in cfg_filter]
    tr, va = get_loaders()
    results = {}

    for i, cfg in active:
        print(f"\n{'─'*60}\nConfig {cfg.key}: {cfg.label}\n{'─'*60}")
        torch.manual_seed(SEED + i)
        K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
        base = SGNNET_SmallWorld(
            N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
            K_in=K_IN, K_iter=K_ITER,
            K_local=K_l, K_random=K_r, n_groups=ng,
            norm_mode="l2", encoding_mode="fourier",
        )
        resonant = SGNNET_Resonant(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
            alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0,
        )
        model = SGNNET_AntiHebbian(resonant, alpha_ahebb=cfg.alpha_ahebb,
                                    variant="wpos").to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}  α={cfg.alpha_ahebb}")

        t0  = time.time()
        kw  = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)

        def _log(m):
            if str(DEVICE) == "mps": torch.mps.empty_cache()
            ep = m["epoch"] + 1
            if ep % 10 == 0:
                flag = " *** PHASE EXIT! ***" if m["val_top1"] >= 0.95 else ""
                print(f"  ep{ep:3d}  val={m['val_top1']:.4f}{flag}", flush=True)

        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0
        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1
        results[cfg.key] = {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "alpha_ahebb": cfg.alpha_ahebb, "warm": False, "proj": False,
            "data_frac": 1.0,
            "top1_best": best, "top1_last": top1h[-1],
            "best_epoch": bep, "epochs_run": len(history),
            "top1_history": top1h, "elapsed_s": round(elapsed, 1),
            "n_params": n_p, "label": cfg.label,
        }
        pe = "✓ PHASE EXIT!" if best >= 0.95 else f"({0.95-best:.3f}pp short)"
        print(f"\n  top1={best:.4f}  vs_step173Ref={best-0.9493:+.4f}  {pe}")
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}\nSTEP 176 SUMMARY — N=2048 D=32 K_hh=4 scratch Tier-2\n{'='*70}")
    print(f"step173-Ref (50% data) reference: 94.93%")
    print(f"Phase-exit target: ≥95% @ FLOPs ~6.1M")
    for k, r in results.items():
        pe = "✓ PHASE EXIT!" if r["top1_best"] >= 0.95 else f"-{0.95-r['top1_best']:.3f}pp"
        print(f"{k}  {r['top1_best']:.4f}  best_ep={r['best_epoch']}  {pe}")
    print(f"\nResults → {OUT_PATH}")


if __name__ == "__main__":
    main()
