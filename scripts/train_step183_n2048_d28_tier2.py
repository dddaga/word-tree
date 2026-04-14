"""Step 183: N=2048 D=28 K_hh=4 scratch Tier-2 — 5.51M FLOPs phase exit confirmation.

MOTIVATION
==========
step179 Tier-1 (50%/75ep): 94.24% best_ep=57 @ ~5.51M FLOPs.

Tier-2 projection (+1.1-1.25pp based on N=2048 pattern):
  step179 Tier-1=94.24% → step183 Tier-2 ≈ 95.3-95.5% ← PHASE EXIT expected

Fills in the FLOPs frontier between step181 (~3.93M) and step176-A (~6.1M):
  ~3.15M: step182 Tier-1 (running)
  ~3.93M: step181 Tier-2 (ep40=94.80%, strong)
  ~4.72M: step177 95.13% ✓ / step178 TBD
  ~5.51M: step183 ← this run
  ~6.1M:  step176-A 96.18% ✓

Phase-exit criterion: ≥95% @ ≤6.18M FLOPs. ~5.51M ✓ (11% below budget).

CONFIGS (N=2048, D=28, K_hh=4, K_iter=8, AH=1.0, 100% data, 150ep — Tier-2)
=========================================================================
  A : scratch α=1.0

To reproduce:
    python -u scripts/train_step183_n2048_d28_tier2.py --device cpu
    python -u scripts/train_step183_n2048_d28_tier2.py --device mps
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

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=150)
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 28; K_HH = 4; K_IN = 25; K_ITER = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

FLOPS = 3 * N * K_HH * D * K_ITER  # 5,505,024 ≈ 5.51M
OUT_PATH = ROOT / "results" / "train_step183_n2048_d28_tier2.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 183 — N=2048 D=28 K_hh=4 scratch Tier-2 (full data 150ep)")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  α={ALPHA_AHEBB}  Data=100%  Epochs={EPOCHS}")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M) — 11% BELOW 6.18M budget")
    print(f"step179 Tier-1 ref: 94.24% @ ~5.51M → Tier-2 projection: ~95.3-95.5%")
    print(f"Phase-exit target: ≥95% @ FLOPs ~5.51M\n{'='*70}")

    torch.manual_seed(SEED)
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
    model = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB,
                                variant="wpos").to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_p:,}")

    tr, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    t0 = time.time()
    kw = trainer_kwargs(N, n_epochs=EPOCHS)
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
    result = {"A": {
        "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
        "alpha_ahebb": ALPHA_AHEBB, "warm": False, "proj": False,
        "data_frac": 1.0,
        "top1_best": best, "top1_last": top1h[-1],
        "best_epoch": bep, "epochs_run": len(history),
        "top1_history": top1h, "elapsed_s": round(elapsed, 1),
        "n_params": n_p, "flops": FLOPS,
        "label": "A  scratch N=2048 D=28 K_hh=4 α=1.0 full data 150ep",
    }}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    pe = "✓ PHASE EXIT!" if best >= 0.95 else f"({0.95-best:.3f}pp short)"
    print(f"\n{'='*70}")
    print(f"STEP 183: {best:.4f}  vs_step179={best-0.9424:+.4f}  vs_step176A={best-0.9618:+.4f}  FLOPs={FLOPS/1e6:.2f}M  {pe}")
    print(f"best_ep={bep}  params={n_p:,}")
    print(f"Results → {OUT_PATH}\n{'='*70}")


if __name__ == "__main__":
    main()
