"""Step 192: N=2048 D=16 K_hh=3 scratch Tier-2 — sub-2.4M FLOPs phase exit attempt.

MOTIVATION
==========
step191 Tier-1 (50%/75ep): 94.68% best_ep=71 @ ~2.36M FLOPs.

This is already ABOVE step188's Tier-2 result (94.62% @ 2.36M D=12 K_hh=4).
D=16 dimensionality (S^15 manifold) is the binding constraint — not K_hh.
K_hh=3 retains the high-dim encoding while cutting FLOPs 25% vs K_hh=4.

Tier-2 projection (+1.1-1.9pp based on N=2048 pattern):
  Conservative: 94.68% + 1.1pp = 95.78% ← PHASE EXIT ✓
  Middle: 94.68% + 1.5pp = 96.18% ← PHASE EXIT ✓
  Optimistic: 94.68% + 1.9pp = 96.58% ← PHASE EXIT ✓

ALL projections exceed 95%. Near-certain phase exit.

If confirmed: NEW MIN-FLOPs EFFICIENCY RECORD @ ~2.36M (beats step185 @ 3.15M by 25%).

FLOPs: 3×2048×3×16×8 = 2,359,296 ≈ 2.36M (62% below 6.18M budget)

Reference chain:
  step185 (D=16 K_hh=4): floor @ 3.15M — 95.87% ✓
  step191 (D=16 K_hh=3 Tier-1): 94.68% @ 2.36M — now Tier-2

CONFIGS (N=2048, D=16, K_hh=3, K_iter=8, AH=1.0, 100% data, 150ep — Tier-2)
"""
from __future__ import annotations
import argparse, json, sys, time
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
D = 16; K_HH = 3; K_IN = 25; K_ITER = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

FLOPS = 3 * N * K_HH * D * K_ITER  # 2,359,296 ≈ 2.36M
OUT_PATH = ROOT / "results" / "train_step192_n2048_d16_khh3_tier2.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 192 — N=2048 D=16 K_hh=3 scratch Tier-2 (full data 150ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M) — 62% BELOW 6.18M budget")
    print(f"step191 Tier-1: 94.68% → Tier-2 proj: 95.8-96.6% — near-certain PHASE EXIT")
    print(f"step185 current record: 95.87% @ 3.15M | THIS: 2.36M (-25% FLOPs)")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    model = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB,
                                variant="wpos").to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_p:,}")

    tr, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    t0 = time.time()
    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

    def _log(m):
        if str(DEVICE) == "mps": torch.mps.empty_cache()
        ep = m["epoch"] + 1
        if ep % 10 == 0:
            flag = " *** PHASE EXIT! NEW RECORD! ***" if m["val_top1"] >= 0.95 else ""
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}{flag}", flush=True)

    history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
    elapsed = time.time() - t0
    top1h = [round(h.get("val_top1", 0.), 4) for h in history]
    best = max(top1h); bep = int(np.argmax(top1h)) + 1
    result = {"A": {"N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
                    "alpha_ahebb": ALPHA_AHEBB, "warm": False, "data_frac": 1.0,
                    "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
                    "epochs_run": len(history), "top1_history": top1h,
                    "elapsed_s": round(elapsed, 1), "n_params": n_p, "flops": FLOPS,
                    "label": "A scratch N=2048 D=16 K_hh=3 α=1.0 full data 150ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    pe = "✓ PHASE EXIT! NEW MIN-FLOPs RECORD!" if best >= 0.95 else f"({0.95-best:.3f}pp short)"
    print(f"\n{'='*70}")
    print(f"STEP 192: {best:.4f}  vs_step191T1={best-0.9468:+.4f}  vs_step185={best-0.9587:+.4f}  FLOPs={FLOPS/1e6:.2f}M  {pe}")
    print(f"best_ep={bep}  params={n_p:,}  → {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
