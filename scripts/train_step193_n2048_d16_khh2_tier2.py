"""Step 193: N=2048 D=16 K_hh=2 scratch Tier-2 — sub-1.6M FLOPs phase exit attempt.

MOTIVATION
==========
step190 Tier-1 (50%/75ep): 93.86% best_ep=75 @ ~1.57M FLOPs.

KEY FINDING: D=16 K_hh=2 (1.57M) outperforms D=8 K_hh=4 (1.57M) by +2.60pp at Tier-1.
D=16 dimensionality (S^15 manifold) is the binding constraint — NOT K_hh connectivity.
At same FLOPs, higher D + lower K_hh >> lower D + higher K_hh.

Tier-2 projection (+1.1-1.9pp based on N=2048 pattern):
  Conservative (+1.1pp): 93.86% + 1.10pp = 94.96% ← barely below 95%
  Middle (+1.5pp):       93.86% + 1.50pp = 95.36% ← PHASE EXIT ✓
  Optimistic (+1.9pp):   93.86% + 1.90pp = 95.76% ← PHASE EXIT ✓

Middle/optimistic both project phase exit. Borderline even at conservative.

If confirmed ≥95%: NEW MIN-FLOPs RECORD @ ~1.57M (beats step185 @ 3.15M by 50%,
beats step192 D=16 K_hh=3 pending @ 2.36M by 33%).

Reference:
  step185 floor: 95.87% @ 3.15M (D=16 K_hh=4)
  step192 pending: ~96% @ 2.36M (D=16 K_hh=3 Tier-2 running)
  THIS:  ~95.4% @ 1.57M (D=16 K_hh=2 Tier-2)

FLOPs: 3×2048×2×16×8 = 1,572,864 ≈ 1.57M (75% below 6.18M budget)

CONFIGS (N=2048, D=16, K_hh=2, K_iter=8, AH=1.0, 100% data, 150ep — Tier-2)
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
D = 16; K_HH = 2; K_IN = 25; K_ITER = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

FLOPS = 3 * N * K_HH * D * K_ITER  # 1,572,864 ≈ 1.57M
OUT_PATH = ROOT / "results" / "train_step193_n2048_d16_khh2_tier2.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 193 — N=2048 D=16 K_hh=2 scratch Tier-2 (full data 150ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M) — 75% BELOW 6.18M budget")
    print(f"step190 Tier-1: 93.86% → Tier-2 proj: 94.96-95.76% (borderline exit)")
    print(f"step185 current record: 95.87% @ 3.15M | THIS: 1.57M (-50% FLOPs if exits)")
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
            flag = " *** PHASE EXIT! NEW MIN-FLOPs RECORD! ***" if m["val_top1"] >= 0.95 else ""
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
                    "label": "A scratch N=2048 D=16 K_hh=2 α=1.0 full data 150ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    pe = "✓ PHASE EXIT! NEW MIN-FLOPs RECORD!" if best >= 0.95 else f"({0.95-best:.3f}pp short)"
    print(f"\n{'='*70}")
    print(f"STEP 193: {best:.4f}  vs_step190T1={best-0.9386:+.4f}  vs_step185={best-0.9587:+.4f}  FLOPs={FLOPS/1e6:.2f}M  {pe}")
    print(f"best_ep={bep}  params={n_p:,}  → {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
