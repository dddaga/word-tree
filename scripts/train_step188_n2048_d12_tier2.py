"""Step 188: N=2048 D=12 K_hh=4 scratch Tier-2 — sub-2.4M FLOPs phase exit attempt.

MOTIVATION
==========
step186 Tier-1 (50%/75ep): 93.10% best_ep=75 @ ~2.36M FLOPs.

Tier-2 projection (+1.89-1.91pp based on N=2048 D=16/D=20 pattern):
  step186 Tier-1=93.10% → step188 Tier-2 ≈ 95.0-95.0% ← borderline PHASE EXIT possible

FLOPs frontier confirmed exits:
  ~3.15M: step185 95.87% ✓ (D=16 current floor)
  ~3.93M: step181 96.03% ✓

If step188 ≥95%: new min-FLOPs record at ~2.36M (25% below current floor 3.15M).
Conservative projection (+1.1pp): 94.20% — no exit. Optimistic (+1.91pp): 95.01% — barely exits.
The Tier-2 lift trend has been INCREASING as D decreases (D=20→+1.89pp, D=16→+1.91pp).

Phase-exit criterion: ≥95% @ ≤6.18M FLOPs. ~2.36M ✓ (62% below budget).

CONFIGS (N=2048, D=12, K_hh=4, K_iter=8, AH=1.0, 100% data, 150ep — Tier-2)
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
D = 12; K_HH = 4; K_IN = 25; K_ITER = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

FLOPS = 3 * N * K_HH * D * K_ITER  # 2,359,296 ≈ 2.36M
OUT_PATH = ROOT / "results" / "train_step188_n2048_d12_tier2.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 188 — N=2048 D=12 K_hh=4 scratch Tier-2 (full data 150ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M) — 62% BELOW 6.18M budget")
    print(f"step186 Tier-1 ref: 93.10% @ ~2.36M → Tier-2 optimistic: ~95.0%")
    print(f"step185 floor: 95.87% @ 3.15M | step181: 96.03% @ 3.93M")
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
            flag = " *** PHASE EXIT! ***" if m["val_top1"] >= 0.95 else ""
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
                    "label": "A scratch N=2048 D=12 K_hh=4 α=1.0 full data 150ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    pe = "✓ PHASE EXIT! NEW MIN-FLOPs RECORD!" if best >= 0.95 else f"({0.95-best:.3f}pp short)"
    print(f"\n{'='*70}")
    print(f"STEP 188: {best:.4f}  vs_step186={best-0.9310:+.4f}  vs_step185={best-0.9587:+.4f}  FLOPs={FLOPS/1e6:.2f}M  {pe}")
    print(f"best_ep={bep}  params={n_p:,}  → {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
