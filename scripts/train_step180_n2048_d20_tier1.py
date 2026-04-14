"""Step 180: N=2048 D=20 K_hh=4 scratch Tier-1 — sub-4M FLOPs floor probe.

MOTIVATION
==========
New min-FLOPs phase exit record:
  step177: 95.13% @ ~4.72M FLOPs (N=1024 D=48 K_hh=4 warm+W_proj, Tier-2)
  step176-A: 96.18% @ ~6.1M FLOPs (N=2048 D=32 K_hh=4 scratch, Tier-2)

Step178 probes N=2048 D=24 at same ~4.72M FLOPs (running).

This step asks: can we go below 4M FLOPs and still hit ≥95%?

N=2048 D=20 K_hh=4: FLOPs = 3×2048×4×20×8 = 3,932,160 ≈ 3.93M
  → 37% below 6.18M budget.
  → Same FLOPs neighbourhood as step169 (N=1024 D=32 K_hh=8, ~6.1M) but way lower.
  → N=2048 has shown strong scaling: D=32 Tier-1 scratch = 94.93%.
  → D=20 (×0.625 of D=32) likely ~92-93% at Tier-1 → Tier-2 could hit 95%?

Phase-exit criterion: ≥95% @ ≤6.18M FLOPs. ~3.93M ✓ (36% below budget).
If Tier-1 ≥92%: Tier-2 likely viable. Big win if confirmed.

CONFIGS (N=2048, D=20, K_hh=4, K_iter=8, AH=1.0, 50% data, 75ep — Tier-1)
=========================================================================
  A : scratch α=1.0  ← sub-4M FLOPs floor probe

To reproduce:
    python -u scripts/train_step180_n2048_d20_tier1.py --device cpu
    python -u scripts/train_step180_n2048_d20_tier1.py --device mps
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
parser.add_argument("--epochs", type=int, default=75)
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 20; K_HH = 4; K_IN = 25; K_ITER = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
DATA_FRAC = 0.5

FLOPS = 3 * N * K_HH * D * K_ITER  # 3,932,160 ≈ 3.93M
OUT_PATH = ROOT / "results" / "train_step180_n2048_d20_tier1.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 180 — N=2048 D=20 K_hh=4 scratch Tier-1 (50% data 75ep)")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  α={ALPHA_AHEBB}  Data={DATA_FRAC*100:.0f}%  Epochs={EPOCHS}")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M) — 36% BELOW 6.18M budget")
    print(f"step177 ref (N=1024 D=48 Tier-2): 95.13% @ ~4.72M — new min-FLOPs record")
    print(f"step176-A ref (N=2048 D=32 Tier-2): 96.18% @ ~6.1M")
    print(f"step178 ref (N=2048 D=24 Tier-1, running): ~4.72M probe")
    print(f"Phase-exit target: ≥95% @ FLOPs ~3.93M\n{'='*70}")

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

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:int(n * DATA_FRAC)]
    sub = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(sub, batch_size=BATCH, shuffle=True, num_workers=0)

    t0 = time.time()
    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **kw)

    def _log(m):
        if str(DEVICE) == "mps": torch.mps.empty_cache()
        ep = m["epoch"] + 1
        if ep % 10 == 0:
            flag = " *** PHASE EXIT candidate! ***" if m["val_top1"] >= 0.95 else ""
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}{flag}", flush=True)

    history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
    elapsed = time.time() - t0
    top1h = [round(h.get("val_top1", 0.), 4) for h in history]
    best = max(top1h); bep = int(np.argmax(top1h)) + 1
    result = {"A": {
        "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
        "alpha_ahebb": ALPHA_AHEBB, "warm": False, "proj": False,
        "data_frac": DATA_FRAC,
        "top1_best": best, "top1_last": top1h[-1],
        "best_epoch": bep, "epochs_run": len(history),
        "top1_history": top1h, "elapsed_s": round(elapsed, 1),
        "n_params": n_p, "flops": FLOPS,
        "label": "A  scratch N=2048 D=20 K_hh=4 α=1.0 50% 75ep",
    }}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    pe = "✓ PHASE EXIT candidate!" if best >= 0.95 else f"({0.95-best:.3f}pp short of 95%)"
    print(f"\n{'='*70}")
    print(f"STEP 180: {best:.4f}  vs_step177={best-0.9513:+.4f}  vs_step176A={best-0.9618:+.4f}  FLOPs={FLOPS/1e6:.2f}M  {pe}")
    print(f"best_ep={bep}  params={n_p:,}")
    print(f"Results → {OUT_PATH}\n{'='*70}")


if __name__ == "__main__":
    main()
