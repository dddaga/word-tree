"""Step 182: N=2048 D=16 K_hh=4 scratch Tier-1 — sub-3.2M FLOPs floor probe.

MOTIVATION
==========
FLOPs floor progress so far (phase exits confirmed):
  step181 (running): N=2048 D=20 Tier-2, ep40=94.80% @ ~3.93M — likely ≥95%
  step177: 95.13% @ ~4.72M (confirmed phase exit)
  step176-A: 96.18% @ ~6.1M (confirmed phase exit)

step179 Tier-1: 94.24% @ ~5.51M. Tier-2 expected ~95.4% (step183 will confirm).

Question: can N=2048 D=16 hit ≥95% at ~3.15M FLOPs?

N=2048 D=16 K_hh=4: FLOPs = 3×2048×4×16×8 = 3,145,728 ≈ 3.15M
  → 49% below 6.18M budget.
  → D=20 Tier-1 = 94.14% and looking strong in Tier-2.
  → D=16 (×0.8 of D=20) expected ~92-93% at Tier-1.
  → If ≥92%: Tier-2 likely ≥95% → floor pushed to ~3.15M.

Phase-exit criterion: ≥95% @ ≤6.18M FLOPs. ~3.15M ✓ (49% below budget).

CONFIGS (N=2048, D=16, K_hh=4, K_iter=8, AH=1.0, 50% data, 75ep — Tier-1)
=========================================================================
  A : scratch α=1.0  ← sub-3.2M FLOPs floor probe

To reproduce:
    python -u scripts/train_step182_n2048_d16_tier1.py --device mps
    python -u scripts/train_step182_n2048_d16_tier1.py --device cpu
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
D = 16; K_HH = 4; K_IN = 25; K_ITER = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
DATA_FRAC = 0.5

FLOPS = 3 * N * K_HH * D * K_ITER  # 3,145,728 ≈ 3.15M
OUT_PATH = ROOT / "results" / "train_step182_n2048_d16_tier1.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 182 — N=2048 D=16 K_hh=4 scratch Tier-1 (50% data 75ep)")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  α={ALPHA_AHEBB}  Data={DATA_FRAC*100:.0f}%  Epochs={EPOCHS}")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M) — 49% BELOW 6.18M budget")
    print(f"step180 ref (D=20 Tier-1): 94.14% @ ~3.93M → Tier-2 ~95%+ (step181 running)")
    print(f"step177 ref: 95.13% @ ~4.72M FLOPs (confirmed phase exit)")
    print(f"Phase-exit target: ≥95% @ FLOPs ~3.15M (new record if Tier-2 succeeds)\n{'='*70}")

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
        "label": "A  scratch N=2048 D=16 K_hh=4 α=1.0 50% 75ep",
    }}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    pe = "✓ Tier-2 viable!" if best >= 0.93 else f"({best:.4f} — check if Tier-2 worthwhile)"
    print(f"\n{'='*70}")
    print(f"STEP 182: {best:.4f}  vs_step180={best-0.9414:+.4f}  FLOPs={FLOPS/1e6:.2f}M  {pe}")
    print(f"best_ep={bep}  params={n_p:,}")
    print(f"Results → {OUT_PATH}\n{'='*70}")


if __name__ == "__main__":
    main()
