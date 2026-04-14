"""Step 196: N=2048 D=16 K_hh=2 K_iter=4 Tier-1 — BELOW ≤1% FLOPs hard target.

MOTIVATION
==========
K_iter reduction axis at D=16 K_hh=2:
  K_iter=8 (step193 Tier-2): 95.67%@ep135 @ 1.57M FLOPs ✓ PHASE EXIT
  K_iter=6 (step195 Tier-2): running @ 1.18M FLOPs ← AT ≤1% target
  K_iter=4 (this):           ? @ 0.79M FLOPs ← BELOW ≤1% target (63% of 1.2M)

FLOPs = 3×2048×2×16×4 = 786,432 ≈ 0.79M — 34% BELOW the ≤1% FLOPs target.

step194 K_iter=6 Tier-1 = 94.88%@ep71 (BETTER than K_iter=8 Tier-1 93.86% despite 25% fewer FLOPs).
Key question: does K_iter=4 continue the trend or collapse?

If K_iter=4 Tier-1 ≥ ~92% → Tier-2 could reach ~94-95% → potential sub-1% FLOPs phase exit.
If < ~90% → K_iter reduction hits a floor; K_iter=6 (1.18M) is likely the minimum.

FLOPs comparison at D=16 K_hh=2:
  K_iter=8 (step193): 1.57M → 95.67% ✓ PHASE EXIT
  K_iter=6 (step195): 1.18M → ~96% projected (Tier-2 running)
  K_iter=4 (this):    0.79M → ?  ← 34% below ≤1% FLOPs target

CONFIGS (N=2048, D=16, K_hh=2, K_iter=4, AH=1.0, 50% data, 75ep — Tier-1)
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
parser.add_argument("--epochs", type=int, default=75)
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 4
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
DATA_FRAC = 0.5

FLOPS = 3 * N * K_HH * D * K_ITER  # 786,432 ≈ 0.79M  ← BELOW ≤1% FLOPs target
OUT_PATH = ROOT / "results" / "train_step196_n2048_d16_khh2_kiter4_tier1.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 196 — N=2048 D=16 K_hh=2 K_iter=4 Tier-1 (50% data 75ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M) — 34% BELOW ≤1% FLOPs target (1.2M)")
    print(f"K_iter=6 ref (step194 T1): 94.88%@ep71 | K_iter=8 ref (step190 T1): 93.86%")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    print(f"  K_local={K_l}  K_random={K_r}  K_iter={K_ITER}  n_groups={ng}")
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

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:int(n * DATA_FRAC)]
    sub = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(sub, batch_size=BATCH, shuffle=True, num_workers=0)

    t0 = time.time()
    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

    def _log(m):
        if str(DEVICE) == "mps": torch.mps.empty_cache()
        ep = m["epoch"] + 1
        if ep % 10 == 0:
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)

    history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
    elapsed = time.time() - t0
    top1h = [round(h.get("val_top1", 0.), 4) for h in history]
    best = max(top1h); bep = int(np.argmax(top1h)) + 1
    result = {"A": {"N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
                    "alpha_ahebb": ALPHA_AHEBB, "warm": False, "data_frac": DATA_FRAC,
                    "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
                    "epochs_run": len(history), "top1_history": top1h,
                    "elapsed_s": round(elapsed, 1), "n_params": n_p, "flops": FLOPS,
                    "label": "A scratch N=2048 D=16 K_hh=2 K_iter=4 α=1.0 50% 75ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    kiter6_ref = 0.9488  # step194 D=16 K_hh=2 K_iter=6 Tier-1
    kiter8_ref = 0.9386  # step190 D=16 K_hh=2 K_iter=8 Tier-1
    print(f"\n{'='*70}")
    print(f"STEP 196: {best:.4f}  vs_Kiter6={best-kiter6_ref:+.4f}  vs_Kiter8={best-kiter8_ref:+.4f}  FLOPs={FLOPS/1e6:.2f}M")
    print(f"best_ep={bep}  params={n_p:,}  → {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
