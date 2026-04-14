"""Step 203: N=4096 D=16 K_hh=2 K_iter=5 Tier-1 — N-scaling at sub-1% FLOPs config.

MOTIVATION
==========
step199 T2: N=2048 D=16 K_hh=2 K_iter=5 → 95.52% @ 0.98M FLOPs (sub-1% exit confirmed).

This tests N=4096 with the same sub-1% config:
  N=2048 K_iter=5 (step199 T2): 95.52% @ 0.98M FLOPs ✓ sub-1% exit
  N=4096 K_iter=5 (this):       ?     @ 1.97M FLOPs ← still ≤ 6.18M budget

FLOPs = 3×4096×2×16×5 = 1,966,080 ≈ 1.97M

Complements step201 (N=4096 K_iter=6 @ 2.36M).
Together they map: does N=4096 maintain the K_iter=5 vs K_iter=6 advantage pattern
observed at N=2048 (K_iter=5 outperforms K_iter=6 at Tier-1 despite fewer FLOPs)?

N-scaling law hypothesis: doubling N should add ~1-2pp at same D/K config.
  N=2048 K_iter=5 T1: 93.96% → Tier-2: 95.52%
  N=4096 K_iter=5 T1: projected ~95-96% ← would confirm strong N-scaling

CONFIGS (N=4096, D=16, K_hh=2, K_iter=5, AH=1.0, 50% data, 75ep — Tier-1)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

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
N = 4096; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
DATA_FRAC = 0.5

FLOPS = 3 * N * K_HH * D * K_ITER  # 1,966,080 ≈ 1.97M
OUT_PATH = ROOT / "results" / "train_step203_n4096_d16_khh2_kiter5_tier1.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 203 — N=4096 D=16 K_hh=2 K_iter=5 Tier-1 (50% data 75ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"N=2048 K_iter=5 T1 ref: 93.96%@ep72 → T2: 95.52%")
    print(f"N-scaling probe: N×2 at sub-1% FLOPs config")
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
                    "label": "A scratch N=4096 D=16 K_hh=2 K_iter=5 α=1.0 50% 75ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    n2048_t1 = 0.9396  # step197 N=2048 K_iter=5 Tier-1
    print(f"\n{'='*70}")
    print(f"STEP 203: {best:.4f}  vs_N2048_T1={best-n2048_t1:+.4f}  FLOPs={FLOPS/1e6:.2f}M")
    print(f"best_ep={bep}  params={n_p:,}  → {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
