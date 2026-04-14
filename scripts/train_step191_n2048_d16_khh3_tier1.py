"""Step 191: N=2048 D=16 K_hh=3 scratch Tier-1 — K_hh=3 vs D=12 @ same 2.36M FLOPs.

MOTIVATION
==========
At 2.36M FLOPs, two configs are possible:
  D=12 K_hh=4: step188 Tier-2 = 94.62% (NO phase exit, 0.38pp short)
  D=16 K_hh=3: this run — same FLOPs, higher dimensional manifold (S^15 vs S^11)

FLOPs check: 3×2048×3×16×8 = 2,359,296 ≈ 2.36M ✓ (same as step188)

Key question: at fixed 2.36M FLOPs, does D=16 K_hh=3 beat D=12 K_hh=4?
  - D=16 K_hh=3 retains the higher-dim Fourier encoding (16D vs 12D)
  - K_hh=3 means K_local=2, K_random=1 — slightly less sparse than K_hh=4

If D=16 K_hh=3 Tier-1 > 93.10% (D=12 Tier-1 step186):
  → Tier-2 projection may reach 95% at 2.36M (would beat step188's 94.62%)
  → Would confirm D is the binding constraint, not K_hh

If D=16 K_hh=3 Tier-1 ≤ D=12 K_hh=4 at same FLOPs:
  → D=16 requires K_hh≥4 — the full connectivity density matters
  → Floor remains D=16 K_hh=4 @ 3.15M

Reference points:
  D=16 K_hh=4 Tier-1: 93.96% @ 3.15M (step182)
  D=12 K_hh=4 Tier-1: 93.10% @ 2.36M (step186)
  D=16 K_hh=2 Tier-1: running @ 1.57M (step190)

CONFIGS (N=2048, D=16, K_hh=3, K_iter=8, AH=1.0, 50% data, 75ep — Tier-1)
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
D = 16; K_HH = 3; K_IN = 25; K_ITER = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
DATA_FRAC = 0.5

FLOPS = 3 * N * K_HH * D * K_ITER  # 2,359,296 ≈ 2.36M
OUT_PATH = ROOT / "results" / "train_step191_n2048_d16_khh3_tier1.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 191 — N=2048 D=16 K_hh=3 scratch Tier-1 (50% data 75ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M) — same as D=12 K_hh=4 @ 2.36M")
    print(f"D=12 K_hh=4 Tier-2 ref: 94.62% (step188) | D=16 K_hh=4 Tier-1: 93.96% (step182)")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    print(f"  K_local={K_l}  K_random={K_r}  n_groups={ng}")
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
                    "label": "A scratch N=2048 D=16 K_hh=3 α=1.0 50% 75ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    d12_khh4_t2 = 0.9462   # step188 D=12 K_hh=4 Tier-2 (same FLOPs)
    d16_khh4_t1 = 0.9396   # step182 D=16 K_hh=4 Tier-1
    print(f"\n{'='*70}")
    print(f"STEP 191: {best:.4f}  vs_D12_Khh4_T2={best-d12_khh4_t2:+.4f}  vs_D16_Khh4_T1={best-d16_khh4_t1:+.4f}  FLOPs={FLOPS/1e6:.2f}M")
    print(f"best_ep={bep}  params={n_p:,}  → {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
