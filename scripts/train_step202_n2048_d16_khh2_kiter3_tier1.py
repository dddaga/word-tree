"""Step 202: N=2048 D=16 K_hh=2 K_iter=3 Tier-1 — Efficiency floor below K_iter=4.

MOTIVATION
==========
K_iter reduction axis at D=16 K_hh=2 N=2048:
  K_iter=6 (step195 T2): 96.08% @ 1.18M FLOPs ✓ ≤1% criterion MET
  K_iter=5 (step199 T2): 95.52% @ 0.98M FLOPs ✓ sub-1% exit confirmed
  K_iter=4 (step196 T1): 92.74% @ 0.79M FLOPs KILLED (−2.14pp)
  K_iter=3 (this):       ?     @ 0.59M FLOPs ← sub-0.5% probe

FLOPs = 3×2048×2×16×3 = 589,824 ≈ 0.59M — same as N=1024 K_iter=6 (step198 killed)

The K_iter=4 result (92.74%) at 0.79M was deemed too low for Tier-2.
K_iter=3 will likely be even lower. But: knowing the floor shape matters.

If K_iter=3 ≥ 90% → K_iter axis degrades gracefully (useful for efficiency characterization).
If K_iter=3 < 85% → sharp collapse (routing needs minimum ~4 steps to function at D=16).

This is a quick Tier-1 probe (75ep, 50% data) — low cost, high information value for the
efficiency floor characterization paper.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=3, AH=1.0, 50% data, 75ep — Tier-1)
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
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 3
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
DATA_FRAC = 0.5

FLOPS = 3 * N * K_HH * D * K_ITER  # 589,824 ≈ 0.59M  ← sub-0.5% of VGG16 FC FLOPs
OUT_PATH = ROOT / "results" / "train_step202_n2048_d16_khh2_kiter3_tier1.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 202 — N=2048 D=16 K_hh=2 K_iter=3 Tier-1 (50% data 75ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M) — sub-0.5% of VGG16 FC FLOPs")
    print(f"K_iter axis: K_iter=5→95.52%, K_iter=4→92.74% killed, K_iter=3→?")
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
                    "label": "A scratch N=2048 D=16 K_hh=2 K_iter=3 α=1.0 50% 75ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    kiter4_ref = 0.9274; kiter5_ref = 0.9396
    print(f"\n{'='*70}")
    print(f"STEP 202: {best:.4f}  vs_Kiter4={best-kiter4_ref:+.4f}  vs_Kiter5={best-kiter5_ref:+.4f}  FLOPs={FLOPS/1e6:.2f}M")
    print(f"best_ep={bep}  params={n_p:,}  → {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
