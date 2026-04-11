"""Step 215: N=2048 D=8 K_hh=16 K_iter=8 Tier-1 — D=8 maximum connectivity.

MOTIVATION
==========
Stress test: can extreme connectivity (K_hh=16) at D=8 approach D=16 accuracy?
K_hh=16 means each neuron connects to 16 neighbors per routing step — 8× more than
our efficiency optimum (K_hh=2).

FLOPs = 3×2048×16×8×8 = 6,291,456 ≈ 6.29M — just above the 6.18M budget.
At this FLOPs budget, D=32 K_hh=4 (step176) achieves 96.18%.

If D=8 K_hh=16 << 96.18% at similar FLOPs, D is unambiguously the constraint.
If D=8 K_hh=16 ≈ 96% → connectivity CAN substitute for dimensionality at high cost.

Reference chain:
  D=8  K_hh=4  (step187): 91.26% @ 1.57M
  D=8  K_hh=8  (step214): running @ 3.15M
  D=8  K_hh=16 (this):    ? @ 6.29M
  D=16 K_hh=2  (step190): 93.86% @ 1.57M
  D=16 K_hh=4  (step182): 93.96% @ 3.15M
  D=32 K_hh=4  (step176): 96.18% @ 6.10M
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
D = 8; K_HH = 16; K_IN = 25; K_ITER = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
DATA_FRAC = 0.5

FLOPS = 3 * N * K_HH * D * K_ITER  # 6,291,456 ≈ 6.29M
OUT_PATH = ROOT / "results" / "train_step215_n2048_d8_khh16_kiter8_tier1.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 215 — N=2048 D=8 K_hh=16 K_iter=8 Tier-1 (50% data 75ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"Stress test: max connectivity at D=8")
    print(f"D=8 K_hh=4: 91.26% | D=32 K_hh=4 @ same FLOPs: 96.18%")
    print(f"If D=8 K_hh=16 << 96% → D is unambiguously the constraint")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    print(f"  K_local={K_l}  K_random={K_r}  n_groups={ng}  batch={BATCH}")
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
            flag = (" *** APPROACHES D=32! ***" if m["val_top1"] >= 0.95 else
                    " *** ABOVE D=16 T1! ***"   if m["val_top1"] >= 0.9396 else
                    " *** ABOVE D=8 K_hh=4 ***" if m["val_top1"] >= 0.9126 else "")
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}{flag}", flush=True)

    history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
    elapsed = time.time() - t0
    top1h = [round(h.get("val_top1", 0.), 4) for h in history]
    best = max(top1h); bep = int(np.argmax(top1h)) + 1
    result = {"A": {"N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
                    "alpha_ahebb": ALPHA_AHEBB, "warm": False, "data_frac": DATA_FRAC,
                    "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
                    "epochs_run": len(history), "top1_history": top1h,
                    "elapsed_s": round(elapsed, 1), "n_params": n_p, "flops": FLOPS,
                    "label": "A scratch N=2048 D=8 K_hh=16 K_iter=8 α=1.0 50% 75ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    d8_khh4 = 0.9126; d16_khh4 = 0.9396; d32_khh4 = 0.9493
    if best >= 0.95:
        verdict = "✓ D=8 VIABLE — extreme connectivity achieves phase exit"
    elif best >= d16_khh4:
        verdict = f"PARTIAL — beats D=16 T1 but at 4× FLOPs"
    elif best >= d8_khh4:
        verdict = f"MARGINAL — K_hh helps (+{best-d8_khh4:.4f}pp) but D still dominates"
    else:
        verdict = "OVER-SMOOTHING — more K_hh made it worse"
    print(f"\n{'='*70}")
    print(f"STEP 215: {best:.4f}  vs_D8_K4={best-d8_khh4:+.4f}  vs_D16_K4={best-d16_khh4:+.4f}  vs_D32_K4={best-d32_khh4:+.4f}  FLOPs={FLOPS/1e6:.2f}M")
    print(f"best_ep={bep}  params={n_p:,}  {verdict}")
    print(f"→ {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
