"""Step 200: N=2048 D=16 K_hh=1 K_iter=8 Tier-1 — Minimum connectivity probe.

MOTIVATION
==========
K_hh reduction axis at D=16 N=2048:
  K_hh=4 (step185): 3.15M → 95.87% ✓
  K_hh=3 (step192): 2.36M → 95.90% ✓
  K_hh=2 (step193): 1.57M → 95.67% ✓ (with K_iter=8)
  K_hh=1 (this):    0.79M → ? ← minimum connectivity (1 edge per neuron)

FLOPs = 3×2048×1×16×8 = 786,432 ≈ 0.79M — 50% below K_hh=2 K_iter=8

K_hh=1 means each neuron connects to only 1 other neuron (K_random=1, K_local=0).
This is the absolute minimum graph connectivity. The question:
  - Does the graph remain connected enough for K_iter=8 to propagate information?
  - Or does K_hh=1 cause information bottleneck / isolated subgraphs?

Prior: K_iter=4 at D=16 K_hh=2 was killed at 92.74% (too few routing steps).
K_hh=1 K_iter=8 has the SAME FLOPs (0.79M) but inverts the tradeoff:
  - K_iter=4: more connectivity (K_hh=2), fewer steps (4)
  - K_iter=8: minimum connectivity (K_hh=1), more steps (8) ← this experiment

If K_hh=1 K_iter=8 ≥ K_iter=4 K_hh=2 (92.74%): more steps compensate for less connectivity.
If K_hh=1 K_iter=8 < 90%: connectivity floor hits before K_iter floor.

FLOPs comparison at 0.79M:
  K_hh=2 K_iter=4 (step196): 0.79M → 92.74% KILLED
  K_hh=1 K_iter=8 (this):    0.79M → ?

CONFIGS (N=2048, D=16, K_hh=1, K_iter=8, AH=1.0, 50% data, 75ep — Tier-1)
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
D = 16; K_HH = 1; K_IN = 25; K_ITER = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
DATA_FRAC = 0.5

FLOPS = 3 * N * K_HH * D * K_ITER  # 786,432 ≈ 0.79M  ← minimum connectivity
OUT_PATH = ROOT / "results" / "train_step200_n2048_d16_khh1_kiter8_tier1.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 200 — N=2048 D=16 K_hh=1 K_iter=8 Tier-1 (50% data 75ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M) — minimum connectivity (K_hh=1)")
    print(f"Same FLOPs as K_iter=4 K_hh=2 (step196 KILLED @92.74%) — inverted tradeoff")
    print(f"K_hh=2 K_iter=8 ref (step190 T1): 93.86% @ 1.57M")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    # K_hh=1: K_r=max(1, 1//4)=1, K_l=0 — all random connections, no local
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
                    "label": "A scratch N=2048 D=16 K_hh=1 K_iter=8 α=1.0 50% 75ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    kiter4_ref = 0.9274  # step196 K_hh=2 K_iter=4 same FLOPs (killed)
    khh2_ki8_ref = 0.9386  # step190 K_hh=2 K_iter=8 Tier-1
    verdict = "→ ADVANCE TO TIER-2" if best >= 0.93 else ("→ beats K_iter=4" if best >= kiter4_ref else "→ KILLED")
    print(f"\n{'='*70}")
    print(f"STEP 200: {best:.4f}  vs_Kiter4_sameFLOPs={best-kiter4_ref:+.4f}  vs_Khh2Ki8={best-khh2_ki8_ref:+.4f}  FLOPs={FLOPS/1e6:.2f}M  {verdict}")
    print(f"best_ep={bep}  params={n_p:,}  → {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
