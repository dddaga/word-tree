"""Step 214: N=2048 D=8 K_hh=8 K_iter=8 Tier-1 — D=8 high connectivity probe.

MOTIVATION
==========
User hypothesis: D=8 has few directions, but with K_hh=2 the routing is too sparse
to explore those directions. Higher K_hh might compensate for low D by giving each
neuron access to more diverse neighbors.

D=8 K_hh=4 (step187): 91.26% @ 1.57M FLOPs — KILLED.
D=16 K_hh=2 (step190): 93.86% @ 1.57M FLOPs — at same FLOPs, D wins by +2.60pp.

But we never tested D=8 with K_hh > 4. Maybe the D>K_hh principle breaks when K_hh
is sufficiently large to overcome directional crowding on S^7.

This experiment: D=8 K_hh=8 @ 3.15M FLOPs.
Comparison: D=16 K_hh=4 @ 3.15M (step182 T1=93.96%, step185 T2=95.87%).
If D=8 K_hh=8 T1 ≥ 93.96% → connectivity CAN compensate for low D.
If D=8 K_hh=8 T1 << 93.96% → D is truly the binding constraint.

FLOPs = 3×2048×8×8×8 = 3,145,728 ≈ 3.15M
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
D = 8; K_HH = 8; K_IN = 25; K_ITER = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
DATA_FRAC = 0.5

FLOPS = 3 * N * K_HH * D * K_ITER  # 3,145,728 ≈ 3.15M
OUT_PATH = ROOT / "results" / "train_step214_n2048_d8_khh8_kiter8_tier1.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 214 — N=2048 D=8 K_hh=8 K_iter=8 Tier-1 (50% data 75ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"Hypothesis: high connectivity compensates for low D")
    print(f"D=8 K_hh=4 (step187): 91.26% | D=16 K_hh=4 (step182): 93.96%")
    print(f"If D=8 K_hh=8 ≥ 93.96% → connectivity compensates for D")
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
            flag = (" *** BEATS D=16 K_hh=4 T1! ***" if m["val_top1"] >= 0.9396 else
                    " *** ABOVE D=8 K_hh=4 ***"       if m["val_top1"] >= 0.9126 else "")
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
                    "label": "A scratch N=2048 D=8 K_hh=8 K_iter=8 α=1.0 50% 75ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    d8_khh4 = 0.9126; d16_khh4 = 0.9396; d16_khh2 = 0.9386
    if best >= d16_khh4:
        verdict = "✓ CONNECTIVITY COMPENSATES — D=8 K_hh=8 matches D=16 K_hh=4"
    elif best >= d8_khh4:
        verdict = f"PARTIAL — above D=8 K_hh=4 (+{best-d8_khh4:.4f}) but below D=16 ({best-d16_khh4:+.4f})"
    else:
        verdict = "WORSE — more K_hh hurts at D=8 (over-smoothing?)"
    print(f"\n{'='*70}")
    print(f"STEP 214: {best:.4f}  vs_D8_K4={best-d8_khh4:+.4f}  vs_D16_K4={best-d16_khh4:+.4f}  FLOPs={FLOPS/1e6:.2f}M")
    print(f"best_ep={bep}  params={n_p:,}  {verdict}")
    print(f"→ {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
