"""Step 213: N=8192 D=16 K_hh=3 K_iter=5 Tier-1 — K_hh axis at N=8192.

MOTIVATION
==========
D=16 K_hh=2 has a confirmed ceiling at ~97.17% (both N=4096 and N=8192 T2 reach exactly 97.17%).
The ceiling appears independent of N — suggesting K_hh=2 connectivity is the binding constraint.

K_hh axis results at N=2048 D=16:
  K_hh=2 K_iter=6 T2 (step195): 96.08% @ 1.18M FLOPs
  K_hh=3 K_iter=8 T2 (step192): 95.90% @ 2.36M FLOPs — K_hh=3 slightly worse (different K_iter!)
  K_hh=2 K_iter=8 T2 (step193): 95.67% @ 1.57M FLOPs

At N=8192 with K_iter=5:
  K_hh=2 T2 (step209): 97.17% ceiling
  K_hh=3 T1 (this):   ? — does more connectivity break the ceiling?

K_hh=3 triples the local connectivity vs K_hh=2.
FLOPs = 3×8192×3×16×5 = 5,898,240 ≈ 5.90M — within ≤6.18M budget.

If K_hh=3 T1 ≥95% → T2 projected to possibly break 97.17% ceiling.
This is the direct test of whether K_hh=2 connectivity is the binding constraint
preventing D=16 from matching D=64 (97.86%).
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

EPOCHS = args.epochs; BATCH = 64; SEED = 42; DATA = "data/store.h5"
N = 8192; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 3; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
DATA_FRAC = 0.5

FLOPS = 3 * N * K_HH * D * K_ITER  # 5,898,240 ≈ 5.90M
OUT_PATH = ROOT / "results" / "train_step213_n8192_d16_khh3_kiter5_tier1.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 213 — N=8192 D=16 K_hh=3 K_iter=5 Tier-1 (50% data 75ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"D=16 K_hh=2 ceiling: 97.17% (N=4096 step205, N=8192 step209 — both same!)")
    print(f"Hypothesis: K_hh=2 connectivity is binding. K_hh=3 may break ceiling.")
    print(f"Ref: N=2048 K_hh=2 T1=93.86%, K_hh=3 T1=94.68% (+0.82pp at N=2048)")
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
            flag = (" *** ABOVE K_hh=2 T1! CEILING BROKEN ***" if m["val_top1"] >= 0.9578 else
                    " *** PHASE EXIT ***"                        if m["val_top1"] >= 0.95 else "")
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
                    "label": "A scratch N=8192 D=16 K_hh=3 K_iter=5 α=1.0 50% 75ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    n8192_khh2_k5_t1 = 0.9577; d16_ceiling = 0.9717
    pe = "✓ PHASE EXIT" if best >= 0.95 else f"({0.95-best:.3f}pp short)"
    ceiling = ("✓ ABOVE K_hh=2 T1 — K_hh=3 advantage confirmed" if best > n8192_khh2_k5_t1 else
               f"BELOW K_hh=2 T1 ({best-n8192_khh2_k5_t1:+.4f}pp)")
    print(f"\n{'='*70}")
    print(f"STEP 213: {best:.4f}  vs_K_hh2_K5_T1={best-n8192_khh2_k5_t1:+.4f}  vs_D16_ceiling={best-d16_ceiling:+.4f}  FLOPs={FLOPS/1e6:.2f}M  {pe}")
    print(f"best_ep={bep}  params={n_p:,}  {ceiling}")
    print(f"→ {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
