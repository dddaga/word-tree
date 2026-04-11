"""Step 208: N=8192 D=16 K_hh=2 K_iter=5 Tier-1 — N-scaling at sub-1.97M FLOPs path.

MOTIVATION
==========
step205 established D=16 record: N=4096 K_iter=5 T2 = 97.17% @ 1.97M FLOPs.
This is the most efficient path to near-record accuracy found so far.

N-scaling at D=16 K_hh=2 K_iter=5:
  N=2048 T2 (step199): 95.52% @ 0.98M FLOPs
  N=4096 T1 (step203): 96.08% @ 1.97M FLOPs (+2.12pp T1 gain vs N=2048 T1)
  N=4096 T2 (step205): 97.17% @ 1.97M FLOPs (+1.65pp T2 lift vs N=2048 T2)
  N=8192 T1 (this):    ?      @ 3.93M FLOPs ← parallel K_iter=5 probe

FLOPs = 3×8192×2×16×5 = 3,932,160 ≈ 3.93M — within ≤6.18M budget.

Key question: does the K_iter=5 N-scaling law continue beyond N=4096?
If T1 ≥ 96% → N=8192 may rival N=4096 T2 at T2 (sub-4M FLOPs).
If T1 regression like step206 → K_iter=5 also has D=16 bottleneck at N=8192.

Comparing both K_iter=6 (step207) and K_iter=5 (this) at N=8192 maps the full
N=8192 Pareto frontier and validates whether the K_iter=5 path remains superior.
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
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
DATA_FRAC = 0.5

FLOPS = 3 * N * K_HH * D * K_ITER  # 3,932,160 ≈ 3.93M
OUT_PATH = ROOT / "results" / "train_step208_n8192_d16_khh2_kiter5_tier1.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 208 — N=8192 D=16 K_hh=2 K_iter=5 Tier-1 (50% data 75ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"N=4096 K_iter=5 T2 (step205): 97.17% — D=16 record")
    print(f"N=8192 K_iter=6 T1 (step206): 95.11% — regression vs N=4096")
    print(f"Question: does K_iter=5 path continue N-scaling at N=8192?")
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
            flag = (" *** ABOVE N=4096 T1! N-SCALING HOLDS ***" if m["val_top1"] >= 0.9617 else
                    " *** PHASE EXIT ***"                         if m["val_top1"] >= 0.95 else "")
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
                    "label": "A scratch N=8192 D=16 K_hh=2 K_iter=5 α=1.0 50% 75ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    n4096_t1 = 0.9608; n8192_k6_t1 = 0.9511
    print(f"\n{'='*70}")
    print(f"STEP 208: {best:.4f}  vs_N4096_K5_T1={best-n4096_t1:+.4f}  vs_N8192_K6_T1={best-n8192_k6_t1:+.4f}  FLOPs={FLOPS/1e6:.2f}M")
    print(f"best_ep={bep}  params={n_p:,}  → {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
