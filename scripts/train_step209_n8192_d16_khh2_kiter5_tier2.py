"""Step 209: N=8192 D=16 K_hh=2 K_iter=5 Tier-2 — N=8192 K_iter=5 full training.

MOTIVATION
==========
step208 Tier-1 (50%/75ep): 95.77%@ep51 @ 3.93M FLOPs. PHASE EXIT threshold met.
Key finding: K_iter=5 beats K_iter=6 at N=8192 T1 (+0.66pp: 95.77% vs 95.11%).
Same T1 regression vs N=4096 pattern (−0.31pp), but converges faster (ep51 vs ep72).

N-scaling at D=16 K_hh=2 K_iter=5 — T2 trajectory:
  N=2048 T2 (step199): 95.52% @ 0.98M FLOPs
  N=4096 T2 (step205): 97.17% @ 1.97M FLOPs (+1.65pp lift vs N=2048 T2)
  N=8192 T2 (this):    ?      @ 3.93M FLOPs ← does +1.65pp pattern hold?

T2 lift projection:
  N=4096 K_iter=5: T1=96.08% → T2=97.17% (+1.09pp lift)
  N=8192 K_iter=5: T1=95.77% + 1.09pp = ~96.9% (conservative)
  If pattern scales: T1=95.77% + 1.65pp = ~97.4% (optimistic — would beat step205 record!)

FLOPs = 3×8192×2×16×5 = 3,932,160 ≈ 3.93M — within ≤6.18M budget.
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
parser.add_argument("--epochs", type=int, default=150)
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 64; SEED = 42; DATA = "data/store.h5"
N = 8192; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

FLOPS = 3 * N * K_HH * D * K_ITER  # 3,932,160 ≈ 3.93M
OUT_PATH = ROOT / "results" / "train_step209_n8192_d16_khh2_kiter5_tier2.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 209 — N=8192 D=16 K_hh=2 K_iter=5 Tier-2 (full data 150ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"step208 T1: 95.77%@ep51 — K_iter=5 beats K_iter=6 at N=8192 T1 (+0.66pp)")
    print(f"T2 proj: ~96.9-97.4% | N=4096 K5 T2 record: 97.17% | D=64 rec: 97.86%")
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

    tr, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    t0 = time.time()
    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

    def _log(m):
        if str(DEVICE) == "mps": torch.mps.empty_cache()
        ep = m["epoch"] + 1
        if ep % 10 == 0:
            flag = (" *** NEW D=16 RECORD! ***"      if m["val_top1"] >= 0.9718 else
                    " *** MATCHES N=4096 K5 T2 ***"  if m["val_top1"] >= 0.9710 else
                    " *** N-SCALING HOLDS ***"        if m["val_top1"] >= 0.970 else
                    " *** PHASE EXIT ***"             if m["val_top1"] >= 0.95 else "")
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}{flag}", flush=True)

    history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
    elapsed = time.time() - t0
    top1h = [round(h.get("val_top1", 0.), 4) for h in history]
    best = max(top1h); bep = int(np.argmax(top1h)) + 1
    result = {"A": {"N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
                    "alpha_ahebb": ALPHA_AHEBB, "warm": False, "data_frac": 1.0,
                    "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
                    "epochs_run": len(history), "top1_history": top1h,
                    "elapsed_s": round(elapsed, 1), "n_params": n_p, "flops": FLOPS,
                    "label": "A scratch N=8192 D=16 K_hh=2 K_iter=5 α=1.0 full 150ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    n4096_k5_t2 = 0.9717; d64_record = 0.9786
    pe = "✓ PHASE EXIT" if best >= 0.95 else f"({0.95-best:.3f}pp short)"
    scaling = ("✓ NEW D=16 RECORD"          if best > n4096_k5_t2 else
               "✓ N-SCALING HOLDS (matches)" if best >= n4096_k5_t2 else
               f"N-SCALING PARTIAL ({best-n4096_k5_t2:+.4f} vs N=4096 K5 T2)")
    print(f"\n{'='*70}")
    print(f"STEP 209: {best:.4f}  vs_N4096_K5_T2={best-n4096_k5_t2:+.4f}  vs_D64={best-d64_record:+.4f}  FLOPs={FLOPS/1e6:.2f}M  {pe}")
    print(f"best_ep={bep}  params={n_p:,}  {scaling}")
    print(f"→ {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
