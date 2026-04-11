"""Step 204: N=4096 D=16 K_hh=2 K_iter=6 Tier-2 — N-scaling law validation.

MOTIVATION
==========
step201 Tier-1 (50%/75ep): 95.64%@ep66 @ 2.36M FLOPs.
Phase exit criterion hit at TIER-1 — extraordinary result.

N-scaling at D=16 K_hh=2 K_iter=6:
  N=1024 (step198 T1): 88.92% @ 0.59M FLOPs — killed
  N=2048 (step195 T2): 96.08% @ 1.18M FLOPs ← ≤1% FLOPs criterion MET
  N=4096 (step201 T1): 95.64%@ep66 @ 2.36M FLOPs ← T1 already ≥95%!
  N=4096 (this T2):    ~97%+ projected @ 2.36M FLOPs ← N-scaling law

Tier-2 lift projection: +1.5-2.0pp over T1 (N=2048 pattern):
  Conservative (+1.0pp): 95.64% + 1.0pp = 96.64%
  Middle (+1.5pp):       95.64% + 1.5pp = 97.14%
  Optimistic (+2.0pp):   95.64% + 2.0pp = 97.64%

Middle/optimistic project ~97%+ — would approach step89 D=64 record (97.86%) using D=16.
This validates that N-scaling can compensate for D-reduction in SGNNET.

Key question: can D=16 N=4096 match D=64 N=4096 accuracy? Comparing D=64 step89=97.86%.

CONFIGS (N=4096, D=16, K_hh=2, K_iter=6, AH=1.0, 100% data, 150ep — Tier-2)
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

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 4096; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 6
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

FLOPS = 3 * N * K_HH * D * K_ITER  # 2,359,296 ≈ 2.36M
OUT_PATH = ROOT / "results" / "train_step204_n4096_d16_khh2_kiter6_tier2.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 204 — N=4096 D=16 K_hh=2 K_iter=6 Tier-2 (full data 150ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"step201 T1: 95.64%@ep66 — T2 proj: ~96.6-97.6% (N-scaling law)")
    print(f"N=2048 T2 ref (step195): 96.08% | D=64 record (step89): 97.86%")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
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
            flag = " *** N-SCALING RECORD! ***" if m["val_top1"] >= 0.97 else (
                   " *** PHASE EXIT ***" if m["val_top1"] >= 0.95 else "")
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
                    "label": "A scratch N=4096 D=16 K_hh=2 K_iter=6 α=1.0 full 150ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    n2048_t2 = 0.9608; d64_record = 0.9786
    pe = "✓ PHASE EXIT" if best >= 0.95 else f"({0.95-best:.3f}pp short)"
    print(f"\n{'='*70}")
    print(f"STEP 204: {best:.4f}  vs_N2048_T2={best-n2048_t2:+.4f}  vs_D64={best-d64_record:+.4f}  FLOPs={FLOPS/1e6:.2f}M  {pe}")
    print(f"best_ep={bep}  params={n_p:,}  → {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
