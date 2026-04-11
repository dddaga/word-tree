"""Step 199: N=2048 D=16 K_hh=2 K_iter=5 Tier-2 — Sub-1% FLOPs phase exit attempt.

MOTIVATION
==========
step197 Tier-1: 93.96%@ep72 @ 0.98M FLOPs. Meets ≥93% threshold → advance to Tier-2.

K_iter=5 at 0.98M FLOPs sits 18% below the ≤1% FLOPs threshold (1.18M).
Tier-2 lift from N=2048 scratch pattern: +1.5-2.1pp over Tier-1.
  Conservative (+1.5pp): 93.96% + 1.50pp = 95.46% ← PHASE EXIT ✓
  Middle    (+1.8pp):    93.96% + 1.80pp = 95.76% ← PHASE EXIT ✓
  Optimistic (+2.1pp):   93.96% + 2.10pp = 96.06% ← PHASE EXIT ✓

All three projections land above 95%. This is near-certain sub-1% FLOPs phase exit.

Comparison at Tier-2 (projected):
  K_iter=6 (step195): 1.18M → 96.08% ✓ criterion MET
  K_iter=5 (this):    0.98M → ~95.5-96% projected ← sub-1% of criterion threshold
  K_iter=4 (step196): 0.79M → killed (92.74% T1)

If confirmed ≥95%: SUB-1% FLOPs (0.98M = 0.79% of VGG16 FC) + ≥95% accuracy.
That is a stronger efficiency result than step195.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, AH=1.0, 100% data, 150ep — Tier-2)
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
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

FLOPS = 3 * N * K_HH * D * K_ITER  # 983,040 ≈ 0.98M  ← sub-1% FLOPs target
OUT_PATH = ROOT / "results" / "train_step199_n2048_d16_khh2_kiter5_tier2.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 199 — N=2048 D=16 K_hh=2 K_iter=5 Tier-2 (full data 150ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M) — 18% BELOW ≤1% FLOPs threshold")
    print(f"step197 Tier-1: 93.96%@ep72 → Tier-2 proj: ~95.5-96% — NEAR-CERTAIN SUB-1% EXIT")
    print(f"K_iter=6 ref (step195): 96.08% @ 1.18M | THIS: K_iter=5 @ 0.98M")
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
            flag = " *** SUB-1% FLOPs PHASE EXIT! ***" if m["val_top1"] >= 0.95 else ""
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
                    "label": "A scratch N=2048 D=16 K_hh=2 K_iter=5 α=1.0 full 150ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    pe = "✓ SUB-1% FLOPs PHASE EXIT!" if best >= 0.95 else f"({0.95-best:.3f}pp short)"
    print(f"\n{'='*70}")
    print(f"STEP 199: {best:.4f}  vs_step197T1={best-0.9396:+.4f}  vs_step195={best-0.9608:+.4f}  FLOPs={FLOPS/1e6:.2f}M  {pe}")
    print(f"best_ep={bep}  params={n_p:,}  → {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
