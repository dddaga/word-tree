"""Step 297: K_in=10 + aug T2 @ N=16384 — validate T1 winner at full scale.

MOTIVATION
==========
step296 T1 (75ep, 50% data): A_k10_aug = 96.56% @ ep57
  Δ_Ref = +3.13pp vs step286 Ref (93.43%)
  Δ_Aug = +1.02pp vs aug-only (95.54%)
  Δ_K10 = +1.60pp vs K_in=10 no-aug (94.96%)

Compounding of K_in=10 + aug at N=16384 is clearly additive and advances to T2.
Compare to: step291 D_k15_aug T2 = 96.89% (K_in=15 + aug).
  If K_in=10+aug T2 ≥ 96.89%: K_in=10 is strictly better than K_in=15 at N=16384.
  If K_in=10+aug T2 < 96.89%: K_in=15 remains the crossover winner at N=16384.

Prior evidence:
  step287 : N=16384 + aug T2 = 96.87% (K_in=25)
  step291 C: N=16384 + K_in=15 (no aug) T2 = 96.13%
  step291 D: N=16384 + K_in=15 + aug T2 = 96.89%
  step296  : N=16384 + K_in=10 + aug T1 = 96.56% (@ep57)

Tier: T2 (150ep, 100% data)
Device: MPS (mini_mps)
"""
from __future__ import annotations
import argparse, json, os, sys, time
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
parser.add_argument("--device",   default="auto")
parser.add_argument("--epochs",   type=int, default=150)
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data",     default="data/store.h5")
parser.add_argument("--data_aug", default="data/store_aug.h5")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 64; SEED = args.seed
N = 16384; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step297_kin10_aug_n16384_t2_seed{SEED}__{SLOT}.json"

# Reference numbers for delta reporting
REF_ACC    = 0.934267520904541   # step286 T1 Ref (K_in=25, no-aug)
AUG_T2_ACC = 0.9687              # step287 T2 (K_in=25 + aug) ≈ 96.87%
K15_AUG_T2 = 0.9689              # step291 D (K_in=15 + aug T2) = 96.89%
T1_ACC     = 0.9656              # step296 T1 result = 96.56%


def make_model():
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=10, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def main():
    for path in [args.data, args.data_aug]:
        if not (ROOT / path).exists():
            print(f"ERROR: {ROOT / path} not found."); sys.exit(1)

    # T2: full dataset (no subset)
    _, va        = make_loaders(str(ROOT / args.data), batch_size=BATCH, seed=SEED)
    tr_aug, _    = make_loaders(str(ROOT / args.data_aug), batch_size=BATCH, seed=SEED)

    print(f"Step 297 — K_in=10+aug T2 @ N=16384 ({EPOCHS}ep, 100% data, device={DEVICE})")
    print(f"  Baselines: Ref=93.43% | aug_T2=96.87% | k15_aug_T2=96.89% | T1=96.56%")
    print(f"  Train={len(tr_aug.dataset)}  Val={len(va.dataset)}")

    model = make_model()
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  N={N} K_in=10 params={n_p:,}")

    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr_aug, val_loader=va, device=DEVICE, **kw)

    t0 = time.time()
    history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
        print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
        if (m['epoch']+1) % 15 == 0 else None))
    elapsed = time.time() - t0

    top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
    best, best_ep = max(top1h), int(np.argmax(top1h)) + 1

    result = {
        "A_k10_aug": {
            "label":        "N=16384 K_in=10 + aug T2",
            "N":            N, "K_in": 10, "use_aug": True,
            "top1_best":    best, "best_epoch": best_ep,
            "top1_history": [round(v, 4) for v in top1h],
            "delta_vs_ref":     round(best - REF_ACC, 4),
            "delta_vs_aug_t2":  round(best - AUG_T2_ACC, 4),
            "delta_vs_k15_aug": round(best - K15_AUG_T2, 4),
            "delta_vs_t1":      round(best - T1_ACC, 4),
            "elapsed_s":    round(elapsed),
            "n_params":     n_p,
        }
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))

    r = result["A_k10_aug"]
    print(f"\n{'='*60}\nSTEP 297 SUMMARY")
    print(f"  A_k10_aug  best={best:.4f} @ep{best_ep}")
    print(f"  Δ_Ref      = {r['delta_vs_ref']*100:+.2f}pp  (vs 93.43%)")
    print(f"  Δ_aug_T2   = {r['delta_vs_aug_t2']*100:+.2f}pp  (vs 96.87% K_in=25+aug T2)")
    print(f"  Δ_k15_aug  = {r['delta_vs_k15_aug']*100:+.2f}pp  (vs 96.89% K_in=15+aug T2)")
    print(f"  Δ_T1       = {r['delta_vs_t1']*100:+.2f}pp  (T1→T2 gain)")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
