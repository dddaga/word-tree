"""Step 269: K=4 + augmentation T2 validation — paper-quality.

MOTIVATION
==========
step268 T1 (75ep, 50% data) showed:
  Ref (K=5, no-aug): 95.21%
  A (K=4, no-aug):   95.54% (+0.33pp)
  B (K=4+aug COMBO): 96.23% (+1.02pp) ← STRONG advance

B config (K=4 + augmented data) advances with +1.02pp — paper-quality
confirmation needed. This T2 run validates at 150ep full data.

Config C (K=5+aug) result will arrive — if C > Ref + 0.5pp, aug alone explains
the gain; if B > C, the K=4+aug combo is synergistic.

CONFIGS
=======
  Ref      : K=5, no-aug (step199 reference)
  B_aug_k4 : K=4, augmented data (T1 STRONG winner)
  C_aug_k5 : K=5, augmented data (isolate aug-only effect)

Scale: N=2048, D=16, K_hh=2 (production config)
Tier: T2 (150ep, 100% data — uses augmented train data where applicable)

To run:
    python -u scripts/train_step269_k4aug_t2.py --device cuda --epochs 150
    python -u scripts/train_step269_k4aug_t2.py --device mps --epochs 150
    python -u scripts/train_step269_k4aug_t2.py --configs Ref,B_aug_k4 --device cuda
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--data_aug", default="data/store_aug.h5")
parser.add_argument("--configs", default="Ref,B_aug_k4,C_aug_k5")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step269_k4aug_t2_seed{SEED}__{SLOT}.json"

# (K_iter, use_aug, description)
CONFIGS = {
    "Ref":      (5, False, "K=5 no-aug (step199 reference)"),
    "B_aug_k4": (4, True,  "K=4 + aug (T1 STRONG winner, +1.02pp)"),
    "C_aug_k5": (5, True,  "K=5 + aug (aug-only isolation)"),
}


def make_model(K_iter: int) -> nn.Module:
    torch.manual_seed(SEED)
    K_r_ = max(1, K_HH // 4); K_l_ = K_HH - K_r_
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=25, K_iter=K_iter, K_local=K_l_, K_random=K_r_,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def main():
    torch.manual_seed(SEED)
    for attr, path in [("data", args.data), ("data_aug", args.data_aug)]:
        if not (ROOT / path).exists():
            print(f"ERROR: {ROOT / path} not found."); sys.exit(1)

    tr_clean, va = make_loaders(str(ROOT / args.data), batch_size=BATCH, seed=SEED)
    tr_aug, _    = make_loaders(str(ROOT / args.data_aug), batch_size=BATCH, seed=SEED)

    print(f"Step 269 — K=4+aug T2 (150ep, full data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        K_iter, use_aug, desc = CONFIGS[key]
        tr = tr_aug if use_aug else tr_clean
        model = make_model(K_iter)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        routing_macs = EPOCHS * N * K_HH * D * 2  # not per-iter, just for reference
        routing_macs = K_iter * N * K_HH * D * 2
        print(f"{'─'*60}\n{key}: {desc}")
        print(f"  params={n_p:,}  K_iter={K_iter}  aug={'yes' if use_aug else 'no'}  routing={routing_macs/1e3:.0f}K MACs")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 15 == 0 else None))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1

        if key == "Ref": ref_acc = best
        delta = best - (ref_acc or 0)
        print(f"  → K_iter={K_iter} aug={'y' if use_aug else 'n'}  best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "label": desc, "K_iter": K_iter, "use_aug": use_aug,
            "n_params": n_p, "routing_macs": routing_macs,
            "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    STEP199_REF = 0.9552
    print(f"\n{'='*60}")
    print(f"STEP 269 SUMMARY — K=4+aug T2")
    print(f"{'='*60}")
    for k, r in results.items():
        d199 = r["best"] - STEP199_REF
        print(f"  {k:<12} K={r['K_iter']} aug={'y' if r['use_aug'] else 'n'}  best={r['best']:.4f}  Δ_ref={r['delta_vs_ref']*100:+.2f}pp  Δ_199={d199*100:+.2f}pp")
    if "Ref" in results and "B_aug_k4" in results:
        d = results["B_aug_k4"]["delta_vs_ref"]
        verdict = "STRONG NEW DEFAULT" if d >= 0.005 else ("MEDIUM" if d >= 0 else "KILLED")
        print(f"\n  B_aug_k4 verdict: {verdict}")
        if "C_aug_k5" in results:
            aug_only = results["C_aug_k5"]["delta_vs_ref"]
            combo_vs_aug = results["B_aug_k4"]["best"] - results["C_aug_k5"]["best"]
            print(f"  Aug-only gain: {aug_only*100:+.2f}pp | K=4 extra on top: {combo_vs_aug*100:+.2f}pp")


if __name__ == "__main__":
    main()
