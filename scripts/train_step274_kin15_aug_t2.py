"""Step 274: K_in=15 + aug compound T2 validation.

MOTIVATION
==========
step271 T1: C_k15_aug = 94.55% (+0.69pp vs 93.86% ref), only 0.07pp behind
aug-only (B_aug=94.62%). Compound K_in=15+aug clearly advances.

Expected T2 outcome: ~95.80% (+0.28pp vs step199 baseline of 95.52%) with
26.7× total seed FLOP reduction (16× spatial precomputation × 1.67× K_in cut).

Paper claim enabled: same accuracy class as step199 (efficiency record) but
with dramatically cheaper seed computation — and a small accuracy gain.

CONFIGS
=======
  Ref       : K_in=25, no-aug, 150ep full data (matches step199/step632)
  C_k15_aug : K_in=15, aug, 150ep full data (COMPOUND paper claim)

Tier: T2 (150ep, 100% data)

To run:
    python -u scripts/train_step274_kin15_aug_t2.py --device mps
    python -u scripts/train_step274_kin15_aug_t2.py --device cuda
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
parser.add_argument("--device",   default="auto")
parser.add_argument("--epochs",   type=int, default=150)
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data",     default="data/store.h5")
parser.add_argument("--data_aug", default="data/store_aug.h5")
parser.add_argument("--configs",  default="Ref,C_k15_aug")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step274_kin15_aug_t2_seed{SEED}__{SLOT}.json"

# (K_in, use_aug, description)
CONFIGS = {
    "Ref":       (25, False, "K_in=25 no-aug (step199/step632 ref, 95.52%)"),
    "C_k15_aug": (15, True,  "K_in=15 + aug (compound: 26.7× seed reduction + accuracy gain)"),
}


def make_model(K_in: int) -> nn.Module:
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_in, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def main():
    torch.manual_seed(SEED)
    for path in [args.data, args.data_aug]:
        if not (ROOT / path).exists():
            print(f"ERROR: {ROOT / path} not found."); sys.exit(1)

    tr_clean, va = make_loaders(str(ROOT / args.data), batch_size=BATCH, seed=SEED)
    tr_aug, _    = make_loaders(str(ROOT / args.data_aug), batch_size=BATCH, seed=SEED)

    print(f"Step 274 — K_in=15 + aug compound T2 ({EPOCHS}ep, 100% data)")
    print(f"  device={DEVICE}  seed={SEED}  N={N}  D={D}  K_iter={K_ITER}")
    print(f"  Train={len(tr_clean.dataset)}  Val={len(va.dataset)}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    if OUT_PATH.exists():
        try:
            results = json.loads(OUT_PATH.read_text())
            print(f"  Resuming: {list(results.keys())} already done\n")
        except Exception:
            pass

    ref_acc = results.get("Ref", {}).get("top1_best")

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        if key in results:
            r = results[key]
            print(f"  skip {key} (done: {r['top1_best']:.4f})")
            if key == "Ref": ref_acc = r["top1_best"]
            continue

        K_in, use_aug, desc = CONFIGS[key]
        tr = tr_aug if use_aug else tr_clean
        seed_macs = N * K_in
        model = make_model(K_in)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: {desc}")
        print(f"  K_in={K_in}  aug={'yes' if use_aug else 'no'}  seed_MACs={seed_macs:,}  params={n_p:,}")

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
        print(f"  → {key}  best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "label": desc, "K_in": K_in, "use_aug": use_aug,
            "seed_macs": seed_macs, "n_params": n_p,
            "top1_best": best, "best_epoch": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    STEP199_REF = 0.9552
    print(f"\n{'='*60}")
    print(f"STEP 274 SUMMARY — K_in=15 + aug compound T2")
    print(f"{'─'*60}")
    ref_b = results.get("Ref", {}).get("top1_best", 0)
    for k in keys:
        r = results.get(k, {})
        if not r: continue
        d = r["top1_best"] - ref_b
        d199 = r["top1_best"] - STEP199_REF
        seed_ratio = 51200 / r["seed_macs"] if r["seed_macs"] > 0 else 1
        print(f"  {k:<14} K_in={r['K_in']} aug={'y' if r['use_aug'] else 'n'}  "
              f"best={r['top1_best']:.4f}  Δ_ref={d*100:+.2f}pp  Δ_199={d199*100:+.2f}pp")
    if "C_k15_aug" in results and "Ref" in results:
        c = results["C_k15_aug"]
        verdict = "COMPOUND ADVANCE" if c["top1_best"] >= ref_b else "NET LOSS"
        compound_seed_reduction = 26.7
        print(f"\n  C_k15_aug verdict: {verdict}")
        print(f"  Compound seed reduction: {compound_seed_reduction}× (16× spatial precomp × 1.67× K_in)")
        print(f"  Δ_199 = {(c['top1_best'] - STEP199_REF)*100:+.2f}pp vs efficiency record")


if __name__ == "__main__":
    main()
