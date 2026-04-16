"""Step 296: K_in=10 + aug T1 @ N=16384 — compound test of new best K_in.

MOTIVATION
==========
step290 T1 at N=16384 (no-aug): K_in=10 = 94.96% (+1.53pp vs Ref 93.43%) — BEST K_in.
step287 T2 at N=16384 (K_in=25 + aug): +0.99pp (strongest aug delta across all N).

Does K_in=10 compound with aug? Expected:
  Base aug effect at N=16384 T1: +2.11pp (step286)
  Base K_in=10 effect at N=16384 T1: +1.53pp
  If orthogonal: compound ~+3.5pp → 96.9%.
  If saturating: compound ~+2.5pp → 95.9%.

CONFIGS
  Ref_k25_naug : N=16384 K_in=25 no-aug T1 (reference, matches step286 Ref)
  A_k10_aug    : N=16384 K_in=10 + aug T1 (compound test)

Tier: T1 (75ep, 50% data)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",   default="auto")
parser.add_argument("--epochs",   type=int, default=75)
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data",     default="data/store.h5")
parser.add_argument("--data_aug", default="data/store_aug.h5")
parser.add_argument("--configs",  default="A_k10_aug")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 64; SEED = args.seed
N = 16384; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step296_kin10_aug_n16384_t1_seed{SEED}__{SLOT}.json"

# Baselines from prior steps:
REF_ACC = 0.934267520904541    # step286 T1 Ref (K_in=25, no-aug)
AUG_ACC = 0.9554  # approx from step286 aug result
K10_ACC = 0.9495942592620850   # step290 K_in=10 T1 no-aug

CONFIGS = {
    "A_k10_aug": (10, True,  "N=16384 K_in=10 + aug T1 (compound test)"),
}


def make_model(K_in):
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

    tr_clean_full, va = make_loaders(str(ROOT / args.data), batch_size=BATCH, seed=SEED)
    tr_aug_full, _    = make_loaders(str(ROOT / args.data_aug), batch_size=BATCH, seed=SEED)

    # 50% data subset
    def subset(loader):
        ds = loader.dataset
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(SEED))[:len(ds)//2].tolist()
        return torch.utils.data.DataLoader(Subset(ds, idx), batch_size=BATCH, shuffle=True)

    tr_clean = subset(tr_clean_full)
    tr_aug   = subset(tr_aug_full)

    print(f"Step 296 — K_in=10+aug T1 @ N=16384 ({EPOCHS}ep, 50% data)")
    print(f"  Baselines: Ref_k25=93.43%, K_in=10_naug=94.96%, aug_k25=~95.54%")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    if OUT_PATH.exists():
        try: results = json.loads(OUT_PATH.read_text())
        except Exception: pass

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        if key in results:
            r = results[key]
            print(f"  skip {key} (done: {r['top1_best']:.4f})")
            continue

        K_in, use_aug, desc = CONFIGS[key]
        tr = tr_aug if use_aug else tr_clean
        model = make_model(K_in)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{key}: {desc}  K_in={K_in}  aug={use_aug}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 15 == 0 else None))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        delta_ref = best - REF_ACC
        delta_aug = best - AUG_ACC
        delta_k10 = best - K10_ACC
        print(f"  → best={best:.4f} @ep{best_ep}  Δ_Ref={delta_ref*100:+.2f}pp  Δ_Aug={delta_aug*100:+.2f}pp  Δ_K10={delta_k10*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {"label": desc, "N": N, "K_in": K_in, "use_aug": use_aug,
                        "top1_best": best, "best_epoch": best_ep,
                        "delta_vs_ref": round(delta_ref, 4),
                        "delta_vs_aug": round(delta_aug, 4),
                        "delta_vs_k10": round(delta_k10, 4),
                        "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}\nSTEP 296 summary")
    print(f"  Ref_k25=93.43% | aug_k25~95.54% | k10_naug=94.96%")
    for k in keys:
        r = results.get(k, {})
        if r: print(f"  {k}  best={r['top1_best']:.4f}  Δ_Ref={r['delta_vs_ref']*100:+.2f}pp")


if __name__ == "__main__":
    main()
