"""Step 294: K_in=15 no-aug T2 @ N=4096 — validate step293 T1 crossover at T2.

MOTIVATION
==========
step293 T1: K_in=15 @ N=4096 = 96.18% (+0.33pp vs Ref 95.85%).
T2 validation: does +0.33pp hold at 150ep/100% data?

step279 did K_in=15+aug compound T2 at N=4096: +0.18pp.
Isolation K_in=15 T2 at N=4096 hasn't been directly measured.

T2 Ref at N=4096 (from step273): 97.12%.
Expected K_in=15 T2: ~97.0-97.3% (consistent with +0.1-0.3pp T1-to-T2 compression).

Tier: T2 (150ep, 100% data)
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
parser.add_argument("--configs", default="A_k15")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 64; SEED = args.seed
N = 4096; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 15
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step294_kin15_t2_n4096_seed{SEED}__{SLOT}.json"

REF_T2 = 0.9712   # step273 N=4096 T2 Ref
CONFIGS = {
    "A_k15": (15, "N=4096 K_in=15 no-aug T2 (validates step293 T1 +0.33pp)"),
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
    if not (ROOT / args.data).exists():
        print(f"ERROR: {ROOT / args.data} not found."); sys.exit(1)

    tr, va = make_loaders(str(ROOT / args.data), batch_size=BATCH, seed=SEED)

    print(f"Step 294 — K_in=15 T2 @ N=4096 ({EPOCHS}ep, 100% data)")
    print(f"  device={DEVICE}  T1 delta (step293)=+0.33pp  Ref T2=97.12%\n")

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

        K_in, desc = CONFIGS[key]
        model = make_model(K_in)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{key}: {desc}  K_in={K_in}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 25 == 0 else None))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        delta = best - REF_T2
        print(f"  → best={best:.4f} @ep{best_ep}  Δ_T2={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {"label": desc, "N": N, "K_in": K_in,
                        "top1_best": best, "best_epoch": best_ep,
                        "ref_acc": REF_T2, "delta_vs_ref": round(delta, 4),
                        "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}\nSTEP 294 summary: K_in=15 T2 @ N=4096")
    for k in keys:
        r = results.get(k, {})
        if r: print(f"  {k}  best={r['top1_best']:.4f}  Δ_T2={r['delta_vs_ref']*100:+.2f}pp")


if __name__ == "__main__":
    main()
