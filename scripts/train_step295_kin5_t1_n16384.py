"""Step 295: K_in=5 T1 @ N=16384 — extend K_in curve to see if trend continues.

MOTIVATION
==========
step288/290 K_in curve at N=16384 T1:
  K_in=25 (Ref): 93.43%
  K_in=20: 93.61% (+0.18pp)
  K_in=15: 94.70% (+1.27pp)
  K_in=10: 94.96% (+1.53pp) — BEST SO FAR

Monotonic improvement as K_in decreases from 25 to 10. Does this trend continue at K_in=5?

Hypothesis: K_in=5 either continues the trend (+1.5 to +2pp) or breaks down due to
insufficient input coverage. At N=16384 each neuron sees 5 of 25088 features = 0.02%.
The regularization effect might be maximal, or the seed might be too sparse to learn.

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
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="A_k5")
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
OUT_PATH = ROOT / "results" / f"train_step295_kin5_t1_n16384_seed{SEED}__{SLOT}.json"

REF_ACC = 0.934267520904541   # step286 T1 Ref
K10_ACC = 0.9495942592620850  # step290 K_in=10 T1
CONFIGS = {
    "A_k5": (5, "N=16384 K_in=5 T1 (extend curve beyond K_in=10)"),
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

    tr_full, va = make_loaders(str(ROOT / args.data), batch_size=BATCH, seed=SEED)
    ds = tr_full.dataset
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(SEED))[:len(ds)//2].tolist()
    tr = torch.utils.data.DataLoader(Subset(ds, idx), batch_size=BATCH, shuffle=True)

    print(f"Step 295 — K_in=5 T1 @ N=16384 ({EPOCHS}ep, 50% data)")
    print(f"  Ref(K_in=25)=93.43%  K_in=10=94.96% (best so far)\n")

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
        print(f"\n{key}: {desc}  K_in={K_in}  params={n_p:,}")

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
        delta_k10 = best - K10_ACC
        print(f"  → best={best:.4f} @ep{best_ep}  Δ_Ref={delta_ref*100:+.2f}pp  Δ_K10={delta_k10*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {"label": desc, "N": N, "K_in": K_in,
                        "top1_best": best, "best_epoch": best_ep,
                        "delta_vs_ref": round(delta_ref, 4),
                        "delta_vs_k10": round(delta_k10, 4),
                        "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}\nSTEP 295 — K_in=5 T1 @ N=16384")
    for k in keys:
        r = results.get(k, {})
        if r: print(f"  {k}  K_in={r['K_in']}  best={r['top1_best']:.4f}  Δ_Ref={r['delta_vs_ref']*100:+.2f}pp")


if __name__ == "__main__":
    main()
