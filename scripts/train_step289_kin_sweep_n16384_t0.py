"""Step 289: K_in sweep @ N=16384 T0 — characterize K_in anomaly.

MOTIVATION
==========
step288 C_k15_naug=94.70% (+1.27pp vs Ref 93.43%). K_in=15 HELPS at N=16384.
Opposite of N<=8192 pattern (K_in=15 costs -0.33pp). Is this a monotonic trend?
Does K_in=10 help even more? Does K_in=20 also beat K_in=25?

Ref_n16384=93.43% (step286). C_k15_naug=94.70% (step288).
Testing K_in=20 and K_in=10 to map the K_in curve at N=16384.

Tier: T0 (20ep, 50% data — scout only)
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
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="A_k20,B_k10")
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
OUT_PATH = ROOT / "results" / f"train_step289_kin_n16384_t0_seed{SEED}__{SLOT}.json"

# Ref from step286: 93.43%. K_in=15 from step288: 94.70% (+1.27pp).
REF_ACC = 0.934267520904541
CONFIGS = {
    "A_k20": (20, "N=16384 K_in=20 no-aug T0"),
    "B_k10": (10, "N=16384 K_in=10 no-aug T0"),
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
    if not (ROOT / args.data).exists():
        print(f"ERROR: {ROOT / args.data} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(ROOT / args.data), batch_size=BATCH, seed=SEED)
    ds = tr_full.dataset
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(SEED))[:len(ds)//2].tolist()
    tr = torch.utils.data.DataLoader(Subset(ds, idx), batch_size=BATCH, shuffle=True)

    print(f"Step 289 — K_in sweep @ N=16384 T0 ({EPOCHS}ep, 50% data)")
    print(f"  device={DEVICE}  Ref=93.43% (step286)  K_in=15=94.70% (step288)")

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
            print(f"  skip {key} (done: {r['top1_best']:.4f}  Δ={( r['top1_best']-REF_ACC)*100:+.2f}pp)")
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
            if (m['epoch']+1) % 10 == 0 else None))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        delta = best - REF_ACC
        print(f"  → best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp vs Ref  {elapsed:.0f}s")

        results[key] = {"label": desc, "N": N, "K_in": K_in,
                        "top1_best": best, "best_epoch": best_ep,
                        "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}\nSTEP 289 — K_in @ N=16384 T0 summary")
    print(f"  Ref_k25 =93.43% (step286)  K_in=15=94.70% (step288 C)")
    for k in keys:
        r = results.get(k, {})
        if r: print(f"  {k}  K_in={r['K_in']}  best={r['top1_best']:.4f}  Δ={r['delta_vs_ref']*100:+.2f}pp")


if __name__ == "__main__":
    main()
