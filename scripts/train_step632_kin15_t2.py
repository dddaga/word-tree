"""Step 632: K_in=15 T2 validation — paper-quality confirmation.

MOTIVATION
==========
step631 T1 result: K_in=15 scored 93.66% (Δ=−0.36pp vs Ref 94.01%) — within
the −0.5pp advance threshold. This T2 run gives paper-quality validation at
150ep on 100% data.

If K_in=15 holds within −0.5pp of step199 Ref (95.52%), it becomes the new
default, yielding a compound 26.7× seed FLOP reduction:
  - 16× from spatial precomputation (model_smallworld.py)
  - 1.67× from K_in 25→15

Seed MACs at K_in=15: 30,720 (vs original 819,200 before precomputation).
Routing still dominates: 655,360 MACs.

CONFIGS
=======
  Ref_k25 : K_in=25 (step199 reference config, retrained for fair comparison)
  A_k15   : K_in=15 (T1 winner, Δ=−0.36pp)

Scale: N=2048, D=16, K_hh=2, K_iter=5 (production config)
Tier: T2 (150ep, 100% data)

To run:
    python -u scripts/train_step632_kin15_t2.py --device mps --epochs 150
    python -u scripts/train_step632_kin15_t2.py --device cuda --epochs 150
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
parser.add_argument("--configs", default="Ref_k25,A_k15")
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
OUT_PATH = ROOT / "results" / f"train_step632_kin15_t2_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Ref_k25": (25, "K_in=25 reference (step199 config)"),
    "A_k15":   (15, "K_in=15 T1 winner — 26.7× compound seed reduction"),
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
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED)

    print(f"Step 632 — K_in=15 T2 validation (150ep, 100% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        K_in, desc = CONFIGS[key]
        model = make_model(K_in)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        seed_macs = N * K_in
        routing_macs = K_ITER * N * K_HH * D * 2
        total_macs = seed_macs + routing_macs
        print(f"{'─'*60}\n{key}: {desc}")
        print(f"  params={n_p:,}  K_in={K_in}  seed_MACs={seed_macs/1e3:.0f}K  routing={routing_macs/1e3:.0f}K  total={total_macs/1e3:.0f}K")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 15 == 0 else None))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1

        if key == "Ref_k25": ref_acc = best
        delta = best - (ref_acc or 0)
        print(f"  → K_in={K_in}  best={best:.4f} @ep{best_ep}  Δ_vs_ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "K_in": K_in, "label": desc, "n_params": n_p,
            "seed_macs": seed_macs, "routing_macs": routing_macs,
            "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(best - (ref_acc or 0), 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # Step199 reference for paper comparison
    STEP199_REF = 0.9552
    print(f"\n{'='*60}")
    print(f"STEP 632 SUMMARY — K_in=15 T2 (150ep 100% data)")
    print(f"{'='*60}")
    print(f"  {'Config':<10} {'K_in':>5} {'best':>7}  {'Δ_vs_Ref':>10}  {'Δ_vs_step199':>14}  paper?")
    for k, r in results.items():
        d_step199 = r["best"] - STEP199_REF
        paper = "YES" if r["delta_vs_ref"] >= -0.005 else "NO — accuracy loss"
        print(f"  {k:<10} {r['K_in']:>5} {r['best']:>7.4f}  {r['delta_vs_ref']*100:>+9.2f}pp  {d_step199*100:>+13.2f}pp  {paper}")
    print(f"\n  Paper accept: Δ_vs_step199 ≥ −0.5pp (K_in=15 seed reduction publishable)")


if __name__ == "__main__":
    main()
