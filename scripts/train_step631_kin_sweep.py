"""Step 631: K_in sweep — find minimum viable fan-in for seed gather.

MOTIVATION
==========
Seed gather (K_in=25) costs ~1.64M FLOPs per inference (~70% of total).
With spatial precomputation (merged in model_smallworld.py), seed now costs
only N×K_in MACs = 51K MACs at K_in=25. But K_in still scales linearly.

If K_in=10 holds accuracy → seed MACs = 20K (2.5× less than K_in=25).
If K_in=5 holds accuracy  → seed MACs = 10K (5× less than K_in=25).

Combined with spatial precomputation already applied:
  K_in=25: 51K seed MACs
  K_in=10: 20K seed MACs
  K_in=5:  10K seed MACs

Total inference FLOPs at K_in=5, K_iter=5:
  seed: 10K + routing: 655K + readout: 41K ≈ 706K total
  vs step199: 51K + 655K + 41K ≈ 747K total (minimal gain from K_in alone)

HOWEVER: memory bandwidth matters more than FLOPs.
  K_in=5 gather: [B, N, 5] = 80KB at B=128 (vs [B, N, 25] = 400KB)
  5× smaller intermediate tensor = 5× less cache pressure.

Prior T0 result (step session): K_in=5 scored BEST at T0 (2.01pp above K_in=25).
CAUTION: T0 raw-loop results at 20ep were noisy (14-16% all near baseline).
This T1 (75ep, 50% data) with proper Trainer gives reliable comparison.

CONFIGS
=======
  Ref   : K_in=25 (step199 reference)
  A_k15 : K_in=15 (−40% seed FLOPs vs ref)
  B_k10 : K_in=10 (−60% seed FLOPs)
  C_k5  : K_in=5  (−80% seed FLOPs)

Scale: N=2048, D=16, K_hh=2, K_iter=5 (production config)
Tier: T1 (75ep, 50% data)

To run:
    python -u scripts/train_step631_kin_sweep.py --device mps --epochs 75
    python -u scripts/train_step631_kin_sweep.py --device cuda --epochs 75
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
parser.add_argument("--configs", default="Ref,A_k15,B_k10,C_k5")
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
OUT_PATH = ROOT / "results" / f"train_step631_kin_sweep_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Ref":   (25, "K_in=25 reference (step199 config)"),
    "A_k15": (15, "K_in=15 — 40% fewer seed FLOPs"),
    "B_k10": (10, "K_in=10 — 60% fewer seed FLOPs"),
    "C_k5":  (5,  "K_in=5  — 80% fewer seed FLOPs"),
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

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED)
    # 50% subset for T1
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))[:n_total//2].tolist()
    tr = torch.utils.data.DataLoader(
        Subset(tr_full.dataset, idx), batch_size=BATCH, shuffle=True, drop_last=False)

    print(f"Step 631 — K_in sweep T1 (75ep, 50% data)")
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
        seed_macs_new = N * K_in        # after spatial precomputation
        seed_macs_old = N * K_in * D   # before optimization
        routing_macs = K_ITER * N * K_HH * D * 2
        print(f"{'─'*60}\n{key}: {desc}")
        print(f"  params={n_p:,}  K_in={K_in}")
        print(f"  seed_MACs(new)={seed_macs_new/1e3:.0f}K  seed_MACs(old)={seed_macs_old/1e3:.0f}K  routing={routing_macs/1e3:.0f}K")

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
        print(f"  → K_in={K_in}  best={best:.4f} @ep{best_ep}  Δ_vs_ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "K_in": K_in, "label": desc, "n_params": n_p,
            "seed_macs_optimized": seed_macs_new,
            "routing_macs": routing_macs,
            "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(best - (ref_acc or 0), 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    print(f"STEP 631 SUMMARY — K_in sweep T1 (75ep 50% data)")
    print(f"{'='*60}")
    print(f"  {'Config':<8} {'K_in':>5} {'best':>7}  {'Δ_vs_ref':>10}  {'seed_MACs':>10}  advance?")
    for k, r in results.items():
        advance = "YES (Tier 2)" if r["delta_vs_ref"] >= -0.005 else "NO — accuracy loss"
        print(f"  {k:<8} {r['K_in']:>5} {r['best']:>7.4f}  {r['delta_vs_ref']*100:>+9.2f}pp  {r['seed_macs_optimized']:>10,}  {advance}")
    print(f"\n  ADVANCE RULE: Δ ≥ -0.5pp → advance to T2 (paper-quality validation)")


if __name__ == "__main__":
    main()
