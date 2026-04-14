"""Step 154: Safety valve loss ablation — is it redundant with AntiHebbian?

MOTIVATION
==========
Safety valve loss is a Coulomb-like O(N^2) positional repulsion on W_pos,
originally tuned at N=256/D=4. AntiHebbian suppression already decorrelates
nearby neurons via W_pos cosine similarity — potentially making safety
redundant. At N=1024, safety contributes ~2% of total loss. If removing it
causes no regression, we simplify the training pipeline and save compute.

CONFIGS (N=1024, D=16, K_hh=8, K_iter=8, K_in=25, AH=1.0, 50%/20ep scout)
============================================================================
  Ref : Standard (lambda_safety from experiment_config, ~0.49)
  A   : lambda_safety = 0.0 (no safety valve)
  B   : lambda_safety = 2.0 (2× stronger — check if more helps)

To reproduce:
    python -u scripts/train_step154_safety_ablation.py --device mps
    python -u scripts/train_step154_safety_ablation.py --device mps --epochs 75
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20,
                    help="Training epochs (default 20 scout)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys. Empty = all.")
args = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 16; K_ITER = 8; K_IN = 25; K_HH = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0


@dataclass
class Config:
    key: str
    label: str
    lambda_safety: float  # -1 = use default from experiment_config


CONFIGS = [
    Config("Ref", "Ref  standard safety (λ≈0.49)", lambda_safety=-1),
    Config("A",   "A    no safety (λ=0.0)",         lambda_safety=0.0),
    Config("B",   "B    2× safety (λ≈0.98)",        lambda_safety=-2),  # -2 = 2× default
]


_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True,
                                          num_workers=0)
        _loaders = (tr, va)
    return _loaders


def make_model(seed_offset=0):
    torch.manual_seed(SEED + seed_offset)
    K_random = max(1, K_HH // 4)
    K_local = K_HH - K_random
    n_groups = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER,
        K_local=K_local, K_random=K_random, n_groups=n_groups,
        norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    print(f"\n{'='*70}")
    print(f"Step 154 — Safety Valve Loss Ablation")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  K_in={K_IN}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    for c in CONFIGS:
        print(f"  {c.key:4s}  {c.label}")
    print()

    get_loaders()
    results = {}
    out_path = ROOT / "results" / "train_step154_safety_ablation.json"

    # Get default lambda_safety for this N
    default_kw = trainer_kwargs(N, n_epochs=EPOCHS)
    default_lambda = default_kw["lambda_safety"]
    print(f"  Default lambda_safety for N={N}: {default_lambda:.4f}\n")

    cfg_filter = ([k.strip() for k in args.configs.split(",") if k.strip()]
                  if args.configs else [])
    active = [(i, c) for i, c in enumerate(CONFIGS)
              if not cfg_filter or c.key in cfg_filter]

    for i, cfg in active:
        model = make_model(seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        # Determine lambda_safety
        if cfg.lambda_safety == -1:
            ls = default_lambda
        elif cfg.lambda_safety == -2:
            ls = default_lambda * 2.0
        else:
            ls = cfg.lambda_safety

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}  lambda_safety={ls:.4f}  params={n_params:,}")
        print(f"{'─'*60}")

        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        kw["lambda_safety"] = ls
        trainer = Trainer(model=model, train_loader=get_loaders()[0],
                         val_loader=get_loaders()[1], device=DEVICE, **kw)
        history = trainer.train(n_epochs=EPOCHS)
        elapsed = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep = int(np.argmax(top1_hist)) + 1

        # Also track safety loss over time
        safety_hist = [round(h.get("safety_loss", 0.0), 6) for h in history]

        results[cfg.key] = {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "lambda_safety": ls,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "safety_history": safety_hist,
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params, "label": cfg.label,
        }

        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs = top1_best - ref_best if ref_best > 0 else 0
        print(f"\n  top1={top1_best:.4f}  vs_Ref={vs:+.4f}  "
              f"safety_last={safety_hist[-1]:.6f}  elapsed={elapsed/60:.1f}min")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 154 SUMMARY — Safety Valve Ablation")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for k, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"  {k:4s}  λ_safety={r['lambda_safety']:.4f}  "
              f"top1={r['top1_best']:.4f}  vs_Ref={vs:+.4f}  "
              f"safety_final={r['safety_history'][-1]:.6f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")

    # Decision guidance
    ref_top1 = results.get("Ref", {}).get("top1_best", 0)
    a_top1 = results.get("A", {}).get("top1_best", 0)
    if a_top1 > 0 and ref_top1 > 0:
        delta = a_top1 - ref_top1
        if delta >= -0.005:
            print(f"\n  VERDICT: Safety removal is SAFE (Δ={delta:+.4f}). "
                  f"Can set lambda_safety=0 globally.")
        else:
            print(f"\n  VERDICT: Safety removal HURTS (Δ={delta:+.4f}). "
                  f"Keep lambda_safety at current value.")


if __name__ == "__main__":
    main()
