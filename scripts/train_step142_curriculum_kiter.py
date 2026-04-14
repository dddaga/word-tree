"""Step 142: Curriculum K_iter — ramp routing depth during training.

MOTIVATION
==========
Fixed K_iter treats all training epochs identically. Early training is dominated
by large-loss gradients — fewer routing steps means shorter backprop chains and
less gradient vanishing, potentially allowing faster convergence. Later epochs
need full K_iter depth to exploit the topology. Curriculum: start shallow, ramp
to full depth.

Reverse curriculum (Config D) tests the opposite: is starting with full depth
better? A clear win for D would suggest the opposite — depth matters most early.

CONFIGS (N=1024, D=16, K_hh=8, K_in=25, AH=1.0, 50%/75ep)
============================================================
  Ref : Fixed K_iter=8 (baseline)
  A   : Curriculum 4→8→12  (25%/25%/50% of epochs)
  B   : Curriculum 4→12    (50%/50%, skip intermediate)
  C   : Curriculum 2→4→8→12 (25% each, very gradual)
  D   : Reverse  12→8→4   (control — expected worse)

Implementation note:
  The model's K_iter attribute is overridden per-epoch via a custom training
  loop that replaces trainer.train(n_epochs=EPOCHS) with a per-epoch loop
  that patches base.K_iter before each epoch.

To reproduce:
    python -u scripts/train_step142_curriculum_kiter.py --device mps
    python -u scripts/train_step142_curriculum_kiter.py --device mps --epochs 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs, topology_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75,
                    help="Training epochs (default 75; use 20 for scout)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (e.g. A,D). Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 16; K_IN = 25; K_HH = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key:    str
    label:  str
    # Schedule: list of (fraction_of_total_epochs, k_iter) tuples.
    # fraction is cumulative: the value applies until that fraction of epochs.
    schedule: list = field(default_factory=list)   # [(frac, k_iter), ...]

CONFIGS = [
    Config("Ref", "Ref  fixed K_iter=8 (baseline)",
           schedule=[(1.0, 8)]),
    Config("A",   "A    curriculum 4→8→12 (25%/25%/50%)",
           schedule=[(0.25, 4), (0.50, 8), (1.0, 12)]),
    Config("B",   "B    curriculum 4→12 (50%/50%)",
           schedule=[(0.50, 4), (1.0, 12)]),
    Config("C",   "C    curriculum 2→4→8→12 (25% each)",
           schedule=[(0.25, 2), (0.50, 4), (0.75, 8), (1.0, 12)]),
    Config("D",   "D    reverse 12→8→4 (control)",
           schedule=[(0.25, 12), (0.50, 8), (1.0, 4)]),
]


def get_kiter_for_epoch(cfg: Config, epoch: int, total_epochs: int) -> int:
    """Return K_iter to use for a given epoch (1-indexed) given schedule."""
    frac = epoch / total_epochs
    for threshold, k in cfg.schedule:
        if frac <= threshold:
            return k
    return cfg.schedule[-1][1]


# ---------------------------------------------------------------------------
# Data (cached, 50%)
# ---------------------------------------------------------------------------

_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    K_random = max(1, K_HH // 4)
    K_local  = K_HH - K_random
    n_groups = max(8, N // 8)

    # Use the max K_iter from the schedule as the model's initial value.
    # Per-epoch patching will override it during training.
    max_k_iter = max(k for _, k in cfg.schedule)

    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=max_k_iter,
        K_local=K_local, K_random=K_random, n_groups=n_groups,
        norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def _get_base(model: nn.Module) -> SGNNET_SmallWorld:
    """Navigate SGNNET_AntiHebbian → SGNNET_Resonant → SGNNET_SmallWorld."""
    return model.m.base


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def compute_flops(k_iter: int) -> int:
    """FLOPs for one forward pass with given K_iter."""
    seed     = N * K_IN * D
    per_step = N * K_HH * D * 2 + N * D + N * D * 2
    routing  = k_iter * per_step
    readout  = N * N_OUT * D
    return seed + routing + readout


# ---------------------------------------------------------------------------
# Custom training loop: patch K_iter per epoch
# ---------------------------------------------------------------------------

def train_with_curriculum(model: nn.Module, cfg: Config, device: torch.device,
                          n_epochs: int) -> list[dict]:
    """Train epoch-by-epoch, patching base.K_iter according to curriculum."""
    tr_loader, va_loader = get_loaders()
    kw = trainer_kwargs(N, n_epochs=n_epochs)

    trainer = Trainer(
        model=model,
        train_loader=tr_loader,
        val_loader=va_loader,
        device=device,
        **kw,
    )

    base = _get_base(model)
    history = []
    prev_k = None

    for ep in range(1, n_epochs + 1):
        k = get_kiter_for_epoch(cfg, ep, n_epochs)
        if k != prev_k:
            base.K_iter = k
            print(f"    [curriculum ep={ep}] K_iter → {k}")
            prev_k = k

        ep_hist = trainer.train(n_epochs=1)
        history.extend(ep_hist)

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 142 — Curriculum K_iter")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_in={K_IN}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    print(f"{'Key':4s}  {'Label'}")
    print(f"{'─'*70}")
    for c in CONFIGS:
        sched_str = " ".join(f"{int(f*100)}%→K={k}" for f, k in c.schedule)
        print(f"{c.key:4s}  {c.label}  [{sched_str}]")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step142_curriculum_kiter.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  schedule={cfg.schedule}")
        print(f"{'─'*60}")

        t0 = time.time()
        history = train_with_curriculum(model, cfg, DEVICE, n_epochs=EPOCHS)
        elapsed = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1

        # FLOPs: use the terminal K_iter (majority of training) as representative
        terminal_k = cfg.schedule[-1][1]
        flops = compute_flops(terminal_k)

        results[cfg.key] = {
            "N": N, "D": D, "K_hh": K_HH, "K_in": K_IN,
            "schedule": cfg.schedule,
            "terminal_k_iter": terminal_k,
            "flops": flops, "flops_M": round(flops / 1e6, 2),
            "alpha_ahebb": ALPHA_AHEBB,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params,
            "label": cfg.label,
        }

        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs_ref = top1_best - ref_best if ref_best > 0 else 0
        print(f"\n  top1={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min  params={n_params:,}")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 142 SUMMARY — Curriculum K_iter")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Key':4s}  {'schedule':30s}  {'params':>8}  {'top1':>7}  {'vs_Ref':>8}")
    print(f"{'─'*65}")
    for key, r in results.items():
        vs    = r["top1_best"] - ref_best if ref_best > 0 else 0
        sched = str(r["schedule"])
        print(f"{key:4s}  {sched:30s}  {r['n_params']:>8}  "
              f"{r['top1_best']:.4f}  {vs:>+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
