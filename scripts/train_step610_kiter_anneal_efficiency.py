"""Step 610: K_iter annealing on efficiency config (N=2048, D=16, K_hh=2).

MOTIVATION
==========
K_iter is the biggest FLOPs lever: each -1 saves ~0.2M FLOPs. Prior findings
(step142, step164) tested annealing at N=1024 D=16 K_hh=8 — different config.
This re-tests on the confirmed efficiency config (step199 baseline: K_iter=5).

Prior findings (STALE, different config):
  - step142 C (LOW→HIGH curriculum) won +2.85pp
  - step142 D, step164 C (HIGH→LOW decreasing) hurt ~6-7pp
  - step163 warm-start (K=12 teacher → K=8 student) won +7.82pp

CONFIGS (N=2048, D=16, K_hh=2, AH=1.0, 75ep, 50% data — Tier-1)
==================================================================
  Ref              : K_iter=5 constant (efficiency baseline, step199)
  A_up             : 3→5→7→9 monotonic UP, 4 phases ~19ep each
  B_down_gentle    : 9→7→5 over 3 phases 25ep each — ends at default
  C_down_aggressive: 12→8→5→3 over 4 phases — aggressive compression
  D_warm_switch    : K=12 for ep1-40, hard switch to K=5 at ep41
  E_warm_switch_k3 : K=12 for ep1-40, hard switch to K=3 at ep41

Each epoch's K_iter is logged alongside val_top1.
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

from src.sgnnet.model_smallworld       import SGNNET_SmallWorld
from src.sgnnet.model_resonant         import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory  import SGNNET_AntiHebbian
from src.training.trainer              import Trainer
from src.training.experiment_config    import trainer_kwargs
from src.training.dataset              import make_loaders

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--configs", default="", help="Comma-separated config keys")
args = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25
ALPHA_AHEBB = 1.0; ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

OUT_PATH = ROOT / "results" / "train_step610_kiter_anneal_efficiency.json"


# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------
@dataclass
class Config:
    key: str
    label: str
    # schedule: list of (end_epoch_frac, k_iter) — cumulative fractions
    schedule: list = field(default_factory=list)

CONFIGS = [
    Config("Ref",               "Ref   K_iter=5 constant",
           schedule=[(1.0, 5)]),
    Config("A_up",              "A_up  3→5→7→9 monotonic UP (4 phases, ~19ep each)",
           schedule=[(0.25, 3), (0.50, 5), (0.75, 7), (1.0, 9)]),
    Config("B_down_gentle",     "B_down_gentle  9→7→5 (3 phases, 25ep each)",
           schedule=[(0.333, 9), (0.667, 7), (1.0, 5)]),
    Config("C_down_aggressive", "C_down_aggressive  12→8→5→3 (4 phases, ~19ep each)",
           schedule=[(0.25, 12), (0.50, 8), (0.75, 5), (1.0, 3)]),
    Config("D_warm_switch",     "D_warm_switch  K=12 for ep1-40, switch to K=5 at ep41",
           schedule=None),   # handled specially
    Config("E_warm_switch_k3",  "E_warm_switch_k3  K=12 for ep1-40, switch to K=3 at ep41",
           schedule=None),   # handled specially
]

WARM_SWITCH_EP = 40   # epoch at which warm-switch configs hard-switch

# student K for each warm-switch config
WARM_STUDENT_K = {"D_warm_switch": 5, "E_warm_switch_k3": 3}


def get_kiter_for_epoch(cfg: Config, ep: int, total: int) -> int:
    """Return K_iter for epoch ep (1-indexed). Warm-switch handled here."""
    if cfg.key in WARM_STUDENT_K:
        return 12 if ep <= WARM_SWITCH_EP else WARM_STUDENT_K[cfg.key]
    frac = ep / total
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
def make_model(cfg: Config, seed_offset: int = 0):
    torch.manual_seed(SEED + seed_offset)
    # Init at max K_iter from schedule (per-epoch patching overrides it)
    if cfg.key in WARM_STUDENT_K:
        init_k = 12
    else:
        init_k = max(k for _, k in cfg.schedule)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=init_k,
                              K_local=K_l, K_random=K_r, n_groups=ng,
                              norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def _get_base(model) -> SGNNET_SmallWorld:
    return model.m.base


def compute_flops(k_iter: int) -> int:
    seed     = N * K_IN * D
    per_step = N * K_HH * D * 2 + N * D + N * D * 2
    routing  = k_iter * per_step
    readout  = N * N_OUT * D
    return seed + routing + readout


# ---------------------------------------------------------------------------
# Per-epoch training loop
# ---------------------------------------------------------------------------
def train_with_schedule(model, cfg: Config, n_epochs: int) -> dict:
    tr, va = get_loaders()
    kw = trainer_kwargs(N, n_epochs=n_epochs)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **kw)
    base = _get_base(model)
    top1_hist = []
    k_hist = []
    prev_k = None

    for ep in range(1, n_epochs + 1):
        k = get_kiter_for_epoch(cfg, ep, n_epochs)
        if k != prev_k:
            base.K_iter = k
            print(f"    [schedule ep={ep}] K_iter → {k}", flush=True)
            prev_k = k
        k_hist.append(k)

        ep_hist = trainer.train(n_epochs=1)
        v = round(ep_hist[-1].get("val_top1", 0.0), 4)
        top1_hist.append(v)
        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep:3d}  K_iter={k}  val={v:.4f}", flush=True)

    best_idx = int(np.argmax(top1_hist))
    return {
        "top1_history": top1_hist,
        "k_iter_history": k_hist,
        "best_top1": max(top1_hist),
        "best_epoch": best_idx + 1,
        "k_at_best": k_hist[best_idx],
        "final_k_iter": k_hist[-1],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active = [(i, c) for i, c in enumerate(CONFIGS)
              if not cfg_filter or c.key in cfg_filter]

    print(f"\n{'='*70}")
    print(f"Step 610 — K_iter annealing on efficiency config")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_in={K_IN}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%  Tier-1")
    print(f"Running: {[c.key for _, c in active]}")
    print(f"{'='*70}\n")

    get_loaders()
    results = {}

    for i, cfg in active:
        n_params = sum(p.numel() for p in make_model(cfg).parameters() if p.requires_grad)
        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}")
        print(f"{'─'*60}")

        model = make_model(cfg, seed_offset=i).to(DEVICE)
        t0 = time.time()
        r = train_with_schedule(model, cfg, EPOCHS)
        elapsed = time.time() - t0

        final_flops = compute_flops(r["final_k_iter"])
        results[cfg.key] = {
            "N": N, "D": D, "K_hh": K_HH, "K_in": K_IN,
            "alpha_ahebb": ALPHA_AHEBB, "data_frac": 0.5,
            "label": cfg.label,
            "schedule": cfg.schedule if cfg.schedule else f"warm_switch_{WARM_SWITCH_EP}ep",
            **r,
            "final_flops": final_flops,
            "final_flops_M": round(final_flops / 1e6, 2),
            "n_params": n_params,
            "elapsed_s": round(elapsed, 1),
        }

        ref_best = results.get("Ref", {}).get("best_top1", 0)
        vs_ref = r["best_top1"] - ref_best if ref_best > 0 else 0
        print(f"\n  best={r['best_top1']:.4f} @ ep{r['best_epoch']} "
              f"(K_iter={r['k_at_best']})  vs_Ref={vs_ref:+.4f}  "
              f"final_K={r['final_k_iter']}  {elapsed/60:.1f}min")

        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # Summary
    ref_best = results.get("Ref", {}).get("best_top1", 0)
    print(f"\n{'='*70}")
    print(f"STEP 610 SUMMARY — K_iter annealing")
    print(f"{'='*70}")
    print(f"{'Key':22s}  {'best':>7}  {'vs_Ref':>8}  {'final_K':>7}  {'K@best':>7}  {'FLOPs_M':>8}")
    print(f"{'─'*70}")
    for key, r in results.items():
        vs = r["best_top1"] - ref_best if ref_best > 0 else 0
        print(f"{key:22s}  {r['best_top1']:.4f}  {vs:>+.4f}  "
              f"{r['final_k_iter']:>7}  {r['k_at_best']:>7}  {r['final_flops_M']:>8.2f}")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
