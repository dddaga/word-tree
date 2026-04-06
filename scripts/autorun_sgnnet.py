"""Autoresearch-style autonomous experiment loop for SGNNET.

Inspired by Karpathy's autoresearch philosophy:
  - Fixed eval harness (FashionMNIST val_top1, 50% data)
  - Fixed compute budget per experiment (20 epochs)
  - Monotonic ratchet: keep wins, revert losses
  - TSV log of all attempts (experiment journal)
  - Single search space definition

EMPIRICAL BASIS (from 46 historical experiments):
  D=64 patched arch, 15ep scouts: 77% winner accuracy, ρ=0.80
  D=64 patched arch, 30ep scouts: 91% winner accuracy
  → 20 epochs is a good compromise: ~85% reliable, ~8 min/run at N=1024

SEARCH SPACE:
  This script explores the neighborhood around confirmed winners.
  Each iteration perturbs one or more parameters and evaluates.
  The ratchet advances only on clear improvements (>0.3pp).

MODES:
  --mode explore    : Random perturbations around current best (default)
  --mode sweep      : Systematic grid sweep of one parameter
  --mode compound   : Try combining top-2 recent wins

To run:
    python -u scripts/autorun_sgnnet.py --device mps --max-iter 50
    python -u scripts/autorun_sgnnet.py --device cpu --max-iter 20

Results logged to: results/autorun_journal.tsv
Best config saved to: results/autorun_best.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld       import SGNNET_SmallWorld
from src.sgnnet.model_resonant         import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory  import SGNNET_AntiHebbian
from src.training.trainer              import Trainer
from src.training.experiment_config    import trainer_kwargs, run_metadata
from src.training.dataset              import make_loaders

# ─── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Autoresearch-style SGNNET loop")
parser.add_argument("--device", default="auto")
parser.add_argument("--max-iter", type=int, default=50,
                    help="Maximum number of experiments to run")
parser.add_argument("--budget", type=int, default=20,
                    help="Epochs per experiment (scout budget)")
parser.add_argument("--threshold", type=float, default=0.003,
                    help="Minimum improvement to keep (0.003 = 0.3pp)")
parser.add_argument("--mode", choices=["explore", "sweep", "compound"],
                    default="explore")
parser.add_argument("--sweep-param", type=str, default=None,
                    help="Parameter to sweep in sweep mode")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n-scale", type=int, default=1024,
                    help="N_hidden to use (1024 for fast, 4096 for validation)")
args = parser.parse_args()

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)


# ─── FIXED EVAL HARNESS (DO NOT MODIFY) ────────────────────────────────────────

DATA       = "data/store.h5"
BATCH      = 128
SEED       = 42
N_IN       = 25088
N_OUT      = 10
DATA_FRAC  = 0.5

_loaders = None

def get_loaders():
    """Fixed data pipeline — same split every time for comparability."""
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(
            subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


# ─── SEARCH SPACE (configurable) ───────────────────────────────────────────────

@dataclass
class SGNNETConfig:
    """All tunable parameters. Defaults = current confirmed winners."""
    N: int = 1024
    D: int = 64
    K_local: int = 2         # K_hh = K_local + K_random = 4
    K_random: int = 2
    K_in: int = 50
    K_iter: int = 12         # step89 confirmed
    K_phase: int = 8
    beam_size: int = 16
    geo_gamma: float = 0.5
    alpha_reflect: float = 0.5
    alpha_turing: float = 0.0
    alpha_ahebb: float = 1.0
    lr: float = 2.364e-3
    encoding_mode: str = "fourier"
    norm_mode: str = "l2"
    # Derived
    n_groups: int = -1       # -1 = auto (max(8, N//8))

    def __post_init__(self):
        if self.n_groups == -1:
            self.n_groups = max(8, self.N // 8)

    @property
    def K_hh(self):
        return self.K_local + self.K_random


# Parameter perturbation ranges for exploration
PERTURBATIONS = {
    "K_iter":        [4, 6, 8, 10, 12, 14, 16, 20],
    "K_local":       [1, 2, 3, 4],
    "K_random":      [1, 2, 3, 4],
    "K_in":          [10, 25, 50, 75, 100],
    "D":             [16, 32, 48, 64, 96, 128],
    "alpha_ahebb":   [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
    "alpha_reflect": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
    "alpha_turing":  [0.0, 0.1, 0.3, 0.5],
    "lr":            [5e-4, 1e-3, 2e-3, 2.364e-3, 3e-3, 5e-3, 1e-2],
    "geo_gamma":     [0.0, 0.25, 0.5, 0.75, 1.0],
    "beam_size":     [4, 8, 16, 32],
    "K_phase":       [4, 8, 12, 16],
    "n_groups":      [4, 8, 16, 32, 64, 128, 256],
}


def perturb_config(base: SGNNETConfig, rng: random.Random,
                   n_changes: int = 1) -> tuple[SGNNETConfig, str]:
    """Create a variant of base with n_changes random perturbations."""
    cfg = SGNNETConfig(**{k: v for k, v in asdict(base).items()})
    changes = []
    params = list(PERTURBATIONS.keys())
    chosen = rng.sample(params, min(n_changes, len(params)))

    for param in chosen:
        options = PERTURBATIONS[param]
        current = getattr(cfg, param)
        # Pick a value different from current
        candidates = [v for v in options if v != current]
        if not candidates:
            continue
        new_val = rng.choice(candidates)
        setattr(cfg, param, new_val)
        changes.append(f"{param}:{current}→{new_val}")

    # Recompute n_groups if N changed
    if "N" in [c.split(":")[0] for c in changes]:
        cfg.n_groups = max(8, cfg.N // 8)

    desc = ", ".join(changes) if changes else "no change"
    return cfg, desc


def sweep_configs(base: SGNNETConfig, param: str) -> list[tuple[SGNNETConfig, str]]:
    """Generate configs sweeping one parameter."""
    if param not in PERTURBATIONS:
        raise ValueError(f"Unknown param: {param}. Choose from: {list(PERTURBATIONS.keys())}")
    results = []
    for val in PERTURBATIONS[param]:
        cfg = SGNNETConfig(**{k: v for k, v in asdict(base).items()})
        setattr(cfg, param, val)
        if param == "N":
            cfg.n_groups = max(8, cfg.N // 8)
        results.append((cfg, f"{param}={val}"))
    return results


# ─── MODEL BUILDER ──────────────────────────────────────────────────────────────

def build_model(cfg: SGNNETConfig, seed_offset: int = 0) -> nn.Module:
    """Build SGNNET with given config. Fixed architecture — only params change."""
    torch.manual_seed(SEED + seed_offset)
    base = SGNNET_SmallWorld(
        N_in=N_IN, N_hidden=cfg.N, N_out=N_OUT,
        K_local=cfg.K_local, K_random=cfg.K_random,
        K_in=cfg.K_in, K_iter=cfg.K_iter, n_groups=cfg.n_groups,
        norm_mode=cfg.norm_mode, D=cfg.D, encoding_mode=cfg.encoding_mode,
    )
    resonant = SGNNET_Resonant(
        base, K_phase=cfg.K_phase, alpha_reflect=cfg.alpha_reflect,
        alpha_turing=cfg.alpha_turing, beam_size=cfg.beam_size,
        geo_gamma=cfg.geo_gamma, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=cfg.alpha_ahebb, variant="wpos")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ─── SINGLE EXPERIMENT RUNNER ───────────────────────────────────────────────────

def run_experiment(cfg: SGNNETConfig, budget: int, seed_offset: int = 0) -> dict:
    """Run one experiment with fixed budget. Returns result dict."""
    model = build_model(cfg, seed_offset).to(DEVICE)
    tr, va = get_loaders()

    tk = trainer_kwargs(cfg.N, lr_wpos=cfg.lr, n_epochs=budget, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)

    t0 = time.time()
    try:
        history = trainer.train(n_epochs=budget)
    except Exception as e:
        return {"status": "crash", "error": str(e), "top1_best": 0.0,
                "elapsed_s": round(time.time() - t0, 1)}
    elapsed = time.time() - t0

    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best = max(top1_hist) if top1_hist else 0.0
    best_ep = int(np.argmax(top1_hist)) + 1 if top1_hist else 0

    return {
        "status": "ok",
        "top1_best": best,
        "top1_last": top1_hist[-1] if top1_hist else 0.0,
        "best_epoch": best_ep,
        "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "params": count_params(model),
        "top1_history": top1_hist,
    }


# ─── TSV LOGGING ────────────────────────────────────────────────────────────────

TSV_PATH = ROOT / "results" / "autorun_journal.tsv"
TSV_FIELDS = [
    "iter", "timestamp", "status", "verdict", "top1_best", "top1_last",
    "best_epoch", "epochs_run", "elapsed_s", "params",
    "description", "N", "D", "K_iter", "K_local", "K_random", "K_in",
    "alpha_ahebb", "alpha_reflect", "alpha_turing", "lr",
    "n_groups", "geo_gamma", "beam_size", "K_phase",
]


def init_tsv():
    if not TSV_PATH.exists():
        with open(TSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t")
            writer.writeheader()


def log_tsv(iteration: int, result: dict, cfg: SGNNETConfig,
            verdict: str, description: str):
    with open(TSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t")
        import datetime
        row = {
            "iter": iteration,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "status": result.get("status", "?"),
            "verdict": verdict,
            "top1_best": f"{result.get('top1_best', 0):.4f}",
            "top1_last": f"{result.get('top1_last', 0):.4f}",
            "best_epoch": result.get("best_epoch", 0),
            "epochs_run": result.get("epochs_run", 0),
            "elapsed_s": result.get("elapsed_s", 0),
            "params": result.get("params", 0),
            "description": description,
            "N": cfg.N, "D": cfg.D, "K_iter": cfg.K_iter,
            "K_local": cfg.K_local, "K_random": cfg.K_random,
            "K_in": cfg.K_in,
            "alpha_ahebb": cfg.alpha_ahebb,
            "alpha_reflect": cfg.alpha_reflect,
            "alpha_turing": cfg.alpha_turing,
            "lr": cfg.lr,
            "n_groups": cfg.n_groups,
            "geo_gamma": cfg.geo_gamma,
            "beam_size": cfg.beam_size,
            "K_phase": cfg.K_phase,
        }
        writer.writerow(row)


# ─── RATCHET LOOP ──────────────────────────────────────────────────────────────

def main():
    print(f"{'='*70}")
    print(f"AUTORUN SGNNET — Autoresearch-style experiment loop")
    print(f"  Device: {DEVICE}")
    print(f"  Budget: {args.budget} epochs per experiment")
    print(f"  Max iterations: {args.max_iter}")
    print(f"  Threshold: {args.threshold:.4f} ({args.threshold*100:.1f}pp)")
    print(f"  Mode: {args.mode}")
    print(f"  N: {args.n_scale}")
    print(f"{'='*70}\n")

    # Load data once
    get_loaders()
    print(f"Data: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    # Initialize TSV
    init_tsv()

    # Current best config
    best_cfg = SGNNETConfig(N=args.n_scale)
    best_score = None
    rng = random.Random(args.seed)

    # Load previous best if exists
    best_path = ROOT / "results" / "autorun_best.json"
    if best_path.exists():
        try:
            prev = json.load(open(best_path))
            best_score = prev.get("top1_best")
            # Restore config
            for k in asdict(best_cfg):
                if k in prev.get("config", {}):
                    setattr(best_cfg, k, prev["config"][k])
            print(f"Loaded previous best: {best_score:.4f}")
        except Exception:
            pass

    # Run baseline if no previous best
    if best_score is None:
        print(f"\n--- Baseline run (confirmed winners) ---")
        result = run_experiment(best_cfg, args.budget)
        best_score = result["top1_best"]
        log_tsv(0, result, best_cfg, "baseline", "confirmed_winners_baseline")
        save_best(best_path, best_cfg, result)
        print(f"  Baseline: {best_score:.4f}  ({result['elapsed_s']:.0f}s)")

    # Generate experiment queue
    if args.mode == "sweep" and args.sweep_param:
        queue = sweep_configs(best_cfg, args.sweep_param)
    elif args.mode == "explore":
        queue = []  # Generate on the fly
    else:
        queue = []

    wins = 0
    losses = 0
    crashes = 0

    for i in range(1, args.max_iter + 1):
        # Generate next config
        if args.mode == "explore":
            # Adaptive: start with single changes, escalate to multi
            n_changes = 1 if i <= args.max_iter // 2 else rng.choice([1, 2])
            cfg, desc = perturb_config(best_cfg, rng, n_changes=n_changes)
        elif args.mode == "sweep" and queue:
            cfg, desc = queue.pop(0)
        else:
            cfg, desc = perturb_config(best_cfg, rng, n_changes=1)

        print(f"\n--- Iter {i}/{args.max_iter}  [{desc}] ---")
        print(f"  Config: N={cfg.N} D={cfg.D} K_iter={cfg.K_iter} K_hh={cfg.K_hh} "
              f"α_AH={cfg.alpha_ahebb} lr={cfg.lr:.4e}")

        result = run_experiment(cfg, args.budget, seed_offset=i)

        if result["status"] == "crash":
            print(f"  CRASH: {result.get('error', '?')}")
            log_tsv(i, result, cfg, "crash", desc)
            crashes += 1
            continue

        score = result["top1_best"]
        delta = score - best_score

        if delta > args.threshold:
            # WIN — advance the ratchet
            print(f"  ✓ WIN: {score:.4f} (+{delta:.4f})  "
                  f"ep={result['best_epoch']}/{result['epochs_run']}  "
                  f"t={result['elapsed_s']:.0f}s")
            best_score = score
            best_cfg = cfg
            wins += 1
            log_tsv(i, result, cfg, "WIN", desc)
            save_best(best_path, best_cfg, result)
        elif delta > -args.threshold:
            # TIE — keep if simpler (fewer params)
            if result["params"] < count_params(build_model(best_cfg).to("cpu")):
                print(f"  ~ TIE (simpler): {score:.4f} ({delta:+.4f})  "
                      f"fewer params → keep")
                best_cfg = cfg
                log_tsv(i, result, cfg, "TIE_SIMPLER", desc)
                save_best(best_path, best_cfg, result)
            else:
                print(f"  ~ TIE: {score:.4f} ({delta:+.4f})  → discard")
                log_tsv(i, result, cfg, "tie", desc)
                losses += 1
        else:
            # LOSS — revert
            print(f"  ✗ LOSS: {score:.4f} ({delta:+.4f})  → revert")
            log_tsv(i, result, cfg, "loss", desc)
            losses += 1

        # Progress report every 5 iterations
        if i % 5 == 0:
            total = wins + losses + crashes
            print(f"\n  [Progress] iter={i}  best={best_score:.4f}  "
                  f"W/L/C={wins}/{losses}/{crashes}  "
                  f"win_rate={wins/total:.0%}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"AUTORUN COMPLETE — {args.max_iter} iterations")
    print(f"  Best score: {best_score:.4f}")
    print(f"  Wins: {wins}  Losses: {losses}  Crashes: {crashes}")
    print(f"  Win rate: {wins/(wins+losses+crashes):.0%}" if wins+losses+crashes > 0 else "")
    print(f"  Best config:")
    for k, v in asdict(best_cfg).items():
        print(f"    {k}: {v}")
    print(f"\n  Journal: {TSV_PATH}")
    print(f"  Best: {best_path}")
    print(f"{'='*70}")


def save_best(path: Path, cfg: SGNNETConfig, result: dict):
    """Save current best config and score."""
    data = {
        "top1_best": result["top1_best"],
        "config": asdict(cfg),
        "result": {k: v for k, v in result.items() if k != "top1_history"},
        "_meta": run_metadata(__file__, asdict(cfg)),
    }
    path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
