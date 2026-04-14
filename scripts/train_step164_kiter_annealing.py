"""Step 164: K_iter Annealing — progressive routing compression in a single run.

MOTIVATION
==========
step163 showed: K=12 teacher trained to convergence, then K=8 student initialized
from teacher's weights → +7.82pp over teacher. The warm-start is the mechanism:
high-K training shapes W_pos geometry; fewer steps then avoids over-smoothing.

HYPOTHESIS
==========
We can replicate and improve this in a SINGLE training run by annealing K downward:
  1. High K early → more routing steps → richer gradient signal → better W_pos geometry
  2. Reduce K progressively → forces compression into fewer steps → avoids over-smoothing
  3. No separate teacher+student needed — one run, one set of weights

Analogous to learning rate annealing: start large (explore), decay (exploit).
Distinct from step142 (K INCREASING 2→4→8→12): this is the REVERSE — K DECREASING.

CONFIGS (N=1024, D=16, K_hh=8, K_in=25, AH=1.0, 75ep, 50% data)
================================================================

  Ref : K=8 constant                    — efficiency baseline (reproduces step163 Ref approx)
  A   : K=12 constant                   — ceiling comparison (same as step163 teacher)
  B   : K=12 → K=8 hard switch at ep50  — coarse version: build then compress
  C   : K=16→14→12→10→8→6→4 monotonic  — 7 phases, ~10ep each, full annealing to K=4
  D   : K=16→12→16→14→12→10→8→6 oscillate+descend
        — oscillate first (exploration), then monotonically descend (compression)

Key comparisons:
  Ref vs A    : K=8 vs K=12 constant (does high K help even without warm-start?)
  A vs B      : hard switch at ep50 vs no switch (is any compression better than none?)
  B vs C      : coarse switch vs smooth annealing (does granularity matter?)
  Ref vs best : does annealing beat constant K=8 without the warm-start overhead?
  C vs D      : monotonic vs oscillating schedule

Best_epoch metric captures where in the schedule the model peaked — tells us the
optimal final K even if we overshot with further reduction.

FLOPs: annealing schedule ends at K=4 (Config C/D) or K=8 (Config B).
At K=8: ~4.8M FLOPs. At K=6: ~3.6M. At K=4: ~2.4M. All within budget.

To reproduce:
    python -u scripts/train_step164_kiter_annealing.py --device cpu
    python -u scripts/train_step164_kiter_annealing.py --device mps
    python -u scripts/train_step164_kiter_annealing.py --configs Ref,C,D --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

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

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (e.g. Ref,C,D). Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
N      = 1024;  N_IN = 25088;  N_OUT = 10
D      = 16;    K_HH = 8;      K_IN  = 25
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
ALPHA_AHEBB   = 1.0

OUT_PATH = ROOT / "results" / "train_step164_kiter_annealing.json"


# ---------------------------------------------------------------------------
# K schedules
# ---------------------------------------------------------------------------

def make_schedule_constant(k: int, n_epochs: int) -> List[int]:
    return [k] * n_epochs


def make_schedule_hard_switch(k_high: int, k_low: int, switch_ep: int, n_epochs: int) -> List[int]:
    """K=k_high for first switch_ep epochs, then K=k_low."""
    return [k_high] * switch_ep + [k_low] * (n_epochs - switch_ep)


def make_schedule_monotonic(k_start: int, k_end: int, n_epochs: int) -> List[int]:
    """Linearly space K values from k_start down to k_end over n_epochs.

    Only even values used; phase boundaries rounded to epoch.
    Example: k_start=16, k_end=4, n_epochs=75 →
      K=16 for ~10ep, K=14 for ~10ep, K=12 for ~10ep, ... K=4 for ~10ep
    """
    k_values = list(range(k_start, k_end - 1, -2))  # [16, 14, 12, 10, 8, 6, 4]
    n_phases  = len(k_values)
    ep_per_phase = n_epochs // n_phases
    schedule  = []
    for i, k in enumerate(k_values):
        if i < n_phases - 1:
            schedule.extend([k] * ep_per_phase)
        else:
            # Last phase absorbs remainder
            schedule.extend([k] * (n_epochs - len(schedule)))
    return schedule


def make_schedule_oscillate_descend(n_epochs: int) -> List[int]:
    """K=16→12→16→14 oscillation, then monotonic 14→12→10→8→6 descent.

    Oscillation: alternate high/low K to avoid local minima (like cyclical LR).
    Descent: steady compression once W_pos is well-shaped.
    """
    # Phase 1: oscillation (30 epochs)
    oscillation = [16]*10 + [12]*10 + [16]*10
    # Phase 2: monotonic descent (45 epochs)
    descent_ks  = [14, 12, 10, 8, 6]
    ep_per_descent = (n_epochs - 30) // len(descent_ks)
    descent = []
    for i, k in enumerate(descent_ks):
        if i < len(descent_ks) - 1:
            descent.extend([k] * ep_per_descent)
        else:
            descent.extend([k] * (n_epochs - 30 - len(descent)))
    return oscillation + descent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key:      str
    label:    str
    schedule: List[int] = field(default_factory=list)

    @property
    def k_start(self): return self.schedule[0] if self.schedule else 0
    @property
    def k_end(self):   return self.schedule[-1] if self.schedule else 0


def build_configs(n_epochs: int) -> List[Config]:
    switch_ep = n_epochs * 2 // 3  # hard switch at 2/3 of training

    return [
        Config("Ref", "K=8 constant — efficiency baseline",
               make_schedule_constant(8, n_epochs)),
        Config("A",   "K=12 constant — ceiling comparison",
               make_schedule_constant(12, n_epochs)),
        Config("B",   f"K=12→8 hard switch at ep{switch_ep}",
               make_schedule_hard_switch(12, 8, switch_ep, n_epochs)),
        Config("C",   "K=16→4 monotonic annealing (7 phases ~10ep each)",
               make_schedule_monotonic(16, 4, n_epochs)),
        Config("D",   "K=16→12→16→14→12→10→8→6 oscillate+descend",
               make_schedule_oscillate_descend(n_epochs)),
    ]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_loaders_cache = None

def get_loaders():
    global _loaders_cache
    if _loaders_cache is None:
        tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
        n   = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        sub = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr  = torch.utils.data.DataLoader(sub, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders_cache = (tr, va)
    return _loaders_cache


# ---------------------------------------------------------------------------
# Model factory  (K_iter is set per-epoch; initial value = schedule[0])
# ---------------------------------------------------------------------------

def make_model(k_init: int, seed: int = SEED) -> SGNNET_AntiHebbian:
    torch.manual_seed(seed)
    K_random = max(1, K_HH // 4)
    K_local  = K_HH - K_random
    n_groups = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=k_init,
        K_local=K_local, K_random=K_random, n_groups=n_groups,
        norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def set_k_iter(model: SGNNET_AntiHebbian, k: int):
    """Change K_iter on the underlying SmallWorld model."""
    model.m.base.K_iter = k


# ---------------------------------------------------------------------------
# Training loop with per-epoch K schedule
# ---------------------------------------------------------------------------

def train_with_schedule(model: SGNNET_AntiHebbian, schedule: List[int]) -> List[dict]:
    """Train model following K_iter schedule. Returns per-epoch history."""
    tr, va  = get_loaders()
    n_epochs = len(schedule)

    kw  = trainer_kwargs(N, n_epochs=n_epochs)
    opt = torch.optim.AdamW(
        [{"params": [model.W_pos], "lr": kw["lr_wpos"], "weight_decay": 0.0}]
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=kw["sched_factor"],
        patience=kw["sched_patience"], min_lr=kw["min_lr"],
    )

    history   = []
    prev_k    = schedule[0]

    for ep in range(1, n_epochs + 1):
        k_this_ep = schedule[ep - 1]

        # Update K_iter if schedule changed
        if k_this_ep != prev_k:
            set_k_iter(model, k_this_ep)
            prev_k = k_this_ep

        if hasattr(model, "tick_epoch"):
            model.tick_epoch()

        # ── Train epoch ──────────────────────────────────────────────────
        model.train()
        loss_sum  = 0.0
        n_batches = 0
        for features, soft_labels, _ in tr:
            features    = features.to(DEVICE)
            soft_labels = soft_labels.to(DEVICE)
            opt.zero_grad()
            logits = model(features)
            loss   = F.kl_div(F.log_softmax(logits, -1), soft_labels, reduction="batchmean")
            loss.backward()
            opt.step()
            with torch.no_grad():
                model.W_pos.clamp_(0, 1.0)
            loss_sum  += loss.item()
            n_batches += 1

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for features, _, labels in va:
                logits  = model(features.to(DEVICE)).cpu()
                correct += (logits.argmax(1) == labels).sum().item()
                total   += len(labels)

        train_loss = loss_sum / max(n_batches, 1)
        val_top1   = correct / total
        sched.step(train_loss)

        if ep % 10 == 0 or ep == n_epochs:
            print(f"    ep{ep:3d}  K={k_this_ep:2d}  loss={train_loss:.4f}  "
                  f"val={val_top1:.4f}  lr={opt.param_groups[0]['lr']:.2e}")

        history.append({
            "epoch": ep, "k_iter": k_this_ep,
            "train_loss": train_loss, "val_top1": round(val_top1, 4),
        })

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 164 — K_iter Annealing")
    print(f"N={N}  D={D}  K_hh={K_HH}  AH={ALPHA_AHEBB}  Epochs={EPOCHS}  Device={DEVICE}")
    print(f"{'='*70}")
    print("""
Hypothesis: start with high K (rich routing, better gradient signal for W_pos),
progressively reduce K (compression, less over-smoothing) — all in one run.
Inspired by step163: teacher-init → K_iter reduction = +7.82pp.
""")

    CONFIGS = build_configs(EPOCHS)

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active     = [c for c in CONFIGS if not cfg_filter or c.key in cfg_filter]

    print("Schedules:")
    for c in active:
        phases = []
        prev = c.schedule[0]
        start = 1
        for ep, k in enumerate(c.schedule, 1):
            if k != prev or ep == len(c.schedule):
                end = ep if k == prev else ep - 1
                phases.append(f"K={prev}@{start}-{end}")
                prev  = k
                start = ep
        print(f"  {c.key}: {' → '.join(phases[:8])}{'...' if len(phases)>8 else ''}")
    print()

    results  = {}
    OUT_PATH.parent.mkdir(exist_ok=True)
    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text())

    for i, cfg in enumerate(active):
        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")

        model    = make_model(cfg.k_start, seed=SEED + i).to(DEVICE)
        n_params = count_params(model)
        print(f"  params={n_params:,}  K_start={cfg.k_start}  K_end={cfg.k_end}")
        print(f"{'─'*60}")

        t0      = time.time()
        history = train_with_schedule(model, cfg.schedule)
        elapsed = time.time() - t0

        top1_hist = [h["val_top1"] for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1
        best_k    = cfg.schedule[best_ep - 1]

        ref_best = results.get("Ref", {}).get("top1_best", 0.0)
        vs_ref   = top1_best - ref_best if ref_best > 0 else 0.0

        print(f"\n  top1={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"best_ep={best_ep}  K@best={best_k}  elapsed={elapsed/60:.1f}min")

        results[cfg.key] = {
            "label":      cfg.label,
            "schedule_summary": f"K={cfg.k_start}→{cfg.k_end}",
            "top1_best":  top1_best,
            "best_epoch": best_ep,
            "k_at_best":  best_k,
            "top1_history": top1_hist,
            "k_history":    cfg.schedule,
            "n_params":   n_params,
            "elapsed_s":  round(elapsed, 1),
        }
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 164 SUMMARY — K_iter Annealing")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0.0)
    print(f"{'Key':4s}  {'Schedule':30s}  {'top1':>7}  {'vs_Ref':>8}  {'K@peak':>7}")
    print(f"{'─'*65}")
    for k, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"{k:4s}  {r['schedule_summary']:30s}  "
              f"{r['top1_best']:.4f}  {vs:>+.4f}  K={r.get('k_at_best','?')}")

    print(f"""
Key questions answered:
  1. Does K-annealing beat constant K=8? (Ref vs C/D)
  2. Does smooth annealing beat hard switch? (B vs C)
  3. Where does accuracy peak — K=8, K=6, or K=4? (k_at_best)
  4. Does oscillation before descent help? (C vs D)
""")
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Results saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
