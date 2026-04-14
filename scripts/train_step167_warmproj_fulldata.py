"""Step 167: step165-B (warm+W_proj) Tier-2 validation — full data, 150ep.

MOTIVATION
==========
step165-B (warm-start K=12→K=8 + W_proj [D,D]) achieved 90.96% at N=1024 D=16
using 50% data / 75ep (Tier-1). This is the efficiency track record.

This script runs the exact same config at full data (100%) for 150ep to:
  1. Establish the true ceiling (does it cross 95%?)
  2. Compare against N=1024 D=16 Ref at full data (expected ~85-87%)

Teacher checkpoint reused from results/train_step165_teacher.pt (already trained).

CONFIGS (N=1024, D=16, K_hh=8, K_iter=8, AH=1.0, 100%/150ep)
==============================================================
  Ref : K=8 scratch, full data 150ep (establish N=1024 D=16 ceiling without warm)
  B   : warm+W_proj, full data 150ep (step165-B winner, Tier-2 validation)

To reproduce:
    python -u scripts/train_step167_warmproj_fulldata.py --device cpu
    python -u scripts/train_step167_warmproj_fulldata.py --device mps
    python -u scripts/train_step167_warmproj_fulldata.py --configs B --device cpu
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
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--configs", default="",
                    help="Comma-sep config keys (e.g. Ref,B). Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 8; K_IN = 25; K_ITER = 8; K_ITER_TEACHER = 12
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

TEACHER_PATH = ROOT / "results" / "train_step165_teacher.pt"
OUT_PATH     = ROOT / "results" / "train_step167_warmproj_fulldata.json"


# ---------------------------------------------------------------------------
# Model: warm+W_proj (identical to step165 Config B)
# ---------------------------------------------------------------------------

class SGNNET_WarmProj(nn.Module):
    """AntiHebbian + W_proj. Warm-loadable from teacher state_dict (strict=False)."""

    def __init__(self, resonant, alpha_ahebb: float = 1.0):
        super().__init__()
        self.m           = resonant
        self.alpha_ahebb = alpha_ahebb
        D_ = resonant.base.D
        self.proj = nn.Linear(D_, D_, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)

    @property
    def W_pos(self):   return self.m.W_pos

    @property
    def W_phase(self): return self.m.W_phase

    def warm_load(self, teacher_state: dict) -> None:
        missing, unexpected = self.load_state_dict(teacher_state, strict=False)
        loaded = [k for k in teacher_state if k not in unexpected]
        print(f"    warm_load: {len(loaded)} keys, {len(missing)} new (proj), "
              f"{len(unexpected)} teacher-only")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base      = self.m.base
        Z         = base._seed(x)
        Z         = self.proj(Z)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = base.conn_hh
        N_h       = base.N_hidden

        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)
        for _ in range(base.K_iter):
            Z_fwd       = F.relu(Z - theta_pos)
            Z_nb        = Z_fwd[:, conn_hh, :]
            Z_struct    = (Z_nb * supp_w).sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z           = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)

        return base._readout(Z)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key:       str
    label:     str
    warm_proj: bool   # True = warm+W_proj; False = scratch (plain AH)

CONFIGS = [
    Config("Ref", "Ref  K=8 scratch full data 150ep (N=1024 D=16 ceiling)", False),
    Config("B",   "B    warm+W_proj full data 150ep (step165-B Tier-2 val)", True),
]


# ---------------------------------------------------------------------------
# Data — FULL data (no 50% subset)
# ---------------------------------------------------------------------------

_loaders_cache = None

def get_loaders():
    global _loaders_cache
    if _loaders_cache is None:
        tr, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
        _loaders_cache = (tr, va)
    return _loaders_cache


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _build_resonant(seed: int = SEED) -> "SGNNET_Resonant":
    torch.manual_seed(seed)
    K_random = max(1, K_HH // 4)
    K_local  = K_HH - K_random
    n_groups = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER,
        K_local=K_local, K_random=K_random, n_groups=n_groups,
        norm_mode="l2", encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )


def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    resonant = _build_resonant(SEED + seed_offset)
    if cfg.warm_proj:
        return SGNNET_WarmProj(resonant, ALPHA_AHEBB)
    else:
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 167 — warm+W_proj Tier-2 validation (full data 150ep)")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=100%")
    print(f"{'='*70}")
    print(f"step165-B reference (50% data 75ep): 90.96%\n")

    if not TEACHER_PATH.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {TEACHER_PATH}\n"
            f"Run train_step165_compound_warmstart.py first.")

    teacher_state = torch.load(TEACHER_PATH, map_location=DEVICE, weights_only=True)
    print(f"  Teacher loaded from {TEACHER_PATH}\n")

    cfg_filter     = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    get_loaders()
    results = {}

    for i, cfg in active_configs:
        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"{'─'*60}")

        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        if cfg.warm_proj:
            print(f"  Warm-loading from teacher...")
            model.warm_load(teacher_state)

        print(f"  params={n_params:,}  warm_proj={cfg.warm_proj}")

        t0      = time.time()
        kw      = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(
            model=model, train_loader=get_loaders()[0],
            val_loader=get_loaders()[1], device=DEVICE, **kw,
        )

        def _log(m):
            if str(DEVICE) == "mps":
                torch.mps.empty_cache()
            if (m["epoch"] + 1) % 10 == 0:
                print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)

        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1

        results[cfg.key] = {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER, "K_in": K_IN,
            "warm_proj": cfg.warm_proj,
            "data_frac": 1.0,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1), "n_params": n_params,
            "label": cfg.label,
        }

        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs_ref   = top1_best - ref_best if ref_best > 0 else 0
        vs_165b  = top1_best - 0.9096
        print(f"\n  top1={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"vs_step165B={vs_165b:+.4f}  elapsed={elapsed/60:.1f}min")

        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 167 SUMMARY — warm+W_proj Tier-2 (full data 150ep)")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"step165-B Tier-1 reference: 90.96% (50% data 75ep)")
    print(f"{'Key':4s}  {'warm_proj':>9}  {'params':>8}  {'top1':>7}  {'vs_Ref':>8}  {'vs_165B':>8}")
    print(f"{'─'*55}")
    for key, r in results.items():
        vs_r   = r["top1_best"] - ref_best if ref_best > 0 else 0
        vs_165 = r["top1_best"] - 0.9096
        print(f"{key:4s}  {str(r['warm_proj']):>9}  {r['n_params']:>8,}  "
              f"{r['top1_best']:.4f}  {vs_r:>+.4f}  {vs_165:>+.4f}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {OUT_PATH}")
    print(f"\nTarget: ≥95% for efficiency track phase exit (FLOPs ~3.1M ✓ params ✓ acc ❌).")


if __name__ == "__main__":
    main()
