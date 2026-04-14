"""Step 173: warm+W_proj at N=2048 D=32 K_hh=4 — N-scaling + representation capacity.

MOTIVATION
==========
Efficiency track ceiling progression:
  N=1024 D=16 K_hh=8 (step167): 93.17% @ ~3.1M FLOPs
  N=1024 D=32 K_hh=8 (step169): 94.01% @ ~6.1M FLOPs  ← best so far
  N=1024 D=32 K_hh=4 (step171): 93.35% @ ~3.1M FLOPs  (B=α=1.05)
  N=2048 D=16 K_hh=8 (step170): ~93% expected @ ~6.2M FLOPs
  N=1024 D=48 K_hh=4 (step172): running @ ~4.65M FLOPs

Key insight: N×K_hh is the FLOPs multiplier for message passing.
  N=1024 × K_hh=8 = 8192  → ~6.1M FLOPs
  N=2048 × K_hh=4 = 8192  → ~6.1M FLOPs (SAME!)

So N=2048 D=32 K_hh=4 K_iter=8 fits within the 6.18M FLOPs budget.
Doubling N while halving K_hh gives: more neurons (better coverage), sparser
per-neuron connectivity (less over-smoothing) — potentially the best of both worlds.

Hypothesis: N=2048 breaks the N=1024 D=32 ceiling (~94%) by providing more
representational neurons with the same FLOPs envelope.

Teacher: K=12 N=2048 D=32 K_hh=4 trained fresh.
Cached at results/train_step173_teacher_n2048_d32.pt.

CONFIGS (N=2048, D=32, K_hh=4, K_in=25, 50% data, 75ep — Tier-1)
==================================================================
  Ref : K=8 scratch N=2048 D=32 K_hh=4 (new N=2048 D=32 baseline)
  B   : warm+W_proj K_hh=4 K_iter=8 alpha=1.0  ← primary test
  C   : warm+W_proj K_hh=4 K_iter=8 alpha=1.05 ← alpha calibration

Phase-exit criterion: ≥95% @ ≤6.18M FLOPs.
N=2048 D=32 K_hh=4 K_iter=8 FLOPs ≈ 6.1M ✓

To reproduce:
    python -u scripts/train_step173_n2048_d32.py --device mps
    python -u scripts/train_step173_n2048_d32.py --device cpu
    python -u scripts/train_step173_n2048_d32.py --configs B --device mps
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

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--configs", default="",
                    help="Comma-sep config keys. Empty = all.")
parser.add_argument("--force-retrain-teacher", action="store_true")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS         = args.epochs
BATCH          = 128
SEED           = 42
DATA           = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 32; K_HH = 4; K_IN = 25; K_ITER_TEACHER = 12
ALPHA_REFLECT  = 0.5; ALPHA_TURING = 0.0

TEACHER_PATH = ROOT / "results" / "train_step173_teacher_n2048_d32.pt"
OUT_PATH     = ROOT / "results" / "train_step173_n2048_d32.json"


class SGNNET_WarmProj(nn.Module):
    """AntiHebbian + W_proj [D,D]. Warm-loadable from teacher state_dict."""

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
        print(f"    warm_load: {len(loaded)} keys, {len(missing)} new, {len(unexpected)} teacher-only")

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


@dataclass
class Config:
    key:         str
    label:       str
    warm:        bool
    proj:        bool
    alpha_ahebb: float

CONFIGS = [
    Config("Ref", "Ref  K=8 scratch N=2048 D=32 K_hh=4",               False, False, 1.0),
    Config("B",   "B    warm+W_proj N=2048 D=32 K_hh=4 alpha=1.0",     True,  True,  1.0),
    Config("C",   "C    warm+W_proj N=2048 D=32 K_hh=4 alpha=1.05",    True,  True,  1.05),
]

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


def _build(k_iter: int, seed: int = SEED) -> "SGNNET_Resonant":
    torch.manual_seed(seed)
    K_random = max(1, K_HH // 4)
    K_local  = K_HH - K_random
    n_groups = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=k_iter,
        K_local=K_local, K_random=K_random, n_groups=n_groups,
        norm_mode="l2", encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )


def get_teacher(force: bool = False) -> dict:
    if not force and TEACHER_PATH.exists():
        print(f"  Loading N=2048 D=32 K_hh=4 teacher from {TEACHER_PATH}")
        return torch.load(TEACHER_PATH, map_location=DEVICE, weights_only=True)
    print(f"\nTraining N=2048 D=32 K_hh=4 teacher K={K_ITER_TEACHER} for {EPOCHS}ep...")
    t = SGNNET_AntiHebbian(_build(K_ITER_TEACHER), alpha_ahebb=1.0, variant="wpos").to(DEVICE)
    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    tr = Trainer(model=t, train_loader=get_loaders()[0], val_loader=get_loaders()[1],
                 device=DEVICE, **kw)
    def _log(m):
        if str(DEVICE) == "mps": torch.mps.empty_cache()
        if (m["epoch"] + 1) % 10 == 0:
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
    h = tr.train(n_epochs=EPOCHS, log_fn=_log)
    best = max(x["val_top1"] for x in h)
    print(f"  Teacher done: {best:.4f}")
    TEACHER_PATH.parent.mkdir(exist_ok=True)
    torch.save(t.state_dict(), TEACHER_PATH)
    return t.state_dict()


def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    r = _build(8, SEED + seed_offset)
    if cfg.warm or cfg.proj:
        return SGNNET_WarmProj(r, cfg.alpha_ahebb)
    return SGNNET_AntiHebbian(r, alpha_ahebb=cfg.alpha_ahebb, variant="wpos")


def main():
    print(f"\n{'='*70}")
    print(f"Step 173 — warm+W_proj N=2048 D=32 K_hh=4 (N-scaling phase-exit test)")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_in={K_IN}  Epochs={EPOCHS}  Data=50%")
    print(f"FLOPs ≈ 6.1M (N×K_hh same as N=1024×K_hh=8) — within 6.18M budget ✓")
    print(f"Phase-exit: ≥95% @ ≤6.18M FLOPs")
    print(f"step169 D=32 K_hh=8 N=1024 reference (full data): 94.01%\n{'='*70}")

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active     = [(i, c) for i, c in enumerate(CONFIGS) if not cfg_filter or c.key in cfg_filter]

    need_teacher = any(c.warm for _, c in active)
    ts = get_teacher(force=args.force_retrain_teacher) if need_teacher else None
    get_loaders()
    results = {}

    for i, cfg in active:
        print(f"\n{'─'*60}\nConfig {cfg.key}: {cfg.label}\n{'─'*60}")
        model = make_model(cfg, i).to(DEVICE)
        if cfg.warm and ts:
            print(f"  Warm-loading from N=2048 D=32 K_hh=4 teacher...")
            model.warm_load(ts)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}  N={N}  D={D}  K_hh={K_HH}  alpha={cfg.alpha_ahebb}")

        t0      = time.time()
        kw      = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=get_loaders()[0],
                          val_loader=get_loaders()[1], device=DEVICE, **kw)

        def _log(m):
            if str(DEVICE) == "mps": torch.mps.empty_cache()
            if (m["epoch"] + 1) % 10 == 0:
                flag = " *** PHASE EXIT! ***" if m["val_top1"] >= 0.95 else ""
                print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}{flag}", flush=True)

        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best  = max(top1h); bep = int(np.argmax(top1h)) + 1
        results[cfg.key] = {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": 8,
            "alpha_ahebb": cfg.alpha_ahebb, "warm": cfg.warm, "proj": cfg.proj,
            "data_frac": 0.5,
            "top1_best": best, "top1_last": top1h[-1],
            "best_epoch": bep, "epochs_run": len(history),
            "top1_history": top1h,
            "elapsed_s": round(elapsed, 1), "n_params": n_p, "label": cfg.label,
        }
        ref = results.get("Ref", {}).get("top1_best", 0)
        pe  = "✓ PHASE EXIT!" if best >= 0.95 else f"({0.95 - best:.3f}pp short)"
        print(f"\n  top1={best:.4f}  vs_Ref={best-ref:+.4f}  vs_step169={best-0.9401:+.4f}  "
              f"{pe}  elapsed={elapsed/60:.1f}min")
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}\nSTEP 173 SUMMARY — warm+W_proj N=2048 D=32 K_hh=4\n{'='*70}")
    ref = results.get("Ref", {}).get("top1_best", 0)
    print(f"step169 D=32 K_hh=8 N=1024 full data reference: 94.01%")
    print(f"step171-B D=32 K_hh=4 N=1024 Tier-1 reference: 93.35%")
    print(f"Phase-exit target: ≥95% @ FLOPs ≤6.18M")
    print(f"{'Key':4s}  {'label':45s}  {'top1':>7}  {'vs_Ref':>8}  {'vs_169':>8}  {'phase_exit':>12}")
    print(f"{'─'*90}")
    for k, r in results.items():
        vs_r   = r["top1_best"] - ref if ref > 0 else 0
        vs_169 = r["top1_best"] - 0.9401
        pe     = "✓ DONE!" if r["top1_best"] >= 0.95 else f"-{0.95-r['top1_best']:.3f}pp"
        print(f"{k:4s}  {r['label'][:45]:45s}  {r['top1_best']:.4f}  {vs_r:>+.4f}  {vs_169:>+.4f}  {pe:>12}")
    print(f"\nResults → {OUT_PATH}")


if __name__ == "__main__":
    main()
