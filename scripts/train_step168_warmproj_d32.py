"""Step 168: warm-start + W_proj at N=1024 D=32 — efficiency track ceiling test.

MOTIVATION
==========
step165-B (warm+W_proj) at N=1024 D=16: 90.96% (+10.32pp over scratch) @ ~3.1M FLOPs.
step144-C (W_proj+RigL) at N=1024 D=32: 93.50% (+5.55pp over scratch) @ ~6.1M FLOPs.

Gap to phase-exit target (≥95% @ ≤6.18M FLOPs):
  D=16: 90.96% — 4.04pp short
  D=32: 93.50% — 1.50pp short

QUESTION: Does warm-start + W_proj push N=1024 D=32 past 95%?
  - D=32 scratch (step144 Ref) ≈ 87.95%
  - D=32 W_proj alone (step144-B) ≈ 93.02%
  - D=32 warm+W_proj: unknown — if +3pp over W_proj alone → 96%+

Phase-exit: if B (warm+W_proj D=32) ≥ 95% → EFFICIENCY TRACK COMPLETE.

CONFIGS (N=1024, D=32, K_hh=8, K_iter=8, AH=1.0, 50%/75ep)
=============================================================
  Ref : K=8 scratch, D=32 (reproduce step144 baseline ~87.95%)
  A   : K=8 warm-start only (isolate warm-start gain at D=32)
  B   : K=8 warm-start + W_proj [D,D] ← primary target

Teacher: K=12 N=1024 D=32 trained fresh (no cached checkpoint at D=32).
Teacher cached at results/train_step168_teacher_d32.pt.

FLOPs at N=1024 D=32 K_iter=8 K_hh=8: ~6.1M (≤6.18M budget ✓)

To reproduce:
    python -u scripts/train_step168_warmproj_d32.py --device mps
    python -u scripts/train_step168_warmproj_d32.py --device cpu
    python -u scripts/train_step168_warmproj_d32.py --configs B --device mps
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
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--configs", default="",
                    help="Comma-sep config keys (e.g. A,B). Empty = all.")
parser.add_argument("--force-retrain-teacher", action="store_true")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS          = args.epochs
BATCH           = 128
SEED            = 42
DATA            = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10
D = 32; K_HH = 8; K_IN = 25; K_ITER = 8; K_ITER_TEACHER = 12
ALPHA_REFLECT   = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

TEACHER_PATH = ROOT / "results" / "train_step168_teacher_d32.pt"
OUT_PATH     = ROOT / "results" / "train_step168_warmproj_d32.json"


# ---------------------------------------------------------------------------
# Model: warm+W_proj (same pattern as step165-B, now at D=32)
# ---------------------------------------------------------------------------

class SGNNET_WarmProj_D32(nn.Module):
    """AntiHebbian + optional W_proj [D,D]. Warm-loadable from teacher state_dict."""

    def __init__(self, resonant, alpha_ahebb: float = 1.0, use_proj: bool = True):
        super().__init__()
        self.m           = resonant
        self.alpha_ahebb = alpha_ahebb
        self.use_proj    = use_proj
        if use_proj:
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
        print(f"    warm_load: {len(loaded)} keys, "
              f"{len(missing)} new, {len(unexpected)} teacher-only")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base      = self.m.base
        Z         = base._seed(x)
        if self.use_proj:
            Z = self.proj(Z)
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
    warm_init: bool
    use_proj:  bool

CONFIGS = [
    Config("Ref", "Ref  K=8 scratch D=32 (step144 Ref repro ~87.95%)",  False, False),
    Config("A",   "A    K=8 warm-start only D=32 (isolate init gain)",   True,  False),
    Config("B",   "B    K=8 warm-start + W_proj D=32 ← PHASE-EXIT TEST", True,  True),
]


# ---------------------------------------------------------------------------
# Data — 50% subset (Tier-1 budget)
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
# Model factory
# ---------------------------------------------------------------------------

def _build_resonant(k_iter: int = K_ITER, seed: int = SEED):
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


def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    if not cfg.warm_init and not cfg.use_proj:
        resonant = _build_resonant(K_ITER, SEED + seed_offset)
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    resonant = _build_resonant(K_ITER, SEED + seed_offset)
    return SGNNET_WarmProj_D32(resonant, ALPHA_AHEBB, use_proj=cfg.use_proj)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Teacher: train or load
# ---------------------------------------------------------------------------

def get_or_train_teacher(force: bool = False) -> dict:
    if not force and TEACHER_PATH.exists():
        print(f"  Loading cached D=32 teacher from {TEACHER_PATH}")
        return torch.load(TEACHER_PATH, map_location=DEVICE, weights_only=True)

    print(f"\n{'─'*60}")
    print(f"Training D=32 teacher  K_iter={K_ITER_TEACHER}  epochs={EPOCHS}")
    print(f"{'─'*60}")
    resonant = _build_resonant(K_ITER_TEACHER, SEED)
    teacher  = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos").to(DEVICE)

    t0      = time.time()
    kw      = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(
        model=teacher, train_loader=get_loaders()[0],
        val_loader=get_loaders()[1], device=DEVICE, **kw,
    )
    def _log(m):
        if str(DEVICE) == "mps":
            torch.mps.empty_cache()
        if (m["epoch"] + 1) % 10 == 0:
            print(f"    ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
    history   = trainer.train(n_epochs=EPOCHS, log_fn=_log)
    best_top1 = max(h["val_top1"] for h in history)
    elapsed   = time.time() - t0
    print(f"  Teacher done: top1={best_top1:.4f}  elapsed={elapsed/60:.1f}min")
    TEACHER_PATH.parent.mkdir(exist_ok=True)
    torch.save(teacher.state_dict(), TEACHER_PATH)
    return teacher.state_dict()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 168 — warm+W_proj at N=1024 D=32 (efficiency track ceiling)")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}")
    print(f"Target: ≥95% → phase-exit criterion met (FLOPs ~6.1M ≤ 6.18M ✓)\n")

    print("Configs:")
    for c in CONFIGS:
        mech = []
        if c.warm_init: mech.append("warm-start")
        if c.use_proj:  mech.append("W_proj")
        print(f"  {c.key:4s}  {'+'.join(mech) or 'scratch':20s}  {c.label}")
    print()

    cfg_filter     = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    need_teacher  = any(cfg.warm_init for _, cfg in active_configs)
    teacher_state = get_or_train_teacher(force=args.force_retrain_teacher) if need_teacher else None

    get_loaders()
    results = {}

    for i, cfg in active_configs:
        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"{'─'*60}")

        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        if cfg.warm_init and teacher_state is not None:
            print(f"  Warm-loading from D=32 teacher...")
            model.warm_load(teacher_state)

        print(f"  params={n_params:,}  warm={cfg.warm_init}  proj={cfg.use_proj}")

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
            "warm_init": cfg.warm_init, "use_proj": cfg.use_proj,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1), "n_params": n_params,
            "label": cfg.label,
        }

        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs_ref   = top1_best - ref_best if ref_best > 0 else 0
        phase_exit = "✓ PHASE EXIT!" if top1_best >= 0.95 else f"({0.95 - top1_best:.3f} short)"
        print(f"\n  top1={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min  {phase_exit}")

        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 168 SUMMARY — warm+W_proj at N=1024 D=32")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"step144-C reference (W_proj+RigL D=32 50%/75ep): 93.50%")
    print(f"step165-B reference (warm+W_proj D=16 50%/75ep): 90.96%")
    print(f"Phase-exit target: ≥95% @ FLOPs ≤6.18M\n")
    print(f"{'Key':4s}  {'mechanism':20s}  {'params':>8}  {'top1':>7}  {'vs_Ref':>8}  {'phase_exit':>12}")
    print(f"{'─'*65}")
    for key, r in results.items():
        vs_r = r["top1_best"] - ref_best if ref_best > 0 else 0
        pe   = "✓ DONE!" if r["top1_best"] >= 0.95 else f"-{0.95-r['top1_best']:.3f}pp"
        mech = ("warm" if r["warm_init"] else "") + ("+W_proj" if r["use_proj"] else "")
        print(f"{key:4s}  {mech or 'scratch':20s}  {r['n_params']:>8,}  "
              f"{r['top1_best']:.4f}  {vs_r:>+.4f}  {pe:>12}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
