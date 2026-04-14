"""Step 158: DropMessage regularization in K_iter routing — combat over-smoothing.

MOTIVATION
==========
At K_iter=12, messages are aggregated 12 times. Risk: over-smoothing where all
neurons converge to similar representations (information blending). DropMessage
(AAAI 2023) stochastically drops entire messages (all D dims of a specific edge)
during training, acting as regularization. Zero new parameters. ~5-line change.

The key insight: dropping messages forces neurons to be robust to missing neighbors,
preventing co-adaptation. Different from dropout (which drops feature dims): here
we drop entire edge transmissions, preserving D-dimensional message structure.

CONFIGS (N=1024, D=16, K_hh=8, K_iter=8, AH α=1.0, data_frac=0.5, 75ep Tier-0)
================================================================================
  Ref : no DropMessage (current default)
  A   : drop_rate=0.1 (drop 10% of messages per routing step)
  B   : drop_rate=0.2
  C   : drop_rate=0.3
  D   : drop_rate=0.1, only during last K_iter//2 steps (early steps unaffected)

Ref baseline at N=1024 D=16: ~89.85% (Tier-1, step131).

To reproduce:
    python -u scripts/train_step158_drop_message.py --device mps
    python -u scripts/train_step158_drop_message.py --device mps --epochs 20  # quick scout
    python -u scripts/train_step158_drop_message.py --device cpu --configs A,Ref
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75,
                    help="Training epochs (default 75 for Tier-0)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys to run (e.g. A,Ref). Empty = run all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 16; K_ITER = 8; K_IN = 25; K_HH = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0


# ---------------------------------------------------------------------------
# Custom model: AntiHebbian routing with DropMessage regularization
# ---------------------------------------------------------------------------

class SGNNET_AH_DropMessage(nn.Module):
    """AntiHebbian routing with configurable DropMessage stochastic regularization.

    drop_rate: fraction of messages to drop per routing step during training.
    drop_late_only: if True, only apply DropMessage in the last K_iter//2 steps.

    DropMessage zeros entire messages [D dims] for randomly selected edges,
    forcing neurons to be robust to missing neighborhood information.
    At inference (eval mode), drop_rate is ignored (no messages dropped).
    """

    def __init__(
        self,
        resonant,
        alpha_ahebb: float,
        drop_rate: float = 0.0,
        drop_late_only: bool = False,
    ):
        super().__init__()
        self.m              = resonant
        self.alpha_ahebb    = alpha_ahebb
        self.drop_rate      = drop_rate
        self.drop_late_only = drop_late_only

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base      = self.m.base
        Z         = base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = base.conn_hh
        N_h       = base.N_hidden
        K_iter    = base.K_iter

        # Pre-compute static AH suppression weights (wpos variant)
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                   ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)

        # Threshold for late-only dropping: apply only in last half of K_iter steps
        late_start = K_iter // 2 if self.drop_late_only else 0

        for t in range(K_iter):
            Z_fwd = F.relu(Z - theta_pos)

            # Gather neighbor messages: [B, N, K_hh, D]
            Z_nb = Z_fwd[:, conn_hh, :]

            # DropMessage: randomly zero entire edge messages during training
            should_drop = (
                self.training
                and self.drop_rate > 0.0
                and t >= late_start
            )
            if should_drop:
                # keep mask: [B, N, K_hh] — same for all D dims of an edge
                keep = torch.bernoulli(
                    torch.ones(Z_nb.shape[:-1], device=Z_nb.device) * (1.0 - self.drop_rate)
                )
                Z_nb = Z_nb * keep.unsqueeze(-1)

            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return base._readout(Z)


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    drop_rate: float
    drop_late_only: bool = False

CONFIGS = [
    Config("Ref", "Ref  No DropMessage (current default)",              0.0,  False),
    Config("A",   "A    drop_rate=0.1 all steps",                       0.1,  False),
    Config("B",   "B    drop_rate=0.2 all steps",                       0.2,  False),
    Config("C",   "C    drop_rate=0.3 all steps",                       0.3,  False),
    Config("D",   "D    drop_rate=0.1, last K_iter//2 steps only",      0.1,  True),
]


# ---------------------------------------------------------------------------
# Data loaders (cached, 50% data)
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
    return SGNNET_AH_DropMessage(
        resonant, ALPHA_AHEBB,
        drop_rate=cfg.drop_rate,
        drop_late_only=cfg.drop_late_only,
    )


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 158 — DropMessage regularization")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  K_in={K_IN}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    print("Configs:")
    for c in CONFIGS:
        late = " (late-only)" if c.drop_late_only else ""
        print(f"  {c.key:4s}  drop={c.drop_rate:.1f}{late:12s}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step158_drop_message.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  drop_rate={cfg.drop_rate}  drop_late_only={cfg.drop_late_only}")
        print(f"{'─'*60}")

        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(
            model=model,
            train_loader=get_loaders()[0],
            val_loader=get_loaders()[1],
            device=DEVICE,
            **kw,
        )
        history = trainer.train(n_epochs=EPOCHS)
        elapsed = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1

        results[cfg.key] = {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER, "K_in": K_IN,
            "drop_rate": cfg.drop_rate,
            "drop_late_only": cfg.drop_late_only,
            "alpha_ahebb": ALPHA_AHEBB,
            "alpha_reflect": ALPHA_REFLECT,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params,
            "label": cfg.label,
        }

        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs_ref   = top1_best - ref_best if ref_best > 0 else 0
        print(f"\n  top1={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min  params={n_params:,}")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary table
    print(f"\n{'='*70}")
    print(f"STEP 158 SUMMARY — DropMessage Regularization")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Key':4s}  {'drop':>5}  {'late':>5}  {'params':>8}  {'top1':>7}  {'vs_Ref':>8}")
    print(f"{'─'*50}")
    for key, r in results.items():
        vs   = r["top1_best"] - ref_best if ref_best > 0 else 0
        late = "Y" if r["drop_late_only"] else "N"
        print(f"{key:4s}  {r['drop_rate']:.1f}    {late:>5}  {r['n_params']:>8,}  "
              f"{r['top1_best']:.4f}  {vs:>+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")
    print(f"\nDecision: if any config shows ≥+0.5pp → Tier-1 candidate.")
    print(f"If all negative → over-smoothing is not the bottleneck at K_iter=8.")


if __name__ == "__main__":
    main()
