"""Step 131: Tier-1 validation of step128-A and step117-A winners.

MOTIVATION
==========
Two Tier-0 winners from today's scouting blitz:
  - step128-A: weighted_neg activation (β=0.3) → +8.07pp at N=1024, 0 extra params
  - step117-A: learned W_proj [D,D] after scatter-sum → +8.66pp at N=1024, 4096 extra params

Both address different bottlenecks (activation recovery vs input compression) and
could compound. This Tier-1 run validates each independently and tests compounding.

CONFIGS (N=1024, D=64, K_hh=4, K_iter=12, AH=1.0, turing=0.0, reflect=0.5, 50%/75ep)
=====================================================================================
  Ref : Standard AH routing (baseline)
  A   : weighted_neg β=0.3 only (step128-A winner)
  B   : W_proj [D,D] only (step117-A winner)
  C   : weighted_neg β=0.3 + W_proj [D,D] (compound)
  D   : weighted_neg β=0.3 + low-rank [D,16]×[16,D] (compound with fewer params)

To reproduce:
    python -u scripts/train_step131_tier1_winners.py --device mps
    python -u scripts/train_step131_tier1_winners.py --device mps --epochs 20
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
from src.training.experiment_config   import trainer_kwargs, topology_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75,
                    help="Training epochs (default 75 for Tier-1; use 20 for scout)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (e.g. A,C). Empty = run all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 64; K_ITER = 12; K_IN = 50
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0


# ---------------------------------------------------------------------------
# Model: SGNNET_AH_WinnersCompound
# ---------------------------------------------------------------------------

class SGNNET_AH_WinnersCompound(nn.Module):
    """AH routing with optional weighted_neg activation and optional input projection.

    Combines two independent mechanisms:
      - weighted_neg: Z_fwd = relu(Z-θ) + β*relu(θ-Z)  (sub-threshold recovery)
      - input_proj:   Z = proj(Z) after _seed()          (input representation)
    """

    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float,
                 use_weighted_neg: bool = False, beta: float = 0.3,
                 proj_mode: str = "none", rank: int = 16):
        super().__init__()
        self.m              = resonant
        self.alpha          = alpha_ahebb
        self.use_weighted_neg = use_weighted_neg
        self.beta           = beta
        self.proj_mode      = proj_mode

        D_ = resonant.base.D
        if proj_mode == "linear":
            self.proj = nn.Linear(D_, D_, bias=False)
        elif proj_mode == "lowrank":
            self.proj_down = nn.Linear(D_, rank, bias=False)
            self.proj_up   = nn.Linear(rank, D_, bias=False)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _apply_proj(self, Z: torch.Tensor) -> torch.Tensor:
        if self.proj_mode == "linear":
            return self.proj(Z)
        elif self.proj_mode == "lowrank":
            return self.proj_up(self.proj_down(Z))
        return Z

    def _activation(self, Z: torch.Tensor, theta_pos: torch.Tensor) -> torch.Tensor:
        if self.use_weighted_neg:
            return F.relu(Z - theta_pos) + self.beta * F.relu(theta_pos - Z)
        return F.relu(Z - theta_pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)

        # Optional input projection
        Z = self._apply_proj(Z)

        # Routing loop
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden

        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z_fwd  = self._activation(Z, theta_pos)
            Z_nb   = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    use_weighted_neg: bool
    proj_mode: str
    use_custom: bool = True

CONFIGS = [
    Config("Ref", "Ref  standard AH (baseline)",           False, "none",    use_custom=False),
    Config("A",   "A    weighted_neg β=0.3 only",          True,  "none"),
    Config("B",   "B    W_proj [D,D] only",                False, "linear"),
    Config("C",   "C    weighted_neg + W_proj [D,D]",      True,  "linear"),
    Config("D",   "D    weighted_neg + low-rank [D,16,D]", True,  "lowrank"),
]


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
    tk = topology_kwargs(N)
    tk.pop("K_in", None); tk.pop("K_iter", None)
    base = SGNNET_SmallWorld(
        N_in=N_IN, N_hidden=N, N_out=N_OUT,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=K_IN, K_iter=K_ITER, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    if not cfg.use_custom:
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_AH_WinnersCompound(
        resonant, alpha_ahebb=ALPHA_AHEBB,
        use_weighted_neg=cfg.use_weighted_neg,
        proj_mode=cfg.proj_mode, rank=16,
    )


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 131 — Tier-1 validation: step128-A + step117-A winners")
    print(f"N={N}  D={D}  K_iter={K_ITER}  K_hh=4  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    for c in CONFIGS:
        print(f"  {c.key:4s}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step131_tier1_winners.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  weighted_neg={cfg.use_weighted_neg}  proj={cfg.proj_mode}")
        print(f"{'─'*60}")

        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(
            model=model, train_loader=get_loaders()[0],
            val_loader=get_loaders()[1], device=DEVICE, **kw,
        )
        history = trainer.train(n_epochs=EPOCHS)
        elapsed = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1

        results[cfg.key] = {
            "N": N, "D": D, "K_iter": K_ITER,
            "use_weighted_neg": cfg.use_weighted_neg,
            "proj_mode": cfg.proj_mode,
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
        print(f"\n  top1_best={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min  params={n_params:,}")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 131 SUMMARY — Tier-1 Winners Validation")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Config':<6}  {'w_neg':>5}  {'proj':>8}  {'params':>8}  "
          f"{'top1':>7}  {'vs_Ref':>8}  {'best_ep':>7}")
    print(f"{'─'*60}")
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"{key:<6}  {str(r['use_weighted_neg']):>5}  {r['proj_mode']:>8}  "
              f"{r['n_params']:>8}  {r['top1_best']:.4f}  {vs:>+.4f}  "
              f"{r['best_epoch']:>4}/{r['epochs_run']}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
