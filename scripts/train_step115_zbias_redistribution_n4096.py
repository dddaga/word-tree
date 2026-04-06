"""Step 115: Scale Z-bias + Redistribution Routing to N=4096.

MOTIVATION
==========
The two largest gains at N=1024 have NEVER been tested at N=4096:
  - Per-step Z-bias (step106-A): +7.42pp at N=1024 (90.78%, 768 params)
  - Input-modulated temperature routing (step75-D): +3.98pp at N=1024 (87.24%)

Both mechanisms are structurally sound at scale:
  - Z-bias is purely additive — no interaction with sparsity
  - Redistribution uses softmax (sum=1) — no gate-death
  - Neither depends on topology density (unlike group topology which NULL'd)

Scale transfer hypothesis: mechanisms operating on SIGNAL (activations, weights)
transfer to N=4096 better than mechanisms operating on STRUCTURE (topology, groups).
Evidence: group topology (+3.01pp→0pp), W_phase (+2.88pp→+0.18pp), turing (+1.68pp→-0.12pp)
all lost gains at N=4096 because they depend on structural differentiation that's
diluted at 0.1% connectivity (K_hh=4 / N=4096).

CONFIGS (N=4096, D=64, K_hh=4, K_iter=12, AH=1.0, turing=0.0, 50%/75ep)
=========================================================================
  Ref : standard AH K_iter=12 (current defaults, matches step89-Ref baseline)
  A   : Z-bias only (768 params, init=zeros, step106-A at N=4096)
  B   : redistribution routing only (tau_0=0.3, learned W_temp, step75-D at N=4096)
  C   : Z-bias + redistribution compound (both mechanisms, orthogonal axes)

To reproduce:
    python -u scripts/train_step115_zbias_redistribution_n4096.py --device mps
    python -u scripts/train_step115_zbias_redistribution_n4096.py --device mps --epochs 20  # scout
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs, topology_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75,
                    help="Training epochs (default 75; use 20 for Tier-0 scout)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys to run (e.g. A,C). Empty = run all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 4096; N_IN = 25088; N_OUT = 10; D = 64; K_ITER = 12; K_IN = 50
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
# step89-Ref at 50%/75ep for vs_ref column (N=4096 baseline)
STEP89_REF_50 = 0.9658   # step89-Ref 50%/75ep ≈ step71 Ref


# ---------------------------------------------------------------------------
# Model A: Z-bias per K_iter step (from step106)
# ---------------------------------------------------------------------------

class SGNNET_AH_ZBias(nn.Module):
    """AH routing with additive per-step Z-bias (step106-A mechanism)."""

    def __init__(self, resonant, alpha_ahebb: float, K_iter: int, D: int):
        super().__init__()
        self.m     = resonant
        self.alpha = alpha_ahebb
        self.K_iter_n = K_iter
        self.step_emb = nn.Parameter(torch.zeros(K_iter, D))  # init=zeros

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

        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha * pos_sim.clamp(min=0)
                   ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)

        for t in range(self.K_iter_n):
            Z_cond = Z + self.step_emb[t].unsqueeze(0).unsqueeze(0)
            Z_fwd  = F.relu(Z_cond - theta_pos)
            Z_nb   = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z_cond
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return base._readout(Z)


# ---------------------------------------------------------------------------
# Model B: Redistribution routing with learned temperature (from step75)
# ---------------------------------------------------------------------------

class SGNNET_AH_Redistribution(nn.Module):
    """AH routing with softmax redistribution + input-modulated temperature."""

    def __init__(self, resonant, alpha_ahebb: float, tau_0: float = 0.3):
        super().__init__()
        self.m         = resonant
        self.alpha     = alpha_ahebb
        self.tau_0     = tau_0
        N_h = resonant.base.N_hidden
        D_  = resonant.base.W_pos.shape[1]
        self.W_temp = nn.Parameter(torch.zeros(N_h, D_))

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

        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        ah_logit = -self.alpha * pos_sim.clamp(min=0)    # [N_h, K_hh]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]              # [B, N, K_hh, D]

            # Z-state score
            z_score = (Z_fwd.unsqueeze(2) * Z_nb).sum(-1)   # [B, N, K_hh]

            # Per-neuron learned temperature
            temp_score = (self.W_temp * Z_fwd).sum(-1)   # [B, N]
            temp = self.tau_0 * (1.0 + torch.sigmoid(temp_score))
            logit = z_score / temp.unsqueeze(-1) + ah_logit.unsqueeze(0)

            w        = F.softmax(logit, dim=2)
            Z_struct = (w.unsqueeze(-1) * Z_nb).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return base._readout(Z)


# ---------------------------------------------------------------------------
# Model C: Z-bias + Redistribution compound
# ---------------------------------------------------------------------------

class SGNNET_AH_ZBias_Redistribution(nn.Module):
    """Compound: per-step Z-bias + softmax redistribution routing."""

    def __init__(self, resonant, alpha_ahebb: float,
                 K_iter: int, D: int, tau_0: float = 0.3):
        super().__init__()
        self.m         = resonant
        self.alpha     = alpha_ahebb
        self.tau_0     = tau_0
        self.K_iter_n  = K_iter
        self.step_emb  = nn.Parameter(torch.zeros(K_iter, D))
        N_h = resonant.base.N_hidden
        D_  = resonant.base.W_pos.shape[1]
        self.W_temp = nn.Parameter(torch.zeros(N_h, D_))

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

        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        ah_logit = -self.alpha * pos_sim.clamp(min=0)

        Z_reflected = torch.zeros_like(Z)

        for t in range(self.K_iter_n):
            # Per-step Z-bias
            Z_cond = Z + self.step_emb[t].unsqueeze(0).unsqueeze(0)

            Z_fwd = F.relu(Z_cond - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]

            # Redistribution: Z-score + learned temperature
            z_score = (Z_fwd.unsqueeze(2) * Z_nb).sum(-1)
            temp_score = (self.W_temp * Z_fwd).sum(-1)
            temp = self.tau_0 * (1.0 + torch.sigmoid(temp_score))
            logit = z_score / temp.unsqueeze(-1) + ah_logit.unsqueeze(0)

            w        = F.softmax(logit, dim=2)
            Z_struct = (w.unsqueeze(-1) * Z_nb).sum(dim=2)

            Z_remainder = Z_fwd - Z_cond
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
    model_type: str   # "ref", "zbias", "redist", "compound"


CONFIGS = [
    Config("Ref", "Ref  standard AH (step89 baseline at N=4096)", "ref"),
    Config("A",   "A    Z-bias only (768 params, step106 mechanism)", "zbias"),
    Config("B",   "B    Redistribution tau=0.3 + W_temp (step75 mechanism)", "redist"),
    Config("C",   "C    Z-bias + redistribution compound", "compound"),
]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_resonant(seed_offset: int = 0) -> SGNNET_Resonant:
    torch.manual_seed(SEED + seed_offset)
    topo = topology_kwargs(N)
    topo.pop("K_in", None)
    topo.pop("K_iter", None)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, encoding_mode="fourier",
        **topo,
    )
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )


def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    resonant = make_resonant(seed_offset)
    if cfg.model_type == "ref":
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    elif cfg.model_type == "zbias":
        return SGNNET_AH_ZBias(resonant, ALPHA_AHEBB, K_ITER, D)
    elif cfg.model_type == "redist":
        return SGNNET_AH_Redistribution(resonant, ALPHA_AHEBB, tau_0=0.3)
    elif cfg.model_type == "compound":
        return SGNNET_AH_ZBias_Redistribution(resonant, ALPHA_AHEBB, K_ITER, D, tau_0=0.3)
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


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
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 115 — Scale Z-bias + Redistribution to N=4096")
    print(f"N={N}  D={D}  K_iter={K_ITER}  K_hh=4  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"Baseline (step89 Ref 50%/75ep): ~{STEP89_REF_50:.4f}")
    print(f"{'='*70}\n")

    print("Configs:")
    for c in CONFIGS:
        print(f"  {c.key:4s}  type={c.model_type:10s}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step115_zbias_redistribution_n4096.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS) if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  type={cfg.model_type}")
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

        history   = trainer.train(n_epochs=EPOCHS)
        elapsed   = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1
        vs_ref    = top1_best - STEP89_REF_50

        results[cfg.key] = {
            "N": N, "D": D, "K_iter": K_ITER,
            "model_type": cfg.model_type,
            "alpha_ahebb": ALPHA_AHEBB,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "vs_ref": round(vs_ref, 6),
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params,
            "label": cfg.label,
        }

        print(f"\n  top1_best={top1_best:.4f}  vs_ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary table
    print(f"\n{'='*70}")
    print(f"STEP 115 SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<6}  {'type':<10}  {'params':>8}  {'top1':>7}  {'vs_ref':>8}  label")
    print(f"{'─'*70}")
    for key, r in results.items():
        print(f"{key:<6}  {r['model_type']:<10}  {r['n_params']:>8,}  "
              f"{r.get('top1_best',0):.4f}  {r['vs_ref']:>+.4f}  {r['label']}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
