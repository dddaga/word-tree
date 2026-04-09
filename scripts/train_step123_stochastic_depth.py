"""Step 123: Stochastic Depth Training for SGNNET.

MOTIVATION
==========
During training, randomly skip K_iter steps with increasing probability for deeper
steps. This regularizes against oversmoothing and creates implicit robustness to
varying depths. At inference, use all K_iter steps. Standard technique from ResNet
stochastic depth (Huang et al. 2016), adapted for iterative message passing.

MECHANISM
=========
At each K_iter step t during training:
  - Compute drop probability: p_drop = (t / K_iter) * p_max
  - With probability p_drop, skip this step entirely (Z unchanged)
  - With probability (1 - p_drop), run the step normally, scaled by 1/(1-p_drop)
    to maintain expected value
  - During eval: run all steps (no dropping, no scaling)

Config D uses uniform drop probability across all steps instead of linearly increasing.

CONFIGS (N=1024, D=32, K_hh=4, K_iter=12, AH=1.0, 50%/75ep)
==============================================================
  Ref : Standard AH, no stochastic depth (D=32 baseline)
  A   : p_max=0.2 (last step dropped 20% of the time)
  B   : p_max=0.4 (last step dropped 40% of the time)
  C   : p_max=0.6 (aggressive dropping)
  D   : Uniform drop p=0.1 (all steps equal probability, not increasing)

To reproduce:
    python -u scripts/train_step123_stochastic_depth.py --device mps
    python -u scripts/train_step123_stochastic_depth.py --device mps --epochs 20  # scout
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
                    help="Training epochs (default 75; use 20 for Tier-0 scout)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys to run (e.g. A,D). Empty = run all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 32; K_ITER = 12; K_IN = 50
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0


# ---------------------------------------------------------------------------
# Model: AH with stochastic depth — drop K_iter steps during training
# ---------------------------------------------------------------------------

class SGNNET_AH_StochasticDepth(nn.Module):
    """SGNNET AntiHebbian with stochastic depth regularization.

    During training, each K_iter step t is dropped with probability p_drop(t).
    Two modes:
      "linear"  — p_drop(t) = (t / K_iter) * p_max  (deeper steps dropped more)
      "uniform" — p_drop(t) = p_max for all steps

    When a step is NOT dropped, its output is scaled by 1/(1-p_drop) to preserve
    expected value. During eval, all steps run without dropping or scaling.
    """

    def __init__(self, resonant, alpha_ahebb: float,
                 p_max: float = 0.2,
                 drop_mode: str = "linear"):
        super().__init__()
        self.m           = resonant
        self.alpha_ahebb = alpha_ahebb
        self.p_max       = p_max
        self.drop_mode   = drop_mode

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _drop_prob(self, t: int, K: int) -> float:
        """Compute drop probability for step t (0-indexed) out of K total."""
        if self.drop_mode == "uniform":
            return self.p_max
        else:  # linear: increases from 0 to p_max
            return (t / K) * self.p_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base      = self.m.base
        Z         = base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = base.conn_hh
        N_h       = base.N_hidden
        K         = base.K_iter

        # Pre-compute static AH suppression weights (wpos variant)
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)       # [N_h, K_hh]
        supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)                      # [1, N, K_hh, 1]

        Z_reflected = torch.zeros_like(Z)

        for t in range(K):
            p_drop = self._drop_prob(t, K)

            # Stochastic depth: skip step during training with probability p_drop
            if self.training and p_drop > 0:
                if torch.rand(1).item() < p_drop:
                    # Drop this step entirely — Z unchanged
                    continue

            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]                           # [B, N, K_hh, D]

            Z_struct = (Z_nb * supp_w).sum(dim=2)

            # Reflection accumulator
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            # Phase inhibition
            if self.m.alpha_turing != 0.0:
                W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
                Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
                Z_new = Z_struct + Z_reflected + self.m.alpha_turing * Z_inh
            else:
                Z_new = Z_struct + Z_reflected

            # Scale by 1/(1-p_drop) during training to preserve expected value
            if self.training and p_drop > 0:
                Z_new = Z_new / (1.0 - p_drop)

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return base._readout(Z)


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    p_max: float
    drop_mode: str
    use_stochastic: bool = True


CONFIGS = [
    Config("Ref", "Ref  Standard AH, no stochastic depth (D=32 baseline)",
           0.0, "linear", use_stochastic=False),
    Config("A",   "A    p_max=0.2 linear (last step 20% drop)",
           0.2, "linear"),
    Config("B",   "B    p_max=0.4 linear (last step 40% drop)",
           0.4, "linear"),
    Config("C",   "C    p_max=0.6 linear (aggressive dropping)",
           0.6, "linear"),
    Config("D",   "D    Uniform p=0.1 (all steps equal probability)",
           0.1, "uniform"),
]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    topo = topology_kwargs(N)
    topo.pop("K_in", None)
    topo.pop("K_iter", None)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, encoding_mode="fourier",
        **topo,
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    if not cfg.use_stochastic:
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_AH_StochasticDepth(
        resonant, alpha_ahebb=ALPHA_AHEBB,
        p_max=cfg.p_max, drop_mode=cfg.drop_mode,
    )


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
    print(f"Step 123 — Stochastic Depth Training")
    print(f"N={N}  D={D}  K_iter={K_ITER}  K_hh=4  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    print("Configs:")
    for c in CONFIGS:
        print(f"  {c.key:4s}  p_max={c.p_max:.1f}  mode={c.drop_mode:7s}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step123_stochastic_depth.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  p_max={cfg.p_max}  mode={cfg.drop_mode}")
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

        results[cfg.key] = {
            "N": N, "D": D, "K_iter": K_ITER,
            "p_max": cfg.p_max,
            "drop_mode": cfg.drop_mode,
            "alpha_ahebb": ALPHA_AHEBB,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params,
            "label": cfg.label,
        }

        # vs Ref (compute once Ref is available)
        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs_ref = top1_best - ref_best if ref_best > 0 else 0

        print(f"\n  top1_best={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary table
    print(f"\n{'='*70}")
    print(f"STEP 123 SUMMARY (D={D})")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Config':<6}  {'p_max':>5}  {'mode':>7}  "
          f"{'params':>8}  {'top1':>7}  {'vs_Ref':>8}")
    print(f"{'─'*70}")
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"{key:<6}  {r['p_max']:>5.1f}  {r['drop_mode']:>7}  "
              f"{r['n_params']:>8,}  {r['top1_best']:.4f}  {vs:>+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
