"""Step 128: ConcatReLU / activation function ablation for SGNNET.

MOTIVATION
==========
SGNNET's routing loop applies a hard threshold:
    Z_fwd = F.relu(Z - theta_pos)
This destroys sub-threshold information. The reflection accumulator
(alpha_reflect=0.5) partially recovers it with a one-step delay.
ConcatReLU-style approaches make sub-threshold info available within-step.

CONFIGS (N=1024, D=64, K_hh=4, K_iter=12, AH=1.0, turing=0.0, reflect=0.5, 50%/75ep)
=====================================================================================
  Ref : Standard AH routing (F.relu(Z - theta) + reflection accumulator)
  A   : Weighted negative beta=0.3 + reflection       (0 extra params)
  B   : Weighted negative beta=0.3, NO reflection     (0 extra params)
  C   : ConcatReLU + learned W_proj(2D→D) + reflect   (8192 extra params)
  D   : SwiGLU-inspired: F.silu(Z-theta)*Z + reflect  (0 extra params)
  E   : LeakyReLU(0.1) on (Z-theta) + reflection      (0 extra params)

To reproduce:
    python -u scripts/train_step128_concat_relu.py --device mps
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
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75,
                    help="Training epochs (default 75; use 20 for Tier-0 scout)")
parser.add_argument("--configs", default="", help="Comma-separated config keys to run (e.g. A,E). Empty = run all.")
parser.add_argument("--full_data", action="store_true", help="Use 100 percent data (Tier-2)")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 64; K_ITER = 12
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
STEP69_REF = 0.8336


# ---------------------------------------------------------------------------
#  Model: SGNNET_AH_ConcatReLU — activation-function variants
# ---------------------------------------------------------------------------

class SGNNET_AH_ConcatReLU(nn.Module):
    """SGNNET_AntiHebbian with alternative activation functions in the routing loop.

    Modes:
      weighted_neg:        Z_fwd = relu(Z-theta) + beta*relu(theta-Z)
      weighted_neg_norefl: same as weighted_neg but reflection accumulator disabled
      concat_relu:         Z_pos||Z_neg → W_proj(2D, D) → Z_fwd
      swiglu:              Z_fwd = silu(Z-theta) * Z  (smooth gating)
      leaky_relu:          Z_fwd = leaky_relu(Z-theta, 0.1)
    """

    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float,
                 mode: str, D: int, beta: float = 0.3):
        super().__init__()
        self.m     = resonant
        self.alpha = alpha_ahebb
        self.mode  = mode
        self.beta  = beta

        # Config C: learned projection from 2D → D (weight-tied across steps)
        if mode == "concat_relu":
            self.W_proj = nn.Linear(2 * D, D, bias=False)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _activation(self, Z: torch.Tensor, theta_pos: torch.Tensor) -> torch.Tensor:
        """Apply the configured activation to (Z, theta_pos) → Z_fwd."""
        if self.mode == "weighted_neg" or self.mode == "weighted_neg_norefl":
            return F.relu(Z - theta_pos) + self.beta * F.relu(theta_pos - Z)
        elif self.mode == "concat_relu":
            Z_pos = F.relu(Z - theta_pos)        # [B, N, D]
            Z_neg = F.relu(theta_pos - Z)         # [B, N, D]
            Z_cat = torch.cat([Z_pos, Z_neg], dim=-1)  # [B, N, 2D]
            return self.W_proj(Z_cat)             # [B, N, D]
        elif self.mode == "swiglu":
            return F.silu(Z - theta_pos) * Z      # smooth gating
        elif self.mode == "leaky_relu":
            return F.leaky_relu(Z - theta_pos, negative_slope=0.1)
        else:
            raise ValueError(f"Unknown activation mode: {self.mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                          # [B, N, D]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden

        # AH wpos suppression (static, precomputed)
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)       # [N, K_hh]
        supp_w  = (1.0 - self.alpha * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)                     # [1, N, K_hh, 1]

        # Reflection accumulator (disabled for weighted_neg_norefl)
        use_reflect = (self.mode != "weighted_neg_norefl")
        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z_fwd  = self._activation(Z, theta_pos)
            Z_nb   = Z_fwd[:, conn_hh, :]                         # [B, N, K_hh, D]
            Z_struct = (Z_nb * supp_w).sum(dim=2)                  # [B, N, D]

            if use_reflect:
                Z_remainder = Z_fwd - Z
                Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
                Z_new = Z_struct + Z_reflected
            else:
                Z_new = Z_struct

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ---------------------------------------------------------------------------
#  Config definitions
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    mode: str
    beta: float = 0.3
    use_custom: bool = True   # False → use standard SGNNET_AntiHebbian (Ref)


CONFIGS = [
    Config("Ref", "Ref  standard AH (relu + reflection)",    "weighted_neg", use_custom=False),
    Config("A",   "A    weighted_neg beta=0.3 + reflect",    "weighted_neg",        beta=0.3),
    Config("B",   "B    weighted_neg beta=0.3, NO reflect",  "weighted_neg_norefl", beta=0.3),
    Config("C",   "C    ConcatReLU + W_proj(2D→D) + reflect","concat_relu"),
    Config("D",   "D    SwiGLU silu(Z-theta)*Z + reflect",   "swiglu"),
    Config("E",   "E    LeakyReLU(0.1) + reflect",           "leaky_relu"),
]


# ---------------------------------------------------------------------------
#  Data loading (cached, 50% subset)
# ---------------------------------------------------------------------------

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
        if getattr(args, "full_data", False):
            _loaders = (tr_full, va)
        else:
            n = len(tr_full.dataset)
            idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
            subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
            tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
            _loaders = (tr, va)
    return _loaders


# ---------------------------------------------------------------------------
#  Model factory
# ---------------------------------------------------------------------------

def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    tk = topology_kwargs(N)
    tk.pop("K_in", None)
    tk.pop("K_iter", None)
    base = SGNNET_SmallWorld(
        N_in=N_IN, N_hidden=N, N_out=N_OUT,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=50, K_iter=K_ITER, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    if not cfg.use_custom:
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_AH_ConcatReLU(
        resonant, ALPHA_AHEBB, cfg.mode, D, beta=cfg.beta)


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

def run(cfg, model, meta):
    print(f"\n{'='*70}\n{cfg.label}\n{'='*70}")
    tr, va = get_loaders()
    tk = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0 = time.time(); history = trainer.train(n_epochs=EPOCHS); elapsed = time.time() - t0
    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best = max(top1_hist); best_ep = int(np.argmax(top1_hist)) + 1
    frac = best_ep / len(history)
    result = {
        "label": cfg.label, "mode": cfg.mode, "beta": cfg.beta,
        "top1_best": best, "top1_last": history[-1].get("val_top1", 0.0),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1), "best_epoch_frac": round(frac, 3),
        "step69_ref": STEP69_REF, "delta_vs_ref": round(best - STEP69_REF, 4),
        "params": count_params(model), "top1_history": top1_hist,
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1={best:.4f}  ep={best_ep}/{len(history)}  vs_ref={best-STEP69_REF:+.4f}  t={elapsed:.0f}s  params={count_params(model)}")
    return result


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  N={N}  K_iter={K_ITER}  Data: 50%")
    print(f"Step 128: ConcatReLU / activation function ablation")
    print(f"Baseline: step69 Ref = {STEP69_REF:.4f}\n")
    for c in CONFIGS:
        print(f"  {c.key:4s}  {c.label}")
    print()
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")
    results = {}
    out_path = ROOT / "results" / "train_step128_concat_relu.json"
    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS) if not cfg_filter or cfg.key in cfg_filter]
    for i, cfg in active_configs:
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        meta = {"N": N, "D": D, "K_iter": K_ITER, "mode": cfg.mode,
                "beta": cfg.beta, "alpha_ahebb": ALPHA_AHEBB,
                "alpha_reflect": ALPHA_REFLECT, "data_frac": 0.5}
        results[cfg.key] = run(cfg, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")
    print(f"\n{'='*70}\nSTEP 128 COMPLETE\n")
    print(f"  {'Key':4s}  {'mode':>20s}  {'params':>8s}  {'top1':>8s}  {'vs_ref':>8s}")
    for c in CONFIGS:
        if c.key in results:
            r = results[c.key]
            print(f"  {c.key:4s}  {c.mode:>20s}  {r['params']:>8d}  {r['top1_best']:.4f}  {r['delta_vs_ref']:+.4f}")
    w = max(results, key=lambda k: results[k]["top1_best"])
    print(f"\n  Winner: {w}")
