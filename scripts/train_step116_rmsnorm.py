"""Step 116: RMSNorm / normalisation ablation.

MOTIVATION
==========
SGNNET currently applies F.normalize (L2 unit-sphere projection) at every K_iter
step — the most aggressive possible normalisation. This may cause oversmoothing:
all neuron activations are forcibly projected onto S^{D-1}, collapsing any
magnitude signal. The smearing effect worsens with K_iter depth.

Alternatives range from softer magnitude preservation (RMSNorm) to position
variants (pre-route normalise) to fully removing normalisation (clamp only).

CONFIGS (N=1024, D=16, K_hh=8, K_iter=8, K_in=25, AH=1.0, 50%/75ep)
======================================================================
  Ref : F.normalize L2 (current default)
  A   : RMSNorm — Z / sqrt(mean(Z², dim=-1) + eps). Preserves scale, no sphere projection.
  B   : Pre-route normalize — normalize Z BEFORE routing, then clamp only after.
  C   : LayerNorm — (Z - mean) / std with learned affine (standard transformer norm).
  D   : No normalize — clamp(-10, 10) only. Baseline: is normalisation needed at all?
  E   : Normalize every 3rd step — apply F.normalize at steps 0, 3, 6 only.

To reproduce:
    python -u scripts/train_step116_rmsnorm.py --device mps
    python -u scripts/train_step116_rmsnorm.py --device mps --epochs 20  # Tier-0 scout
    python -u scripts/train_step116_rmsnorm.py --device mps --configs A,D
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
N = 1024; N_IN = 25088; N_OUT = 10; D = 16; K_ITER = 8; K_IN = 25; K_HH = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
RMS_EPS = 1e-6


# ---------------------------------------------------------------------------
# Custom model: SGNNET_AH with configurable normalisation in routing loop
# ---------------------------------------------------------------------------

class SGNNET_AH_NormAblation(nn.Module):
    """AntiHebbian routing with configurable normalisation strategy.

    norm_mode controls what happens to Z after each routing step:
      'l2'        : F.normalize (current default, unit sphere projection)
      'rms'       : RMSNorm — Z / sqrt(mean(Z², -1, keepdim=True) + eps)
      'pre_route' : normalize BEFORE routing; only clamp after
      'layernorm' : learnable LayerNorm per D dimensions
      'none'      : clamp(-10, 10) only, no normalisation
      'every3'    : F.normalize at routing steps 0, 3, 6 only; clamp otherwise
    """

    def __init__(self, resonant, alpha_ahebb: float, norm_mode: str = "l2"):
        super().__init__()
        self.m           = resonant
        self.alpha_ahebb = alpha_ahebb
        self.norm_mode   = norm_mode

        if norm_mode == "layernorm":
            # Shared LayerNorm over D dims applied to each neuron independently
            self.layer_norm = nn.LayerNorm(resonant.base.D)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _apply_norm(self, Z: torch.Tensor, step: int) -> torch.Tensor:
        """Apply normalisation according to self.norm_mode."""
        if self.norm_mode == "l2":
            return F.normalize(Z.clamp(-10, 10), dim=-1)

        elif self.norm_mode == "rms":
            Z_c = Z.clamp(-10, 10)
            rms = Z_c.pow(2).mean(dim=-1, keepdim=True).add(RMS_EPS).sqrt()
            return Z_c / rms

        elif self.norm_mode == "layernorm":
            return self.layer_norm(Z.clamp(-10, 10))

        elif self.norm_mode == "none":
            return Z.clamp(-10, 10)

        elif self.norm_mode == "every3":
            Z_c = Z.clamp(-10, 10)
            if step % 3 == 0:
                return F.normalize(Z_c, dim=-1)
            return Z_c

        else:
            # fallback: l2
            return F.normalize(Z.clamp(-10, 10), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base      = self.m.base
        Z         = base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = base.conn_hh
        N_h       = base.N_hidden

        # Pre-compute static AH suppression weights (wpos variant)
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                   ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)

        for t in range(base.K_iter):

            # Pre-route normalise variant: normalise Z before computing Z_fwd
            if self.norm_mode == "pre_route":
                Z_normed = F.normalize(Z, dim=-1)
                Z_fwd    = F.relu(Z_normed - theta_pos)
            else:
                Z_fwd = F.relu(Z - theta_pos)

            Z_nb     = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected

            if self.norm_mode == "pre_route":
                # After routing: only clamp, no normalisation
                Z = Z_new.clamp(-10, 10)
            else:
                Z = self._apply_norm(Z_new, step=t)

        return base._readout(Z)


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    norm_mode: str

CONFIGS = [
    Config("Ref", "Ref  F.normalize L2 (current default)",              "l2"),
    Config("A",   "A    RMSNorm (scale-preserving, no sphere proj)",    "rms"),
    Config("B",   "B    Pre-route normalize (normalize before routing)", "pre_route"),
    Config("C",   "C    LayerNorm with learned affine",                 "layernorm"),
    Config("D",   "D    No normalize — clamp(-10,10) only",             "none"),
    Config("E",   "E    Normalize every 3rd step (steps 0,3,6)",        "every3"),
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
    return SGNNET_AH_NormAblation(resonant, ALPHA_AHEBB, norm_mode=cfg.norm_mode)


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 116 — RMSNorm / Normalisation Ablation")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  K_in={K_IN}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    print("Configs:")
    for c in CONFIGS:
        print(f"  {c.key:4s}  norm={c.norm_mode:12s}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step116_rmsnorm.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  norm_mode={cfg.norm_mode}")
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
            "norm_mode": cfg.norm_mode,
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
    print(f"STEP 116 SUMMARY — Normalisation Ablation")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Key':4s}  {'norm_mode':12s}  {'params':>8}  {'top1':>7}  {'vs_Ref':>8}")
    print(f"{'─'*55}")
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"{key:4s}  {r['norm_mode']:12s}  {r['n_params']:>8,}  "
              f"{r['top1_best']:.4f}  {vs:>+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
