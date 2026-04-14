"""Step 156: LayerNorm ablation at N=4096 D=64 — validate step116-C winner at scale.

MOTIVATION
==========
step116-C showed LayerNorm with learned affine beats L2 sphere norm by +2.24pp
at N=1024 D=16 (Tier-1 confirmed). This is a significant win but untested at
N=4096 D=64. If it holds, LayerNorm replaces L2 as the default normalisation.

CONFIGS (N=4096, D=64, K_hh=4, K_iter=12, AH α=1.05, data_frac=0.5, 20ep scout)
================================================================================
  Ref : norm_mode="l2"             — F.normalize per neuron (current default)
  A   : norm_mode="layernorm"      — nn.LayerNorm(D) learned affine, applied after routing
  B   : norm_mode="layernorm_pre"  — LayerNorm BEFORE routing, F.normalize after
  C   : norm_mode="rms"            — RMSNorm (no mean centering, scale only)

Ref at N=4096 Tier-1: ~96.48% (75ep 50% data); step89 project best 97.86% (150ep).

To reproduce:
    python -u scripts/train_step156_layernorm_n4096.py --device mps
    python -u scripts/train_step156_layernorm_n4096.py --device mps --epochs 20
    python -u scripts/train_step156_layernorm_n4096.py --device cpu --configs A,Ref
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
parser.add_argument("--epochs", type=int, default=20,
                    help="Training epochs (default 20 for scout)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys to run (e.g. A,Ref). Empty = run all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 4096; N_IN = 25088; N_OUT = 10; D = 64; K_ITER = 12; K_IN = 50; K_HH = 4
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.05
RMS_EPS = 1e-6


# ---------------------------------------------------------------------------
# Custom model: subclass SGNNET_AntiHebbian with configurable norm
# ---------------------------------------------------------------------------

class SGNNET_AH_NormAblation_N4096(SGNNET_AntiHebbian):
    """AntiHebbian routing with configurable normalisation — inherits from
    SGNNET_AntiHebbian (MPS-proven) and overrides only the norm step.

    norm_mode:
      'l2'           : F.normalize (current default, unit sphere projection)
      'layernorm'    : nn.LayerNorm(D) with learned affine, applied after routing
      'layernorm_pre': LayerNorm BEFORE routing, F.normalize after
      'rms'          : RMSNorm — Z / sqrt(mean(Z², -1) + eps), no mean centering
    """

    def __init__(self, resonant, alpha_ahebb: float, norm_mode: str = "l2"):
        super().__init__(resonant, alpha_ahebb=alpha_ahebb, variant="wpos")
        self.norm_mode = norm_mode

        if norm_mode in ("layernorm", "layernorm_pre"):
            # Shared LayerNorm over D dims, applied to each neuron independently
            self.layer_norm = nn.LayerNorm(resonant.base.D)

    def _apply_norm(self, Z: torch.Tensor) -> torch.Tensor:
        if self.norm_mode == "l2":
            return F.normalize(Z.clamp(-10, 10), dim=-1)
        elif self.norm_mode == "layernorm":
            return self.layer_norm(Z.clamp(-10, 10))
        elif self.norm_mode == "rms":
            Z_c = Z.clamp(-10, 10)
            rms = Z_c.pow(2).mean(dim=-1, keepdim=True).add(RMS_EPS).sqrt()
            return Z_c / rms
        else:
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

        for _ in range(base.K_iter):

            if self.norm_mode == "layernorm_pre":
                Z_normed = self.layer_norm(Z)
                Z_fwd    = F.relu(Z_normed - theta_pos)
            else:
                Z_fwd = F.relu(Z - theta_pos)

            Z_nb     = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected

            if self.norm_mode == "layernorm_pre":
                Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)
            else:
                Z = self._apply_norm(Z_new)

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
    Config("Ref", "Ref  L2 F.normalize (current default)",                  "l2"),
    Config("A",   "A    LayerNorm learned affine (step116-C winner N=1024)", "layernorm"),
    Config("B",   "B    LayerNorm pre-route (LayerNorm before, F.norm after)", "layernorm_pre"),
    Config("C",   "C    RMSNorm (scale-preserving, no mean centering)",     "rms"),
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
    topo = topology_kwargs(N)
    topo.pop("K_in", None); topo.pop("K_iter", None); topo.pop("norm_mode", None)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, encoding_mode="fourier",
        norm_mode="l2",   # base always l2; normalisation overridden in wrapper
        **topo,
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AH_NormAblation_N4096(resonant, ALPHA_AHEBB, norm_mode=cfg.norm_mode)


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 156 — LayerNorm at N=4096 D=64 (validate step116-C)")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    print("Configs:")
    for c in CONFIGS:
        print(f"  {c.key:4s}  norm={c.norm_mode:15s}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step156_layernorm_n4096.json"

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
        def _log(m):
            # torch.mps.synchronize() drains pending MPS ops between eval and next
            # train_epoch — without this, MPS hangs after evaluate() on custom models.
            if str(DEVICE) == "mps":
                torch.mps.synchronize()
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
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
    print(f"STEP 156 SUMMARY — LayerNorm validation at N=4096")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Key':4s}  {'norm_mode':15s}  {'params':>8}  {'top1':>7}  {'vs_Ref':>8}")
    print(f"{'─'*55}")
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"{key:4s}  {r['norm_mode']:15s}  {r['n_params']:>8,}  "
              f"{r['top1_best']:.4f}  {vs:>+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")
    print(f"\nDecision: if A (layernorm) shows ≥0pp vs Ref → advance to Tier-1 (75ep).")
    print(f"If +2pp+ → potential new default (replaces L2 sphere norm).")


if __name__ == "__main__":
    main()
