"""Step 120: Higher K_iter with Z-bias and Gradient Checkpointing.

MOTIVATION
==========
step71 showed K_iter=12 optimal at N=4096, but K_iter=16 gave +0.44pp. The cliff
at K_iter=24 was only measured at N=1024 on buggy arch — never retested at N=4096
with patched arch + Z-bias.

Z-bias (step106, +7.42pp at N=1024) adds per-step embeddings. Higher K_iter means
more step embeddings = more temporal specialization. The hypothesis: Z-bias prevents
the K_iter cliff by giving each step a unique identity, breaking the repetitive
dynamics that cause signal degradation at high iteration counts.

Gradient checkpointing enables K_iter=24+ without OOM by recomputing forward
activations during backward instead of storing them (trades ~30% extra compute
for ~K_iter× memory reduction in the routing loop).

CONFIGS (N=4096, D=64, K_hh=4, AH=1.0, 50%/75ep)
===================================================
  Ref : K_iter=12 + Z-bias (step115-A baseline, standard memory)
  A   : K_iter=16 + Z-bias (16 step embeddings, 1024 params)
  B   : K_iter=20 + Z-bias + gradient checkpointing
  C   : K_iter=24 + Z-bias + gradient checkpointing
  D   : K_iter=16 + Z-bias + gradient checkpointing (numerics comparison vs A)

To reproduce:
    python -u scripts/train_step120_high_kiter_zbias.py --device mps
    python -u scripts/train_step120_high_kiter_zbias.py --device mps --epochs 20  # scout
    python -u scripts/train_step120_high_kiter_zbias.py --device mps --configs A,D
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
from torch.utils.checkpoint import checkpoint as grad_checkpoint

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
N = 4096; N_IN = 25088; N_OUT = 10; D = 64; K_IN = 50
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
# step89-Ref at 50%/75ep for vs_ref column (N=4096 baseline)
STEP89_REF_50 = 0.9658


# ---------------------------------------------------------------------------
# Model: Z-bias with configurable K_iter and optional gradient checkpointing
# ---------------------------------------------------------------------------

class SGNNET_AH_ZBias_Checkpointed(nn.Module):
    """AH routing with per-step Z-bias (step106 mechanism) and optional
    gradient checkpointing for memory-efficient high K_iter.

    When use_checkpoint=True, each routing step's computation is wrapped in
    torch.utils.checkpoint.checkpoint(). This recomputes forward activations
    during backward instead of storing them, trading ~30% extra compute for
    ~K_iter× memory savings in the routing loop.
    """

    def __init__(self, resonant, alpha_ahebb: float,
                 K_iter: int, D: int, use_checkpoint: bool = False):
        super().__init__()
        self.m              = resonant
        self.alpha          = alpha_ahebb
        self.K_iter_n       = K_iter
        self.use_checkpoint = use_checkpoint
        self.step_emb       = nn.Parameter(torch.zeros(K_iter, D))  # init=zeros

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _routing_step(self, Z: torch.Tensor, Z_reflected: torch.Tensor,
                      step_bias: torch.Tensor, theta_pos: torch.Tensor,
                      supp_w: torch.Tensor, conn_hh: torch.Tensor,
                      alpha_reflect: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Single routing step — extracted so it can be wrapped in checkpoint()."""
        Z_cond = Z + step_bias.unsqueeze(0).unsqueeze(0)
        Z_fwd  = F.relu(Z_cond - theta_pos)
        Z_nb   = Z_fwd[:, conn_hh, :]
        Z_struct = (Z_nb * supp_w).sum(dim=2)

        Z_remainder = Z_fwd - Z_cond
        Z_reflected_new = alpha_reflect * Z_reflected + Z_remainder

        Z_new = Z_struct + Z_reflected_new
        Z_out = F.normalize(Z_new.clamp(-10, 10), dim=-1)
        return Z_out, Z_reflected_new

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
        alpha_reflect = self.m.alpha_reflect

        for t in range(self.K_iter_n):
            step_bias = self.step_emb[t]
            if self.use_checkpoint and self.training:
                # Gradient checkpointing: recompute forward during backward
                # use_reentrant=False is the recommended modern API
                Z, Z_reflected = grad_checkpoint(
                    self._routing_step,
                    Z, Z_reflected, step_bias, theta_pos, supp_w,
                    conn_hh, alpha_reflect,
                    use_reentrant=False,
                )
            else:
                Z, Z_reflected = self._routing_step(
                    Z, Z_reflected, step_bias, theta_pos, supp_w,
                    conn_hh, alpha_reflect,
                )

        return base._readout(Z)


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    K_iter: int
    use_checkpoint: bool


CONFIGS = [
    Config("Ref", "Ref  K_iter=12 + Z-bias (step115-A baseline)",
           K_iter=12, use_checkpoint=False),
    Config("A",   "A    K_iter=16 + Z-bias (no checkpointing)",
           K_iter=16, use_checkpoint=False),
    Config("B",   "B    K_iter=20 + Z-bias + grad checkpoint",
           K_iter=20, use_checkpoint=True),
    Config("C",   "C    K_iter=24 + Z-bias + grad checkpoint",
           K_iter=24, use_checkpoint=True),
    Config("D",   "D    K_iter=16 + Z-bias + grad checkpoint (numerics check vs A)",
           K_iter=16, use_checkpoint=True),
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
        K_in=K_IN, K_iter=cfg.K_iter, encoding_mode="fourier",
        **topo,
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AH_ZBias_Checkpointed(
        resonant, ALPHA_AHEBB, K_iter=cfg.K_iter, D=D,
        use_checkpoint=cfg.use_checkpoint,
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
    print(f"Step 120 — Higher K_iter with Z-bias + Gradient Checkpointing")
    print(f"N={N}  D={D}  K_hh=4  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"Baseline (step89 Ref 50%/75ep): ~{STEP89_REF_50:.4f}")
    print(f"{'='*70}\n")

    print("Configs:")
    for c in CONFIGS:
        print(f"  {c.key:4s}  K_iter={c.K_iter:2d}  ckpt={'Y' if c.use_checkpoint else 'N'}  "
              f"{c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step120_high_kiter_zbias.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)
        n_step_emb = cfg.K_iter * D

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  K_iter={cfg.K_iter}  "
              f"step_emb={n_step_emb} params  checkpoint={cfg.use_checkpoint}")
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
            "N": N, "D": D, "K_iter": cfg.K_iter,
            "use_checkpoint": cfg.use_checkpoint,
            "alpha_ahebb": ALPHA_AHEBB,
            "step_emb_params": n_step_emb,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "vs_step89_ref": round(vs_ref, 6),
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params,
            "label": cfg.label,
        }

        print(f"\n  top1_best={top1_best:.4f}  vs_step89_ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary table
    print(f"\n{'='*70}")
    print(f"STEP 120 SUMMARY — K_iter scaling with Z-bias")
    print(f"{'='*70}")
    print(f"{'Config':<6}  {'K_iter':>6}  {'ckpt':>4}  {'params':>8}  "
          f"{'step_emb':>8}  {'top1':>7}  {'vs_ref':>8}  label")
    print(f"{'─'*80}")
    for key, r in results.items():
        print(f"{key:<6}  {r['K_iter']:>6}  {'Y' if r['use_checkpoint'] else 'N':>4}  "
              f"{r['n_params']:>8,}  {r['step_emb_params']:>8}  "
              f"{r['top1_best']:.4f}  {r['vs_step89_ref']:>+.4f}  {r['label']}")

    # A vs D comparison (checkpoint numerics check)
    if "A" in results and "D" in results:
        delta = results["A"]["top1_best"] - results["D"]["top1_best"]
        print(f"\nCheckpoint numerics check: A-D = {delta:+.4f} "
              f"(expect ~0, checkpoint should not change converged accuracy)")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
