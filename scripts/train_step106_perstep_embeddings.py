"""Step 106: Per-step embeddings for K_iter (PLE-inspired).

MOTIVATION
==========
Gemma 4 uses Per-Layer Embeddings (PLE) — a small conditioning vector per
transformer layer. SGNNET's K_iter=12 steps are analogous to 12 identical
transformer layers. Per-step conditioning lets each step specialize:
  - Early steps: coarse routing (large Z-bias shifts)
  - Late steps: fine-grained (small shifts)

Gate-death safe: embeddings are ADDITIVE (Z + emb[t]) — no multiplicative
compounding. Signal magnitude preserved. AH operates on shifted Z.

MODES
=====
  Z-bias:       Z_t = Z_t + emb[t]                    (12×D params)
  Edge-scale:   weights *= (1 + scale[t])              (12 scalars)
  AH-modulate:  alpha_t = alpha_base + delta[t]        (12 scalars)

CONFIGS (N=1024, D=64, K_hh=4, K_iter=12, AH=1.0, turing=0.0, 50%/75ep)
=========================================================================
  Ref : no per-step conditioning (standard AH K_iter=12)
  A   : Z-bias, init=zeros                (768 params, core mechanism)
  B   : Z-bias, init=small random 0.01    (768 params, does init matter?)
  C   : Edge-scale, init=ones             (12 params, minimal)
  D   : AH-modulation, init=zeros         (12 params, step-specialized diversity)
  E   : Z-bias + GCNII residual a=0.1     (768 params, compound with step91)

To reproduce:
    python -u scripts/train_step106_perstep_embeddings.py --device mps
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
from src.training.experiment_config   import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75,
                    help="Training epochs (default 75; use 20 for Tier-0 scout)")
parser.add_argument("--configs", default="", help="Comma-separated config keys to run (e.g. A,E). Empty = run all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 64; K_ITER = 12
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
STEP69_REF = 0.8336


class SGNNET_AH_PerStep(nn.Module):
    """SGNNET_AntiHebbian with per-step conditioning in K_iter loop.

    Modes:
      zbias:  Z_t += emb[t]  before routing (additive, D-dim per step)
      escale: edge weights *= (1 + scale[t])  (scalar per step)
      ahmod:  alpha_t = alpha_base + delta[t]  (scalar per step)
    """

    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float,
                 mode: str, K_iter: int, D: int,
                 init_scale: float = 0.0, residual_alpha: float = 0.0):
        super().__init__()
        self.m     = resonant
        self.alpha = alpha_ahebb
        self.mode  = mode
        self.K_iter_override = K_iter
        self.residual_alpha = residual_alpha

        if mode == "zbias":
            init = torch.zeros(K_iter, D) if init_scale == 0.0 else torch.randn(K_iter, D) * init_scale
            self.step_emb = nn.Parameter(init)
        elif mode == "escale":
            self.step_scale = nn.Parameter(torch.zeros(K_iter))  # 1+scale → init at 1.0
        elif mode == "ahmod":
            self.step_delta = nn.Parameter(torch.zeros(K_iter))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                          # [B, N, D]
        h_0       = Z.clone() if self.residual_alpha > 0 else None
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden

        # AH wpos suppression — base alpha (may be modulated per step)
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)       # [N, K_hh]

        Z_reflected = torch.zeros_like(Z)

        for t in range(self.K_iter_override):
            # Per-step Z-bias injection
            if self.mode == "zbias":
                Z_cond = Z + self.step_emb[t].unsqueeze(0).unsqueeze(0)  # [B, N, D]
            else:
                Z_cond = Z

            Z_fwd  = F.relu(Z_cond - theta_pos)
            Z_nb   = Z_fwd[:, conn_hh, :]                         # [B, N, K, D]

            # Per-step AH modulation
            if self.mode == "ahmod":
                alpha_t = self.alpha + self.step_delta[t]
            else:
                alpha_t = self.alpha

            supp_w = (1.0 - alpha_t * pos_sim.clamp(min=0)
                     ).unsqueeze(0).unsqueeze(-1)                  # [1, N, K, 1]

            Z_struct = (Z_nb * supp_w).sum(dim=2)                  # [B, N, D]

            # Per-step edge scaling
            if self.mode == "escale":
                Z_struct = Z_struct * (1.0 + self.step_scale[t])

            Z_remainder = Z_fwd - Z_cond
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected

            # Optional GCNII residual
            if self.residual_alpha > 0 and h_0 is not None:
                Z_new = (1.0 - self.residual_alpha) * Z_new + self.residual_alpha * h_0

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


@dataclass
class Config:
    key: str; label: str; mode: str; init_scale: float
    residual_alpha: float = 0.0
    use_perstep: bool = True


CONFIGS = [
    Config("Ref", "Ref  no per-step (K_iter=12 baseline)", "zbias", 0.0, use_perstep=False),
    Config("A",   "A    Z-bias init=zeros",       "zbias",  0.0),
    Config("B",   "B    Z-bias init=random(0.01)", "zbias",  0.01),
    Config("C",   "C    Edge-scale init=ones",     "escale", 0.0),
    Config("D",   "D    AH-modulation init=zeros", "ahmod",  0.0),
    Config("E",   "E    Z-bias + GCNII a=0.1",    "zbias",  0.0, residual_alpha=0.1),
]

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=N_IN, N_hidden=N, N_out=N_OUT,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_ITER, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    if not cfg.use_perstep:
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_AH_PerStep(
        resonant, ALPHA_AHEBB, cfg.mode, K_ITER, D,
        init_scale=cfg.init_scale, residual_alpha=cfg.residual_alpha)


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


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
        "label": cfg.label, "mode": cfg.mode, "init_scale": cfg.init_scale,
        "residual_alpha": cfg.residual_alpha,
        "top1_best": best, "top1_last": history[-1].get("val_top1", 0.0),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1), "best_epoch_frac": round(frac, 3),
        "step69_ref": STEP69_REF, "delta_vs_ref": round(best - STEP69_REF, 4),
        "params": count_params(model), "top1_history": top1_hist,
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1={best:.4f}  ep={best_ep}/{len(history)}  vs_ref={best-STEP69_REF:+.4f}  t={elapsed:.0f}s")
    return result


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  N={N}  K_iter={K_ITER}  Data: 50%")
    print(f"Step 106: Per-step embeddings for K_iter (PLE-inspired)")
    print(f"Baseline: step69 Ref = {STEP69_REF:.4f}\n")
    for c in CONFIGS:
        print(f"  {c.key:4s}  mode={c.mode:6s}  init={c.init_scale:.3f}  res_a={c.residual_alpha:.2f}  {c.label}")
    print()
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")
    results = {}
    out_path = ROOT / "results" / "train_step106_perstep_embeddings.json"
    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS) if not cfg_filter or cfg.key in cfg_filter]
    for i, cfg in active_configs:
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        meta = {"N": N, "D": D, "K_iter": K_ITER, "mode": cfg.mode,
                "init_scale": cfg.init_scale, "residual_alpha": cfg.residual_alpha,
                "alpha_ahebb": ALPHA_AHEBB, "data_frac": 0.5}
        results[cfg.key] = run(cfg, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")
    print(f"\n{'='*70}\nSTEP 106 COMPLETE\n")
    print(f"  {'Key':4s}  {'mode':>6s}  {'init':>6s}  {'top1':>8s}  {'vs_ref':>8s}")
    for c in CONFIGS:
        if c.key in results:
            r = results[c.key]
            print(f"  {c.key:4s}  {c.mode:>6s}  {c.init_scale:>6.3f}  {r['top1_best']:.4f}  {r['delta_vs_ref']:+.4f}")
    w = max(results, key=lambda k: results[k]["top1_best"])
    print(f"\n  Winner: {w}")
