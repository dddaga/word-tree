"""Step 91: GCNII-style initial residual in K_iter routing loop.

MOTIVATION
==========
GCNII (ICML 2020) prevents over-smoothing at 64 GNN layers using initial
residual: h_t = (1-a)*message_pass(h_{t-1}) + a*h_0.

SGNNET's K_iter=8-12 routing steps are deep message passing. All 9 dynamic
routing experiments died from gate-death (multiplicative signal collapse).
Initial residual is gate-free — just a fixed lerp with h_0 (seed output).

Key properties:
  - Zero new parameters (a is a fixed hyperparameter)
  - h_0 injected at EVERY step (not just skip from t-1 to t)
  - AH still operates on the message_pass component
  - Gradient flows directly from any step to h_0 (no vanishing)

This is orthogonal to AH — AH controls WHICH neighbors contribute;
initial residual controls HOW MUCH of the original signal persists.

CONFIGS (N=1024, D=64, K_iter=8, AH=1.0, turing=0.0, 50%/75ep)
=================================================================
  Ref : no residual (standard AH routing — step69 Ref baseline)
  A   : a=0.05  (5% h_0 injection — very gentle)
  B   : a=0.1   (10% h_0 injection)
  C   : a=0.2   (20% h_0 injection)
  D   : a=0.3   (30% h_0 injection — strong anchor to seed)
  E   : a=0.1, K_iter=16  (test: does residual enable deeper routing?)

If E > B at K_iter=16: residual unlocks depth that was previously blocked.
If B > Ref: adopt a=0.1 for all future experiments.

To reproduce:
    python -u scripts/train_step91_gcnii_residual.py --device mps
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
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 75; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 64
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
STEP69_REF = 0.8336


class SGNNET_AH_Residual(nn.Module):
    """SGNNET_AntiHebbian with GCNII initial residual injection.

    At each K_iter step:
      h_msg = AH_message_pass(h_{t-1})    (existing AH routing)
      h_t   = (1-a) * h_msg + a * h_0     (initial residual)
      h_t   = normalize(h_t)
    """

    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float,
                 residual_alpha: float):
        super().__init__()
        self.m     = resonant
        self.alpha = alpha_ahebb
        self.a     = residual_alpha

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        h_0       = Z.clone()                                     # initial residual anchor
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden

        # AH wpos suppression weights (pre-computed, static)
        W_n    = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_nb     = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected
            # GCNII initial residual injection
            if self.a > 0:
                Z_new = (1.0 - self.a) * Z_new + self.a * h_0
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


@dataclass
class Config:
    key: str; label: str; residual_alpha: float; K_iter: int


CONFIGS = [
    Config("Ref", "Ref  no residual, K_iter=8  (step69 baseline)", 0.0, 8),
    Config("A",   "A    a=0.05  K_iter=8",  0.05, 8),
    Config("B",   "B    a=0.1   K_iter=8",  0.10, 8),
    Config("C",   "C    a=0.2   K_iter=8",  0.20, 8),
    Config("D",   "D    a=0.3   K_iter=8",  0.30, 8),
    Config("E",   "E    a=0.1   K_iter=16 (depth test)", 0.10, 16),
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
        K_in=tk["K_in"], K_iter=cfg.K_iter, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    if cfg.residual_alpha == 0.0:
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_AH_Residual(resonant, ALPHA_AHEBB, cfg.residual_alpha)


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
        "label": cfg.label, "residual_alpha": cfg.residual_alpha,
        "K_iter": cfg.K_iter, "top1_best": best,
        "top1_last": history[-1].get("val_top1", 0.0),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "best_epoch_frac": round(frac, 3),
        "step69_ref": STEP69_REF, "delta_vs_ref": round(best - STEP69_REF, 4),
        "params": count_params(model),
        "top1_history": top1_hist,
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1={best:.4f}  ep={best_ep}/{len(history)}  vs_ref={best-STEP69_REF:+.4f}  t={elapsed:.0f}s")
    return result


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  N={N}  Data: 50%")
    print(f"Step 91: GCNII initial residual + AH routing")
    print(f"Baseline: step69 Ref = {STEP69_REF:.4f}\n")
    for c in CONFIGS:
        print(f"  {c.key:4s}  a={c.residual_alpha:.2f}  K_iter={c.K_iter}  {c.label}")
    print()
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")
    results = {}
    out_path = ROOT / "results" / "train_step91_gcnii_residual.json"
    for i, cfg in enumerate(CONFIGS):
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        meta = {"N": N, "D": D, "K_iter": cfg.K_iter, "residual_alpha": cfg.residual_alpha,
                "alpha_ahebb": ALPHA_AHEBB, "data_frac": 0.5}
        results[cfg.key] = run(cfg, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")
    print(f"\n{'='*70}\nSTEP 91 COMPLETE\n")
    print(f"  {'Key':4s}  {'a':>5s}  {'K':>3s}  {'top1':>8s}  {'vs_ref':>8s}")
    for c in CONFIGS:
        if c.key in results:
            r = results[c.key]
            print(f"  {c.key:4s}  {c.residual_alpha:>5.2f}  {c.K_iter:>3d}  {r['top1_best']:.4f}  {r['delta_vs_ref']:+.4f}")
    w = max(results, key=lambda k: results[k]["top1_best"])
    print(f"\n  Winner: {w}  (a={CONFIGS[[c.key for c in CONFIGS].index(w)].residual_alpha})")
