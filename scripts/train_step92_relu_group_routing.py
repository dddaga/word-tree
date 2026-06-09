"""Step 92: ReLU group routing — fix step83's softmax collapse.

MOTIVATION
==========
step83 failed because softmax(S_g) over group centroids collapsed early in
training: all groups converged to uniform routing weights. Root causes:
  1. S_g = mean(Z) too coarse → all groups look similar
  2. softmax normalization forces Σw=1 → kills sparse routing
  3. Temporal mismatch (AH epoch-level vs router batch-level)

ReMoE (ICLR 2025) showed ReLU routing replaces softmax with:
  - ReLU(W_route @ h): naturally sparse (zeros for inactive routes)
  - Gradient = 1 for active routes (no exponential decay under iteration)
  - L1 regularization for load balancing (not aux losses)

This directly addresses gate-death: ReLU(x) for x>0 has constant gradient,
and for x≤0 it's cleanly off. No multiplicative compounding.

DESIGN
======
Groups of neurons (n_groups=8) act as "experts". Each neuron routes its
message through groups via ReLU-gated weights:

  S_g = mean(Z[neurons_in_group_g])    (group centroid, as in step83)
  r_g = ReLU(W_route @ S_g)            (routing weight per group, sparse)
  Z_inter[h] = sum_g(r_g * S_g)        (inter-group message)
  Z[h] = Z_struct[h] + beta * Z_inter[h]

W_route: [n_groups, D] — one routing vector per group (80 params at D=64, n_g=8).
L1 penalty on r_g encourages sparsity (not all groups active for every neuron).

CONFIGS (N=1024, D=64, K_iter=8, AH=1.0, n_groups=8, 50%/75ep)
=================================================================
  Ref : no inter-group routing (step82 Ref with spatial topology)
  A   : ReLU routing, beta=0.3, L1=0.01, every step
  B   : ReLU routing, beta=0.3, L1=0.01, final 3 steps only
  C   : ReLU routing, beta=0.1, L1=0.01, every step (conservative)
  D   : ReLU routing, beta=0.3, L1=0.001, every step (less sparsity)

ABLATION AXIS
=============
  A vs Ref  : does ReLU group routing help at all?
  B vs A    : does routing only in late steps avoid early-training collapse?
  C vs A    : beta sensitivity (how much inter-group signal?)
  D vs A    : L1 sensitivity (how sparse should routing be?)

To reproduce:
    python -u scripts/train_step92_relu_group_routing.py --device mps
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
N_GROUPS = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
STEP69_REF = 0.8336


class SGNNET_AH_ReLUGroupRouting(nn.Module):
    """AH routing + ReLU inter-group routing at each K_iter step."""

    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float,
                 n_groups: int, beta: float, l1_lambda: float,
                 route_from_step: int = 0):
        super().__init__()
        self.m     = resonant
        self.alpha = alpha_ahebb
        self.n_groups = n_groups
        self.beta  = beta
        self.l1_lambda = l1_lambda
        self.route_from_step = route_from_step

        # ReLU routing: one vector per group
        self.W_route = nn.Parameter(torch.randn(n_groups, D) * 0.01)

        # Group assignment (static, round-robin)
        N_h = resonant.base.N_hidden
        self.register_buffer("group_ids",
            torch.arange(N_h) % n_groups)                        # [N_h]
        # Precompute group membership masks [n_groups, N_h]
        masks = torch.zeros(n_groups, N_h)
        for g in range(n_groups):
            masks[g, self.group_ids == g] = 1.0
        self.register_buffer("group_masks", masks)
        # Count per group for averaging
        counts = masks.sum(dim=1, keepdim=True).clamp(min=1)     # [n_groups, 1]
        self.register_buffer("group_counts", counts)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase
    @property
    def routing_l1(self):
        """L1 penalty on routing weights — call after forward."""
        return self._routing_l1

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                          # [B, N, D]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden

        # AH wpos suppression (static)
        W_n    = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)
        total_l1 = torch.tensor(0.0, device=x.device)

        for step_idx in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_nb     = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)                 # [B, N, D]

            # ReLU inter-group routing (if past route_from_step)
            if step_idx >= self.route_from_step and self.beta > 0:
                # Group centroids: [B, n_groups, D]
                # Z: [B, N, D], group_masks: [n_groups, N]
                S_g = torch.einsum("bnd,gn->bgd", Z, self.group_masks)
                S_g = S_g / self.group_counts.unsqueeze(0)        # [B, G, D]

                # ReLU routing weights: [B, G]
                r_g = F.relu(torch.einsum("bgd,gd->bg", S_g, self.W_route))

                # L1 penalty
                total_l1 = total_l1 + r_g.mean()

                # Inter-group message: [B, G, D] weighted by r_g
                Z_inter_g = r_g.unsqueeze(-1) * S_g               # [B, G, D]
                # Broadcast to neurons: each neuron gets its group's message
                # group_ids: [N], Z_inter_g: [B, G, D] → [B, N, D]
                Z_inter = Z_inter_g[:, self.group_ids, :]         # [B, N, D]

                Z_struct = Z_struct + self.beta * Z_inter

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        self._routing_l1 = self.l1_lambda * total_l1
        return self.m.base._readout(Z)


@dataclass
class Config:
    key: str; label: str; beta: float; l1: float; route_from: int
    use_routing: bool = True


CONFIGS = [
    Config("Ref", "Ref  no inter-group routing (baseline)", 0.0, 0.0, 0, False),
    Config("A",   "A    ReLU β=0.3  L1=0.01  every step",  0.3, 0.01, 0),
    Config("B",   "B    ReLU β=0.3  L1=0.01  last 3 steps", 0.3, 0.01, 5),
    Config("C",   "C    ReLU β=0.1  L1=0.01  every step",  0.1, 0.01, 0),
    Config("D",   "D    ReLU β=0.3  L1=0.001 every step",  0.3, 0.001, 0),
]

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=10)
        _loaders = (tr, va)
    return _loaders


def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=N_IN, N_hidden=N, N_out=N_OUT,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=8, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    if not cfg.use_routing:
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_AH_ReLUGroupRouting(
        resonant, ALPHA_AHEBB, N_GROUPS, cfg.beta, cfg.l1, cfg.route_from)


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
        "label": cfg.label, "beta": cfg.beta, "l1_lambda": cfg.l1,
        "route_from_step": cfg.route_from,
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
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  N={N}  n_groups={N_GROUPS}  Data: 50%")
    print(f"Step 92: ReLU group routing (fixes step83 softmax collapse)")
    print(f"Baseline: step69 Ref = {STEP69_REF:.4f}\n")
    for c in CONFIGS:
        tag = f"β={c.beta} L1={c.l1} from_step={c.route_from}" if c.use_routing else "disabled"
        print(f"  {c.key:4s}  {tag}")
    print()
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")
    results = {}
    out_path = ROOT / "results" / "train_step92_relu_group_routing.json"
    for i, cfg in enumerate(CONFIGS):
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        meta = {"N": N, "D": D, "K_iter": 8, "n_groups": N_GROUPS,
                "beta": cfg.beta, "l1_lambda": cfg.l1,
                "alpha_ahebb": ALPHA_AHEBB, "data_frac": 0.5}
        results[cfg.key] = run(cfg, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")
    print(f"\n{'='*70}\nSTEP 92 COMPLETE\n")
    print(f"  {'Key':4s}  {'β':>5s}  {'L1':>6s}  {'from':>5s}  {'top1':>8s}  {'vs_ref':>8s}")
    for c in CONFIGS:
        if c.key in results:
            r = results[c.key]
            print(f"  {c.key:4s}  {c.beta:>5.2f}  {c.l1:>6.3f}  {c.route_from:>5d}  "
                  f"{r['top1_best']:.4f}  {r['delta_vs_ref']:+.4f}")
    w = max(results, key=lambda k: results[k]["top1_best"])
    print(f"\n  Winner: {w}")
