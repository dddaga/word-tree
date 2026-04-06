"""Step 108: Hierarchical polar routing (PolarQuant-inspired).

MOTIVATION
==========
PolarQuant decomposes D-dim vectors into hierarchical angles via recursive
polar transform. SGNNET's W_pos lives on S^{D-1} — unit hypersphere with
natural recursive polar decomposition into D-1 angles.

Key insight: polar angles form a TREE. Level 1 (θ_1) splits S^{D-1} into
2 hemispheres. Level 2: 4 quadrants. Level L: 2^L regions. This gives
coarse-to-fine topology from GEOMETRY, not learned parameters.

Zero new parameters. Topology-only change (like step82 group topology).
AH-compatible: polar hierarchy is a read-only structure for neighbor
selection. AH freely moves W_pos; hierarchy adapts passively.

CONFIGS (N=1024, D=64, K_hh=4, K_iter=12, AH=1.0, turing=0.0, 50%/75ep)
=========================================================================
  Ref : standard KNN (cosine similarity on W_pos)
  A   : 3-level hierarchy (8 regions), K_local=2, K_cross=2
  B   : 4-level (16 regions), K_local=2, K_cross=2
  C   : 5-level (32 regions), K_local=3, K_cross=1
  D   : 3-level, K_local=3, K_cross=1 (more local, less cross)
  E   : 3-level, rebuilt every 25 epochs (tracks AH movement)

To reproduce:
    python -u scripts/train_step108_polar_routing.py --device mps
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
N = 1024; N_IN = 25088; N_OUT = 10; D = 64; K_ITER = 12
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
STEP69_REF = 0.8336


def polar_region_ids(W_pos: torch.Tensor, n_levels: int) -> torch.Tensor:
    """Assign each neuron to a region based on hierarchical polar angles.

    W_pos: [N, D] on S^{D-1} (unit normalized).
    n_levels: number of hierarchy levels (2^n_levels regions).

    Returns: [N] integer region IDs in [0, 2^n_levels).

    Level 1: sign(W_pos[0]) → 2 hemispheres
    Level 2: sign(W_pos[1]) within each hemisphere → 4 quadrants
    Level L: sign(W_pos[L-1]) within each parent region → 2^L regions

    This is equivalent to taking the sign bits of the first n_levels
    coordinates, which IS the recursive polar decomposition at the
    coarsest quantization (1 bit per level).
    """
    N = W_pos.shape[0]
    region_ids = torch.zeros(N, dtype=torch.long, device=W_pos.device)
    for level in range(n_levels):
        # Bit at this level: 1 if W_pos[level] >= 0, else 0
        bit = (W_pos[:, level] >= 0).long()
        region_ids = region_ids * 2 + bit
    return region_ids


def build_polar_conn_hh(W_pos: torch.Tensor, N_h: int, n_levels: int,
                        K_local: int, K_cross: int) -> torch.Tensor:
    """Build conn_hh using hierarchical polar regions.

    For each neuron:
      - K_local neighbors from the same finest-level region (by cosine sim)
      - K_cross neighbors from the parent region (one level up) but
        DIFFERENT finest-level region

    Returns: [N_h, K_local + K_cross] index tensor
    """
    K_hh = K_local + K_cross
    W = F.normalize(W_pos[:N_h], dim=-1)
    region_ids = polar_region_ids(W, n_levels)                     # [N_h]
    parent_ids = region_ids // 2                                   # parent region (one level up)

    # Cosine similarity matrix (full, for N=1024 this is fine)
    sim = W @ W.T                                                  # [N_h, N_h]
    sim.fill_diagonal_(-10.0)                                      # exclude self

    conn_hh = torch.zeros(N_h, K_hh, dtype=torch.long, device=W_pos.device)

    for h in range(N_h):
        my_region = region_ids[h].item()
        my_parent = parent_ids[h].item()

        # Local: same region, top-K_local by similarity
        local_mask = (region_ids == my_region)
        local_mask[h] = False
        local_sim = sim[h].clone()
        local_sim[~local_mask] = -10.0
        n_local_avail = local_mask.sum().item()

        if n_local_avail >= K_local:
            _, local_idx = local_sim.topk(K_local)
        else:
            # Not enough in same region — fill from parent
            _, local_idx = local_sim.topk(max(n_local_avail, 1))
            # Pad with parent-region neighbors
            extra_needed = K_local - n_local_avail
            cross_mask = (parent_ids == my_parent) & (region_ids != my_region)
            cross_sim = sim[h].clone()
            cross_sim[~cross_mask] = -10.0
            if cross_mask.sum() > 0:
                _, extra_idx = cross_sim.topk(min(extra_needed, cross_mask.sum().item()))
                local_idx = torch.cat([local_idx[:n_local_avail], extra_idx])
            # If still not enough, pad with random
            while local_idx.shape[0] < K_local:
                rand_idx = torch.randint(0, N_h, (1,), device=W_pos.device)
                local_idx = torch.cat([local_idx, rand_idx])
            local_idx = local_idx[:K_local]

        conn_hh[h, :K_local] = local_idx

        # Cross: same parent region, different finest region
        if K_cross > 0:
            cross_mask = (parent_ids == my_parent) & (region_ids != my_region)
            cross_sim = sim[h].clone()
            cross_sim[~cross_mask] = -10.0
            n_cross_avail = cross_mask.sum().item()

            if n_cross_avail >= K_cross:
                _, cross_idx = cross_sim.topk(K_cross)
            else:
                # Fall back to global neighbors
                global_sim = sim[h].clone()
                global_sim[local_mask] = -10.0  # exclude local
                global_sim[h] = -10.0
                _, cross_idx = global_sim.topk(min(K_cross, N_h - 1))
                cross_idx = cross_idx[:K_cross]
                while cross_idx.shape[0] < K_cross:
                    rand_idx = torch.randint(0, N_h, (1,), device=W_pos.device)
                    cross_idx = torch.cat([cross_idx, rand_idx])
                cross_idx = cross_idx[:K_cross]

            conn_hh[h, K_local:] = cross_idx

    return conn_hh


class SGNNET_AH_PolarRouting(nn.Module):
    """SGNNET_AntiHebbian with hierarchical polar topology.

    Replaces standard KNN with polar-hierarchy-based neighbor selection.
    conn_hh built from polar region decomposition of W_pos.
    """

    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float,
                 n_levels: int, K_local: int, K_cross: int,
                 rebuild_every: int = 0):
        super().__init__()
        self.m      = resonant
        self.alpha  = alpha_ahebb
        self.n_levels = n_levels
        self.K_local  = K_local
        self.K_cross  = K_cross
        self.K_hh     = K_local + K_cross
        self.rebuild_every = rebuild_every
        self._epoch = 0
        self._polar_conn_hh = None

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def _build_topology(self):
        N_h = self.m.base.N_hidden
        with torch.no_grad():
            self._polar_conn_hh = build_polar_conn_hh(
                self.m.W_pos, N_h, self.n_levels,
                self.K_local, self.K_cross)

    def tick_epoch(self):
        self._epoch += 1
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()
        # Rebuild polar topology if requested
        if self.rebuild_every > 0 and self._epoch % self.rebuild_every == 0:
            self._build_topology()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build topology on first forward pass
        if self._polar_conn_hh is None:
            self._build_topology()

        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self._polar_conn_hh                           # polar topology
        N_h       = self.m.base.N_hidden

        # AH wpos suppression using polar conn_hh
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
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
            Z_new       = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


@dataclass
class Config:
    key: str; label: str; n_levels: int; K_local: int; K_cross: int
    rebuild_every: int = 0
    use_polar: bool = True


CONFIGS = [
    Config("Ref", "Ref  standard KNN (cosine sim)", 0, 4, 0, use_polar=False),
    Config("A",   "A    3-level (8 reg), Kl=2 Kx=2",  3, 2, 2),
    Config("B",   "B    4-level (16 reg), Kl=2 Kx=2",  4, 2, 2),
    Config("C",   "C    5-level (32 reg), Kl=3 Kx=1",  5, 3, 1),
    Config("D",   "D    3-level, Kl=3 Kx=1 (more local)", 3, 3, 1),
    Config("E",   "E    3-level, rebuild every 25ep",  3, 2, 2, rebuild_every=25),
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
    if not cfg.use_polar:
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_AH_PolarRouting(
        resonant, ALPHA_AHEBB, cfg.n_levels, cfg.K_local, cfg.K_cross,
        cfg.rebuild_every)


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
    n_regions = 2 ** cfg.n_levels if cfg.use_polar else 0
    result = {
        "label": cfg.label, "n_levels": cfg.n_levels,
        "n_regions": n_regions,
        "K_local": cfg.K_local, "K_cross": cfg.K_cross,
        "rebuild_every": cfg.rebuild_every,
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
    print(f"Step 108: Hierarchical polar routing (PolarQuant-inspired)")
    print(f"Baseline: step69 Ref = {STEP69_REF:.4f}\n")
    for c in CONFIGS:
        if c.use_polar:
            n_reg = 2 ** c.n_levels
            tag = f"{c.n_levels}-level ({n_reg} reg)  Kl={c.K_local} Kx={c.K_cross}  rebuild={c.rebuild_every}"
        else:
            tag = "standard KNN"
        print(f"  {c.key:4s}  {tag}")
    print()
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")
    results = {}
    out_path = ROOT / "results" / "train_step108_polar_routing.json"
    for i, cfg in enumerate(CONFIGS):
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        n_regions = 2 ** cfg.n_levels if cfg.use_polar else 0
        meta = {"N": N, "D": D, "K_iter": K_ITER,
                "n_levels": cfg.n_levels, "n_regions": n_regions,
                "K_local": cfg.K_local, "K_cross": cfg.K_cross,
                "rebuild_every": cfg.rebuild_every,
                "alpha_ahebb": ALPHA_AHEBB, "data_frac": 0.5}
        results[cfg.key] = run(cfg, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")
    print(f"\n{'='*70}\nSTEP 108 COMPLETE\n")
    print(f"  {'Key':4s}  {'levels':>6s}  {'Kl':>3s}  {'Kx':>3s}  {'top1':>8s}  {'vs_ref':>8s}")
    for c in CONFIGS:
        if c.key in results:
            r = results[c.key]
            lvl = str(c.n_levels) if c.use_polar else "-"
            print(f"  {c.key:4s}  {lvl:>6s}  {c.K_local:>3d}  {c.K_cross:>3d}  "
                  f"{r['top1_best']:.4f}  {r['delta_vs_ref']:+.4f}")
    w = max(results, key=lambda k: results[k]["top1_best"])
    print(f"\n  Winner: {w}")
