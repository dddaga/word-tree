"""Step 144: Efficiency stack — confirmed N=1024 winners at D=32.

MOTIVATION
==========
N=1024 is the efficiency-track scale. Several mechanisms have won at D=64
(N=4096 accuracy track); this experiment tests whether those gains transfer
to the smaller D=32 regime used for the efficiency target.

Key prior winners:
  - step117-A: W_proj [D,D] after scatter-sum → +5.48pp at Tier-1 (N=1024, D=64)
  - step131: weighted_neg β=0.3 independently confirmed
  - step124: RigL topology refinement (needs validation at D=32)

Two compound pairs (Configs C and D) test different combinations. The pair
tested in step131 (weighted_neg + W_proj) is NOT repeated here — instead we
test W_proj + RigL (C) and weighted_neg + RigL (D), both novel compounds.

CONFIGS (N=1024, D=32, K_hh=4, K_iter=12, K_in=50, AH=1.0, 50%/75ep)
======================================================================
  Ref : Standard AH (N=1024, D=32 baseline)
  A   : W_proj [D,D] only (step117-A winner, +5.48pp at Tier-1)
  B   : W_proj + α_ahebb=1.10 (two independent N=4096 gains, test at N=1024)
  C   : W_proj + RigL topology refinement every 5ep (novel compound)
  D   : weighted_neg β=0.3 + RigL (different compound from step131)

RigL note: uses refine_topology_scored() with next(iter(loader))[0] fix.
Topology is frozen for last 30% of training.

To reproduce:
    python -u scripts/train_step144_efficiency_stack.py --device mps
    python -u scripts/train_step144_efficiency_stack.py --device mps --epochs 20
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
                    help="Training epochs (default 75; use 20 for scout)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (e.g. A,D). Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 32; K_ITER = 12; K_IN = 50
K_HH = 4   # K_local=3 + K_random=1 via topology_kwargs
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
N_CANDIDATES = 16   # RigL: random non-neighbors sampled per neuron per refinement


# ---------------------------------------------------------------------------
# Model: combined W_proj + optional mechanisms
# ---------------------------------------------------------------------------

class SGNNET_EfficiencyStack(nn.Module):
    """AH routing with optional W_proj and/or weighted_neg activation.

    use_proj=True:   apply shared [D,D] linear projection after _seed().
    use_weighted_neg: Z_fwd = relu(Z-θ) + β*relu(θ-Z) (sub-threshold recovery).
    Both are independent and can be combined.
    """

    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float = 1.0,
                 use_proj: bool = False, use_weighted_neg: bool = False,
                 beta: float = 0.3):
        super().__init__()
        self.m               = resonant
        self.alpha_ahebb     = alpha_ahebb
        self.use_proj        = use_proj
        self.use_weighted_neg = use_weighted_neg
        self.beta            = beta

        D_ = resonant.base.D
        if use_proj:
            self.proj = nn.Linear(D_, D_, bias=False)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _activation(self, Z: torch.Tensor, theta_pos: torch.Tensor) -> torch.Tensor:
        if self.use_weighted_neg:
            return F.relu(Z - theta_pos) + self.beta * F.relu(theta_pos - Z)
        return F.relu(Z - theta_pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)

        # Optional input projection
        if self.use_proj:
            Z = self.proj(Z)

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden

        # Static AH suppression weights (wpos)
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z_fwd    = self._activation(Z, theta_pos)
            Z_nb     = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ---------------------------------------------------------------------------
# RigL topology refinement (adapted from step124)
# ---------------------------------------------------------------------------

def _get_base(model: nn.Module) -> SGNNET_SmallWorld:
    """Navigate wrapper chain to SGNNET_SmallWorld."""
    if isinstance(model, SGNNET_AntiHebbian):
        return model.m.base
    if isinstance(model, SGNNET_EfficiencyStack):
        return model.m.base
    raise ValueError(f"Cannot extract base from {type(model)}")


def _get_resonant(model: nn.Module) -> SGNNET_Resonant:
    """Navigate wrapper chain to SGNNET_Resonant."""
    if isinstance(model, SGNNET_AntiHebbian):
        return model.m
    if isinstance(model, SGNNET_EfficiencyStack):
        return model.m
    raise ValueError(f"Cannot extract resonant from {type(model)}")


@torch.no_grad()
def refine_topology_scored(model: nn.Module, loader, device: torch.device,
                           max_swaps: int = 1) -> int:
    """Score existing edges and candidates by activation difference; swap worst→best.

    Fix: uses next(iter(loader))[0] since dataset returns 3-tuples.
    """
    # Get one batch — dataset returns (x, y, idx) so index [0]
    batch_x = next(iter(loader))[0].to(device)

    base     = _get_base(model)
    resonant = _get_resonant(model)
    conn_hh  = base.conn_hh
    N_h      = base.N_hidden

    # Seed activations
    Z = base._seed(batch_x)  # [B, N_h, D]

    # Run K_iter//2 routing steps to get mid-routing activations
    theta_pos = resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
    for _ in range(base.K_iter // 2):
        Z_fwd = F.relu(Z - theta_pos)
        Z_nb  = Z_fwd[:, conn_hh, :]
        Z     = F.normalize(Z_nb.sum(dim=2).clamp(-10, 10), dim=-1)

    # Score existing edges: mean |Z[h] - Z[neighbor]| over batch
    Z_neighbors = Z[:, conn_hh, :]               # [B, N_h, K_hh, D]
    Z_expanded  = Z.unsqueeze(2).expand_as(Z_neighbors)
    existing_scores = (Z_expanded - Z_neighbors).abs().mean(dim=(0, -1))  # [N_h, K_hh]

    conn_np  = conn_hh.cpu().numpy()
    new_conn = conn_hh.clone()
    total_swaps = 0

    for h in range(N_h):
        neighbors_set = set(conn_np[h].tolist())
        neighbors_set.add(h)
        non_neighbors = [i for i in range(N_h) if i not in neighbors_set]
        if len(non_neighbors) < 1:
            continue

        n_sample  = min(N_CANDIDATES, len(non_neighbors))
        cand_idx  = np.random.choice(non_neighbors, size=n_sample, replace=False)
        cand_idx_t = torch.tensor(cand_idx, dtype=torch.long, device=device)

        Z_h     = Z[:, h, :]                          # [B, D]
        Z_cands = Z[:, cand_idx_t, :]                 # [B, n_sample, D]
        cand_scores = (Z_h.unsqueeze(1) - Z_cands).abs().mean(dim=(0, -1))  # [n_sample]

        edge_scores  = existing_scores[h]
        sorted_edges = edge_scores.argsort()           # ascending (worst first)
        sorted_cands = cand_scores.argsort(descending=True)

        swaps_done = 0
        for s in range(min(max_swaps, K_HH)):
            worst_edge_pos = sorted_edges[s].item()
            best_cand_pos  = sorted_cands[s].item() if s < len(sorted_cands) else None
            if best_cand_pos is None:
                break
            if cand_scores[best_cand_pos] > edge_scores[worst_edge_pos]:
                new_conn[h, worst_edge_pos] = cand_idx_t[best_cand_pos]
                swaps_done += 1
            else:
                break
        total_swaps += swaps_done

    base.conn_hh.copy_(new_conn)
    return total_swaps


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key:              str
    label:            str
    use_proj:         bool
    use_weighted_neg: bool
    alpha_ahebb:      float = ALPHA_AHEBB
    beta:             float = 0.3
    use_rigl:         bool  = False
    rigl_every:       int   = 5      # epochs between topology refinements
    rigl_freeze_frac: float = 0.70   # freeze topology for last N% of training
    rigl_max_swaps:   int   = 1

CONFIGS = [
    Config("Ref", "Ref  standard AH (baseline)",
           use_proj=False, use_weighted_neg=False),
    Config("A",   "A    W_proj [D,D] only (step117-A, +5.48pp Tier-1)",
           use_proj=True, use_weighted_neg=False),
    Config("B",   "B    W_proj + α_ahebb=1.10",
           use_proj=True, use_weighted_neg=False, alpha_ahebb=1.10),
    Config("C",   "C    W_proj + RigL every 5ep (novel compound)",
           use_proj=True, use_weighted_neg=False,
           use_rigl=True, rigl_every=5, rigl_freeze_frac=0.70, rigl_max_swaps=1),
    Config("D",   "D    weighted_neg β=0.3 + RigL every 5ep",
           use_proj=False, use_weighted_neg=True, beta=0.3,
           use_rigl=True, rigl_every=5, rigl_freeze_frac=0.70, rigl_max_swaps=1),
]


# ---------------------------------------------------------------------------
# Data (cached, 50%)
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
    tk = topology_kwargs(N)
    tk.pop("K_in", None); tk.pop("K_iter", None)

    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER,
        K_local=tk["K_local"], K_random=tk["K_random"],
        n_groups=tk["n_groups"],
        norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )

    if not cfg.use_proj and not cfg.use_weighted_neg and cfg.alpha_ahebb == ALPHA_AHEBB:
        # Ref: use standard AntiHebbian
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=cfg.alpha_ahebb, variant="wpos")

    return SGNNET_EfficiencyStack(
        resonant, alpha_ahebb=cfg.alpha_ahebb,
        use_proj=cfg.use_proj,
        use_weighted_neg=cfg.use_weighted_neg,
        beta=cfg.beta,
    )


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def compute_flops() -> int:
    seed     = N * K_IN * D
    per_step = N * K_HH * D * 2 + N * D + N * D * 2
    routing  = K_ITER * per_step
    readout  = N * N_OUT * D
    return seed + routing + readout


# ---------------------------------------------------------------------------
# Training loop with optional RigL hooks
# ---------------------------------------------------------------------------

def train_with_rigl(model: nn.Module, cfg: Config, device: torch.device,
                    n_epochs: int) -> list[dict]:
    """Train with optional RigL topology refinement between epochs."""
    tr_loader, va_loader = get_loaders()
    kw = trainer_kwargs(N, n_epochs=n_epochs)

    trainer = Trainer(
        model=model,
        train_loader=tr_loader,
        val_loader=va_loader,
        device=device,
        **kw,
    )

    freeze_epoch = int(cfg.rigl_freeze_frac * n_epochs) if cfg.use_rigl else n_epochs + 1
    history    = []
    refine_log = []

    for ep in range(1, n_epochs + 1):
        ep_hist = trainer.train(n_epochs=1)
        history.extend(ep_hist)

        if (cfg.use_rigl and ep < freeze_epoch
                and cfg.rigl_every > 0 and ep % cfg.rigl_every == 0):
            n_swaps = refine_topology_scored(model, tr_loader, device,
                                             max_swaps=cfg.rigl_max_swaps)
            refine_log.append({"epoch": ep, "swaps": n_swaps})
            print(f"    [rigl ep={ep}] scored: {n_swaps} swaps")

        if cfg.use_rigl and ep == freeze_epoch:
            print(f"    [rigl ep={ep}] topology frozen for remaining epochs")

    if history and refine_log:
        history[-1]["refine_log"] = refine_log

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    flops = compute_flops()
    print(f"\n{'='*70}")
    print(f"Step 144 — Efficiency Stack (N=1024 D=32 Winners)")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  K_in={K_IN}")
    print(f"FLOPs={flops/1e6:.2f}M  Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    for c in CONFIGS:
        rigl_info = f"  rigl_every={c.rigl_every}" if c.use_rigl else ""
        print(f"  {c.key:4s}  proj={str(c.use_proj):5s}  wneg={str(c.use_weighted_neg):5s}  "
              f"α={c.alpha_ahebb:.2f}{rigl_info}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step144_efficiency_stack.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  FLOPs={flops/1e6:.2f}M  "
              f"use_rigl={cfg.use_rigl}")
        print(f"{'─'*60}")

        t0 = time.time()
        history = train_with_rigl(model, cfg, DEVICE, n_epochs=EPOCHS)
        elapsed = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1

        # Extract refine log if present
        refine_log = []
        for h in history:
            if "refine_log" in h:
                refine_log = h["refine_log"]

        results[cfg.key] = {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER, "K_in": K_IN,
            "use_proj": cfg.use_proj,
            "use_weighted_neg": cfg.use_weighted_neg,
            "alpha_ahebb": cfg.alpha_ahebb,
            "use_rigl": cfg.use_rigl,
            "rigl_every": cfg.rigl_every if cfg.use_rigl else None,
            "rigl_freeze_frac": cfg.rigl_freeze_frac if cfg.use_rigl else None,
            "flops": flops, "flops_M": round(flops / 1e6, 2),
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params,
            "label": cfg.label,
            "refine_log": refine_log,
        }

        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs_ref = top1_best - ref_best if ref_best > 0 else 0
        print(f"\n  top1={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"FLOPs={flops/1e6:.2f}M  elapsed={elapsed/60:.1f}min  "
              f"params={n_params:,}")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 144 SUMMARY — Efficiency Stack (N=1024, D=32)")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Key':4s}  {'proj':5s}  {'wneg':5s}  {'α':5s}  {'rigl':5s}  "
          f"{'params':>8}  {'top1':>7}  {'vs_Ref':>8}")
    print(f"{'─'*60}")
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"{key:4s}  {str(r['use_proj']):5s}  {str(r['use_weighted_neg']):5s}  "
              f"{r['alpha_ahebb']:.2f}  {str(r['use_rigl']):5s}  "
              f"{r['n_params']:>8}  {r['top1_best']:.4f}  {vs:>+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
