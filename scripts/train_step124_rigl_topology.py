"""Step 124: RigL-style Topology Refinement for SGNNET.

MOTIVATION
==========
SGNNET's K_hh=4 connections are fixed after initialization. RigL (Evci et al.
2020) showed sparse networks improve by periodically regrowing connections based
on gradient magnitude. At K_hh=4 / N=1024 (0.39% connectivity), topology
matters — finding better edges could unlock gains.

MECHANISM (activation-similarity proxy)
=======================================
Every T_refine epochs during training:
1. Run one forward pass, record mean |Z[h] - Z[k]| for each existing edge (h,k)
2. For each neuron h, sample 16 random non-neighbors, compute mean |Z[h] - Z[cand]|
3. If any candidate has HIGHER activation difference than the lowest existing edge,
   swap (AH wants diverse neighbors — higher diff = more diverse = better)
4. Freeze topology for final 30% of training (let weights adapt to final graph)

Random control (Config D): swap edges randomly without scoring, to test whether
ANY topology change helps vs. specifically scored changes.

CONFIGS (N=1024, D=32, K_hh=4, K_iter=12, AH=1.0, 50%/75ep)
=============================================================
  Ref : Fixed topology (standard baseline)
  A   : Refine every 10ep, swap 1 edge/neuron, freeze at ep52
  B   : Refine every 5ep,  swap 1 edge/neuron, freeze at ep52
  C   : Refine every 10ep, swap up to 2 edges/neuron, freeze at ep52
  D   : Refine every 10ep, swap 1 edge, RANDOM replacement (control)

To reproduce:
    python -u scripts/train_step124_rigl_topology.py --device mps
    python -u scripts/train_step124_rigl_topology.py --device mps --epochs 20  # scout
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
K_HH = 4  # K_local + K_random from topology_kwargs
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
N_CANDIDATES = 16  # random non-neighbors to evaluate per neuron


# ---------------------------------------------------------------------------
# Topology refinement logic
# ---------------------------------------------------------------------------

@torch.no_grad()
def refine_topology_scored(model: nn.Module, loader, device: torch.device,
                           max_swaps: int = 1):
    """Score existing edges and candidates by activation difference, swap worst→best.

    For each neuron h:
      - Compute mean |Z[h] - Z[k]| over a batch for each existing neighbor k
      - Sample N_CANDIDATES random non-neighbors, compute same score
      - If best candidate > worst existing edge, swap (up to max_swaps per neuron)
    """
    # Get one batch for scoring
    batch_x = next(iter(loader))[0]
    batch_x = batch_x.to(device)

    # Forward to get Z after seeding (pre-routing activations carry spatial info)
    base = _get_base(model)
    Z = base._seed(batch_x)  # [B, N_hidden, D]

    # Also run a few routing steps to get richer activations
    conn_hh = base.conn_hh  # [N_hidden, K_hh]
    N_h = base.N_hidden

    # Run K_iter/2 routing steps to get mid-routing activations
    resonant = _get_resonant(model)
    theta_pos = resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
    for _ in range(base.K_iter // 2):
        Z_fwd = F.relu(Z - theta_pos)
        Z_nb = Z_fwd[:, conn_hh, :]  # [B, N, K_hh, D]
        Z = F.normalize(Z_nb.sum(dim=2).clamp(-10, 10), dim=-1)

    # Score existing edges: mean |Z[h] - Z[neighbor]| over batch
    # Z shape: [B, N_h, D], conn_hh: [N_h, K_hh]
    Z_neighbors = Z[:, conn_hh, :]  # [B, N_h, K_hh, D]
    Z_expanded = Z.unsqueeze(2).expand_as(Z_neighbors)  # [B, N_h, K_hh, D]
    existing_scores = (Z_expanded - Z_neighbors).abs().mean(dim=(0, -1))  # [N_h, K_hh]

    # For each neuron, find non-neighbors and score candidates
    conn_np = conn_hh.cpu().numpy()
    new_conn = conn_hh.clone()
    total_swaps = 0

    for h in range(N_h):
        neighbors_set = set(conn_np[h].tolist())
        neighbors_set.add(h)  # exclude self
        non_neighbors = [i for i in range(N_h) if i not in neighbors_set]
        if len(non_neighbors) < 1:
            continue

        # Sample candidates
        n_sample = min(N_CANDIDATES, len(non_neighbors))
        cand_idx = np.random.choice(non_neighbors, size=n_sample, replace=False)
        cand_idx_t = torch.tensor(cand_idx, dtype=torch.long, device=device)

        # Score candidates: mean |Z[h] - Z[candidate]| over batch
        Z_h = Z[:, h, :]  # [B, D]
        Z_cands = Z[:, cand_idx_t, :]  # [B, n_sample, D]
        cand_scores = (Z_h.unsqueeze(1) - Z_cands).abs().mean(dim=(0, -1))  # [n_sample]

        # Find worst existing edges and best candidates
        edge_scores = existing_scores[h]  # [K_hh]
        sorted_edges = edge_scores.argsort()  # ascending (worst first)
        sorted_cands = cand_scores.argsort(descending=True)  # descending (best first)

        swaps_done = 0
        for s in range(min(max_swaps, K_HH)):
            worst_edge_pos = sorted_edges[s].item()
            best_cand_pos = sorted_cands[s].item() if s < len(sorted_cands) else None
            if best_cand_pos is None:
                break
            if cand_scores[best_cand_pos] > edge_scores[worst_edge_pos]:
                new_conn[h, worst_edge_pos] = cand_idx_t[best_cand_pos]
                swaps_done += 1
            else:
                break  # no more beneficial swaps
        total_swaps += swaps_done

    # Update conn_hh buffer in-place
    base.conn_hh.copy_(new_conn)
    return total_swaps


@torch.no_grad()
def refine_topology_random(model: nn.Module, max_swaps: int = 1):
    """Random control: swap one random edge per neuron (no scoring)."""
    base = _get_base(model)
    conn_hh = base.conn_hh
    N_h = base.N_hidden
    conn_np = conn_hh.cpu().numpy()
    new_conn = conn_hh.clone()
    total_swaps = 0

    for h in range(N_h):
        neighbors_set = set(conn_np[h].tolist())
        neighbors_set.add(h)
        non_neighbors = [i for i in range(N_h) if i not in neighbors_set]
        if len(non_neighbors) < 1:
            continue
        for s in range(min(max_swaps, K_HH)):
            edge_pos = np.random.randint(0, K_HH)
            new_neighbor = np.random.choice(non_neighbors)
            new_conn[h, edge_pos] = new_neighbor
            total_swaps += 1

    base.conn_hh.copy_(new_conn)
    return total_swaps


def _get_base(model: nn.Module) -> SGNNET_SmallWorld:
    """Navigate wrapper chain to get the SmallWorld base."""
    if isinstance(model, SGNNET_AntiHebbian):
        return model.m.base
    raise ValueError(f"Cannot extract base from {type(model)}")


def _get_resonant(model: nn.Module) -> SGNNET_Resonant:
    """Navigate wrapper chain to get the Resonant layer."""
    if isinstance(model, SGNNET_AntiHebbian):
        return model.m
    raise ValueError(f"Cannot extract resonant from {type(model)}")


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    refine_every: int       # epochs between refinement (0 = no refinement)
    max_swaps: int          # max edges to swap per neuron per refinement
    random_swap: bool       # True = random control, False = scored swap
    freeze_frac: float      # freeze topology for last N% of training


CONFIGS = [
    Config("Ref", "Ref  Fixed topology (baseline)",
           refine_every=0, max_swaps=0, random_swap=False, freeze_frac=0.0),
    Config("A",   "A    Refine/10ep, swap=1, freeze@70%",
           refine_every=10, max_swaps=1, random_swap=False, freeze_frac=0.70),
    Config("B",   "B    Refine/5ep,  swap=1, freeze@70%",
           refine_every=5, max_swaps=1, random_swap=False, freeze_frac=0.70),
    Config("C",   "C    Refine/10ep, swap=2, freeze@70%",
           refine_every=10, max_swaps=2, random_swap=False, freeze_frac=0.70),
    Config("D",   "D    Refine/10ep, swap=1, RANDOM (control)",
           refine_every=10, max_swaps=1, random_swap=True, freeze_frac=0.70),
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
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


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
# Custom training loop with topology refinement hooks
# ---------------------------------------------------------------------------

def train_with_refinement(model: nn.Module, cfg: Config, device: torch.device,
                          n_epochs: int) -> list[dict]:
    """Train with periodic topology refinement between epochs."""
    tr_loader, va_loader = get_loaders()
    kw = trainer_kwargs(N, n_epochs=n_epochs)

    trainer = Trainer(
        model=model,
        train_loader=tr_loader,
        val_loader=va_loader,
        device=device,
        **kw,
    )

    freeze_epoch = int(cfg.freeze_frac * n_epochs) if cfg.freeze_frac > 0 else n_epochs + 1
    history = []
    refine_log = []

    for ep in range(1, n_epochs + 1):
        # Train one epoch
        ep_hist = trainer.train(n_epochs=1)
        history.extend(ep_hist)

        # Topology refinement check
        if cfg.refine_every > 0 and ep < freeze_epoch and ep % cfg.refine_every == 0:
            if cfg.random_swap:
                n_swaps = refine_topology_random(model, max_swaps=cfg.max_swaps)
                method = "random"
            else:
                n_swaps = refine_topology_scored(model, tr_loader, device,
                                                 max_swaps=cfg.max_swaps)
                method = "scored"

            # Recompute static AH suppression weights after topology change
            _recompute_ah_weights(model)

            refine_log.append({"epoch": ep, "method": method, "swaps": n_swaps})
            print(f"    [refine ep={ep}] {method}: {n_swaps} swaps")

        if ep == freeze_epoch:
            print(f"    [freeze ep={ep}] topology frozen for remaining epochs")

    # Attach refine log to last history entry
    if history and refine_log:
        history[-1]["refine_log"] = refine_log

    return history


def _recompute_ah_weights(model: nn.Module):
    """After conn_hh changes, the wpos AH suppression is stale.

    SGNNET_AntiHebbian computes supp_w inside forward() each call using
    the current conn_hh, so no explicit recomputation needed — the next
    forward pass will use the updated topology automatically.

    This function is a no-op but documents the reasoning.
    """
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 124 — RigL-style Topology Refinement")
    print(f"N={N}  D={D}  K_iter={K_ITER}  K_hh={K_HH}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"N_candidates={N_CANDIDATES}")
    print(f"{'='*70}\n")

    print("Configs:")
    for c in CONFIGS:
        print(f"  {c.key:4s}  refine_every={c.refine_every:2d}  max_swaps={c.max_swaps}  "
              f"random={c.random_swap}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step124_rigl_topology.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  refine_every={cfg.refine_every}  "
              f"max_swaps={cfg.max_swaps}  random={cfg.random_swap}")
        print(f"{'─'*60}")

        t0 = time.time()

        if cfg.refine_every > 0:
            history = train_with_refinement(model, cfg, DEVICE, n_epochs=EPOCHS)
        else:
            # Ref: standard training, no refinement
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

        # Extract refine log if present
        refine_log = []
        for h in history:
            if "refine_log" in h:
                refine_log = h["refine_log"]

        results[cfg.key] = {
            "N": N, "D": D, "K_iter": K_ITER, "K_hh": K_HH,
            "refine_every": cfg.refine_every,
            "max_swaps": cfg.max_swaps,
            "random_swap": cfg.random_swap,
            "freeze_frac": cfg.freeze_frac,
            "alpha_ahebb": ALPHA_AHEBB,
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

        print(f"\n  top1_best={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary table
    print(f"\n{'='*70}")
    print(f"STEP 124 SUMMARY (D={D})")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Config':<6}  {'refine':>6}  {'swaps':>5}  {'random':>6}  "
          f"{'params':>8}  {'top1':>7}  {'vs_Ref':>8}")
    print(f"{'─'*70}")
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"{key:<6}  {r['refine_every']:>6}  {r['max_swaps']:>5}  "
              f"{str(r['random_swap']):>6}  {r['n_params']:>8,}  "
              f"{r['top1_best']:.4f}  {vs:>+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
