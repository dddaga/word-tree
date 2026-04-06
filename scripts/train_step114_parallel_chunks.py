"""Step 114: Parallel Chunk Processing + Superimpose + Sequential Mixing.

MOTIVATION
==========
Current _seed collapses K_in=50 inputs per neuron into one vector immediately.
Step 112 (RNN-style) injects chunks sequentially — better per-step context
but inherently serial.

This experiment exploits the key advantage transformers had over RNNs:
PARALLELISATION of input processing. Each input chunk gets its own few routing
steps independently (parallel), then all chunk states are superimposed, then
a final set of mixing steps integrates the combined state.

Architecture:
─────────────
Phase 1 — Parallel:
  Split conn_in into K chunks (K_in/K connections each).
  For each chunk k:
      Z_k = partial_seed(chunk_k)     # [B, N, D]
      Z_k = AH_route(Z_k, N_pre steps)
  All K chunks run simultaneously: batch as [K×B, N, D] → single GPU pass.

Phase 2 — Superimpose:
  Z = mean(Z_0, ..., Z_{K-1})        # merge on the sphere

Phase 3 — Sequential mixing:
  Z = AH_route(Z, N_post steps)
  output = readout(Z)

Total routing steps = K × N_pre + N_post.
Parallelism ratio = K × N_pre / (K × N_pre + N_post).

Analogy to multi-head attention:
  - Each chunk = one "head" attending to its input region
  - N_pre steps = intra-head processing depth
  - Superimpose = concatenate + project (here: mean on sphere, no params)
  - N_post steps = cross-head integration

CONFIGS (N=1024, D=64, K_hh=4, AH=1.0, turing=0.0, 50%/75ep)
=========================================================================
  Ref : K=1, N_pre=0, N_post=12 — current baseline (all-at-once, K_iter=12)
  A   : K=6,  N_pre=2, N_post=4  — 6 chunks, light parallel + deep merge
  B   : K=4,  N_pre=3, N_post=4  — 4 chunks, more per-chunk depth
  C   : K=12, N_pre=1, N_post=6  — max chunks (1 step each), longer merge
  D   : K=6,  N_pre=2, N_post=4, GCNII α=0.05 on merge phase

To reproduce:
    python -u scripts/train_step114_parallel_chunks.py --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

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
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 64; K_IN = 50
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
STEP69_REF = 0.8336


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SGNNET_ParallelChunks(nn.Module):
    """Parallel chunk processing + superimpose + sequential mixing.

    Phase 1 (parallel): K chunks of input, each gets N_pre routing steps.
    All K paths batched as [K*B, N, D] — true parallel execution.

    Phase 2 (superimpose): mean of K chunk states → [B, N, D], then normalise.

    Phase 3 (sequential mixing): N_post routing steps on merged state.
    Optional GCNII residual on merge phase: blend with post-superimpose Z.
    """

    def __init__(
        self,
        resonant,
        alpha_ahebb: float,
        K_in: int,
        K_chunks: int,
        N_pre: int,
        N_post: int,
        residual_alpha: float = 0.0,
    ):
        super().__init__()
        self.m              = resonant
        self.alpha          = alpha_ahebb
        self.K_in           = K_in
        self.K_chunks       = K_chunks
        self.N_pre          = N_pre
        self.N_post         = N_post
        self.residual_alpha = residual_alpha

        k_per = K_in // K_chunks
        if k_per == 0:
            raise ValueError(f"K_in={K_in} too small for K_chunks={K_chunks}")
        self.k_per = k_per   # connections per chunk (last chunk may have fewer)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _partial_seed(self, x: torch.Tensor, col_start: int, col_end: int) -> torch.Tensor:
        """Seed Z from conn_in[:, col_start:col_end]. Returns [B, N_hidden, D]."""
        base    = self.m.base
        B       = x.shape[0]
        spatial = base.spatial_coords.unsqueeze(0).expand(B, -1, -1)
        A_input = torch.cat([x.unsqueeze(-1), spatial], dim=-1)         # [B, N_in, D]
        chunk   = base.conn_in[:, col_start:col_end]                    # [N_hidden, k]
        return A_input[:, chunk, :].sum(dim=2)                          # [B, N_hidden, D]

    def _route_steps(
        self,
        Z: torch.Tensor,
        n_steps: int,
        supp_w: torch.Tensor,
        theta_pos: torch.Tensor,
        conn_hh: torch.Tensor,
        h_0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run n_steps of AH routing on Z. Optionally apply GCNII residual vs h_0."""
        Z_reflected = torch.zeros_like(Z)
        for _ in range(n_steps):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_nb     = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected

            if self.residual_alpha > 0 and h_0 is not None:
                Z_new = (1.0 - self.residual_alpha) * Z_new + self.residual_alpha * h_0

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)
        return Z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base    = self.m.base
        B       = x.shape[0]
        N_h     = base.N_hidden
        conn_hh = base.conn_hh

        # Shared AH suppression weights (same graph for all chunks)
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)    # [N, K_hh]
        supp_w  = (1.0 - self.alpha * pos_sim.clamp(min=0)
                   ).unsqueeze(0).unsqueeze(-1)                  # [1, N, K_hh, 1]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)

        # ── Phase 1: seed + parallel routing for each chunk ──────────────────
        chunk_states = []
        for k in range(self.K_chunks):
            col_start = k * self.k_per
            col_end   = min(col_start + self.k_per, self.K_in)
            Z_k       = base._normalise(self._partial_seed(x, col_start, col_end))

            if self.N_pre > 0:
                # Batch all K chunks for true parallel routing:
                # accumulate then route in one batched pass below
                chunk_states.append(Z_k)
            else:
                chunk_states.append(Z_k)

        if self.N_pre > 0:
            # Stack → [K, B, N, D], reshape → [K*B, N, D], route, reshape back
            Z_batch = torch.stack(chunk_states, dim=0)              # [K, B, N, D]
            K, _B, _N, _D = Z_batch.shape
            Z_flat  = Z_batch.reshape(K * _B, _N, _D)              # [K*B, N, D]

            # Expand suppression weights to match K*B batch
            supp_w_exp  = supp_w.expand(K * _B, -1, -1, -1)
            theta_exp   = theta_pos.expand(K * _B, -1, -1)

            Z_flat_ref  = torch.zeros_like(Z_flat)  # no GCNII in parallel phase
            Z_routed    = self._route_steps(
                Z_flat, self.N_pre,
                supp_w_exp, theta_exp, conn_hh,
                h_0=None,
            )
            Z_batch_out = Z_routed.reshape(K, _B, _N, _D)          # [K, B, N, D]
            chunk_states = [Z_batch_out[k] for k in range(K)]

        # ── Phase 2: superimpose (mean + normalise) ───────────────────────────
        Z = torch.stack(chunk_states, dim=0).mean(dim=0)            # [B, N, D]
        Z = base._normalise(Z)

        # ── Phase 3: sequential mixing ────────────────────────────────────────
        if self.N_post > 0:
            h_0_merge = Z.clone() if self.residual_alpha > 0 else None
            Z = self._route_steps(
                Z, self.N_post,
                supp_w, theta_pos, conn_hh,
                h_0=h_0_merge,
            )

        return base._readout(Z)


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    K_chunks: int
    N_pre: int
    N_post: int
    residual_alpha: float = 0.0
    use_parallel: bool = True


CONFIGS = [
    Config("Ref", "Ref  K=1 N_pre=0 N_post=12 (standard K_iter=12)",  1,  0, 12, use_parallel=False),
    Config("A",   "A    K=6  N_pre=2 N_post=4  (parallel:merge = 3:2)", 6,  2,  4),
    Config("B",   "B    K=4  N_pre=3 N_post=4  (deeper per-chunk)",     4,  3,  4),
    Config("C",   "C    K=12 N_pre=1 N_post=6  (max chunks, long merge)", 12, 1,  6),
    Config("D",   "D    K=6  N_pre=2 N_post=4  GCNII α=0.05 on merge", 6,  2,  4, residual_alpha=0.05),
]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    topo = topology_kwargs(N)
    topo.pop("K_in", None); topo.pop("K_iter", None)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=1, encoding_mode="fourier",   # K_iter unused — we control loop
        **topo,
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    if not cfg.use_parallel:
        # Ref: standard AntiHebbian with K_iter=12
        base.K_iter = 12
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_ParallelChunks(
        resonant,
        alpha_ahebb=ALPHA_AHEBB,
        K_in=K_IN,
        K_chunks=cfg.K_chunks,
        N_pre=cfg.N_pre,
        N_post=cfg.N_post,
        residual_alpha=cfg.residual_alpha,
    )


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data loaders (cached)
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
    print(f"Step 114 — Parallel Chunk Processing + Superimpose + Sequential Mixing")
    print(f"N={N}  D={D}  K_in={K_IN}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"Baseline (step69-A): {STEP69_REF:.4f}")
    print(f"{'='*70}\n")

    print(f"{'Config':<6}  {'K':>3}  {'N_pre':>5}  {'N_post':>6}  {'total_steps':>11}  label")
    print(f"{'─'*70}")
    for c in CONFIGS:
        total = c.K_chunks * c.N_pre + c.N_post
        print(f"  {c.key:<4}  {c.K_chunks:>3}  {c.N_pre:>5}  {c.N_post:>6}  {total:>11}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step114_parallel_chunks.json"

    for i, cfg in enumerate(CONFIGS):
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)
        total_steps = cfg.K_chunks * cfg.N_pre + cfg.N_post

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  total_routing_steps={total_steps}")
        print(f"{'─'*60}")

        t0      = time.time()
        kw      = trainer_kwargs(N, n_epochs=EPOCHS)
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
        vs_ref    = top1_best - STEP69_REF

        results[cfg.key] = {
            "N": N, "D": D, "K_in": K_IN,
            "K_chunks": cfg.K_chunks,
            "N_pre": cfg.N_pre,
            "N_post": cfg.N_post,
            "total_routing_steps": total_steps,
            "residual_alpha": cfg.residual_alpha,
            "alpha_ahebb": ALPHA_AHEBB,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "vs_ref": round(vs_ref, 6),
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params,
            "label": cfg.label,
        }

        print(f"\n  top1_best={top1_best:.4f}  vs_ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 114 SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<6}  {'K':>3}  {'N_pre':>5}  {'N_post':>6}  {'top1':>7}  {'vs_ref':>8}  label")
    print(f"{'─'*70}")
    for key, r in results.items():
        print(f"{key:<6}  {r['K_chunks']:>3}  {r['N_pre']:>5}  {r['N_post']:>6}  "
              f"{r.get('top1_best',0):.4f}  {r['vs_ref']:>+.4f}  {r['label']}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
