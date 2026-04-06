"""Step 112: RNN-style Sequential Input Injection.

MOTIVATION
==========
Current _seed sums K_in=50 inputs per neuron all at once — aggressive averaging
that destroys input signal (50 values collapsed to 1 vector). The graph then
routes a single, already-lossy representation for K_iter steps.

This experiment treats K_iter steps as RNN time steps:
  - Z is the hidden state (graph activation, resets per sample)
  - At each step t: inject chunk_t of input INTO current Z, then route
  - Z accumulates context from prior chunks before each new injection
  - Readout from Z_final (after all K_iter steps)

RNN analogy:
  Vanilla RNN:    h_t = tanh(W_h * h_{t-1} + W_x * x_t)
  SGNNET-RNN:     Z_t = normalize(route(Z_{t-1}) + seed_partial(x, step=t))

Benefits:
  - Less aggressive aggregation: K_in_per_step << K_in total
  - Each input chunk integrates into an already-evolved Z (richer context)
  - Lighter: K_in_per_step=2 halves input FLOPs vs current K_in=50
  - Dynamic internal connectivity via AH naturally modulates per-step state

CONFIGS (N=1024, D=64, K_hh=4, K_iter=12, AH=1.0, turing=0.0, 50%/75ep)
=========================================================================
  Ref : current baseline — all K_in=50 at step 0, pure routing thereafter
  A   : RNN-style, K_in_per_step=4, 12 steps  (total 48 ≈ same as Ref)
  B   : RNN-style, K_in_per_step=2, 12 steps  (total 24 — half K_in, lighter)
  C   : RNN-style, K_in_per_step=4 + GCNII residual α=0.05
  D   : RNN front-loaded, K_in_per_step=8 for first 6 steps then pure routing

To reproduce:
    python -u scripts/train_step112_rnn_input_injection.py --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 64; K_ITER = 12; K_IN = 50
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
STEP69_REF = 0.8336   # step69 Config A baseline for vs_ref column


# ---------------------------------------------------------------------------
# RNN-style input injection model
# ---------------------------------------------------------------------------

class SGNNET_AH_RNNInject(nn.Module):
    """SGNNET with RNN-style sequential input injection across K_iter steps.

    Instead of seeding all K_in inputs at once and routing, this model:
      1. Starts with Z = zeros (empty hidden state)
      2. At each step t (up to n_inject_steps):
           Z += partial_seed(x, conn_in_chunk_t)   # inject K_in_per_step inputs
           Z  = AH_route_one_step(Z)               # route + normalize
      3. For remaining steps (t >= n_inject_steps):
           Z  = AH_route_one_step(Z)               # pure routing
      4. Readout from Z_final

    conn_in is split column-wise across inject steps:
      step 0: conn_in[:, 0:k]
      step 1: conn_in[:, k:2k]
      ...
    where k = K_in_per_step. The block-local bias in conn_in means early
    columns contain spatially closest inputs — natural spatial chunking.

    Optional GCNII residual: after first injection Z_1 is saved as h_0;
    each subsequent step blends: Z = (1-α)*Z_routed + α*h_0.
    """

    def __init__(
        self,
        resonant,           # SGNNET_Resonant wrapper
        alpha_ahebb: float,
        K_iter: int,
        K_in: int,
        K_in_per_step: int,
        n_inject_steps: Optional[int] = None,   # None → inject all K_iter steps
        residual_alpha: float = 0.0,
    ):
        super().__init__()
        self.m              = resonant
        self.alpha          = alpha_ahebb
        self.K_iter         = K_iter
        self.K_in           = K_in
        self.K_in_per_step  = K_in_per_step
        self.n_inject_steps = n_inject_steps if n_inject_steps is not None else K_iter
        self.residual_alpha = residual_alpha

        # Validate: don't exceed conn_in columns
        max_cols = K_in_per_step * self.n_inject_steps
        if max_cols > K_in:
            raise ValueError(
                f"K_in_per_step={K_in_per_step} × n_inject_steps={self.n_inject_steps}"
                f" = {max_cols} > K_in={K_in}. Reduce K_in_per_step or n_inject_steps."
            )

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _partial_seed(self, x: torch.Tensor, col_start: int, col_end: int) -> torch.Tensor:
        """Inject one chunk of input: conn_in[:, col_start:col_end] → [B, N_hidden, D]."""
        base    = self.m.base
        B       = x.shape[0]
        spatial = base.spatial_coords.unsqueeze(0).expand(B, -1, -1)   # [B, N_in, D-1]
        A_input = torch.cat([x.unsqueeze(-1), spatial], dim=-1)        # [B, N_in, D]
        conn_chunk = base.conn_in[:, col_start:col_end]                 # [N_hidden, k]
        # Gather and sum: [B, N_hidden, k, D] → [B, N_hidden, D]
        return A_input[:, conn_chunk, :].sum(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base      = self.m.base
        N_h       = base.N_hidden
        B         = x.shape[0]
        conn_hh   = base.conn_hh
        k         = self.K_in_per_step

        # AH positional suppression weights (fixed per forward pass)
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)   # [N, K_hh]
        supp_w  = (1.0 - self.alpha * pos_sim.clamp(min=0)
                   ).unsqueeze(0).unsqueeze(-1)                # [1, N, K_hh, 1]

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)

        # Start with empty hidden state
        Z           = torch.zeros(B, N_h, base.D, device=x.device, dtype=x.dtype)
        Z_reflected = torch.zeros_like(Z)
        h_0         = None   # GCNII reference (set after first injection)

        for t in range(self.K_iter):
            # --- Input injection phase ---
            if t < self.n_inject_steps:
                col_start = t * k
                col_end   = col_start + k
                Z_inject  = self._partial_seed(x, col_start, col_end)
                Z         = Z + Z_inject
                # Normalize after injection so injected signal is on same scale as routed Z
                Z         = base._normalise(Z)

                # Save reference for GCNII after first injection
                if t == 0 and self.residual_alpha > 0:
                    h_0 = Z.clone()

            # --- Routing phase (AH one step) ---
            Z_fwd    = F.relu(Z - theta_pos)
            Z_nb     = Z_fwd[:, conn_hh, :]             # [B, N, K_hh, D]
            Z_struct = (Z_nb * supp_w).sum(dim=2)        # [B, N, D]

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected

            # GCNII residual: blend with state after first injection
            if self.residual_alpha > 0 and h_0 is not None:
                Z_new = (1.0 - self.residual_alpha) * Z_new + self.residual_alpha * h_0

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return base._readout(Z)


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    K_in_per_step: int
    n_inject_steps: Optional[int]   # None = all K_iter steps
    residual_alpha: float = 0.0
    use_rnn: bool = True

CONFIGS = [
    Config("Ref", "Ref  baseline (all K_in=50 at step 0)",          50, 1,    use_rnn=False),
    Config("A",   "A    RNN K_in_per_step=4, all 12 steps",          4, None),
    Config("B",   "B    RNN K_in_per_step=2, all 12 steps (half K_in)", 2, None),
    Config("C",   "C    RNN K_in_per_step=4 + GCNII α=0.05",         4, None, residual_alpha=0.05),
    Config("D",   "D    RNN front-loaded K_in_per_step=8, 6 steps",  8, 6),
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
        K_in=K_IN, K_iter=K_ITER, encoding_mode="fourier",
        **topo,
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    if not cfg.use_rnn:
        # Ref: standard AntiHebbian (all-at-once seeding)
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")

    n_inject = cfg.n_inject_steps if cfg.n_inject_steps is not None else K_ITER
    return SGNNET_AH_RNNInject(
        resonant,
        alpha_ahebb=ALPHA_AHEBB,
        K_iter=K_ITER,
        K_in=K_IN,
        K_in_per_step=cfg.K_in_per_step,
        n_inject_steps=n_inject,
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
    print(f"Step 112 — RNN-style Sequential Input Injection")
    print(f"N={N}  D={D}  K_iter={K_ITER}  K_in={K_IN}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"Baseline (step69-A): {STEP69_REF:.4f}")
    print(f"{'='*70}\n")

    print("Configs:")
    for c in CONFIGS:
        k_total = c.K_in_per_step * (c.n_inject_steps or K_ITER)
        print(f"  {c.key:4s}  k/step={c.K_in_per_step:2d}  "
              f"inject_steps={c.n_inject_steps or K_ITER:2d}  "
              f"total_k={k_total:3d}  res_α={c.residual_alpha:.2f}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step112_rnn_input_injection.json"

    for i, cfg in enumerate(CONFIGS):
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)
        k_total  = cfg.K_in_per_step * (cfg.n_inject_steps or K_ITER)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  K_in_total={k_total}")
        print(f"{'─'*60}")

        t0  = time.time()
        kw  = trainer_kwargs(N, n_epochs=EPOCHS)
        meta = {
            "N": N, "D": D, "K_iter": K_ITER, "K_in_total": k_total,
            "K_in_per_step": cfg.K_in_per_step,
            "n_inject_steps": cfg.n_inject_steps or K_ITER,
            "residual_alpha": cfg.residual_alpha,
            "alpha_ahebb": ALPHA_AHEBB, "data_frac": 0.5,
        }

        trainer = Trainer(
            model=model, device=DEVICE,
            train_loader=get_loaders()[0], val_loader=get_loaders()[1],
            **kw,
        )
        history   = trainer.train(n_epochs=EPOCHS)
        elapsed   = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1
        vs_ref    = top1_best - STEP69_REF

        results[cfg.key] = {
            **meta,
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

    # Summary table
    print(f"\n{'='*70}")
    print(f"STEP 112 SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<6}  {'k/step':>6}  {'total_k':>7}  {'top1':>7}  {'vs_ref':>8}  label")
    print(f"{'─'*70}")
    for key, r in results.items():
        print(f"{key:<6}  {r['K_in_per_step']:>6}  {r['K_in_total']:>7}  "
              f"{r.get('top1_best',0):.4f}  {r['vs_ref']:>+.4f}  {r['label']}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
