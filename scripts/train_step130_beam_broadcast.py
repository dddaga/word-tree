"""Step 130: Beam-as-Global-Broadcast at extreme sparsity.

MOTIVATION
==========
At K_hh=4 / N=4096, connectivity is 0.1%. Local routing can't propagate signal
across the graph fast enough — after K_iter=12 hops through 4-connected neighbors,
effective receptive field is limited by graph diameter.

The beam mechanism (top-M active neurons broadcast to all N) already exists in
SGNNET_Resonant but is DEAD CODE because alpha_turing=0.0. This experiment
repurposes the beam NOT for Turing inhibition but for EXCITATORY global broadcast:

  1. Find top-M neurons by activation magnitude (beam)
  2. Broadcast their signal to all N neurons
  3. Each neuron receives a weighted mix of beam signals (soft attention)
  4. Add to local routing signal (not replace)

This is O(M×N×D) per step — at M=8, N=1024: 524K ops. Compare to local routing
at N×K×D = 1024×4×32 = 131K. So beam adds ~4× cost, but provides global info.

The beam is a lightweight global attention mechanism at O(M×N) instead of O(N²).
At extreme sparsity (K_hh=4), this is the cheapest way to add long-range signal.

CONFIGS (N=1024, D=32, K_hh=4, K_iter=12, AH=1.0, turing=0.0, 50%/75ep)
=========================================================================
  Ref : standard AH routing, no beam (D=32 baseline)
  A   : beam M=4, every step, additive (λ=0.1)
  B   : beam M=8, every step, additive (λ=0.1)
  C   : beam M=16, every step, additive (λ=0.1)
  D   : beam M=8, every 3rd step (steps 2,5,8,11), additive (λ=0.1)
  E   : beam M=8, every step, additive (λ=0.3, stronger broadcast)

To reproduce:
    python -u scripts/train_step130_beam_broadcast.py --device mps
    python -u scripts/train_step130_beam_broadcast.py --device mps --epochs 20
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
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
STEP69_REF = 0.8336


# ---------------------------------------------------------------------------
# Model: AH routing with beam-based global broadcast
# ---------------------------------------------------------------------------

class SGNNET_AH_BeamBroadcast(nn.Module):
    """AH routing + global broadcast from top-M active neurons.

    At designated steps:
      1. Find top-M neurons by activation magnitude
      2. Compute attention: score[n,m] = dot(Z[n], Z_beam[m]) / tau
      3. w = softmax(score, dim=M)  → [B, N, M]
      4. Z_global = (w @ Z_beam)    → [B, N, D]
      5. Z_new = Z_local + lambda_beam * Z_global

    Beam provides long-range information flow at O(M×N×D) cost.
    Signal-preserving: softmax weights sum to 1 (redistribution, not gating).
    """

    def __init__(self, resonant, alpha_ahebb: float,
                 beam_size: int = 8,
                 lambda_beam: float = 0.1,
                 beam_every: int = 1,
                 tau: float = 1.0):
        super().__init__()
        self.m            = resonant
        self.alpha        = alpha_ahebb
        self.beam_size    = beam_size
        self.lambda_beam  = lambda_beam
        self.beam_every   = beam_every  # apply beam every N-th step (1=every step)
        self.tau          = tau

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _beam_broadcast(self, Z: torch.Tensor) -> torch.Tensor:
        """Global broadcast from top-M active neurons to all N neurons.

        Returns: Z_global [B, N, D] — beam-weighted signal for all neurons.
        """
        B, N_h, D_ = Z.shape
        M = min(self.beam_size, N_h)

        # Find top-M by activation magnitude
        activity = Z.norm(dim=-1)                                   # [B, N]
        top_idx  = activity.topk(M, dim=-1).indices                 # [B, M]

        # Gather beam activations
        Z_beam = torch.gather(                                      # [B, M, D]
            Z, 1, top_idx.unsqueeze(-1).expand(-1, -1, D_)
        )

        # Attention: each neuron attends to beam neurons
        # score[b,n,m] = dot(Z[b,n], Z_beam[b,m]) / tau
        score = torch.einsum("bnd,bmd->bnm", Z, Z_beam) / self.tau  # [B, N, M]
        w     = F.softmax(score, dim=-1)                             # [B, N, M]

        # Weighted mix of beam signals
        Z_global = torch.einsum("bnm,bmd->bnd", w, Z_beam)          # [B, N, D]
        return Z_global

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base      = self.m.base
        Z         = base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = base.conn_hh
        N_h       = base.N_hidden

        # Static AH suppression
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha * pos_sim.clamp(min=0)
                   ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)

        for t in range(base.K_iter):
            Z_fwd  = F.relu(Z - theta_pos)
            Z_nb   = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)               # local routing

            # Beam broadcast at designated steps
            if t % self.beam_every == 0:
                Z_global = self._beam_broadcast(Z_fwd)
                Z_local  = Z_struct + self.lambda_beam * Z_global
            else:
                Z_local = Z_struct

            # Reflection accumulator
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_local + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return base._readout(Z)


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    beam_size: int
    lambda_beam: float
    beam_every: int
    use_beam: bool = True


CONFIGS = [
    Config("Ref", "Ref  standard AH, no beam (D=32 baseline)", 0, 0.0, 1, use_beam=False),
    Config("A",   "A    beam M=4, every step, λ=0.1",           4, 0.1, 1),
    Config("B",   "B    beam M=8, every step, λ=0.1",           8, 0.1, 1),
    Config("C",   "C    beam M=16, every step, λ=0.1",         16, 0.1, 1),
    Config("D",   "D    beam M=8, every 3rd step, λ=0.1",       8, 0.1, 3),
    Config("E",   "E    beam M=8, every step, λ=0.3 (stronger)", 8, 0.3, 1),
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
    if not cfg.use_beam:
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_AH_BeamBroadcast(
        resonant, alpha_ahebb=ALPHA_AHEBB,
        beam_size=cfg.beam_size, lambda_beam=cfg.lambda_beam,
        beam_every=cfg.beam_every, tau=1.0,
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
    print(f"Step 130 — Beam-as-Global-Broadcast")
    print(f"N={N}  D={D}  K_iter={K_ITER}  K_hh=4  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    print("Configs:")
    for c in CONFIGS:
        print(f"  {c.key:4s}  M={c.beam_size:2d}  λ={c.lambda_beam:.1f}  "
              f"every={c.beam_every}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step130_beam_broadcast.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  beam_size={cfg.beam_size}  "
              f"lambda={cfg.lambda_beam}  every={cfg.beam_every}")
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

        results[cfg.key] = {
            "N": N, "D": D, "K_iter": K_ITER,
            "beam_size": cfg.beam_size,
            "lambda_beam": cfg.lambda_beam,
            "beam_every": cfg.beam_every,
            "alpha_ahebb": ALPHA_AHEBB,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params,
            "label": cfg.label,
        }

        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs_ref = top1_best - ref_best if ref_best > 0 else 0

        print(f"\n  top1_best={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary table
    print(f"\n{'='*70}")
    print(f"STEP 130 SUMMARY (D={D})")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Config':<6}  {'M':>3}  {'λ':>4}  {'every':>5}  "
          f"{'top1':>7}  {'vs_Ref':>8}")
    print(f"{'─'*50}")
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"{key:<6}  {r['beam_size']:>3}  {r['lambda_beam']:>4.1f}  "
              f"{r['beam_every']:>5}  {r['top1_best']:.4f}  {vs:>+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
