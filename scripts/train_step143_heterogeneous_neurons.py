"""Step 143: Heterogeneous neuron subpopulations.

MOTIVATION
==========
All neurons in SGNNET share the same theta threshold parameter. Biological
neural circuits have distinct cell types: interneurons vs. pyramidal cells,
fast-spiking vs. regular-spiking, etc. Heterogeneous populations may improve
representational diversity and routing efficiency.

Four mechanisms tested:

A) 2-population split: "aggregators" (low θ=0.05, pass more signal) and
   "filters" (high θ=0.2, selective), 50/50 split. Tests whether routing
   benefits from mixed selectivity.

B) 2-population weighted edges: "relay" neurons (edge weight ×2 = K_hh_eff=16)
   vs. "specialist" neurons (edge weight ×0.5 = K_hh_eff=4). Same K_hh
   topology, different effective fan-in. Relay neurons broadcast widely;
   specialists integrate selectively.

C) 4-population frequency-band grouping: neurons assigned to Fourier frequency
   bands via their W_pos angle, each band gets its own learned scalar θ.
   Hypothesis: different spatial frequencies benefit from different thresholds.

D) Fully per-neuron theta: θ ∈ R^N (no sharing whatsoever). Simplest
   heterogeneity — upper bound on how much θ-diversity can help.

CONFIGS (N=1024, D=16, K_hh=8, K_in=25, AH=1.0, 50%/75ep)
============================================================
  Ref : Standard homogeneous AH (shared scalar θ)
  A   : 2-pop aggregator/filter (fixed θ 0.05/0.20 init, learnable)
  B   : 2-pop relay/specialist (edge weight scale 2.0/0.5)
  C   : 4-pop frequency-band (band-specific learnable θ, 4 params)
  D   : Per-neuron θ ∈ R^N (N=1024 params for θ)

To reproduce:
    python -u scripts/train_step143_heterogeneous_neurons.py --device mps
    python -u scripts/train_step143_heterogeneous_neurons.py --device mps --epochs 20
"""
from __future__ import annotations

import argparse
import json
import math
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
N = 1024; N_IN = 25088; N_OUT = 10; D = 16; K_IN = 25; K_HH = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0


# ---------------------------------------------------------------------------
# Heterogeneous AH model
# ---------------------------------------------------------------------------

class SGNNET_HeteroAH(nn.Module):
    """Anti-Hebbian routing with heterogeneous neuron populations.

    het_mode controls which heterogeneity is applied:
      "homogeneous"  — shared scalar theta (Ref baseline)
      "twopop_theta" — Config A: two theta values, aggregator vs. filter
      "twopop_weight"— Config B: two edge-weight scales, relay vs. specialist
      "freqband"     — Config C: 4 frequency-band theta params
      "perneuron"    — Config D: full N-length theta vector
    """

    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float = 1.0,
                 het_mode: str = "homogeneous",
                 theta_agg: float = 0.05, theta_filt: float = 0.20,
                 relay_scale: float = 2.0, spec_scale: float = 0.5,
                 n_freqbands: int = 4):
        super().__init__()
        self.m            = resonant
        self.alpha_ahebb  = alpha_ahebb
        self.het_mode     = het_mode

        N_h = resonant.base.N_hidden
        D_  = resonant.base.D

        if het_mode == "homogeneous":
            # Ref: use resonant.theta as-is (single scalar)
            pass

        elif het_mode == "twopop_theta":
            # Config A: split neurons 50/50 into aggregators and filters.
            # Each group gets its own learnable scalar theta.
            half = N_h // 2
            pop_mask = torch.zeros(N_h, dtype=torch.long)
            pop_mask[half:] = 1   # pop 0 = aggregator, pop 1 = filter
            self.register_buffer("pop_mask", pop_mask)
            # Two learned thresholds: [2]
            self.theta_pop = nn.Parameter(torch.tensor([theta_agg, theta_filt]))

        elif het_mode == "twopop_weight":
            # Config B: split neurons 50/50 into relay and specialist.
            # Relay neurons: sum neighbours × 2.0 (amplify broadcast)
            # Specialist neurons: sum neighbours × 0.5 (selective)
            half = N_h // 2
            pop_mask = torch.zeros(N_h, dtype=torch.long)
            pop_mask[half:] = 1   # pop 0 = relay, pop 1 = specialist
            self.register_buffer("pop_mask", pop_mask)
            # Learnable edge-weight scales per population: [2]
            self.edge_scale = nn.Parameter(torch.tensor([relay_scale, spec_scale]))

        elif het_mode == "freqband":
            # Config C: 4 frequency-band groups, one learnable theta each.
            # Band assignment: based on index in range(N) split into 4 equal parts.
            # (Could use W_pos angle, but W_pos is not fixed at init time —
            #  so we use simple index-based assignment instead, then let the
            #  learned W_pos organize neurons. Each band's theta is independent.)
            band_size = N_h // n_freqbands
            band_idx = torch.zeros(N_h, dtype=torch.long)
            for b in range(n_freqbands):
                lo = b * band_size
                hi = lo + band_size if b < n_freqbands - 1 else N_h
                band_idx[lo:hi] = b
            self.register_buffer("band_idx", band_idx)
            self.n_freqbands = n_freqbands
            # One learnable theta per band: [n_freqbands]
            self.theta_band = nn.Parameter(torch.full((n_freqbands,), 0.1))

        elif het_mode == "perneuron":
            # Config D: fully per-neuron theta
            self.theta_all = nn.Parameter(torch.full((N_h,), 0.1))

        else:
            raise ValueError(f"Unknown het_mode: {het_mode}")

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _get_theta_pos(self) -> torch.Tensor:
        """Return per-neuron theta [N_h] (positive), then unsqueeze for broadcast."""
        N_h = self.m.base.N_hidden

        if self.het_mode == "homogeneous":
            # Shared scalar from resonant.theta
            return self.m.theta.abs().unsqueeze(0).unsqueeze(-1)  # [1, 1, 1]

        elif self.het_mode in ("twopop_theta", "freqband"):
            if self.het_mode == "twopop_theta":
                theta_vec = self.theta_pop.abs()[self.pop_mask]  # [N_h]
            else:
                theta_vec = self.theta_band.abs()[self.band_idx]  # [N_h]
            return theta_vec.unsqueeze(0).unsqueeze(-1)  # [1, N_h, 1]

        elif self.het_mode == "perneuron":
            theta_vec = self.theta_all.abs()  # [N_h]
            return theta_vec.unsqueeze(0).unsqueeze(-1)  # [1, N_h, 1]

        else:
            # twopop_weight uses the shared theta
            return self.m.theta.abs().unsqueeze(0).unsqueeze(-1)

    def _get_edge_scale(self) -> torch.Tensor | None:
        """Return per-neuron edge scale [1, N_h, 1, 1] or None."""
        if self.het_mode == "twopop_weight":
            scale_vec = self.edge_scale.abs()[self.pop_mask]  # [N_h]
            return scale_vec.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, N_h, 1, 1]
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        theta_pos = self._get_theta_pos()
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden
        edge_scale = self._get_edge_scale()

        # Pre-compute static AH suppression weights (wpos variant)
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)   # [1, N, K_hh, 1]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z_fwd  = F.relu(Z - theta_pos)
            Z_nb   = Z_fwd[:, conn_hh, :]            # [B, N, K_hh, D]

            # Apply per-neuron edge scale if present
            if edge_scale is not None:
                Z_nb = Z_nb * edge_scale

            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key:      str
    label:    str
    het_mode: str

CONFIGS = [
    Config("Ref", "Ref  homogeneous AH (shared θ, baseline)",  "homogeneous"),
    Config("A",   "A    2-pop aggregator/filter (θ=0.05/0.20)", "twopop_theta"),
    Config("B",   "B    2-pop relay/specialist (edge×2/×0.5)",  "twopop_weight"),
    Config("C",   "C    4-band frequency θ (4 learnable vals)",  "freqband"),
    Config("D",   "D    per-neuron θ ∈ R^N (N=1024 params)",    "perneuron"),
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
    K_random = max(1, K_HH // 4)
    K_local  = K_HH - K_random
    n_groups = max(8, N // 8)

    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=8,
        K_local=K_local, K_random=K_random, n_groups=n_groups,
        norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )

    if cfg.het_mode == "homogeneous":
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")

    return SGNNET_HeteroAH(resonant, alpha_ahebb=ALPHA_AHEBB,
                            het_mode=cfg.het_mode)


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 143 — Heterogeneous Neuron Subpopulations")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_in={K_IN}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    for c in CONFIGS:
        print(f"  {c.key:4s}  mode={c.het_mode:15s}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step143_heterogeneous_neurons.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  het_mode={cfg.het_mode}")
        print(f"{'─'*60}")

        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(
            model=model, train_loader=get_loaders()[0],
            val_loader=get_loaders()[1], device=DEVICE, **kw,
        )
        history = trainer.train(n_epochs=EPOCHS)
        elapsed = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1

        results[cfg.key] = {
            "N": N, "D": D, "K_hh": K_HH, "K_in": K_IN,
            "het_mode": cfg.het_mode,
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
        print(f"\n  top1={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min  params={n_params:,}")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 143 SUMMARY — Heterogeneous Neurons")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Key':4s}  {'het_mode':17s}  {'params':>8}  {'top1':>7}  {'vs_Ref':>8}")
    print(f"{'─'*55}")
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"{key:4s}  {r['het_mode']:17s}  {r['n_params']:>8}  "
              f"{r['top1_best']:.4f}  {vs:>+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
