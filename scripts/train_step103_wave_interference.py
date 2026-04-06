"""Step 103: Wave-optical message passing with interference.

MOTIVATION
==========
Physics of light through a lattice: signals are WAVES that accumulate
phase shifts proportional to distance traveled. When multiple paths
converge, waves INTERFERE — constructive if phases align, destructive
if not. Information is encoded in the interference pattern.

This is fundamentally different from all prior routing:
- NOT a gate (gate-death impossible — waves don't gate, they interfere)
- NOT a filter (step102 polar filter is static; this is dynamic per-path)
- Phase shift is a GEOMETRIC operation (rotation on S^{D-1}), not learned

WAVE MODEL
==========
Each neuron's state Z[h] ∈ C^{D/2} is a complex wave (real + imaginary pairs).
Signal from j→h accumulates phase shift based on W_pos distance:

  d_jh    = ||W_pos[j] - W_pos[h]||         (Euclidean distance)
  Δφ_k    = freq_k × d_jh                    (phase shift per frequency)
  Z_shifted = Z[j] × exp(i × Δφ_k)          (rotation in complex plane)

Multiple signals arrive and INTERFERE:
  Z_agg[h] = Σ_{j∈neighbors} Z_shifted[j]    (coherent sum → interference)

Intensity (magnitude) of the aggregate encodes which paths constructively
interfered. Neurons at positions where many paths align get strong signal.

FOURIER SPECTRUM AT LOW D
=========================
At D=8 (4 complex channels): each channel k has frequency freq_k.
Interference pattern is a 4-dim Fourier spectrum of the graph's geometry.
Low D → fewer frequencies → coarser interference → more interpretable.

Key: freq_k are FIXED (log-spaced from 1 to D/2). Not learned.
The network learns W_pos (which controls distances → phase shifts)
and the readout (which interprets the interference pattern).

INTENSITY DECAY
===============
Signal decays as it travels: amplitude *= exp(-α × d_jh).
This is natural — longer paths contribute less. Combined with phase:
  Z_shifted = Z[j] × exp(i × Δφ) × exp(-α × d_jh)
            = Z[j] × exp((-α + i×freq) × d_jh)

This is damped wave propagation — exactly light in a medium.

IMPLEMENTATION NOTE
===================
We represent complex numbers as real pairs: Z[h] ∈ R^D where
Z[h, 2k] = Re(channel k), Z[h, 2k+1] = Im(channel k).
Phase rotation: [Re, Im] × [cos Δφ, -sin Δφ; sin Δφ, cos Δφ].

CONFIGS (N=1024, D=16, K_iter=8, 50%/75ep)
============================================
D=16 → 8 complex channels (8 frequency bands).
K_hh=4 (current default). AH=1.0 wpos.

  Ref : standard AH routing, D=16 (no wave mechanics)
  A   : wave interference, α=0.1 (mild decay)
  B   : wave interference, α=0.5 (moderate decay)
  C   : wave interference, α=0.1, K_iter=12 (more propagation steps)
  D   : wave interference + AH, α=0.1 (compound: AH suppresses + waves interfere)
  E   : wave interference, α=0.1, D=32 (16 frequency channels)

ABLATION:
  A vs Ref : does wave interference help at all?
  B vs A   : decay sensitivity
  C vs A   : do more K_iter steps = richer interference patterns?
  D vs A   : does AH compound with wave mechanics?
  E vs A   : does higher frequency resolution (more channels) help?

To reproduce:
    python -u scripts/train_step103_wave_interference.py --device mps
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
from src.training.experiment_config   import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 75; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10
ALPHA_REFLECT = 0.5; ALPHA_AHEBB = 1.0


class SGNNET_WaveInterference(nn.Module):
    """Wave-optical message passing with interference on graph lattice.

    Each neuron state Z ∈ R^D is interpreted as D/2 complex channels.
    Signal from j→h gets phase-rotated by freq_k × distance(j,h).
    Aggregate = coherent sum (interference).
    """

    def __init__(self, base: SGNNET_SmallWorld, alpha_decay: float,
                 use_ah: bool = False, alpha_ahebb: float = 1.0):
        super().__init__()
        self.base        = base
        self.alpha_decay = alpha_decay
        self.use_ah      = use_ah
        self.alpha_ahebb = alpha_ahebb
        self.D           = base.D
        self.n_channels  = self.D // 2

        # Fixed frequency spectrum: log-spaced from 1 to n_channels
        freqs = torch.logspace(0, math.log10(self.n_channels), self.n_channels)
        self.register_buffer("freqs", freqs)                       # [n_channels]

        # Threshold
        self.theta = nn.Parameter(torch.full((base.N_hidden,), 0.1))

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.base.W_pos[:self.base.N_hidden]

    def _complex_rotate(self, Z: torch.Tensor, phase_shifts: torch.Tensor,
                        decay: torch.Tensor) -> torch.Tensor:
        """Apply phase rotation and decay to complex-valued Z.

        Z: [B, N, K_hh, D] — D = 2*n_channels (real, imag pairs)
        phase_shifts: [N, K_hh, n_channels] — per-edge per-channel phase
        decay: [N, K_hh, 1] — amplitude decay per edge

        Returns: rotated Z [B, N, K_hh, D]
        """
        B, N_h, K, D = Z.shape
        nc = self.n_channels
        # Reshape to [B, N, K, nc, 2] for real/imag
        Z_c = Z.view(B, N_h, K, nc, 2)

        cos_p = torch.cos(phase_shifts).unsqueeze(0)               # [1,N,K,nc]
        sin_p = torch.sin(phase_shifts).unsqueeze(0)               # [1,N,K,nc]
        decay_b = decay.unsqueeze(0).unsqueeze(-1)                 # [1,N,K,1,1]

        # Complex rotation: (a+bi)(cos+isin) = (a cos - b sin) + (a sin + b cos)i
        Re = Z_c[..., 0]                                           # [B,N,K,nc]
        Im = Z_c[..., 1]
        Re_rot = Re * cos_p - Im * sin_p
        Im_rot = Re * sin_p + Im * cos_p

        Z_rot = torch.stack([Re_rot, Im_rot], dim=-1)             # [B,N,K,nc,2]
        Z_rot = Z_rot * decay_b                                    # apply decay
        return Z_rot.view(B, N_h, K, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.base._seed(x)                                     # [B, N, D]
        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh = self.base.conn_hh
        N_h     = self.base.N_hidden

        # Precompute inter-neuron distances and phase shifts
        W_pos_h = self.base.W_pos[:N_h]                           # [N, D_pos]
        # Note: W_pos may have different D than Z's D. Use L2 distance.
        pos_self = W_pos_h.unsqueeze(1)                            # [N, 1, D_pos]
        pos_nb   = W_pos_h[conn_hh]                                # [N, K_hh, D_pos]
        dist     = (pos_self - pos_nb).norm(dim=-1)                # [N, K_hh]

        # Phase shifts: freq_k × distance → [N, K_hh, n_channels]
        phase_shifts = dist.unsqueeze(-1) * self.freqs             # broadcast

        # Decay: exp(-α × d) → [N, K_hh, 1]
        decay = torch.exp(-self.alpha_decay * dist).unsqueeze(-1)

        # AH suppression (optional)
        if self.use_ah:
            W_n = F.normalize(W_pos_h, dim=-1)
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
            ah_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                   ).unsqueeze(0).unsqueeze(-1)                    # [1,N,K,1]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]                          # [B,N,K,D]

            # Wave: rotate phases and decay
            Z_wave = self._complex_rotate(Z_nb, phase_shifts, decay)

            if self.use_ah:
                Z_struct = (Z_wave * ah_w).sum(dim=2)
            else:
                Z_struct = Z_wave.sum(dim=2)                       # interference!

            Z_remainder = Z_fwd - Z
            Z_reflected = ALPHA_REFLECT * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.base._readout(Z)


@dataclass
class Config:
    key: str; label: str; D: int; wave: bool; alpha_decay: float
    K_iter: int; use_ah: bool


CONFIGS = [
    Config("Ref", "Ref  standard AH, D=16 (no wave)", 16, False, 0.0, 8, True),
    Config("A",   "A    wave α=0.1, D=16",            16, True,  0.1, 8, False),
    Config("B",   "B    wave α=0.5, D=16",            16, True,  0.5, 8, False),
    Config("C",   "C    wave α=0.1, D=16, K_iter=12", 16, True,  0.1, 12, False),
    Config("D",   "D    wave+AH α=0.1, D=16",         16, True,  0.1, 8, True),
    Config("E",   "E    wave α=0.1, D=32 (16 freq)",  32, True,  0.1, 8, False),
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
        K_local=2, K_random=2, K_in=tk["K_in"], K_iter=cfg.K_iter,
        n_groups=tk["n_groups"], norm_mode="l2", D=cfg.D, encoding_mode="fourier",
    )
    if not cfg.wave:
        resonant = SGNNET_Resonant(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0,
        )
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_WaveInterference(
        base, alpha_decay=cfg.alpha_decay,
        use_ah=cfg.use_ah, alpha_ahebb=ALPHA_AHEBB,
    )


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run(cfg, model, meta):
    print(f"\n{'='*70}\n{cfg.label}\n{'='*70}")
    tr, va = get_loaders()
    tk = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0 = time.time(); history = trainer.train(n_epochs=EPOCHS); elapsed = time.time() - t0
    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best = max(top1_hist); best_ep = int(np.argmax(top1_hist)) + 1
    result = {
        "label": cfg.label, "D": cfg.D, "wave": cfg.wave,
        "alpha_decay": cfg.alpha_decay, "K_iter": cfg.K_iter, "use_ah": cfg.use_ah,
        "top1_best": best, "top1_last": history[-1].get("val_top1", 0.0),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "params": count_params(model), "top1_history": top1_hist,
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1={best:.4f}  ep={best_ep}/{len(history)}  D={cfg.D}  t={elapsed:.0f}s")
    return result


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  N={N}  Data: 50%")
    print(f"Step 103: Wave-optical message passing with interference\n")
    for c in CONFIGS:
        print(f"  {c.key:4s}  D={c.D:2d}  wave={c.wave}  α={c.alpha_decay}  "
              f"K={c.K_iter}  AH={c.use_ah}")
    print()
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")
    results = {}
    out_path = ROOT / "results" / "train_step103_wave_interference.json"
    for i, cfg in enumerate(CONFIGS):
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        meta = {"N": N, "D": cfg.D, "K_iter": cfg.K_iter, "wave": cfg.wave,
                "alpha_decay": cfg.alpha_decay, "use_ah": cfg.use_ah, "data_frac": 0.5}
        results[cfg.key] = run(cfg, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")
    print(f"\n{'='*70}\nSTEP 103 COMPLETE\n")
    for c in CONFIGS:
        if c.key in results:
            r = results[c.key]
            print(f"  {c.key:4s}  D={c.D:2d}  {r['top1_best']:.4f}  {c.label}")
    w = max(results, key=lambda k: results[k]["top1_best"])
    print(f"\n  Winner: {w}")
