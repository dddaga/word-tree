"""Step 104: Compound wave-polar architecture with stabilization.

MOTIVATION
==========
step102 tests phase polarization (Malus's law filter + alternating training).
step103 tests wave interference (distance-based phase shift + decay).
This experiment COMPOUNDS the winners from both + adds stabilization.

ARCHITECTURE: THREE-LAYER PHYSICS
==================================
Layer 1 — POLARIZATION (from step102):
  Filter signal by phase alignment: cos²(angle(W_phase[h], W_phase[j]))
  → Controls WHAT passes through each edge (selective filtering)

Layer 2 — WAVE PROPAGATION (from step103):
  Phase-shift by distance: Z_shifted = Z[j] × exp(i×freq×d - α×d)
  → Controls HOW signal transforms during travel (interference)

Layer 3 — INTERFERENCE SUM:
  Aggregate = Σ(polarized × shifted signals) → coherent interference
  → The computational primitive: information encoded in interference pattern

Combined edge weight:
  w(j→h) = cos²(phase_align) × exp(-α×d) × exp(i×freq×d)

STABILIZATION TECHNIQUES
=========================
1. GCNII initial residual: h_t = (1-a)*route(h_{t-1}) + a*h_0
   → Guaranteed signal floor at every K_iter step. Proven at 64 layers.

2. Phase accumulation cap: Δφ = clamp(freq×d, -π, π)
   → Prevents random-phase regime after many K_iter steps.

3. Per-channel normalization: normalize each complex channel pair
   independently, not global L2. Preserves relative channel magnitudes.

4. Warm-start then alternate: first N_warm epochs train all params
   simultaneously (establish baseline), then alternate W_pos/W_phase.
   → Prevents early oscillation when both are at random init.

5. Learnable decay per neuron: α_h = sigmoid(α_raw_h) × α_max
   → Each neuron learns its own absorption rate.

6. DropMessage (from step96): random message dropping during K_iter.
   → Anti-over-smoothing, proven effective.

CONFIGS (N=1024, D=16, K_iter=8, 50%/75ep)
============================================
D=16 → 8 complex channels. Small for fast iteration + visible effects.

  Ref : standard AH routing, D=16 (no physics)
  A   : compound (polar + wave) — no stabilization (raw baseline)
  B   : compound + GCNII residual (a=0.1)
  C   : compound + GCNII residual + phase cap + warm-start(15ep)
  D   : compound + ALL stabilization (GCNII + cap + warm + drop + learnable α)
  E   : compound + ALL stab + AH wpos (full kitchen sink — does AH help?)
  F   : compound + ALL stab, D=32 (16 channels, higher resolution)

ABLATION:
  A vs Ref : does compound physics help at all (even unstabilized)?
  B vs A   : does GCNII residual stabilize?
  C vs B   : does phase cap + warm-start add further stability?
  D vs C   : does learnable decay + drop add value?
  E vs D   : does AH compound with wave-polar physics?
  F vs D   : does higher frequency resolution help?

To reproduce:
    python -u scripts/train_step104_compound_wave_polar.py --device mps
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


class SGNNET_WavePolar(nn.Module):
    """Compound wave-polar architecture with stabilization suite.

    Per-edge computation:
      polar    = cos²(angle(W_phase[h], W_phase[j]))
      decay    = exp(-α_h × d(h,j))
      Δφ_k     = clamp(freq_k × d(h,j), -π, π)     [if phase_cap]
      Z_wave   = complex_rotate(Z[j], Δφ) × decay × polar
      Z_agg[h] = Σ_j Z_wave[j]

    Stabilization:
      residual : Z_new = (1-a)*Z_agg + a*Z_0
      drop     : random message dropping during training
      warm     : first N_warm epochs train all params; then alternate
      per_ch   : normalize each complex channel pair independently
    """

    def __init__(self, base: SGNNET_SmallWorld, *,
                 use_ah: bool = False,
                 residual_alpha: float = 0.0,
                 phase_cap: bool = False,
                 warm_epochs: int = 0,
                 alt_block: int = 1,
                 drop_rate: float = 0.0,
                 learnable_decay: bool = False,
                 base_decay: float = 0.1):
        super().__init__()
        self.base = base
        self.D = base.D
        self.n_ch = self.D // 2
        self.use_ah = use_ah
        self.res_alpha = residual_alpha
        self.phase_cap = phase_cap
        self.warm_epochs = warm_epochs
        self.alt_block = alt_block
        self.drop_rate = drop_rate

        N_h = base.N_hidden
        # Phase polarization embeddings
        self.W_phase = nn.Parameter(F.normalize(torch.randn(N_h, self.D), dim=-1))

        # Threshold
        self.theta = nn.Parameter(torch.full((N_h,), 0.1))

        # Fixed frequencies: log-spaced
        freqs = torch.logspace(0, math.log10(max(2, self.n_ch)), self.n_ch)
        self.register_buffer("freqs", freqs)

        # Decay: learnable per-neuron or fixed
        if learnable_decay:
            # sigmoid(raw) × 2.0 → range [0, 2.0]
            init_raw = torch.full((N_h,), math.log(base_decay / (2.0 - base_decay)))
            self.decay_raw = nn.Parameter(init_raw)
            self._learnable_decay = True
        else:
            self.register_buffer("decay_fixed", torch.tensor(base_decay))
            self._learnable_decay = False

        # AH suppression
        self.alpha_ahebb = ALPHA_AHEBB if use_ah else 0.0

        self._epoch = 0

    @property
    def W_pos(self): return self.base.W_pos

    def tick_epoch(self):
        self._epoch += 1

    def _get_decay(self) -> torch.Tensor:
        """Per-neuron decay coefficient [N_h]."""
        if self._learnable_decay:
            return torch.sigmoid(self.decay_raw) * 2.0
        return self.decay_fixed.expand(self.base.N_hidden)

    def _is_warmup(self) -> bool:
        return self._epoch < self.warm_epochs

    def set_alternating_grad(self):
        """Set requires_grad for alternating training."""
        if self._is_warmup():
            # Warm-start: train everything
            self.base.W_pos.requires_grad_(True)
            self.W_phase.requires_grad_(True)
            return
        block = ((self._epoch - self.warm_epochs) // self.alt_block) % 2
        self.base.W_pos.requires_grad_(block == 0)
        self.W_phase.requires_grad_(block == 1)
        self.theta.requires_grad_(True)
        if self._learnable_decay:
            self.decay_raw.requires_grad_(True)

    def _per_channel_normalize(self, Z: torch.Tensor) -> torch.Tensor:
        """Normalize each complex channel pair independently.
        Z: [B, N, D] where D = 2*n_ch. Pairs: (0,1), (2,3), ...
        """
        B, N_h, D = Z.shape
        Z_c = Z.view(B, N_h, self.n_ch, 2)
        mag = Z_c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        Z_c = Z_c / mag
        return Z_c.view(B, N_h, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.base._seed(x)                                     # [B, N, D]
        h_0 = Z.clone() if self.res_alpha > 0 else None
        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh = self.base.conn_hh                                # [N, K_hh]
        N_h = self.base.N_hidden

        # Precompute distances
        W_pos_h = self.base.W_pos[:N_h]
        dist = (W_pos_h.unsqueeze(1) - W_pos_h[conn_hh]).norm(dim=-1)  # [N, K_hh]

        # Phase shifts: [N, K_hh, n_ch]
        phase_shifts = dist.unsqueeze(-1) * self.freqs
        if self.phase_cap:
            phase_shifts = phase_shifts.clamp(-math.pi, math.pi)

        # Decay per edge: [N, K_hh]
        decay_coeff = self._get_decay()
        edge_decay = torch.exp(-decay_coeff.unsqueeze(1) * dist)   # [N, K_hh]

        # Polarization: cos²(angle(phase_h, phase_j)) → [N, K_hh]
        Ph = F.normalize(self.W_phase, dim=-1)
        polar = (Ph.unsqueeze(1) * Ph[conn_hh]).sum(-1) ** 2       # [N, K_hh]

        # Combined static weight: polar × decay → [1, N, K_hh, 1]
        static_w = (polar * edge_decay).unsqueeze(0).unsqueeze(-1)

        # AH suppression (optional)
        if self.use_ah:
            W_n = F.normalize(W_pos_h, dim=-1)
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
            ah_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0))
            static_w = static_w * ah_w.unsqueeze(0).unsqueeze(-1)

        # Precompute cos/sin for complex rotation
        cos_p = torch.cos(phase_shifts)                            # [N, K, nc]
        sin_p = torch.sin(phase_shifts)                            # [N, K, nc]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]                          # [B,N,K,D]

            # DropMessage
            if self.training and self.drop_rate > 0:
                mask = torch.bernoulli(
                    torch.full(Z_nb.shape[:3], 1.0 - self.drop_rate,
                               device=Z_nb.device)
                ).unsqueeze(-1)
                Z_nb = Z_nb * mask / (1.0 - self.drop_rate)

            # Complex rotation
            B_sz, N_sz, K_sz, D_sz = Z_nb.shape
            nc = self.n_ch
            Z_c = Z_nb.view(B_sz, N_sz, K_sz, nc, 2)
            Re, Im = Z_c[..., 0], Z_c[..., 1]
            cos_e = cos_p.unsqueeze(0)
            sin_e = sin_p.unsqueeze(0)
            Re_rot = Re * cos_e - Im * sin_e
            Im_rot = Re * sin_e + Im * cos_e
            Z_rot = torch.stack([Re_rot, Im_rot], dim=-1).view(B_sz, N_sz, K_sz, D_sz)

            # Apply static weights (polar × decay × optional AH) and sum
            Z_struct = (Z_rot * static_w).sum(dim=2)              # [B, N, D]

            # Reflection
            Z_remainder = Z_fwd - Z
            Z_reflected = ALPHA_REFLECT * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected

            # GCNII initial residual
            if self.res_alpha > 0 and h_0 is not None:
                Z_new = (1.0 - self.res_alpha) * Z_new + self.res_alpha * h_0

            # Per-channel normalization
            Z = self._per_channel_normalize(Z_new.clamp(-10, 10))

        return self.base._readout(Z)


@dataclass
class Config:
    key: str; label: str; D: int
    compound: bool      # True = wave+polar; False = standard AH Ref
    use_ah: bool
    residual: float     # GCNII alpha (0=off)
    phase_cap: bool
    warm: int           # warm-start epochs before alternating
    drop: float
    learn_decay: bool


CONFIGS = [
    Config("Ref", "Ref  standard AH D=16 (no physics)", 16,
           False, True, 0.0, False, 0, 0.0, False),
    Config("A", "A    compound+AH raw (no stabilization)", 16,
           True, True, 0.0, False, 0, 0.0, False),
    Config("B", "B    compound+AH + GCNII(a=0.1)", 16,
           True, True, 0.1, False, 0, 0.0, False),
    Config("C", "C    compound+AH + GCNII + cap + warm(15)", 16,
           True, True, 0.1, True, 15, 0.0, False),
    Config("D", "D    compound+AH + ALL stab", 16,
           True, True, 0.1, True, 15, 0.2, True),
    Config("E", "E    compound NO AH + ALL stab (ablation)", 16,
           True, False, 0.1, True, 15, 0.2, True),
    Config("F", "F    compound+AH + ALL stab, D=32", 32,
           True, True, 0.1, True, 15, 0.2, True),
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
        K_local=2, K_random=2, K_in=tk["K_in"], K_iter=8,
        n_groups=tk["n_groups"], norm_mode="l2", D=cfg.D, encoding_mode="fourier",
    )
    if not cfg.compound:
        resonant = SGNNET_Resonant(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0,
        )
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_WavePolar(
        base, use_ah=cfg.use_ah, residual_alpha=cfg.residual,
        phase_cap=cfg.phase_cap, warm_epochs=cfg.warm,
        drop_rate=cfg.drop, learnable_decay=cfg.learn_decay,
    )


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run(cfg, model, meta):
    print(f"\n{'='*70}\n{cfg.label}\n{'='*70}")
    tr, va = get_loaders()
    tk = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)

    if cfg.compound:
        history = []
        for ep in range(EPOCHS):
            model.set_alternating_grad()
            ep_hist = trainer.train(n_epochs=1)
            history.extend(ep_hist)
            if hasattr(model, "tick_epoch"):
                model.tick_epoch()
    else:
        history = trainer.train(n_epochs=EPOCHS)

    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best = max(top1_hist); best_ep = int(np.argmax(top1_hist)) + 1
    result = {
        "label": cfg.label, "D": cfg.D, "compound": cfg.compound,
        "use_ah": cfg.use_ah, "residual": cfg.residual,
        "phase_cap": cfg.phase_cap, "warm": cfg.warm,
        "drop": cfg.drop, "learn_decay": cfg.learn_decay,
        "top1_best": best, "top1_last": history[-1].get("val_top1", 0.0),
        "best_epoch": best_ep, "epochs_run": len(history),
        "params": count_params(model), "top1_history": top1_hist,
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1={best:.4f}  ep={best_ep}/{len(history)}  D={cfg.D}")
    return result


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  N={N}  Data: 50%")
    print(f"Step 104: Compound wave-polar with stabilization suite\n")
    for c in CONFIGS:
        stabs = []
        if c.residual > 0: stabs.append(f"res={c.residual}")
        if c.phase_cap: stabs.append("cap")
        if c.warm > 0: stabs.append(f"warm={c.warm}")
        if c.drop > 0: stabs.append(f"drop={c.drop}")
        if c.learn_decay: stabs.append("learnα")
        if c.use_ah: stabs.append("AH")
        stab_str = "+".join(stabs) if stabs else "none"
        print(f"  {c.key:4s}  D={c.D:2d}  stab=[{stab_str}]")
    print()
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")
    results = {}
    out_path = ROOT / "results" / "train_step104_compound_wave_polar.json"
    for i, cfg in enumerate(CONFIGS):
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        meta = {"N": N, "D": cfg.D, "compound": cfg.compound,
                "stabilization": {"residual": cfg.residual, "cap": cfg.phase_cap,
                    "warm": cfg.warm, "drop": cfg.drop, "learn_decay": cfg.learn_decay},
                "data_frac": 0.5}
        results[cfg.key] = run(cfg, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")
    print(f"\n{'='*70}\nSTEP 104 COMPLETE\n")
    for c in CONFIGS:
        if c.key in results:
            r = results[c.key]
            print(f"  {c.key:4s}  D={c.D:2d}  {r['top1_best']:.4f}  {c.label}")
    w = max(results, key=lambda k: results[k]["top1_best"])
    print(f"\n  Winner: {w}")
    # Stabilization effectiveness
    if "A" in results and "D" in results:
        raw = results["A"]["top1_best"]
        stab = results["D"]["top1_best"]
        print(f"  Stabilization effect: raw={raw:.4f} → stabilized={stab:.4f} "
              f"({stab-raw:+.4f})")
