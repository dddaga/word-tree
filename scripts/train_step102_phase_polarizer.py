"""Step 102: Phase-polarized neurons with alternating training.

MOTIVATION
==========
Physics-inspired architecture where each neuron has two properties:
  - W_pos  ∈ S^{D-1}: spatial position (determines neighborhood)
  - W_phase ∈ S^{D-1}: polarization axis (filters signal flow)

THREE MECHANISMS:
1. Phase polarization (Malus's law): signal from j→h is filtered by
   cos²(angle(W_phase[h], W_phase[j])). Aligned phases → full signal.
   Orthogonal phases → zero signal. NOT a multiplicative gate over K_iter
   — it's a per-edge filter like AH suppression.

2. Alternating training: even epochs train W_pos (frozen W_phase),
   odd epochs train W_phase (frozen W_pos). This avoids the temporal
   mismatch that killed step83 — each component gets a stability period.

3. Pauli exclusion penalty: repels neurons whose combined (pos, phase)
   state is too similar. Extends AH (position diversity only) to full
   state diversity. Two neurons CAN share position if phases differ.

DESIGN INSIGHTS FROM PRIOR FAILURES:
- NOT multiplicative gating (gate-death impossible)
- NOT simultaneous position+phase learning (temporal mismatch)
- Phase filtering is static within a forward pass (like AH wpos)
- Pauli exclusion is a REPULSIVE loss, not a routing mechanism

D=32 chosen for:
- Lower FLOPs (2× vs D=64), faster iteration
- More constrained phase space → effects more visible
- step86 Config F shows D=32 gives ~93% at N=4096 → viable at N=1024

CONFIGS (N=1024, D=32, K_iter=8, AH=1.0, 50%/75ep)
=====================================================
  Ref : standard AH, D=32 (no phase polarization, no alternating)
  A   : phase polarization + alternating training (1 epoch each)
  B   : phase polarization + simultaneous training (isolate alternating)
  C   : phase polarization + alternating + Pauli exclusion (λ=0.01)
  D   : phase polarization + alternating (2-epoch blocks)
  E   : AH wpos REPLACED by phase polarization (no AH, pure phase filter)

ABLATION:
  A vs Ref   : does phase polarization + alternating help?
  B vs A     : is alternating necessary, or does simultaneous work?
  C vs A     : does Pauli exclusion add diversity beyond AH?
  D vs A     : does 2-epoch blocks help stability vs 1-epoch?
  E vs Ref   : can phase polarization replace AH entirely?

To reproduce:
    python -u scripts/train_step102_phase_polarizer.py --device mps
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
parser.add_argument("--epochs", type=int, default=75,
                    help="Training epochs (default 75; use 20 for Tier-0 scout)")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 32   # D=32 for efficiency + visibility
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
STEP69_REF = 0.8336   # D=64 reference — D=32 will be lower


class SGNNET_PhasePolarizer(nn.Module):
    """Phase-polarized routing with alternating training and Pauli exclusion.

    Malus's law filtering:
      polar(h,j) = cos²(angle(W_phase[h], W_phase[j]))
                 = (dot(W_phase_n[h], W_phase_n[j]))²  (for normalized phases)

    Combined with AH wpos suppression:
      weight(h,j) = (1 - α_ah * pos_sim(h,j)) * polar(h,j)

    Or in E mode (phase replaces AH):
      weight(h,j) = polar(h,j)
    """

    def __init__(self, base: SGNNET_SmallWorld, alpha_ahebb: float,
                 use_ah: bool = True, pauli_lambda: float = 0.0,
                 alternating_block: int = 1):
        super().__init__()
        self.base         = base
        self.alpha_ahebb  = alpha_ahebb
        self.use_ah       = use_ah
        self.pauli_lambda = pauli_lambda
        self.alt_block    = alternating_block  # epochs per block

        # Phase embeddings — same shape as W_pos hidden portion
        self.W_phase_polar = nn.Parameter(
            F.normalize(torch.randn(base.N_hidden, D), dim=-1))

        # Threshold (same as SGNNET_Resonant)
        self.theta = nn.Parameter(torch.full((base.N_hidden,), 0.1))

        self._current_epoch = 0
        self._pauli_loss = torch.tensor(0.0)

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.W_phase_polar   # Trainer may query this

    def tick_epoch(self):
        self._current_epoch += 1

    def _is_phase_epoch(self) -> bool:
        """Is this a phase-training epoch (vs position-training)?"""
        block = (self._current_epoch // self.alt_block) % 2
        return block == 1   # block 0 = position, block 1 = phase

    def set_alternating_grad(self):
        """Called before each epoch to set requires_grad."""
        phase_epoch = self._is_phase_epoch()
        self.base.W_pos.requires_grad_(not phase_epoch)
        self.W_phase_polar.requires_grad_(phase_epoch)
        # theta always trainable
        self.theta.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.base._seed(x)                             # [B, N, D]
        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.base.conn_hh
        N_h       = self.base.N_hidden

        # Phase polarization: Malus's law
        Ph_norm = F.normalize(self.W_phase_polar, dim=-1)          # [N, D]
        phase_sim = (Ph_norm.unsqueeze(1) * Ph_norm[conn_hh]).sum(-1)  # [N, K_hh]
        polar_w   = (phase_sim ** 2).unsqueeze(0).unsqueeze(-1)    # [1,N,K_hh,1]

        # AH wpos suppression (if enabled)
        if self.use_ah:
            W_n     = F.normalize(self.base.W_pos[:N_h], dim=-1)
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
            ah_w    = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                      ).unsqueeze(0).unsqueeze(-1)                 # [1,N,K_hh,1]
            edge_w  = ah_w * polar_w                               # combined
        else:
            edge_w  = polar_w                                      # phase only

        # Reflection accumulator
        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_nb     = Z_fwd[:, conn_hh, :]                       # [B,N,K_hh,D]
            Z_struct = (Z_nb * edge_w).sum(dim=2)                  # [B,N,D]

            Z_remainder = Z_fwd - Z
            Z_reflected = ALPHA_REFLECT * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        # Pauli exclusion loss (computed here, added to training loss externally)
        if self.pauli_lambda > 0 and self.training:
            # Combined state: cat(pos, phase) for hidden neurons
            pos_n   = F.normalize(self.base.W_pos[:N_h], dim=-1)
            state   = torch.cat([pos_n, Ph_norm], dim=-1)          # [N, 2D]
            # Pairwise similarity for connected neurons
            s_self  = state.unsqueeze(1)                           # [N,1,2D]
            s_nb    = state[conn_hh]                               # [N,K_hh,2D]
            sim_sq  = ((s_self - s_nb) ** 2).sum(-1)               # [N,K_hh]
            self._pauli_loss = self.pauli_lambda * torch.exp(-sim_sq).mean()
        else:
            self._pauli_loss = torch.tensor(0.0, device=x.device)

        return self.base._readout(Z)


@dataclass
class Config:
    key: str; label: str
    use_ah: bool; use_polar: bool; alternating: bool
    alt_block: int; pauli_lambda: float


CONFIGS = [
    Config("Ref", "Ref  standard AH, D=32 (no polarization)",
           True, False, False, 1, 0.0),
    Config("A",   "A    AH + phase polar + alternating (1-epoch blocks)",
           True, True, True, 1, 0.0),
    Config("B",   "B    AH + phase polar + simultaneous (no alternating)",
           True, True, False, 1, 0.0),
    Config("C",   "C    AH + phase polar + alternating + Pauli (λ=0.01)",
           True, True, True, 1, 0.01),
    Config("D",   "D    AH + phase polar + alternating (2-epoch blocks)",
           True, True, True, 2, 0.0),
    Config("E",   "E    phase polar ONLY (no AH) + alternating",
           False, True, True, 1, 0.0),
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
        K_in=tk["K_in"], K_iter=8, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    if not cfg.use_polar:
        # Standard AH Ref — need SGNNET_Resonant wrapper
        resonant = SGNNET_Resonant(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
            alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0,
        )
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_PhasePolarizer(
        base, alpha_ahebb=ALPHA_AHEBB if cfg.use_ah else 0.0,
        use_ah=cfg.use_ah, pauli_lambda=cfg.pauli_lambda,
        alternating_block=cfg.alt_block,
    )


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run(cfg, model, meta):
    print(f"\n{'='*70}\n{cfg.label}\n{'='*70}")
    tr, va = get_loaders()
    tk = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)

    # If alternating: we need to hook into the epoch loop to toggle gradients.
    # Since Trainer doesn't natively support this, we manually train.
    t0 = time.time()

    if cfg.alternating and cfg.use_polar:
        history = []
        for ep in range(EPOCHS):
            model.set_alternating_grad()
            # Re-add params to optimizer each epoch based on requires_grad
            # (simpler: just let gradients be zero for frozen params)
            ep_hist = trainer.train(n_epochs=1)
            history.extend(ep_hist)
            if hasattr(model, "tick_epoch"):
                model.tick_epoch()
    else:
        history = trainer.train(n_epochs=EPOCHS)

    elapsed = time.time() - t0
    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best = max(top1_hist); best_ep = int(np.argmax(top1_hist)) + 1
    frac = best_ep / len(history)
    result = {
        "label": cfg.label, "use_ah": cfg.use_ah, "use_polar": cfg.use_polar,
        "alternating": cfg.alternating, "alt_block": cfg.alt_block,
        "pauli_lambda": cfg.pauli_lambda,
        "top1_best": best, "top1_last": history[-1].get("val_top1", 0.0),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1), "best_epoch_frac": round(frac, 3),
        "params": count_params(model), "top1_history": top1_hist,
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1={best:.4f}  ep={best_ep}/{len(history)}  t={elapsed:.0f}s")
    return result


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  N={N}  D={D}  Data: 50%")
    print(f"Step 102: Phase-polarized neurons + alternating training")
    print(f"Malus's law filtering + Pauli exclusion on (pos,phase)\n")
    for c in CONFIGS:
        tags = []
        if c.use_ah: tags.append("AH")
        if c.use_polar: tags.append("polar")
        if c.alternating: tags.append(f"alt({c.alt_block})")
        if c.pauli_lambda > 0: tags.append(f"pauli({c.pauli_lambda})")
        print(f"  {c.key:4s}  {'+'.join(tags) if tags else 'baseline':30s}  {c.label}")
    print()
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")
    results = {}
    out_path = ROOT / "results" / "train_step102_phase_polarizer.json"
    for i, cfg in enumerate(CONFIGS):
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        meta = {"N": N, "D": D, "K_iter": 8, "use_ah": cfg.use_ah,
                "use_polar": cfg.use_polar, "alternating": cfg.alternating,
                "pauli_lambda": cfg.pauli_lambda, "data_frac": 0.5}
        results[cfg.key] = run(cfg, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")
    print(f"\n{'='*70}\nSTEP 102 COMPLETE\n")
    for c in CONFIGS:
        if c.key in results:
            r = results[c.key]
            print(f"  {c.key:4s}  {r['top1_best']:.4f}  {c.label}")
    w = max(results, key=lambda k: results[k]["top1_best"])
    print(f"\n  Winner: {w}")
