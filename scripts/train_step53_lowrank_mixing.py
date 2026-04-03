"""Step 53: Low-rank dimension mixing at D=64.

GEN4 DESIGN INSIGHT
===================
Full D x D cross-dim mixing fails at D=64 (step30, step37, fast_W_phase) — it destroys
the Fourier encoding structure. But the dimensions are NOT independent: Fourier encoding
packs information into (sin, cos) frequency pairs. Can we mix WITHIN these natural
sub-structures without disrupting the global Fourier layout?

Three mixing strategies, all O(N * D) per routing step (no N^2 cost):

1. LOW-RANK MIXING (A, B): Z_mixed = Z + alpha * Z @ U @ V.T
   U, V in R^{D x R}. Rank R << D constrains the mixing to a low-D subspace.
   At R=4: only 4 directions of cross-dim interaction (2*D*R = 512 params).
   At R=8: 8 directions (1024 params). Both far below the D^2=4096 of full mixing.
   Hypothesis: low-rank mixing captures the principal cross-dim interactions
   without scrambling the Fourier layout.

2. FREQUENCY-PAIR MIXING (C, E): 2x2 within each (dim_2k, dim_2k+1)
   Fourier encoding creates (sin, cos) pairs at each frequency. The pair shares
   information about the same spatial axis. A learned 2x2 rotation per pair lets the
   model adjust the phase/magnitude balance within each frequency without touching
   other frequencies. D/2 = 32 independent 2x2 matrices = 128 params total.
   Hypothesis: this is the natural mixing granularity for Fourier encodings.

3. GROUP MIXING (D): 8x8 within groups of 8 dims
   D=64 / 8 groups = 8 dims per group. Each group mixes with a learned 8x8 matrix.
   Groups = 8 * 64 = 512 params. Broader than pair-wise but still much less than D x D.
   Hypothesis: groups of 4 frequency pairs may interact (e.g., harmonics of spatial axes).

CONFIGS (D=64 N=1024 K_iter=8 Fourier dynamic_z_geo):
  Ref    AntiHebb alpha=0.5 uniform sum  [anchor ~70.14%, step29 Config A]
  A      + low-rank mixing rank=4 alpha_mix=0.1
  B      + low-rank mixing rank=8 alpha_mix=0.1
  C      + frequency-pair mixing (32 x 2x2 matrices) alpha_mix=0.1
  D      + group mixing (8 x 8x8 matrices) alpha_mix=0.1
  E      + frequency-pair mixing alpha_mix=0.1 + AntiHebb alpha=0.5 [C + AntiHebb]
  F      + low-rank rank=4 alpha_mix=0.1 NO AntiHebb  [ablation: mixing vs inhibition]

Expected outcomes:
  C > A, B → frequency-aligned mixing is the natural granularity
  A, B > D → low-rank beats group (groups impose arbitrary boundaries)
  E > Ref  → frequency-pair mixing compounds with AntiHebb; include in Gen4
  F < Ref  → mixing alone cannot replace inhibition
  F ~ Ref  → mixing is an alternative to AntiHebb (lower param cost)

To reproduce:
    python -u scripts/train_step53_lowrank_mixing.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 150
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
N      = 1024
D      = 64

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_resonant() -> SGNNET_Resonant:
    """D=64 N=1024 K_iter=8 alpha_reflect=0.5 (step22b calibration winner)."""
    torch.manual_seed(SEED)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=8,
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.5, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )


# ── Mixing modules ──────────────────────────────────────────────────────────

class LowRankMixer(nn.Module):
    """Low-rank cross-dim mixing: Z_out = Z + alpha * Z @ U @ V^T."""

    def __init__(self, D: int, rank: int, alpha_mix: float = 0.1):
        super().__init__()
        self.alpha_mix = alpha_mix
        # Xavier-scaled init — keeps initial mixing small
        self.U = nn.Parameter(torch.randn(D, rank) * (2.0 / (D + rank)) ** 0.5)
        self.V = nn.Parameter(torch.randn(D, rank) * (2.0 / (D + rank)) ** 0.5)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Z: [B, N, D] -> [B, N, D]."""
        mixed = Z @ self.U @ self.V.T          # [B, N, D]
        return Z + self.alpha_mix * mixed


class FrequencyPairMixer(nn.Module):
    """2x2 mixing within each (sin, cos) frequency pair.

    Fourier encoding packs dims as (sin_1, cos_1, sin_2, cos_2, ...).
    Each 2x2 matrix can rotate/scale within its frequency pair.
    """

    def __init__(self, D: int, alpha_mix: float = 0.1):
        super().__init__()
        assert D % 2 == 0, f"D must be even for frequency-pair mixing, got D={D}"
        self.alpha_mix = alpha_mix
        self.n_pairs   = D // 2
        # Initialize as identity + small noise so initial mixing is near-zero
        W = torch.eye(2).unsqueeze(0).expand(self.n_pairs, -1, -1).clone()
        W += torch.randn_like(W) * 0.01
        self.W_pair = nn.Parameter(W)          # [n_pairs, 2, 2]

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Z: [B, N, D] -> [B, N, D]."""
        B, Nh, D = Z.shape
        # Reshape to [B, N, n_pairs, 2], apply 2x2, reshape back
        Z_pairs = Z.view(B, Nh, self.n_pairs, 2)
        # einsum: for each pair p, Z_new[...,p,j] = sum_k Z[...,p,k] * W[p,k,j]
        Z_mixed = torch.einsum("bnpk,pkj->bnpj", Z_pairs, self.W_pair)
        Z_mixed = Z_mixed.reshape(B, Nh, D)
        # Residual connection with mixing strength
        return Z + self.alpha_mix * (Z_mixed - Z)


class GroupMixer(nn.Module):
    """8x8 mixing within groups of 8 dimensions.

    D=64 / 8 = 8 groups. Each group has a learned 8x8 mixing matrix.
    Groups align with 4 frequency pairs (4 spatial frequencies per group).
    """

    def __init__(self, D: int, group_size: int = 8, alpha_mix: float = 0.1):
        super().__init__()
        assert D % group_size == 0, f"D={D} not divisible by group_size={group_size}"
        self.alpha_mix  = alpha_mix
        self.group_size = group_size
        self.n_groups   = D // group_size
        # Initialize as identity + small noise
        W = torch.eye(group_size).unsqueeze(0).expand(self.n_groups, -1, -1).clone()
        W += torch.randn_like(W) * 0.01
        self.W_group = nn.Parameter(W)         # [n_groups, gs, gs]

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Z: [B, N, D] -> [B, N, D]."""
        B, Nh, D = Z.shape
        gs = self.group_size
        Z_groups = Z.view(B, Nh, self.n_groups, gs)
        Z_mixed  = torch.einsum("bngk,gkj->bngj", Z_groups, self.W_group)
        Z_mixed  = Z_mixed.reshape(B, Nh, D)
        return Z + self.alpha_mix * (Z_mixed - Z)


# ── Wrapper model ────────────────────────────────────────────────────────────

class SGNNET_LowRankMixing(nn.Module):
    """AntiHebb + optional low-rank/freq-pair/group mixing.

    Mixing is applied once per K_iter step after structural aggregation,
    before normalization. This matches the position where W_mix was applied
    in step30 but with structured mixing instead of full D x D.
    """

    def __init__(
        self,
        base_model: SGNNET_Resonant,
        alpha_ahebb: float = 0.5,
        mixer: nn.Module | None = None,
    ):
        super().__init__()
        self.m           = base_model
        self.alpha_ahebb = alpha_ahebb
        self.mixer       = mixer

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        B, Nh, Dd = Z.shape
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh
        W_pos_h   = self.m.W_pos[:Nh]

        # Pre-compute AntiHebb W_pos similarity (static)
        if self.alpha_ahebb > 0.0:
            W_pos_nb = W_pos_h[conn_hh]                              # [N, K_hh, D]
            pos_sim  = F.cosine_similarity(
                W_pos_h.unsqueeze(1), W_pos_nb, dim=-1
            )                                                         # [N, K_hh]

        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)

            # Structural aggregation with AntiHebb suppression
            if self.alpha_ahebb > 0.0:
                ahebb    = (pos_sim.unsqueeze(0).unsqueeze(-1)        # [1,N,K_hh,1]
                            * Z_fwd[:, conn_hh, :]).sum(2)            # [B,N,D]
                Z_struct = Z_fwd[:, conn_hh, :].sum(2)                # [B,N,D]
                Z_struct = Z_struct - self.alpha_ahebb * ahebb
            else:
                Z_struct = Z_fwd[:, conn_hh, :].sum(2)

            # Long-range phase inhibition
            Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            Z_update = Z_struct + self.m.alpha_turing * Z_inh

            # Apply dimension mixing BEFORE normalization
            if self.mixer is not None:
                Z_update = self.mixer(Z_update)

            Z = F.normalize(Z_update.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ── Run helper ───────────────────────────────────────────────────────────────

def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0
    best    = max(h.get("val_top1", 0.0) for h in history)
    last5   = history[-5:]
    best_ep = int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1
    frac    = best_ep / len(history)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result  = {
        "label": label, "top1_best": best,
        "top1_last": history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "best_epoch_frac": round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "n_params": n_params,
        "top1_history": [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
          f"  params={n_params}  diag={result['convergence_diag']}  t={elapsed:.0f}s")
    return result


# ── Configs ──────────────────────────────────────────────────────────────────

# (key, label, alpha_ahebb, mixer_factory, extra_meta)
CONFIGS = [
    ("Ref", "Ref   AntiHebb alpha=0.5 uniform  [~70.14%, step29A]",
     0.5, lambda: None,
     {"mixing": "none"}),

    ("A", "A     + low-rank mixing rank=4 alpha_mix=0.1",
     0.5, lambda: LowRankMixer(D, rank=4, alpha_mix=0.1),
     {"mixing": "lowrank", "rank": 4, "alpha_mix": 0.1}),

    ("B", "B     + low-rank mixing rank=8 alpha_mix=0.1",
     0.5, lambda: LowRankMixer(D, rank=8, alpha_mix=0.1),
     {"mixing": "lowrank", "rank": 8, "alpha_mix": 0.1}),

    ("C", "C     + frequency-pair mixing (32 x 2x2) alpha_mix=0.1",
     0.5, lambda: FrequencyPairMixer(D, alpha_mix=0.1),
     {"mixing": "freq_pair", "n_pairs": 32, "alpha_mix": 0.1}),

    ("D", "D     + group mixing (8 x 8x8) alpha_mix=0.1",
     0.5, lambda: GroupMixer(D, group_size=8, alpha_mix=0.1),
     {"mixing": "group", "group_size": 8, "alpha_mix": 0.1}),

    ("E", "E     + frequency-pair + AntiHebb alpha=0.5  [C + AH]",
     0.5, lambda: FrequencyPairMixer(D, alpha_mix=0.1),
     {"mixing": "freq_pair", "n_pairs": 32, "alpha_mix": 0.1,
      "_note": "Same as C but explicit compound test"}),

    ("F", "F     + low-rank rank=4  NO AntiHebb  [ablation: mixing vs inhibition]",
     0.0, lambda: LowRankMixer(D, rank=4, alpha_mix=0.1),
     {"mixing": "lowrank", "rank": 4, "alpha_mix": 0.1, "antihebb": False}),
]


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print("Step 53: Low-rank dimension mixing at D=64")
    print("  Gen4 principle: structured sub-D mixing preserves Fourier layout")
    print("  Base: D=64 N=1024 K_iter=8 AntiHebb alpha=0.5 (70.14%)")
    print("  Mechanisms: low-rank U*V^T / frequency-pair 2x2 / group 8x8")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    REF_BASELINE = 0.7014
    results = {}
    for key, label, alpha_ah, mixer_fn, extra_meta in CONFIGS:
        resonant = make_resonant().to(DEVICE)
        mixer    = mixer_fn()
        model    = SGNNET_LowRankMixing(
            resonant, alpha_ahebb=alpha_ah, mixer=mixer,
        ).to(DEVICE)
        meta = {"N": N, "D": D, "K_iter": 8, "alpha_ahebb": alpha_ah,
                **extra_meta}
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step53_lowrank_mixing.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    print(f"\n-- Low-rank mixing (ref={ref_val:.4f}) ---")
    print(f"  {'Config':<60}  {'top1':>6}  {'vs_Ref':>8}"
          f"  {'params':>8}  {'t(s)':>6}")
    print("  " + "-"*100)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        print(f"  {r['label'][:60]:<60}  {r['top1_best']:>6.4f}  {d:>+8.4f}"
              f"  {r.get('n_params', 0):>8}  {r['elapsed_s']:>6.0f}")

    print("\n  Key questions:")
    print("  C > A, B → frequency-pair is the natural Fourier mixing granularity")
    print("  E > Ref  → freq-pair + AntiHebb compound; include in Gen4")
    print("  F < Ref  → mixing alone insufficient; AntiHebb is primary mechanism")
    print("  A, B > D → low-rank beats arbitrary group boundaries")
