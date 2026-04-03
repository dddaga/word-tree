"""Step 52: High-D geometry exploitation — subspace, structural, and projection routing.

GEN4 DESIGN PRINCIPLE (from D=64 experiments)
=============================================
Mechanisms that assume directional alignment fail on S^63:
  - Signed coupling (Z Z^T Z): cos-sim ≈ 0.016 on S^63 (noise) → 5 experiments, all ≤32%
  - Cross-dim W_mix: interferes with Fourier routing → -15pp at D=64

Mechanisms that impose competitive structure succeed regardless of direction:
  - AntiHebb (W_pos spatial surround suppression): +13.86pp → 70.14% confirmed best

QUESTION: Can we EXPLOIT high-D geometry rather than fight it?
Near-orthogonality is capacity, not a bug. On S^63, N=1024 neurons use a tiny
fraction of available space. This experiment tests whether we can exploit that:

MECHANISMS TESTED
=================

1. Z-SUBSPACE ROUTING (A, B) — gating in activation subspace
   Full-D cosine similarity on S^63 ≈ 0.016 (near noise). But after K_iter routing
   steps, low-index dims of Z may develop more structure — neurons in the same crystal
   region develop correlated low-frequency activations. Testing whether cos(Z[0:S])
   in a smaller subspace gives a cleaner routing signal.
     gate_ij = relu(cos(Z_i[0:S], Z_j[0:S]))   S ∈ {16, 32}

2. W_POS-SUBSPACE ROUTING (C, D) — gating in structural position subspace
   Gate connections by cosine similarity in a low-D subspace of W_pos.
   If the position space has structure (which it should — W_pos is the crystal lattice),
   then the first S dims of W_pos carry the coarsest spatial variation.
   Neurons with similar W_pos[0:S] are in the same "region"; they should couple more.
     gate_ij = relu(cos(W_pos_i[0:S], W_pos_j[0:S]))   S ∈ {16, 32}
   Note: static gate (recomputed per epoch as W_pos updates, not per routing step).

3. PROJECTION ROUTING (E) — cross-modal gate (diffraction metaphor fully realized)
   gate_ij = |Z_i · normalize(W_pos_j)|
   "How strongly does neuron i's current activation resonate with neuron j's structural
   direction?" The wave (Z_i) resonates with the crystal plane (W_pos_j) proportional to
   how strongly it hits it. Bridges activation space ↔ structural position space.
   Dynamic per routing step (Z changes). O(N × K × D) per iter.

4. CENTERING DIVERSITY (F) — suppress global drift
   Add -λ × Z_mean to the routing update. Prevents all neurons from drifting toward a
   single direction across K_iter steps. Complementary to AntiHebb:
     AntiHebb: suppresses pairwise similarity via W_pos spatial proximity
     Centering: suppresses global convergence to collective mean
   λ=0.05 is the centering strength. Uses detached mean to avoid training mean itself.

5. ABLATION — W_pos-subspace routing WITHOUT AntiHebb (G)
   Does structural frequency gating achieve the same competitive inhibition that
   AntiHebb achieves, but through a routing gate instead of a subtract term?
   If G ≈ Ref (70.14%), structural gate alone is sufficient.
   If G << Ref, structural gate + AntiHebb are complementary mechanisms.

CONFIGS (D=64 N=1024 K_iter=8 Fourier encoding):
  Ref   AntiHebb α=0.5 uniform sum  [~70.14%, step29 Config A]
  A     + Z-subspace routing split=16   (routing in D/4 dims of Z)
  B     + Z-subspace routing split=32   (routing in D/2 dims of Z)
  C     + W_pos-subspace routing split=16   (spatial gate in D/4 dims of W_pos)
  D     + W_pos-subspace routing split=32   (spatial gate in D/2 dims of W_pos)
  E     + projection routing |Z_i · W_pos_j|  (cross-modal diffraction gate)
  F     + centering diversity λ=0.05   (global drift suppression)
  G     W_pos-subspace split=16  NO AntiHebb  [ablation: gate vs subtract]

Expected outcomes:
  C or D > Ref → structural subspace gate compounds with AntiHebb; use in Gen4
  E > Ref     → diffraction gate works; W_pos and Z interact productively
  G ≈ Ref     → structural gate replaces AntiHebb (more efficient; pure routing)
  G << Ref    → gate + subtract are complementary; both needed
  A/B > Ref   → Z-subspace has structure after routing; unexpected but valuable

To reproduce:
    python -u scripts/train_step52_high_d_routing.py --device mps
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


class SGNNET_HighDRouting(nn.Module):
    """High-D geometry routing wrapper.

    Replaces the uniform structural aggregation Z_struct = Σ_j Z_j with a gated version
    that exploits the structure of D=64 space. AntiHebb inhibition is preserved as the
    confirmed best mechanism (+13.86pp); the gate changes HOW neighbors are aggregated.

    routing_mode:
      'uniform'    — standard sum (Σ_j Z_j), AntiHebb only, replicates step29
      'z_subspace' — gate by relu(cos(Z_i[0:split], Z_j[0:split])) — activation subspace
      'w_subspace' — gate by relu(cos(W_pos_i[0:split], W_pos_j[0:split])) — position subspace
                     static gate (pre-computed outside routing loop; updates as W_pos trains)
      'projection' — gate by |Z_i · normalize(W_pos_j)| — cross-modal diffraction gate
    """

    def __init__(
        self,
        base_model: SGNNET_Resonant,
        routing_mode: str = "uniform",
        alpha_ahebb: float = 0.5,
        split: int = 16,
        lambda_center: float = 0.0,
    ):
        super().__init__()
        self.m             = base_model
        self.routing_mode  = routing_mode
        self.alpha_ahebb   = alpha_ahebb
        self.split         = split
        self.lambda_center = lambda_center

        assert routing_mode in ("uniform", "z_subspace", "w_subspace", "projection"), \
            f"Unknown routing_mode: {routing_mode}"

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        B, N, D   = Z.shape
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh                  # [N, K_hh] — static small-world
        W_pos_h   = self.m.W_pos[:N]                     # [N, D]

        # Pre-compute static W_pos-subspace gate (changes only as W_pos trains, not per step)
        if self.routing_mode == "w_subspace":
            S              = self.split
            W_pos_low      = W_pos_h[:, :S]              # [N, S]
            W_pos_nb_low   = W_pos_low[conn_hh]          # [N, K_hh, S]
            wpos_gate      = F.cosine_similarity(
                W_pos_low.unsqueeze(1), W_pos_nb_low, dim=-1
            ).clamp(min=0)                               # [N, K_hh]
            wpos_gate_n    = wpos_gate / (wpos_gate.sum(-1, keepdim=True) + 1e-8)

        # Pre-compute projection gate denominator (W_pos_nb_norm is static per forward pass)
        if self.routing_mode == "projection":
            W_pos_nb      = W_pos_h[conn_hh]            # [N, K_hh, D]
            W_pos_nb_norm = F.normalize(W_pos_nb, dim=-1)  # [N, K_hh, D]

        # AntiHebb: pre-compute W_pos cosine similarities (static per forward pass)
        if self.alpha_ahebb > 0.0:
            W_pos_nb_ah  = W_pos_h[conn_hh]             # [N, K_hh, D]
            pos_sim      = F.cosine_similarity(
                W_pos_h.unsqueeze(1), W_pos_nb_ah, dim=-1
            )                                            # [N, K_hh]

        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)

            # ---- Structural aggregation (routing mode) ----
            if self.routing_mode == "uniform":
                Z_struct = Z_fwd[:, conn_hh, :].sum(2)  # [B, N, D]

            elif self.routing_mode == "z_subspace":
                S        = self.split
                # Gate: cosine similarity in low-D activation subspace
                Z_low    = Z[:, :, :S]                  # [B, N, S]
                Z_nb_low = Z_low[:, conn_hh, :]         # [B, N, K_hh, S]
                gate     = F.cosine_similarity(
                    Z_low.unsqueeze(2), Z_nb_low, dim=-1
                ).clamp(min=0)                           # [B, N, K_hh]
                gate_n   = gate / (gate.sum(-1, keepdim=True) + 1e-8)
                Z_struct = (Z_fwd[:, conn_hh, :] * gate_n.unsqueeze(-1)).sum(2)

            elif self.routing_mode == "w_subspace":
                # Static gate (pre-computed above): [N, K_hh] → broadcast over batch
                Z_struct = (
                    Z_fwd[:, conn_hh, :]                           # [B, N, K_hh, D]
                    * wpos_gate_n.unsqueeze(0).unsqueeze(-1)       # [1, N, K_hh, 1]
                ).sum(2)                                            # [B, N, D]

            elif self.routing_mode == "projection":
                # Gate: |Z_i · normalize(W_pos_j)| — dynamic per step as Z changes
                # [B, N, K_hh] = sum over D of Z[B,N,D] * W_pos_nb_norm[N,K_hh,D]
                proj   = (Z.unsqueeze(2) * W_pos_nb_norm.unsqueeze(0)).sum(-1).abs()
                proj_n = proj / (proj.sum(-1, keepdim=True) + 1e-8)
                Z_struct = (Z_fwd[:, conn_hh, :] * proj_n.unsqueeze(-1)).sum(2)

            # ---- Long-range phase inhibition (standard dynamic_z_geo) ----
            Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            # ---- AntiHebb spatial surround suppression ----
            if self.alpha_ahebb > 0.0:
                ahebb  = (
                    pos_sim.unsqueeze(0).unsqueeze(-1)             # [1, N, K_hh, 1]
                    * Z_fwd[:, conn_hh, :]                         # [B, N, K_hh, D]
                ).sum(2)                                            # [B, N, D]
                update = (Z_struct + self.m.alpha_turing * Z_inh
                          - self.alpha_ahebb * ahebb)
            else:
                update = Z_struct + self.m.alpha_turing * Z_inh

            # ---- Centering diversity ----
            if self.lambda_center > 0.0:
                Z_mean = Z.mean(dim=1, keepdim=True).detach()      # [B, 1, D]
                update = update - self.lambda_center * Z_mean

            Z = F.normalize(update.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


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
    result  = {
        "label": label, "top1_best": best,
        "top1_last": history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "best_epoch_frac": round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history": [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
          f"  diag={result['convergence_diag']}  t={elapsed:.0f}s")
    return result


# (key, label, routing_mode, alpha_ahebb, split, lambda_center)
CONFIGS = [
    ("Ref", "Ref   AntiHebb α=0.5 uniform  [~70.14%, step29A]",
     "uniform",    0.5, 16, 0.0),
    ("A",   "A     Z-subspace routing split=16 + AntiHebb",
     "z_subspace", 0.5, 16, 0.0),
    ("B",   "B     Z-subspace routing split=32 + AntiHebb",
     "z_subspace", 0.5, 32, 0.0),
    ("C",   "C     W_pos-subspace routing split=16 + AntiHebb",
     "w_subspace", 0.5, 16, 0.0),
    ("D",   "D     W_pos-subspace routing split=32 + AntiHebb",
     "w_subspace", 0.5, 32, 0.0),
    ("E",   "E     Projection routing |Z_i·W_pos_j| + AntiHebb",
     "projection", 0.5, 16, 0.0),
    ("F",   "F     Centering diversity λ=0.05 + AntiHebb",
     "uniform",    0.5, 16, 0.05),
    ("G",   "G     W_pos-subspace split=16  NO AntiHebb  [ablation: gate vs subtract]",
     "w_subspace", 0.0, 16, 0.0),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print("Step 52: High-D geometry exploitation")
    print("  Gen4 principle: exploit near-orthogonality, don't fight it")
    print("  Base: D=64 N=1024 K_iter=8 AntiHebb α=0.5 (70.14%)")
    print("  Mechanisms: Z-subspace gate / W_pos-subspace gate / projection / centering")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    REF_BASELINE = 0.7014
    results = {}
    for key, label, rmode, alpha_ah, split, lam_c in CONFIGS:
        resonant = make_resonant().to(DEVICE)
        model    = SGNNET_HighDRouting(
            resonant, routing_mode=rmode,
            alpha_ahebb=alpha_ah, split=split,
            lambda_center=lam_c,
        ).to(DEVICE)
        meta = {"N": N, "D": D, "K_iter": 8,
                "routing_mode": rmode, "alpha_ahebb": alpha_ah,
                "split": split, "lambda_center": lam_c}
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step52_high_d_routing.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    print(f"\n-- High-D routing (ref={ref_val:.4f}) ---")
    print(f"  {'Config':<60}  {'top1':>6}  {'vs_Ref':>8}  {'t(s)':>6}")
    print("  " + "-"*88)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        print(f"  {r['label'][:60]:<60}  {r['top1_best']:>6.4f}  {d:>+8.4f}"
              f"  {r['elapsed_s']:>6.0f}")

    print("\n  Key questions answered:")
    print("  C/D > Ref → W_pos-subspace gate compounds with AntiHebb; add to Gen4")
    print("  E > Ref   → diffraction gate (Z·W_pos) works; bridges activation + structure")
    print("  G ≈ Ref   → W_pos gate replaces AntiHebb (more efficient)")
    print("  G << Ref  → gate + inhibition are complementary; both needed")
    print("  A/B > Ref → Z-subspace has routing structure after K_iter; unexpected win")
    print("  F > Ref   → centering complements AntiHebb; global + local diversity stack")
