"""Step 51: Spatial dynamic connectivity + phase gating (W_phase as modulator).

ORIGINAL VISION — Phase as Gate, Not Antenna
=============================================
From sparse_geometric_network_report.md, W_phase was always meant to be a
MODULATION mechanism — neurons with aligned phase vectors couple more strongly.
The metaphor: two tuning forks at the same frequency resonate; mismatched ones cancel.

In current SGNNET, W_phase acts as an "antenna" — it defines a K-NN graph
(conn_phase) and is used for beam scoring in dynamic_z_geo routing. This is a
different role than originally intended.

This experiment separates the two roles:
  - TOPOLOGY:  W_pos K-NN determines WHO connects to whom (from step50)
  - STRENGTH:  W_phase alignment determines HOW STRONGLY they connect

Specifically, for each spatial neighbor j of neuron i:
  gate_ij = max(0, dot(W_phase_i, W_phase_j))   # cosine similarity, non-negative
  msg_ij  = gate_ij * Z_j                         # phase-modulated message
  Z_struct_i = sum_j(msg_ij)                       # weighted structural aggregation

This means:
  - Two neurons with W_phase perfectly aligned (same "frequency"): couple maximally
  - Two neurons with orthogonal W_phase (different "frequencies"): don't couple
  - Two neurons with anti-aligned W_phase: gated to zero (max(0, ...))

W_phase is now learned to decide WHICH spatial neighbors are "in resonance" with
each neuron — not to define graph structure (that's W_pos), not to select beam
(that's activation magnitude). It is purely a strength modulator.

WHY THIS COULD WORK
===================
Step31 (Z-KNN) showed dynamic per-step topology hurts (-12pp vs static).
Step50 tests dynamic W_pos K-NN (topology evolving across epochs, not per step).
Step51 adds a second learned degree of freedom: even within the spatial neighborhood,
W_phase selects which neighbors to listen to. This creates a two-level filtering:
  Level 1: W_pos — spatial proximity gates CONNECTION EXISTENCE
  Level 2: W_phase — frequency alignment gates CONNECTION STRENGTH

This is the crystal analogy fully realized: diffraction requires BOTH geometric
proximity AND frequency matching (Bragg's law: 2d·sin(θ) = nλ).

CONFIGS (D=64 N=1024 K_iter=8):
  Ref    static small-world, no phase gating  [~56.28%]
  A      dynamic W_pos K-NN K=6 + phase gate  [W_pos topology + W_phase strength]
  B      dynamic W_pos K-NN K=8 + phase gate
  C      dynamic W_pos K-NN K=6 + phase gate + 50% interneurons
  D      dynamic W_pos K-NN K=6 + phase gate + 50% interneurons + AntiHebb α=0.5
  E      static small-world + phase gate only  [ablation: does phase gate help even
         with fixed topology? isolates phase gating contribution]

Config E is a key ablation: if E > Ref, phase gating helps regardless of dynamic topology.
If A > E, the combination of dynamic topology + phase gate is synergistic.

To reproduce:
    python -u scripts/train_step51_spatial_phase_gating.py --device mps
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


def build_wpos_knn_conn(W_pos: torch.Tensor, K: int, N_hidden: int) -> torch.Tensor:
    """K-NN connectivity from W_pos space with annular inner exclusion."""
    W_h = W_pos[:N_hidden].detach()
    with torch.no_grad():
        dists    = torch.cdist(W_h, W_h)
        dists.fill_diagonal_(float('inf'))
        K_wide   = min(K * 2, N_hidden - 1)
        _, top_i = dists.topk(K_wide, dim=-1, largest=False)
        inner    = K // 2
        conn     = top_i[:, inner:inner + K]
    return conn.to(W_pos.device)


class SGNNET_SpatialPhaseGated(nn.Module):
    """Spatial dynamic connectivity + W_phase gating of connection strength.

    Two-level connectivity:
      1. W_pos K-NN determines structural neighborhood (rebuilt each epoch)
      2. W_phase cosine similarity modulates each connection's weight

    W_phase role: pure strength modulator. It does NOT:
      - Define the graph structure (that's W_pos K-NN)
      - Select beam neurons for inhibition routing (that's activation magnitude)
    It only gates: gate_ij = max(0, cos(W_phase_i, W_phase_j))
    """

    def __init__(self, base_model: SGNNET_Resonant,
                 K_spatial: int = 6,
                 n_input: int = 1024,
                 alpha_ahebb: float = 0.0,
                 use_static_conn: bool = False):
        """
        use_static_conn: if True, use original static small-world conn_hh instead
                         of W_pos K-NN (ablation config E — phase gate only)
        """
        super().__init__()
        self.m              = base_model
        self.K_spatial      = K_spatial
        self.n_input        = n_input
        self.alpha_ahebb    = alpha_ahebb
        self.use_static_conn = use_static_conn

        # Initial spatial connectivity (or use static)
        if use_static_conn:
            conn = base_model.base.conn_hh.clone()
        else:
            conn = build_wpos_knn_conn(
                self.m.W_pos, K_spatial, base_model.base.N_hidden
            )
        self.register_buffer("conn_hh_spatial", conn)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if not self.use_static_conn:
            new_conn = build_wpos_knn_conn(
                self.m.W_pos, self.K_spatial, self.m.base.N_hidden
            )
            self.conn_hh_spatial.copy_(new_conn)
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        if self.n_input < self.m.base.N_hidden:
            Z[:, self.n_input:, :] = 0.0

        B, N, D   = Z.shape
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn      = self.conn_hh_spatial                         # [N, K]

        # Pre-compute phase gate weights — these update as W_phase is learned
        W_ph_norm    = F.normalize(self.m.W_phase, dim=-1)       # [N, D]
        W_ph_nb      = W_ph_norm[conn]                           # [N, K, D]
        phase_sim    = (W_ph_norm.unsqueeze(1) * W_ph_nb).sum(-1) # [N, K]
        phase_gate   = phase_sim.clamp(min=0.0)                  # [N, K] ≥ 0
        # Add small floor so neurons with zero phase alignment still pass some signal
        phase_gate   = phase_gate + 0.1                          # floor = 0.1
        phase_gate_n = phase_gate / (phase_gate.sum(-1, keepdim=True) + 1e-8)

        W_pos_h = self.m.W_pos[:N]

        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)

            # Phase-gated structural aggregation
            Z_nb     = Z_fwd[:, conn, :]                        # [B, N, K, D]
            Z_struct = (Z_nb * phase_gate_n.unsqueeze(0).unsqueeze(-1)).sum(2)

            # Dynamic inhibition (standard dynamic_z_geo from base)
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            if self.alpha_ahebb > 0.0:
                W_pos_nb = W_pos_h[conn]
                pos_sim  = F.cosine_similarity(
                    W_pos_h.unsqueeze(1), W_pos_nb, dim=-1)
                ahebb    = (pos_sim.unsqueeze(0).unsqueeze(-1)
                            * Z_fwd[:, conn, :]).sum(2)
                Z = F.normalize(
                    (Z_struct + self.m.alpha_turing * Z_inh
                     - self.alpha_ahebb * ahebb).clamp(-10, 10),
                    dim=-1,
                )
            else:
                Z = F.normalize(
                    (Z_struct + self.m.alpha_turing * Z_inh).clamp(-10, 10),
                    dim=-1,
                )

        return self.m.base._readout(Z)


def make_resonant() -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk = topology_kwargs(N)
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


# (key, label, K_spatial, use_interneurons, alpha_ahebb, use_static_conn)
CONFIGS = [
    ("Ref", "Ref   static small-world  no phase gate  [~56.28%]",
     6,    False, 0.0, False),   # Ref uses static conn from base model
    ("A",   "A     W_pos K-NN K=6  + phase gate  no-int",
     6,    False, 0.0, False),
    ("B",   "B     W_pos K-NN K=8  + phase gate  no-int",
     8,    False, 0.0, False),
    ("C",   "C     W_pos K-NN K=6  + phase gate  50% interneurons",
     6,    True,  0.0, False),
    ("D",   "D     W_pos K-NN K=6  + phase gate  + interneurons + AntiHebb α=0.5",
     6,    True,  0.5, False),
    ("E",   "E     static small-world + phase gate only  [ablation: topology fixed]",
     6,    False, 0.0, True),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print("Step 51: Spatial dynamic connectivity + W_phase gating")
    print("  W_pos K-NN: WHO connects (topology, evolves during training)")
    print("  W_phase gate: HOW STRONGLY (modulates strength, gate_ij = cos(W_ph_i, W_ph_j))")
    print("  W_phase is a resonance modulator — not an antenna, not a beam selector")
    print("  Bragg analogy: proximity (W_pos) + frequency match (W_phase) = coupling")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    REF_BASELINE = 0.5628
    results = {}
    for key, label, K_sp, use_int, alpha_ah, use_static in CONFIGS:
        resonant = make_resonant().to(DEVICE)
        if key == "Ref":
            model = resonant
        else:
            n_input = N // 2 if use_int else N
            model = SGNNET_SpatialPhaseGated(
                resonant, K_spatial=K_sp, n_input=n_input,
                alpha_ahebb=alpha_ah, use_static_conn=use_static,
            ).to(DEVICE)
        meta = {"N": N, "D": D, "K_iter": 8,
                "K_spatial": K_sp, "interneurons": use_int,
                "n_input": N // 2 if use_int else N,
                "alpha_ahebb": alpha_ah, "use_static_conn": use_static}
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step51_spatial_phase_gating.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    print(f"\n-- Spatial + phase gating (ref={ref_val:.4f}) ---")
    print(f"  {'Config':<60}  {'top1':>6}  {'vs_Ref':>8}  {'t(s)':>6}")
    print("  " + "-"*90)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        print(f"  {r['label'][:60]:<60}  {r['top1_best']:>6.4f}  {d:>+8.4f}  "
              f"{r['elapsed_s']:>6.0f}")

    print("\n  Key questions answered:")
    print("  E > Ref  → phase gating alone helps on static topology")
    print("  A > E    → dynamic W_pos topology + phase gate is synergistic")
    print("  A > step50.A  → phase gating adds value on top of spatial K-NN alone")
    print("  C > A    → interneurons compound with spatial+phase")
    print("  D is the full original vision: spatial topology + phase resonance + competitive inhibition")
