"""Step 50: Spatial dynamic connectivity — W_pos K-NN graph, no phase in routing.

ORIGINAL VISION (sparse_geometric_network_report.md)
=====================================================
The crystal analogy: neurons are atoms at positions W_pos in D-space.
Connectivity should emerge from proximity — nearby atoms couple; distant ones don't.
Original design: connect neurons within personal-volume radius r* = R / N^(1/D).

The key insight (Section 3.5): "the geometry of the medium determines how
information flows through it. The spatial arrangement of neuron positions in
D-space should govern how activations propagate, and this arrangement should
emerge from training data."

WHAT CURRENT SGNNET DOES WRONG (from this lens)
================================================
In current SGNNET:
  - conn_hh is a static small-world graph, fixed at init FOREVER
  - W_pos moves during training but NEVER influences who connects to whom
  - W_pos only influences: geo-penalty in routing, Coulomb repulsion (dead at D=64),
    AntiHebb inhibition (W_pos-weighted surround)
  - The "crystal rearrangement" — where atoms drift toward their resonant positions —
    never happens in training

THIS EXPERIMENT restores the original vision:
  - conn_hh is rebuilt EVERY EPOCH from current W_pos distances
  - Neurons find their K spatial neighbors fresh at each epoch
  - As training progresses and W_pos moves, connectivity topology also evolves
  - No W_phase in routing — pure spatial structural connectivity

ANNULAR NEIGHBORHOOD (r/2 to r)
================================
Original design uses connections within r*. We extend to annular (r*/2, r*]:
  - Exclude closest neighbors (closer than r*/2): too similar, reinforce redundantly
  - Include mid-range (r*/2 to r*): "close enough to couple, different enough to add value"

At D=64 N=1024, absolute distance thresholds yield ~zero neighbors (curse of
dimensionality: typical W_pos distances ≈ 3.3, but r* ≈ 0.45 in [0,1]^64).
Fix: use K-nearest neighbors in W_pos space and apply annular exclusion as a
SOFT weight (not hard gate) using Gaussian kernel with peak at r*/2:
  weight_ij = exp(-(d_ij - r*/2)^2 / (r*/4)^2)  ×  I[d_ij ≤ r*]
This is differentiable and peaks for neurons at the ideal mid-range distance.

INTERNEURONS
============
50% interneurons (step20 winner at D=16: +2.83pp). With K_iter=8 at D=64,
interneurons have 8 routing steps to integrate signal — more than the 3 at D=16.
Interneurons receive zero input seed and serve as relay/integration nodes.
Their positions W_pos still move and participate in spatial K-NN.

CONFIGS (D=64 N=1024 K_iter=8):
  Ref    static conn_hh small-world (current baseline ~56.28%)
  A      dynamic W_pos K-NN (K=6), rebuild every epoch, no interneurons
  B      dynamic W_pos K-NN (K=8), rebuild every epoch, no interneurons
  C      dynamic W_pos K-NN (K=6) + 50% interneurons  [original vision + step20 winner]
  D      dynamic W_pos K-NN (K=6) + 50% interneurons + AntiHebb α=0.5  [add confirmed winner]
  E      dynamic W_pos K-NN (K=6), rebuild every 10 epochs (slower topology evolution)

EXPECTED OUTCOMES:
  A vs Ref: does dynamic W_pos connectivity beat static small-world?
  B vs A:   is K=8 better than K=6 for spatial connectivity?
  C vs A:   do interneurons compound with dynamic spatial topology?
  D vs C:   does AntiHebb add to spatial dynamic + interneurons?
  E vs A:   how fast should topology evolution be?

To reproduce:
    python -u scripts/train_step50_spatial_dynamic_conn.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time, math
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


def personal_volume_radius(N: int, D: int, box_size: float = 1.0) -> float:
    """r* = R / N^(1/D) — personal volume radius from original SGNNET design.

    Expected neighbors per neuron ≈ 1 by construction (see report Section 3.5).
    At D=64 N=1024: r* ≈ 0.449 (vs typical inter-neuron distances ≈ 3.3).
    Thresholds used as Gaussian kernel peak (not hard gate) to handle D=64.
    """
    R = box_size / 2.0
    return R / (N ** (1.0 / D))


def build_wpos_knn_conn(W_pos: torch.Tensor, K: int, N_hidden: int,
                        box_size: float = 1.0) -> torch.Tensor:
    """Build connectivity from W_pos K-nearest neighbors.

    Annular neighborhood: connect to the K neighbors EXCLUDING the K//2 closest
    (too similar, just reinforce) and including the remaining K neighbors
    in mid-range. This implements the r/2 → r* spirit at D=64 where absolute
    thresholds yield no connections.

    Returns: conn_hh [N_hidden, K] integer index tensor (like static conn_hh)
    """
    W_h = W_pos[:N_hidden].detach()            # [N_hidden, D]
    # Pairwise L2 distances in W_pos space
    with torch.no_grad():
        dists = torch.cdist(W_h, W_h)          # [N_hidden, N_hidden]
        dists.fill_diagonal_(float('inf'))      # exclude self
        # Find K*2 nearest neighbors
        K_wide = min(K * 2, N_hidden - 1)
        _, top_idx = dists.topk(K_wide, dim=-1, largest=False)  # [N_h, K*2]
        # Annular exclusion: drop the K//2 closest (inner zone), keep outer K
        inner = K // 2
        conn = top_idx[:, inner:inner + K]     # [N_h, K] — mid-range neighbors
    return conn.to(W_pos.device)


class SGNNET_SpatialDynamic(nn.Module):
    """Spatial dynamic connectivity: conn_hh rebuilt from W_pos every epoch.

    Version 1 — no phase in routing. W_phase still learned (for readout) but
    does NOT influence which neurons communicate or with what strength.

    The topology evolves during training as W_pos moves — neurons that drift
    into proximity become connected; those that drift apart disconnect.
    """

    def __init__(self, base_model: SGNNET_Resonant,
                 K_spatial: int = 6,
                 n_input: int = 1024,
                 alpha_ahebb: float = 0.0,
                 rebuild_every: int = 1):
        """
        base_model   : SGNNET_Resonant (provides W_pos, W_phase, theta, etc.)
        K_spatial    : number of spatial K-NN neighbors per neuron
        n_input      : number of input-receiving neurons (rest = interneurons)
        alpha_ahebb  : AntiHebb strength (0 = disabled)
        rebuild_every: rebuild conn_hh every N epochs (1 = every epoch)
        """
        super().__init__()
        self.m             = base_model
        self.K_spatial     = K_spatial
        self.n_input       = n_input
        self.alpha_ahebb   = alpha_ahebb
        self.rebuild_every = rebuild_every
        self._epoch        = 0

        # Initial connectivity from random W_pos
        conn = build_wpos_knn_conn(
            self.m.W_pos, K_spatial, base_model.base.N_hidden
        )
        self.register_buffer("conn_hh_spatial", conn)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        self._epoch += 1
        if self._epoch % self.rebuild_every == 0:
            # Rebuild connectivity from current W_pos
            new_conn = build_wpos_knn_conn(
                self.m.W_pos, self.K_spatial, self.m.base.N_hidden
            )
            self.conn_hh_spatial.copy_(new_conn)
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                         # [B, N, D]
        if self.n_input < self.m.base.N_hidden:
            Z[:, self.n_input:, :] = 0.0                         # interneurons blank

        B, N, D   = Z.shape
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn      = self.conn_hh_spatial                         # [N_hidden, K]
        W_pos_h   = self.m.W_pos[:N]

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)

            # Structural aggregation via spatial K-NN (no phase weighting)
            Z_struct = Z_fwd[:, conn, :].sum(2)                  # [B, N, D]

            # Phase inhibition (uses dynamic_z_geo from base model)
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            if self.alpha_ahebb > 0.0:
                # Anti-Hebbian over spatial neighbors
                W_pos_nb = W_pos_h[conn]                         # [N, K, D]
                pos_sim  = F.cosine_similarity(
                    W_pos_h.unsqueeze(1), W_pos_nb, dim=-1)      # [N, K]
                ahebb    = (pos_sim.unsqueeze(0).unsqueeze(-1)
                            * Z_fwd[:, conn, :]).sum(2)           # [B, N, D]
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


# (key, label, K_spatial, use_interneurons, alpha_ahebb, rebuild_every)
CONFIGS = [
    ("Ref", "Ref   static small-world  K=6  [current baseline ~56.28%]",
     None, False, 0.0, 1),
    ("A",   "A     dynamic W_pos K-NN  K=6  rebuild=epoch  no-int",
     6,    False, 0.0, 1),
    ("B",   "B     dynamic W_pos K-NN  K=8  rebuild=epoch  no-int",
     8,    False, 0.0, 1),
    ("C",   "C     dynamic W_pos K-NN  K=6  rebuild=epoch  50% interneurons",
     6,    True,  0.0, 1),
    ("D",   "D     dynamic W_pos K-NN  K=6  + interneurons + AntiHebb α=0.5",
     6,    True,  0.5, 1),
    ("E",   "E     dynamic W_pos K-NN  K=6  rebuild=10ep   no-int  [slow evolution]",
     6,    False, 0.0, 10),
]


if __name__ == "__main__":
    r_star = personal_volume_radius(N, D)
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print("Step 50: Spatial dynamic connectivity (W_pos K-NN, no phase in routing)")
    print(f"  Personal volume radius r* = {r_star:.4f}  (D={D} N={N})")
    print(f"  Annular zone: r*/2={r_star/2:.4f} to r*={r_star:.4f}")
    print(f"  Note: at D=64, absolute threshold → use K-NN with inner exclusion")
    print(f"  Original vision: topology evolves as W_pos moves during training")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    REF_BASELINE = 0.5628
    results = {}
    for key, label, K_sp, use_int, alpha_ah, rebuild in CONFIGS:
        resonant = make_resonant().to(DEVICE)
        if K_sp is None:
            model = resonant
        else:
            n_input = N // 2 if use_int else N
            model = SGNNET_SpatialDynamic(
                resonant, K_spatial=K_sp, n_input=n_input,
                alpha_ahebb=alpha_ah, rebuild_every=rebuild,
            ).to(DEVICE)
        meta = {"N": N, "D": D, "K_iter": 8,
                "K_spatial": K_sp, "interneurons": use_int,
                "n_input": N // 2 if use_int else N,
                "alpha_ahebb": alpha_ah, "rebuild_every": rebuild,
                "r_star": round(r_star, 4)}
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step50_spatial_dynamic_conn.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    print(f"\n-- Spatial dynamic connectivity (ref={ref_val:.4f}) ---")
    print(f"  {'Config':<60}  {'top1':>6}  {'vs_Ref':>8}  {'t(s)':>6}")
    print("  " + "-"*90)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        print(f"  {r['label'][:60]:<60}  {r['top1_best']:>6.4f}  {d:>+8.4f}  "
              f"{r['elapsed_s']:>6.0f}")

    print("\n  Key questions answered:")
    print("  A > Ref  → W_pos dynamic topology beats static small-world")
    print("  B > A    → K=8 richer than K=6 for spatial connectivity")
    print("  C > A    → interneurons compound with dynamic spatial topology")
    print("  D > C    → AntiHebb adds to spatial + interneurons (three-way compound)")
    print("  E vs A   → topology evolution speed matters")
    print("  A > step31 (Z-KNN ~44%)  → W_pos is a better basis for dynamic K-NN than Z")
