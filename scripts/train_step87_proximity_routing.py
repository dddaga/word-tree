"""Step 87: Pure proximity routing — no static conn_hh.

MOTIVATION
==========
All dynamic routing experiments failed via gate-death (multiplicative g^K_iter
collapses signal). Static AH works because W_pos updates happen once per epoch
(no multiplicative compounding within the forward pass).

USER HYPOTHESIS (2026-04-09): Build an architecture with NO pre-built conn_hh.
Connectivity is computed at each forward pass from W_pos proximity:
  - W_pos[h] ∈ S^{D-1}: position embedding (slow, trained by gradient)
  - Z[h]: activation state (fast, within-forward dynamics)
  - Top-K nearest neighbors in W_pos space determine who Z aggregates from
  - No gates — pure sum + normalize (gate-death impossible)

KEY CONFLICT IDENTIFIED:
AH wpos suppression (1 - α·pos_sim) is INCOMPATIBLE with proximity routing.
Proximity selects highest-similarity neighbors → pos_sim ≈ 1 → supp_w ≈ 0.
All signal would be zeroed. This experiment directly tests this conflict.

CONFIGS (N=1024, D=64, K_iter=8, 50%/75ep)
============================================
  Ref : static conn_hh + AH wpos (existing best — control)
  A   : proximity routing, no AH  (test: does topology learning help at all?)
  B   : proximity routing + AH zact (Z-based suppression — avoids conflict)
  C   : proximity routing + AH wpos (expected FAIL: pos_sim≈1 → supp_w≈0)
  D   : proximity routing, no AH, K_iter=12 (depth benefit with dynamic topo?)

All: turing=0.0, reflect=0.5, N=1024, D=64.

KEY HYPOTHESES
==============
  A vs Ref   : can learned dynamic topology replace static topology?
  B vs A     : does AH zact + proximity = compatible decorrelation?
  C vs A     : does AH wpos kill proximity routing? (conflict hypothesis)
  D vs A     : does K_iter=12 benefit transfer to proximity routing?

If A > Ref: dynamic topology works! Scale to N=4096.
If B > A:   zact is the right AH variant for proximity.
If C ≈ A/Ref dead: conflict confirmed — never use wpos+proximity.

KEY DIFFERENCES FROM STEP 50 (FAILED: dynamic W_pos K-NN)
==========================================================
  - step50: dynamic ON TOP of static conn_hh (hybrid). step87: NO conn_hh.
  - step50: pre-AH, buggy arch. step87: post-patch (+9.83pp) with AH.
  - step50: used Z (volatile) for KNN. step87: uses W_pos (stable) for KNN.

To reproduce:
    python -u scripts/train_step87_proximity_routing.py --device mps
    python -u scripts/train_step87_proximity_routing.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld       import SGNNET_SmallWorld
from src.sgnnet.model_resonant         import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory  import SGNNET_AntiHebbian
from src.training.trainer              import Trainer
from src.training.experiment_config    import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset              import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS    = 75
BATCH     = 128
SEED      = 42
DATA      = "data/store.h5"
N         = 1024
N_IN      = 25088
N_OUT     = 10
D         = 64
ALPHA_REFLECT = 0.5
STEP69_REF    = 0.8336   # static routing + AH wpos, patched arch, 50%/75ep


# ── Self-contained proximity routing model ────────────────────────────────────

class SGNNET_Proximity(nn.Module):
    """SGNNET with dynamic W_pos-proximity routing (no static conn_hh).

    Forward pass:
      1. Seed from input via fixed conn_in (same as static model)
      2. K_iter routing steps using top-K nearest W_pos neighbors
         (neighbors recomputed once per forward pass, stable within pass)
      3. Optional AH suppression via 'zact' variant (Z-space, not W_pos-space)
      4. Readout via W_pos dot-product (same as static model)
    """

    def __init__(
        self,
        N_in: int, N_hidden: int, N_out: int,
        K_in: int, K_hh: int, K_iter: int, D: int,
        n_groups: int,
        alpha_ahebb: float = 0.0,   # 0 = no AH; >0 = zact AH
        alpha_reflect: float = 0.5,
        seed: int = 42,
    ):
        super().__init__()
        self.N_hidden     = N_hidden
        self.K_hh         = K_hh
        self.K_iter       = K_iter
        self.D            = D
        self.alpha_ahebb  = alpha_ahebb
        self.alpha_reflect = alpha_reflect

        # Build the static components from SGNNET_SmallWorld
        # (We borrow its conn_in, spatial_coords, C_ho_mask, W_pos initialization)
        torch.manual_seed(seed)
        _base = SGNNET_SmallWorld(
            N_in=N_in, N_hidden=N_hidden, N_out=N_out,
            K_local=2, K_random=2, K_in=K_in, K_iter=K_iter,
            n_groups=n_groups, norm_mode="l2", D=D, encoding_mode="fourier",
        )
        # Re-use fixed buffers from base
        self.register_buffer("conn_in",       _base.conn_in)
        self.register_buffer("spatial_coords",_base.spatial_coords)
        self.register_buffer("C_ho_mask",     _base.C_ho_mask)

        # Learnable parameters
        self.W_pos = nn.Parameter(_base.W_pos.data.clone())  # [N+N_out, D]

        # Per-neuron threshold (same as SGNNET_Resonant.theta, init=0.1)
        self.theta = nn.Parameter(torch.full((N_hidden,), 0.1))

    @property
    def W_phase(self):
        # Stub: proximity model doesn't use W_phase but Trainer may query it
        return self.W_pos[:self.N_hidden]

    def _seed(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        spatial = self.spatial_coords.unsqueeze(0).expand(B, -1, -1)
        A_input = torch.cat([x.unsqueeze(-1), spatial], dim=-1)   # [B, N_in, D]
        Z = A_input[:, self.conn_in, :].sum(dim=2)                 # [B, N, D]
        return F.normalize(Z, dim=-1)

    def _get_neighbors(self) -> torch.Tensor:
        """Compute top-K nearest neighbors in W_pos space. [N, K_hh]"""
        W_norm = F.normalize(self.W_pos[:self.N_hidden], dim=-1)   # [N, D]
        sim    = W_norm @ W_norm.T                                  # [N, N]
        # topk includes self (index 0), skip it
        nbrs   = sim.topk(self.K_hh + 1, dim=1).indices[:, 1:]    # [N, K_hh]
        return nbrs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z    = self._seed(x)                                         # [B, N, D]
        nbrs = self._get_neighbors()                                 # [N, K_hh]
        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)      # [1, N, 1]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.K_iter):
            Z_fwd  = F.relu(Z - theta_pos)                          # [B, N, D]
            Z_nb   = Z_fwd[:, nbrs, :]                              # [B, N, K_hh, D]

            if self.alpha_ahebb > 0.0:
                # zact: dynamic cosine suppression in current Z space
                Z_n   = F.normalize(Z, dim=-1)                      # [B, N, D]
                z_sim = (Z_n.unsqueeze(2)
                         * F.normalize(Z_nb, dim=-1)).sum(-1)       # [B, N, K_hh]
                supp  = (1.0 - self.alpha_ahebb
                         * z_sim.clamp(min=0)).unsqueeze(-1)        # [B, N, K_hh, 1]
                Z_struct = (Z_nb * supp).sum(dim=2)                 # [B, N, D]
            else:
                Z_struct = Z_nb.sum(dim=2)                          # [B, N, D]

            # Reflection: accumulate what the threshold suppressed
            Z_remainder = Z_fwd - Z
            Z_reflected = self.alpha_reflect * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        # Readout: same as SGNNET_SmallWorld._readout
        C_ho  = self.C_ho_mask.float()
        A_out = torch.einsum("bhd,ho->bod", Z, C_ho)               # [B, N_out, D]
        W_out = F.normalize(self.W_pos[self.N_hidden:], dim=-1)    # [N_out, D]
        return (A_out * W_out.unsqueeze(0)).sum(dim=-1)             # [B, N_out]


# ── Configs ───────────────────────────────────────────────────────────────────

@dataclass
class Config:
    key:        str
    label:      str
    K_hh:       int
    K_iter:     int
    proximity:  bool
    alpha_ah:   float = 0.0   # 0 = no AH; 1.0 = AH zact; used in static Ref via wrapper


CONFIGS = [
    Config("Ref", "Ref  static conn_hh + AH wpos  K_hh=6 K_iter=8 (patched baseline)",
           6, 8, False, 1.0),
    Config("A",   "A    proximity, no AH           K_hh=4 K_iter=8",
           4, 8, True, 0.0),
    Config("B",   "B    proximity + AH zact        K_hh=4 K_iter=8",
           4, 8, True, 1.0),
    Config("C",   "C    proximity + AH wpos        K_hh=4 K_iter=8  (CONFLICT test)",
           4, 8, False, 1.0),   # wpos via static wrapper forces conflict
    Config("D",   "D    proximity, no AH           K_hh=4 K_iter=12 (depth test)",
           4, 12, True, 0.0),
]


_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n   = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(
            subset, batch_size=BATCH, shuffle=True, num_workers=0
        )
        _loaders = (tr, va)
    return _loaders


def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    tk = topology_kwargs(N)

    if cfg.proximity:
        return SGNNET_Proximity(
            N_in=N_IN, N_hidden=N, N_out=N_OUT,
            K_in=tk["K_in"], K_hh=cfg.K_hh, K_iter=cfg.K_iter, D=D,
            n_groups=tk["n_groups"],
            alpha_ahebb=cfg.alpha_ah, alpha_reflect=ALPHA_REFLECT,
            seed=SEED + seed_offset,
        )
    else:
        # Ref or Config C: static routing (small-world conn_hh) + AH wpos wrapper
        K_local = max(0, cfg.K_hh - tk["K_random"])
        base = SGNNET_SmallWorld(
            N_in=N_IN, N_hidden=N, N_out=N_OUT,
            K_local=K_local, K_random=tk["K_random"],
            K_in=tk["K_in"], K_iter=cfg.K_iter, n_groups=tk["n_groups"],
            norm_mode="l2", D=D, encoding_mode="fourier",
        )
        resonant = SGNNET_Resonant(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0,
        )
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=cfg.alpha_ah, variant="wpos")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run(cfg: Config, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{cfg.label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best      = max(top1_hist)
    best_ep   = int(np.argmax(top1_hist)) + 1
    frac      = best_ep / len(history)

    result = {
        "label":            cfg.label,
        "K_hh":             cfg.K_hh,
        "K_iter":           cfg.K_iter,
        "proximity":        cfg.proximity,
        "alpha_ahebb":      cfg.alpha_ah,
        "top1_best":        best,
        "top1_last":        history[-1].get("val_top1", 0.0),
        "final_task_loss":  float(np.mean([h.get("task_loss", 0.0) for h in history[-5:]])),
        "best_epoch":       best_ep,
        "epochs_run":       len(history),
        "elapsed_s":        round(elapsed, 1),
        "best_epoch_frac":  round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history":     top1_hist,
        "step69_ref":       STEP69_REF,
        "delta_vs_step69":  round(best - STEP69_REF, 4),
        "params":           count_params(model),
        "_meta":            run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  vs_step69={best-STEP69_REF:+.4f}  t={elapsed:.0f}s"
    )
    return result


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  N={N}  Data: 50%")
    print(f"Step 87: Pure proximity routing — no static conn_hh")
    print(f"Baseline: step69 Ref = {STEP69_REF:.4f}")
    print()
    for cfg in CONFIGS:
        ah = f"AH_zact={cfg.alpha_ah}" if cfg.proximity else f"AH_wpos={cfg.alpha_ah}"
        print(f"  {cfg.key:4s}  {'proximity' if cfg.proximity else 'static':9s}"
              f"  K_hh={cfg.K_hh}  K_iter={cfg.K_iter}  {ah}")
    print()

    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step87_proximity_routing.json"

    for i, cfg in enumerate(CONFIGS):
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        meta  = {
            "N": N, "D": D, "K_iter": cfg.K_iter, "K_hh": cfg.K_hh,
            "proximity": cfg.proximity, "alpha_ahebb": cfg.alpha_ah,
            "data_frac": 0.5,
        }
        results[cfg.key] = run(cfg, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    print(f"\n{'='*70}")
    print(f"STEP 87 COMPLETE — Proximity Routing Ablation")
    print(f"Baseline: {STEP69_REF:.4f}")
    print()
    print(f"  {'Key':4s}  {'routing':9s}  K_hh  K_iter  AH     top1      vs_ref")
    for cfg in CONFIGS:
        if cfg.key not in results:
            continue
        r  = results[cfg.key]
        ah = f"zact={cfg.alpha_ah}" if cfg.proximity else f"wpos={cfg.alpha_ah}"
        print(f"  {cfg.key:4s}  {'proximity' if cfg.proximity else 'static':9s}"
              f"  {cfg.K_hh:>4}  {cfg.K_iter:>6}  {ah:6s}"
              f"  {r['top1_best']:.4f}    {r['delta_vs_step69']:+.4f}")
    winner = max(results, key=lambda k: results[k]["top1_best"])
    w = results[winner]
    print(f"\n  Winner: {winner}  ({w['top1_best']:.4f})")
    prox_winner = max(
        (k for k in results if CONFIGS[[c.key for c in CONFIGS].index(k)].proximity),
        key=lambda k: results[k]["top1_best"],
        default=None
    )
    if prox_winner and results[prox_winner]["top1_best"] > STEP69_REF:
        print(f"  → Proximity routing BEATS static baseline. Scale to N=4096.")
    else:
        print(f"  → Proximity routing does not beat static. Check curves for gate-death.")
