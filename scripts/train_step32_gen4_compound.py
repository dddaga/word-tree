"""Step 32: Gen4 compound configuration — systematic stacking of ARM 1+2 winners.

GEN4 CONFIG SELECTION RATIONALE
================================
From step29c Phase 1 calibration (40ep each, calibrated base from step22b):
  AntiHebb alpha sweep:     0.1→48.56%  0.3→54.42%  0.5→59.82%  0.7→64.69%  1.0→70.98%
  Phase excitatory alpha:   0.1→57.91%  0.3→59.34%  0.5→45.83%  1.0→33.43%
  Fast W_phase:             0.1/0.25→49.45%  0.3/0.25→49.30%  (DEAD — D x D interference)

From step48 (K_iter sweep, partial):
  K_iter=8  no-AH: 58.24%    K_iter=12 no-AH: 55.08%  → deeper routing HURTS without AH
  K_iter=16+ with AH: pending — decision deferred

From step54 (LR schedule, partial):
  Plateau: 70.78%   WarmRestarts: 66.98%  → plateau wins; constant LR when loss improving

GEN4 BASE = D=64 N=1024 K_iter=8 AntiHebb alpha=1.0 + plateau LR

CONFIGS (progressive stacking):
  Ref    AntiHebb alpha=0.7 [step29 Config C = 75.24%, current best]
  A      AntiHebb alpha=1.0 [step29c Phase 1 winner — calibrated base]
  B      A + phase_exc alpha=0.1 [low-dose teleportation]
  C      A + phase_exc alpha=0.3 [medium teleportation]
  D      A + centering diversity lambda=0.05 [global drift suppression from step52 design]
  E      A + phase_exc alpha=0.1 + centering diversity lambda=0.05
  F      AntiHebb alpha=0.7 uncalibrated [replicate 75.24% to confirm reproducibility]
  G      A + alpha_reflect=0.5 [calibrated reflect from step22b]

Key questions:
  A > Ref (75.24%) → calibrated base + stronger AntiHebb exceeds uncalibrated best
  B > A            → phase_exc at low dose compounds with maximal AntiHebb
  E > B, D         → triple-mechanism stacking (AH + exc + centering) compounds
  F ~ 75.24%       → step29 Config C is reproducible

To reproduce:
    python -u scripts/train_step32_gen4_compound.py --device mps
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

EPOCHS = 75
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
N      = 1024
D      = 64

# step22b calibrated params (loaded from results)
CALIB_ALPHA_REFLECT = 0.5    # step22b winner (vs default 0.3)
CALIB_ALPHA_TURING  = 0.3    # step22b: not significantly different from default
CALIB_K_PHASE       = 8
CALIB_BEAM_SIZE     = 32
CALIB_GEO_GAMMA     = 1.0

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n   = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


def make_resonant(alpha_reflect: float = 0.3) -> SGNNET_Resonant:
    """Build D=64 N=1024 K_iter=8 model with configurable alpha_reflect."""
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
        base=base, K_phase=CALIB_K_PHASE, beam_size=CALIB_BEAM_SIZE,
        theta_init=0.1, alpha_reflect=alpha_reflect,
        alpha_turing=CALIB_ALPHA_TURING,
        mode="dynamic_z_geo", resonance_threshold=0.0,
        geo_gamma=CALIB_GEO_GAMMA,
    )


# ── Mechanism wrappers ──────────────────────────────────────────────────────

class SGNNET_Gen4(nn.Module):
    """Gen4 compound model: AntiHebb + optional phase_exc + optional centering.

    Combines:
      1. AntiHebb (W_pos spatial surround suppression) — confirmed primary driver
      2. Phase excitatory teleportation (optional) — low-dose long-range excitation
      3. Centering diversity (optional) — global drift suppression
    """

    def __init__(
        self,
        base_model:     SGNNET_Resonant,
        alpha_ahebb:    float = 1.0,
        alpha_exc:      float = 0.0,
        lambda_center:  float = 0.0,
    ):
        super().__init__()
        self.m             = base_model
        self.alpha_ahebb   = alpha_ahebb
        self.alpha_exc     = alpha_exc
        self.lambda_center = lambda_center

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
            W_pos_nb = W_pos_h[conn_hh]
            pos_sim  = F.cosine_similarity(
                W_pos_h.unsqueeze(1), W_pos_nb, dim=-1
            )

        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)

            # Structural aggregation with AntiHebb suppression
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)
            if self.alpha_ahebb > 0.0:
                ahebb = (pos_sim.unsqueeze(0).unsqueeze(-1)
                         * Z_fwd[:, conn_hh, :]).sum(2)
                Z_struct = Z_struct - self.alpha_ahebb * ahebb

            # Phase excitatory teleportation (optional)
            if self.alpha_exc > 0.0:
                Z_exc = Z_fwd[:, self.m.conn_phase, :].sum(2)
                Z_struct = Z_struct + self.alpha_exc * Z_exc

            # Long-range phase inhibition
            Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            Z_update = Z_struct + self.m.alpha_turing * Z_inh

            # Centering diversity (optional)
            if self.lambda_center > 0.0:
                Z_mean   = Z.mean(dim=1, keepdim=True).detach()
                Z_update = Z_update - self.lambda_center * Z_mean

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

# (key, label, alpha_reflect, alpha_ahebb, alpha_exc, lambda_center)
CONFIGS = [
    ("Ref", "Ref   AntiHebb alpha=0.7 [step29 Config C = 75.24%]",
     0.3, 0.7, 0.0, 0.0),

    ("A", "A     AntiHebb alpha=1.0 calibrated base [step29c P1 winner]",
     CALIB_ALPHA_REFLECT, 1.0, 0.0, 0.0),

    ("B", "B     A + phase_exc alpha=0.1 [low-dose teleportation]",
     CALIB_ALPHA_REFLECT, 1.0, 0.1, 0.0),

    ("C", "C     A + phase_exc alpha=0.3 [medium teleportation]",
     CALIB_ALPHA_REFLECT, 1.0, 0.3, 0.0),

    ("D", "D     A + centering diversity lambda=0.05",
     CALIB_ALPHA_REFLECT, 1.0, 0.0, 0.05),

    ("E", "E     A + phase_exc alpha=0.1 + centering lambda=0.05 [triple]",
     CALIB_ALPHA_REFLECT, 1.0, 0.1, 0.05),

    ("F", "F     AntiHebb alpha=0.7 uncalibrated [75.24% replication]",
     0.3, 0.7, 0.0, 0.0),

    ("G", "G     AntiHebb alpha=1.0 + calibrated alpha_reflect=0.5",
     CALIB_ALPHA_REFLECT, 1.0, 0.0, 0.0),
]


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print("Step 32: Gen4 compound configuration")
    print("  Base: D=64 N=1024 K_iter=8 Fourier dynamic_z_geo plateau")
    print("  Primary: AntiHebb alpha=1.0 (step29c calibrated winner)")
    print("  Secondary: phase_exc (low dose), centering diversity")
    print(f"  Calibrated params: alpha_reflect={CALIB_ALPHA_REFLECT},"
          f" alpha_turing={CALIB_ALPHA_TURING}")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    REF_BASELINE = 0.7524
    results = {}
    for key, label, alpha_r, alpha_ah, alpha_exc, lam_c in CONFIGS:
        resonant = make_resonant(alpha_reflect=alpha_r).to(DEVICE)
        model    = SGNNET_Gen4(
            resonant, alpha_ahebb=alpha_ah,
            alpha_exc=alpha_exc, lambda_center=lam_c,
        ).to(DEVICE)
        meta = {"N": N, "D": D, "K_iter": 8,
                "alpha_ahebb": alpha_ah, "alpha_reflect": alpha_r,
                "alpha_exc": alpha_exc, "lambda_center": lam_c}
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step32_gen4_compound.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    print(f"\n-- Gen4 Compound (ref={ref_val:.4f}) ---")
    print(f"  {'Config':<60}  {'top1':>6}  {'vs_Ref':>8}"
          f"  {'params':>8}  {'t(s)':>6}")
    print("  " + "-"*100)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        print(f"  {r['label'][:60]:<60}  {r['top1_best']:>6.4f}  {d:>+8.4f}"
              f"  {r.get('n_params', 0):>8}  {r['elapsed_s']:>6.0f}")

    winner = max(results.values(), key=lambda r: r["top1_best"])
    print(f"\n  GEN4 WINNER: {winner['label']}")
    print(f"  top1_best = {winner['top1_best']:.4f}")
    print(f"  vs step29 Config C (75.24%): {winner['top1_best'] - 0.7524:+.4f}")
