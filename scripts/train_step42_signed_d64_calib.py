"""Step 42: Signed coupling calibration at D=64.

GAP G1 — Signed Coupling x High-D Calibration
================================================
Signed coupling is +10.93pp at D=16 (the largest single mechanism gain) but
REGRESSES to 32.15% at D=64 K_iter=3 (step28) vs 40% unsigned (step29 Ref0).

Root cause: alpha_signed=0.3 (tuned at D=16 S^15) is too strong on S^63 where
cosine similarities concentrate near zero. The coupling signal overwhelms the
structural signal at the same alpha. Also K_iter≥8 + signed = eigenvector collapse.

This experiment calibrates signed coupling alpha at D=64 K_iter=3 (the safe
K_iter for signed) with much lower alpha values than the D=16 default.

PHASE 1 (40ep calibration):
  alpha_signed ∈ {0.005, 0.01, 0.03, 0.05, 0.1, 0.3}
  All at D=64 N=1024 K_iter=3, cosine schedule

PHASE 2 (150ep):
  Ref       D=64 K_iter=3 no-signed  [expect ~40% from step29 Ref0]
  Best_α    best alpha from Phase 1
  2nd_best  runner-up alpha
  + AntiHebb  best_α + AntiHebb α=0.5  [test compound with G2 in mind]

SECONDARY QUESTION: if low-alpha signed works at K_iter=3, does K_iter=5 work?
  Test at best_α with K_iter=5 as a bonus config.

To reproduce:
    python -u scripts/train_step42_signed_d64_calib.py --device mps
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

CALIB_EPOCHS = 40
FULL_EPOCHS  = 150
BATCH        = 128
SEED         = 42
DATA         = "data/store.h5"

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_resonant(N=1024, D=64, K_iter=3) -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_iter,
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )


class SGNNET_SignedCoupling(nn.Module):
    """Signed coupling: Z_signed = alpha * (Z @ Z.T) @ Z / N.

    O(N^2 D) per routing step — acceptable for calibration at N=1024.
    """

    def __init__(self, base: SGNNET_Resonant, alpha_signed: float = 0.1):
        super().__init__()
        self.m = base
        self.alpha_signed = alpha_signed

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                              # [B, N, D]
        B, N, D   = Z.shape
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)     # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)               # [N, D]
        conn_hh   = self.m.base.conn_hh                               # [N, K_hh]

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)                          # [B, N, D]
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)                   # [B, N, D]
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            # Signed coupling: Z_signed = (Z_norm @ Z_norm.T) @ Z_norm / N
            Z_norm   = F.normalize(Z, dim=-1)                         # [B, N, D]
            coupling = torch.bmm(Z_norm, Z_norm.transpose(1, 2))     # [B, N, N]
            Z_signed = torch.bmm(coupling, Z_norm) / N                # [B, N, D]

            Z = F.normalize(
                (Z_struct + self.m.alpha_turing * Z_inh
                 + self.alpha_signed * Z_signed).clamp(-10, 10),
                dim=-1,
            )

        return self.m.base._readout(Z)


class SGNNET_SignedAntiHebb(nn.Module):
    """Signed coupling + Anti-Hebbian combined in one routing loop."""

    def __init__(self, base: SGNNET_Resonant, alpha_signed: float = 0.1,
                 alpha_ahebb: float = 0.5):
        super().__init__()
        self.m = base
        self.alpha_signed = alpha_signed
        self.alpha_ahebb  = alpha_ahebb

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
        conn_hh   = self.m.base.conn_hh
        W_pos_h   = self.m.W_pos[:N]                                  # [N, D]

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            # Signed coupling
            Z_norm   = F.normalize(Z, dim=-1)
            coupling = torch.bmm(Z_norm, Z_norm.transpose(1, 2))
            Z_signed = torch.bmm(coupling, Z_norm) / N

            # Anti-Hebbian: suppress Z toward similar W_pos neighbors
            W_pos_nb = W_pos_h[conn_hh]                               # [N, K_hh, D]
            W_self   = W_pos_h.unsqueeze(1)                           # [N, 1, D]
            pos_sim  = F.cosine_similarity(W_self, W_pos_nb, dim=-1)  # [N, K_hh]
            ahebb    = (pos_sim.unsqueeze(0).unsqueeze(-1)             # [1, N, K_hh, 1]
                        * Z_fwd[:, conn_hh, :]).sum(2)                # [B, N, D]

            Z = F.normalize(
                (Z_struct + self.m.alpha_turing * Z_inh
                 + self.alpha_signed * Z_signed
                 - self.alpha_ahebb * ahebb).clamp(-10, 10),
                dim=-1,
            )

        return self.m.base._readout(Z)


def run(label: str, model: nn.Module, meta: dict, n_epochs: int) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(meta["N"], n_epochs=n_epochs, sched_type="cosine")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=n_epochs)
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
        "_meta": run_metadata(__file__, {**meta, "epochs": n_epochs}),
    }
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
          f"  diag={result['convergence_diag']}  t={elapsed:.0f}s")
    return result


ALPHA_SWEEP = [0.005, 0.01, 0.03, 0.05, 0.1, 0.3]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Calib: {CALIB_EPOCHS}ep  Full: {FULL_EPOCHS}ep")
    print("Step 42: Signed coupling calibration at D=64")
    print("Gap G1: alpha=0.3 from D=16 is too strong on S^63")
    print("Known: D=64 K_iter=3 no-signed = 40% (step29 Ref0)")
    print("Known: D=64 K_iter=3 signed α=0.3 = 32.15% (step28)")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {"calibration": {}, "full": {}}

    # ── Phase 1: alpha sweep at D=64 K_iter=3 ────────────────────────────────
    print(f"\n{'#'*70}")
    print("PHASE 1: signed alpha sweep — D=64 N=1024 K_iter=3, cosine, 40ep")
    print(f"{'#'*70}")

    calib_results = {}
    for alpha in ALPHA_SWEEP:
        key   = f"alpha_{alpha}"
        label = f"calib  signed α={alpha}  D=64 K_iter=3"
        model = SGNNET_SignedCoupling(
            make_resonant(N=1024, D=64, K_iter=3), alpha_signed=alpha,
        ).to(DEVICE)
        meta  = {"N": 1024, "D": 64, "K_iter": 3, "alpha_signed": alpha}
        r     = run(label, model, meta, n_epochs=CALIB_EPOCHS)
        r.update(meta)
        calib_results[key] = r
        results["calibration"][key] = r

    best_key  = max(calib_results, key=lambda k: calib_results[k]["top1_best"])
    best_alpha = calib_results[best_key]["alpha_signed"]
    best_top1  = calib_results[best_key]["top1_best"]
    sorted_keys = sorted(calib_results, key=lambda k: calib_results[k]["top1_best"], reverse=True)
    second_alpha = calib_results[sorted_keys[1]]["alpha_signed"] if len(sorted_keys) > 1 else best_alpha

    print(f"\n  Alpha winner: α={best_alpha}  top1@40ep={best_top1:.4f}")
    print(f"  Runner-up:    α={second_alpha}")

    results["best_params"] = {
        "alpha_signed": best_alpha,
        "top1_calib": best_top1,
    }

    # ── Phase 2: full 150ep runs ──────────────────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"PHASE 2: Full {FULL_EPOCHS}ep runs")
    print(f"{'#'*70}")

    full_configs = [
        ("Ref", f"Ref  D=64 K_iter=3 no-signed [expect ~40%]",
         lambda: make_resonant(N=1024, D=64, K_iter=3)),
        ("Best", f"Best  D=64 K_iter=3 signed α={best_alpha}",
         lambda: SGNNET_SignedCoupling(
             make_resonant(N=1024, D=64, K_iter=3), alpha_signed=best_alpha)),
        ("Second", f"2nd  D=64 K_iter=3 signed α={second_alpha}",
         lambda: SGNNET_SignedCoupling(
             make_resonant(N=1024, D=64, K_iter=3), alpha_signed=second_alpha)),
        ("BestAH", f"Best+AH  D=64 K_iter=3 signed α={best_alpha} + AntiHebb 0.5",
         lambda: SGNNET_SignedAntiHebb(
             make_resonant(N=1024, D=64, K_iter=3),
             alpha_signed=best_alpha, alpha_ahebb=0.5)),
        ("K5", f"K5  D=64 K_iter=5 signed α={best_alpha} [push K_iter]",
         lambda: SGNNET_SignedCoupling(
             make_resonant(N=1024, D=64, K_iter=5), alpha_signed=best_alpha)),
    ]

    for key, label, model_fn in full_configs:
        model = model_fn().to(DEVICE)
        meta  = {"N": 1024, "D": 64, "K_iter": 3 if "K5" not in key else 5,
                 "alpha_signed": best_alpha}
        r     = run(label, model, meta, n_epochs=FULL_EPOCHS)
        r.update(meta)
        results["full"][key] = r

    # ── Save ──────────────────────────────────────────────────────────────────
    out = ROOT / "results" / "train_step42_signed_d64_calib.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_val = results["full"].get("Ref", {}).get("top1_best", 0.40)
    print(f"\n-- Signed coupling at D=64 (ref no-signed={ref_val:.4f}) ---")
    print(f"  {'Config':<55}  {'top1':>6}  {'vs_Ref':>8}  {'t(s)':>6}")
    print("  " + "-"*80)
    for k, r in results["full"].items():
        d = r["top1_best"] - ref_val
        print(f"  {r['label'][:55]:<55}  {r['top1_best']:>6.4f}  {d:>+8.4f}  "
              f"{r['elapsed_s']:>6.0f}")

    print("\n  Interpretation:")
    print("  Best > 40%   → signed CAN work at D=64 with lower alpha")
    print("  Best < 40%   → signed fundamentally incompatible with S^63 routing")
    print("  BestAH > Best → AntiHebb + signed compound; include both in Gen4")
    print("  K5 > Best    → K_iter=5 is safe with calibrated alpha; try K_iter=6-7")
    print("  K5 < Best    → eigenvector collapse starts at K_iter=5; K_iter=3 is limit")
