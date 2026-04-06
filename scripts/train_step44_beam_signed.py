"""Step 44: Beam-restricted signed coupling — O(beam²) instead of O(N²).

GAP G4 — Beam Routing x Signed Coupling
=========================================
Signed coupling is +10.93pp but O(N²D). Step24 static K-NN sparsification
recovered only 37% of the N² gain. But step25 showed beam routing selects
route=64 active neurons INPUT-DEPENDENTLY.

Hypothesis: applying signed coupling ONLY within the beam subset gives
O(beam² × D) = O(64² × 64) = 262k FLOPs vs O(N² × D) = O(1024² × 64) = 67M
FLOPs — 256× cheaper. Because beam selection is input-dependent (unlike step24's
static K-NN), the coupling targets the most active neurons each forward pass.

CONFIGS (D=64 N=1024 K_iter=3 — safe K_iter for signed):
  Ref    no signed coupling  [expect ~40%]
  Full   full N² signed α=0.05  [step42-informed alpha]
  A      beam=64  signed within beam only
  B      beam=128 signed within beam only
  C      beam=256 signed within beam only
  D      beam=64  signed + anti-Hebbian  [compound test]

Alpha is fixed at 0.05 (moderate; step42 will refine, but 0.05 is safe from
step42's sweep range). If step42 finds a better alpha before this runs,
update before launching.

To reproduce:
    python -u scripts/train_step44_beam_signed.py --device mps
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
ALPHA_SIGNED = 0.05   # Conservative default; update from step42 if available

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


class SGNNET_BeamSigned(nn.Module):
    """Beam-restricted signed coupling: O(beam² × D) per routing step.

    Only the top-beam_signed active neurons participate in the coupling matrix.
    The coupling signal is then broadcast back to ALL N neurons, but the
    N² cost becomes beam² since only beam neurons form the interaction matrix.
    """

    def __init__(self, base: SGNNET_Resonant, beam_signed: int = 64,
                 alpha_signed: float = 0.05):
        super().__init__()
        self.m = base
        self.beam_signed = beam_signed
        self.alpha_signed = alpha_signed

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
        M         = min(self.beam_signed, N)

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            # Beam selection: top-M neurons by activation magnitude
            Z_norm   = F.normalize(Z, dim=-1)                         # [B, N, D]
            activity = Z.norm(dim=-1)                                  # [B, N]
            top_idx  = activity.topk(M, dim=-1).indices                # [B, M]

            # Gather beam neurons
            Z_beam = torch.gather(
                Z_norm, 1, top_idx.unsqueeze(-1).expand(-1, -1, D)
            )                                                          # [B, M, D]

            # Signed coupling within beam: O(M² × D)
            coupling = torch.bmm(Z_beam, Z_beam.transpose(1, 2))     # [B, M, M]

            # Broadcast coupling signal from beam to ALL neurons
            # score[b, n] = Z_norm[b, n] @ Z_beam[b, :, :].T → [B, N, M]
            score_all = torch.bmm(Z_norm, Z_beam.transpose(1, 2))    # [B, N, M]
            # Z_signed[b, n] = Σ_m score_all[b,n,m] * Z_beam[b,m,:] / M
            Z_signed = torch.bmm(score_all, Z_beam) / M              # [B, N, D]

            Z = F.normalize(
                (Z_struct + self.m.alpha_turing * Z_inh
                 + self.alpha_signed * Z_signed).clamp(-10, 10),
                dim=-1,
            )

        return self.m.base._readout(Z)


class SGNNET_FullSigned(nn.Module):
    """Full N² signed coupling baseline (same as step42 but standalone)."""

    def __init__(self, base: SGNNET_Resonant, alpha_signed: float = 0.05):
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
        Z         = self.m.base._seed(x)
        B, N, D   = Z.shape
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            Z_norm   = F.normalize(Z, dim=-1)
            coupling = torch.bmm(Z_norm, Z_norm.transpose(1, 2))
            Z_signed = torch.bmm(coupling, Z_norm) / N

            Z = F.normalize(
                (Z_struct + self.m.alpha_turing * Z_inh
                 + self.alpha_signed * Z_signed).clamp(-10, 10),
                dim=-1,
            )

        return self.m.base._readout(Z)


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="cosine")
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


CONFIGS = [
    ("Ref",  "Ref   D=64 K_iter=3 no-signed  [expect ~40%]",
     lambda: make_resonant(N=1024, D=64, K_iter=3)),
    ("Full", f"Full  D=64 K_iter=3 full N² signed α={ALPHA_SIGNED}",
     lambda: SGNNET_FullSigned(make_resonant(N=1024, D=64, K_iter=3), ALPHA_SIGNED)),
    ("A",    f"A     beam=64  signed α={ALPHA_SIGNED}  [O(64²D)]",
     lambda: SGNNET_BeamSigned(make_resonant(N=1024, D=64, K_iter=3), 64, ALPHA_SIGNED)),
    ("B",    f"B     beam=128 signed α={ALPHA_SIGNED}  [O(128²D)]",
     lambda: SGNNET_BeamSigned(make_resonant(N=1024, D=64, K_iter=3), 128, ALPHA_SIGNED)),
    ("C",    f"C     beam=256 signed α={ALPHA_SIGNED}  [O(256²D)]",
     lambda: SGNNET_BeamSigned(make_resonant(N=1024, D=64, K_iter=3), 256, ALPHA_SIGNED)),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print(f"Step 44: Beam-restricted signed coupling at D=64")
    print(f"Gap G4: O(N²) → O(beam²) with input-dependent beam selection")
    print(f"alpha_signed={ALPHA_SIGNED}")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    for key, label, model_fn in CONFIGS:
        model = model_fn().to(DEVICE)
        meta  = {"N": 1024, "D": 64, "K_iter": 3, "alpha_signed": ALPHA_SIGNED}
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step44_beam_signed.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_val = results.get("Ref", {}).get("top1_best", 0.40)
    full_val = results.get("Full", {}).get("top1_best", 0.0)
    print(f"\n-- Beam-restricted signed coupling (ref={ref_val:.4f}, full_N²={full_val:.4f}) ---")
    print(f"  {'Config':<55}  {'top1':>6}  {'vs_Ref':>8}  {'%_N²':>6}  {'t(s)':>6}")
    print("  " + "-"*85)
    n2_gain = max(full_val - ref_val, 0.001)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        pct = d / n2_gain * 100 if n2_gain > 0 else 0
        print(f"  {r['label'][:55]:<55}  {r['top1_best']:>6.4f}  {d:>+8.4f}  {pct:>5.0f}%  "
              f"{r['elapsed_s']:>6.0f}")

    print("\n  Key metric: %_N² = beam gain / full N² gain × 100")
    print("  Target: beam=64 recovers >80% of N² gain at 1/256th compute")
