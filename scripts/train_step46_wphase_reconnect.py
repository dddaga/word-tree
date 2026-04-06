"""Step 46: Reconnect W_phase to routing in dynamic_z_geo mode.

GAP G11 — W_phase Disconnected in Best Config
================================================
In dynamic_z_geo mode (used by ALL D=64 experiments), _inhibit_dynamic_z()
ignores W_ph_norm entirely. W_phase gets gradients only from readout scoring,
NOT from routing. The architecture's namesake mechanism is inactive.

This experiment reconnects W_phase to the routing loop in dynamic_z_geo:
  A) Phase-gated inhibition: multiply dynamic Z-similarity scores by
     dot(W_phase[h], W_phase[k]) — neurons only inhibit phase-aligned neighbors
  B) Phase-weighted structural: W_phase biases structural message aggregation
     score_hk = dot(W_phase[h], W_phase[conn_hh[k]]) — phase-aligned neighbors
     contribute more to Z_struct
  C) Phase routing bias: add W_phase similarity to the dynamic Z score
     score_total = dot(Z_beam, Z) + beta * dot(W_phase_beam, W_phase)
  D) All three combined

CONFIGS (D=64 N=1024 K_iter=8):
  Ref    standard dynamic_z_geo (W_phase disconnected from routing)
  A      + phase-gated inhibition
  B      + phase-weighted structural aggregation
  C      + phase routing bias beta=0.3
  D      A + B + C combined

To reproduce:
    python -u scripts/train_step46_wphase_reconnect.py --device mps
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

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_resonant(N=1024, D=64, K_iter=8) -> SGNNET_Resonant:
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


class SGNNET_WPhaseReconnect(nn.Module):
    """Reconnects W_phase to routing in dynamic_z_geo mode.

    Modes:
      'phase_gate'   — W_phase gates inhibitory connections
      'phase_struct' — W_phase weights structural aggregation
      'phase_bias'   — W_phase biases dynamic Z similarity scores
      'all'          — all three combined
    """

    def __init__(self, base: SGNNET_Resonant, reconnect_mode: str = 'phase_gate',
                 beta: float = 0.3):
        super().__init__()
        self.m = base
        self.reconnect_mode = reconnect_mode
        self.beta = beta

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
        M         = min(self.m.beam_size, N)

        use_gate   = 'gate'   in self.reconnect_mode or self.reconnect_mode == 'all'
        use_struct = 'struct' in self.reconnect_mode or self.reconnect_mode == 'all'
        use_bias   = 'bias'   in self.reconnect_mode or self.reconnect_mode == 'all'

        # Pre-compute phase similarities for structural neighbors
        if use_struct:
            # phase_sim[h, k] = dot(W_phase[h], W_phase[conn_hh[h,k]])
            W_ph_nb = W_ph_norm[conn_hh]                              # [N, K_hh, D]
            phase_struct_sim = (W_ph_norm.unsqueeze(1) * W_ph_nb).sum(-1)  # [N, K_hh]
            # Softmax to get weights (positive, sum to 1)
            phase_struct_w = F.softmax(phase_struct_sim, dim=-1)      # [N, K_hh]

        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)

            # Structural aggregation (optionally phase-weighted)
            if use_struct:
                Z_nb     = Z_fwd[:, conn_hh, :]                       # [B, N, K_hh, D]
                Z_struct = (Z_nb * phase_struct_w.unsqueeze(0).unsqueeze(-1)).sum(2)
            else:
                Z_struct = Z_fwd[:, conn_hh, :].sum(2)

            # Inhibition (standard dynamic_z_geo, optionally phase-gated)
            Z_ref    = -F.relu(-(Z + theta_pos))
            activity = Z.norm(dim=-1)
            top_idx  = activity.topk(M, dim=-1).indices
            Z_beam   = torch.gather(Z, 1, top_idx.unsqueeze(-1).expand(-1, -1, D))
            Z_ref_beam = torch.gather(Z_ref, 1, top_idx.unsqueeze(-1).expand(-1, -1, D))

            # Feature similarity
            Z_beam_n = F.normalize(Z_beam, dim=-1)
            Z_norm   = F.normalize(Z, dim=-1)
            score    = torch.bmm(Z_beam_n, Z_norm.transpose(1, 2))   # [B, M, N]

            # Phase bias: add W_phase similarity to score
            if use_bias:
                W_ph_beam = W_ph_norm[top_idx[0]]  # [M, D] (use first batch's beam)
                phase_score = W_ph_beam @ W_ph_norm.T                 # [M, N]
                score = score + self.beta * phase_score.unsqueeze(0)

            # Geometric penalty
            W_pos_h = self.m.W_pos[:N]
            W_pos_beam = W_pos_h[top_idx[0]]
            geo_dist = torch.cdist(W_pos_beam, W_pos_h).pow(2)
            score = score - self.m.geo_gamma * geo_dist.unsqueeze(0)

            gate = F.relu(score - self.m.resonance_threshold)

            # Phase gating: multiply by W_phase alignment
            if use_gate:
                W_ph_beam2 = W_ph_norm[top_idx[0]]
                phase_gate = (W_ph_beam2 @ W_ph_norm.T).clamp(min=0)  # [M, N]
                gate = gate * phase_gate.unsqueeze(0)

            Z_inhib  = torch.bmm(gate.transpose(1, 2), Z_ref_beam)
            gate_sum = gate.sum(dim=1).unsqueeze(-1).clamp(min=1.0)
            Z_inh    = Z_inhib / gate_sum

            Z = F.normalize(
                (Z_struct + self.m.alpha_turing * Z_inh).clamp(-10, 10),
                dim=-1,
            )

        return self.m.base._readout(Z)


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="plateau")
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
    ("Ref", "Ref   dynamic_z_geo (W_phase disconnected from routing)",
     None, 0.3),
    ("A",   "A     + phase-gated inhibition",
     "phase_gate", 0.3),
    ("B",   "B     + phase-weighted structural aggregation",
     "phase_struct", 0.3),
    ("C",   "C     + phase routing bias β=0.3",
     "phase_bias", 0.3),
    ("D",   "D     all: gate + struct + bias combined",
     "all", 0.3),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print("Step 46: Reconnect W_phase to routing in dynamic_z_geo")
    print("Gap G11: _inhibit_dynamic_z ignores W_ph_norm — W_phase is dead in routing")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    for key, label, mode, beta in CONFIGS:
        resonant = make_resonant(N=1024, D=64, K_iter=8).to(DEVICE)
        if mode is None:
            model = resonant
        else:
            model = SGNNET_WPhaseReconnect(resonant, reconnect_mode=mode,
                                           beta=beta).to(DEVICE)
        meta = {"N": 1024, "D": 64, "K_iter": 8,
                "reconnect_mode": mode, "beta": beta}
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step46_wphase_reconnect.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_val = results.get("Ref", {}).get("top1_best", 0.5628)
    print(f"\n-- W_phase reconnection (ref={ref_val:.4f}) ---")
    print(f"  {'Config':<55}  {'top1':>6}  {'vs_Ref':>8}  {'t(s)':>6}")
    print("  " + "-"*80)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        print(f"  {r['label'][:55]:<55}  {r['top1_best']:>6.4f}  {d:>+8.4f}  "
              f"{r['elapsed_s']:>6.0f}")

    print("\n  Interpretation:")
    print("  Any > Ref → W_phase WAS underutilized; reconnecting improves routing")
    print("  B > A/C   → phase-weighted struct is the key pathway")
    print("  D > best individual → reconnections compound")
    print("  all ≈ Ref → W_phase has nothing useful to add at D=64 via routing")
