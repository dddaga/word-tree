"""Step 41: Oja's rule routing update for SGNNET.

HYPOTHESIS (InfraNodus Gap: hebbian/hopfield/oja ↔ implementation)
===================================================================
The current routing update is an ad-hoc sum:
  Z = normalize(Z_struct + alpha_turing * Z_inh)

Oja's rule (Oja 1982) is a normalized Hebbian learning rule that makes each
neuron learn the principal component direction of its incoming signal:
  ΔZ_j = η * (Z_struct_j - (Z_j · Z_struct_j) * Z_j)
  Z_j ← normalize(Z_j + ΔZ_j)

Intuitively: Z_j moves toward Z_struct_j, but subtracts the component already
in Z_j's direction. This makes each routing step an online PCA step — neurons
converge to the most informative direction of their local input rather than
averaging it. Unlike plain normalization, Oja's update preserves memory of
the current state Z_j while incorporating new signal from Z_struct_j.

Hopfield energy variant (E):
  Z_new = sign(W_eff @ Z_old) where W_eff = Z^T Z / N (Hebbian weight matrix)
  Soft version: Z_new = tanh(Z_nb.mean(2) / tau)
  This is one synchronous update step of a modern Hopfield network.

CONFIGS
=======
  Ref   D=64 N=1024 K_iter=8  [standard routing, baseline ≈56.28%]
  A     Oja η=0.1  [conservative — small step toward principal component]
  B     Oja η=0.3  [moderate]
  C     Oja η=1.0  [aggressive — full Oja step per routing iteration]
  D     Hybrid: Oja on Z_struct, keep Z_inh additive  [combine both signals]
  E     Hopfield soft: Z = tanh(Z_nb.mean(2) / τ)  τ=0.5  [associative memory]

Base: D=64 N=1024 K_iter=8 (uncalibrated — same as step29b)

To reproduce:
    python -u scripts/train_step41_ojas_rule.py --device mps
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


class SGNNET_OjasRule(nn.Module):
    """Oja's rule routing update.

    Replaces the standard sum-normalize with an online PCA step:
      ΔZ_j = η * (Z_struct_j - dot(Z_j, Z_struct_j) * Z_j)
      Z_j  = normalize(Z_j + ΔZ_j)

    mode:
      'oja'      — pure Oja update (replaces Z_inh term)
      'hybrid'   — Oja on Z_struct + Z_inh additive (keeps inhibition signal)
      'hopfield' — soft Hopfield: Z = tanh(Z_struct / tau)
    """

    def __init__(self, base: SGNNET_Resonant, mode: str = 'oja',
                 eta: float = 0.3, tau: float = 0.5):
        super().__init__()
        self.m    = base
        self.mode = mode
        self.eta  = eta
        self.tau  = tau

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                              # [B, N, D]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)     # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)               # [N, D]
        conn_hh   = self.m.base.conn_hh                               # [N, K_hh]

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)                          # [B, N, D]
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)                   # [B, N, D]

            if self.mode == 'hopfield':
                # Soft Hopfield associative memory step
                # Z_new = tanh(mean(Z_neighbors) / tau)
                Z_nb_mean = Z_fwd[:, conn_hh, :].mean(2)             # [B, N, D]
                Z = F.normalize(
                    torch.tanh(Z_nb_mean / self.tau).clamp(-10, 10),
                    dim=-1,
                )

            elif self.mode == 'hybrid':
                # Oja on Z_struct + keep Z_inh additive
                Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
                # Oja step: move Z toward principal component of Z_struct
                dot      = (Z * Z_struct).sum(-1, keepdim=True)      # [B, N, 1]
                Z_oja    = Z + self.eta * (Z_struct - dot * Z)       # [B, N, D]
                Z = F.normalize(
                    (Z_oja + self.m.alpha_turing * Z_inh).clamp(-10, 10),
                    dim=-1,
                )

            else:  # 'oja' — pure Oja, replaces Z_inh
                # Oja step: move Z toward principal component of incoming signal
                dot   = (Z * Z_struct).sum(-1, keepdim=True)         # [B, N, 1]
                Z_oja = Z + self.eta * (Z_struct - dot * Z)          # [B, N, D]
                Z = F.normalize(Z_oja.clamp(-10, 10), dim=-1)

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


# (key, label, mode, eta, tau)
CONFIGS = [
    ("Ref", "Ref   D=64 N=1024 K_iter=8  [standard routing ≈56.28%]",
     None,       0.0,  1.0),
    ("A",   "A     Oja η=0.1  [conservative PCA step]",
     "oja",      0.1,  1.0),
    ("B",   "B     Oja η=0.3  [moderate PCA step]",
     "oja",      0.3,  1.0),
    ("C",   "C     Oja η=1.0  [aggressive PCA step]",
     "oja",      1.0,  1.0),
    ("D",   "D     Hybrid: Oja(Z_struct) + Z_inh additive  η=0.3",
     "hybrid",   0.3,  1.0),
    ("E",   "E     Hopfield soft: tanh(Z_nb.mean / τ)  τ=0.5",
     "hopfield", 0.3,  0.5),
]
KEYS = ["Ref", "A", "B", "C", "D", "E"]


if __name__ == "__main__":
    REF_BASELINE = 0.5628
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Seed: {SEED}")
    print("Step 41: Oja's rule routing update")
    print("InfraNodus gap: Hebbian/Hopfield/Oja learning rules disconnected from implementation")
    print(f"Ref baseline (step22E / D=64 / N=1024 / K_iter=8): {REF_BASELINE:.4f}")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    for key, label, mode, eta, tau in CONFIGS:
        resonant = make_resonant(N=1024, D=64, K_iter=8).to(DEVICE)
        if mode is None:
            model = resonant
        else:
            model = SGNNET_OjasRule(resonant, mode=mode, eta=eta, tau=tau).to(DEVICE)
        meta = {"N": 1024, "D": 64, "K_iter": 8,
                "oja_mode": mode, "eta": eta, "tau": tau}
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step41_ojas_rule.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    print(f"\n-- Oja's rule routing update sweep (ref={ref_val:.4f}) ---")
    print("  %-58s  %9s  %9s  %8s  %6s" % ("Config", "top1", "vs_Ref", "best_ep", "t(s)"))
    print("  " + "-"*98)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        print("  %-58s  %9.4f  %+9.4f  %7d    %6.0f" % (
            r["label"][:58], r["top1_best"], d,
            r.get("best_epoch", 0), r["elapsed_s"]))

    print("\n  Interpretation:")
    print("  B/C > Ref  → Oja step improves over ad-hoc sum; routing = online PCA")
    print("  D > B      → Oja + Z_inh is additive; keep inhibition signal")
    print("  D < B      → Oja and Z_inh conflict; Oja learns best without inhibition")
    print("  E > Ref    → Hopfield energy step recovers associative memory gains")
    print("  all < Ref  → ad-hoc sum is fine; theoretical elegance ≠ empirical gain")
