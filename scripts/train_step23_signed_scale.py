"""Step 23: Signed coupling × scale — combining the two major breakthroughs.

CONTEXT
=======
Step 18 result (signed full coupling, N=512, D=16, K_iter=3):
  Config A: 40.15%  (+11.59pp vs ref 28.56%)  ← BIGGEST WIN SO FAR
  Sparse variants CRASHED (scatter grad issue — skip for now)

Step 13 result (K_iter=8, N=512, D=16):
  K_iter=8: 36.69%  (+7.65pp vs K_iter=3)

Signed coupling has NOT been tested with K_iter>3 or N>512.
These two orthogonal improvements should stack:
  K_iter=8 adds +7.65pp over baseline
  Signed coupling adds +11.59pp over baseline
  Together (assuming ~additive): 28.56 + 7.65 + 11.59 ≈ 47.8%?

Note: Step 18 config A peaked at epoch 60/120 → training too short!
Using 150ep here.

CONFIGS
=======
  Ref  : base resonant D=16 N=512  K_iter=3  (step18 ref = 28.56%)
  A    : signed α=0.3  D=16 N=512  K_iter=3  (reproduce step18A = 40.15%)
  B    : signed α=0.3  D=16 N=512  K_iter=8  (signed + deeper routing)
  C    : signed α=0.3  D=16 N=1024 K_iter=3  (signed + more neurons)
  D    : signed α=0.3  D=16 N=1024 K_iter=8  ← primary hypothesis
  E    : signed α=0.1  D=16 N=1024 K_iter=8  (weaker coupling + scale)
  F    : signed α=0.5  D=16 N=512  K_iter=8  (stronger coupling)

All: Fourier encoding, dynamic_z_geo base, 150ep, plateau, store.h5.

Reference: step18 signed α=0.3 N=512 K_iter=3 = 40.15%

To reproduce:
    python -u scripts/train_step23_signed_scale.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def make_resonant(N: int = 512, K_iter: int = 3) -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_iter,
        n_groups=tk["n_groups"],
        norm_mode="l2", D=16, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )


class SGNNET_SignedCoupling(nn.Module):
    """All-pairs signed coupling: cos(Z_h, Z_j)>0 → excite; <0 → inhibit.

    Binding-by-synchrony (von der Malsburg 1981).
    Z_h += alpha * Σ_j cos(Z_h, Z_j) * Z_j  (unified exc+inh signal)
    """

    def __init__(self, base: SGNNET_Resonant, alpha_signed: float = 0.3):
        super().__init__()
        self.m            = base
        self.alpha_signed = alpha_signed

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, self.m.base.conn_hh, :].sum(2)
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            # All-pairs signed coupling: cos similarity drives exc or inh
            Z_n      = F.normalize(Z, dim=-1)
            sim      = torch.bmm(Z_n, Z_n.transpose(1, 2))   # [B, N, N]
            Z_coupled = torch.bmm(sim, Z)                     # [B, N, D]

            Z_new = F.normalize(
                (Z_struct + self.m.alpha_turing * Z_inh
                 + self.alpha_signed * Z_coupled).clamp(-10, 10), dim=-1)
            Z = Z_new

        return self.m.base._readout(Z)


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    N       = meta.get("N", 512)
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
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
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})  "
          f"diag={result['convergence_diag']}  task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")
    return result


# (label, N, K_iter, alpha_signed or None)
CONFIGS = [
    ("Ref. base resonant  D=16 N=512  K_iter=3  [step18 ref]",
     512,  3, None),
    ("A.  signed α=0.3   D=16 N=512  K_iter=3  [reproduce step18A]",
     512,  3, 0.3),
    ("B.  signed α=0.3   D=16 N=512  K_iter=8  [signed + deep routing]",
     512,  8, 0.3),
    ("C.  signed α=0.3   D=16 N=1024 K_iter=3  [signed + wide]",
     1024, 3, 0.3),
    ("D.  signed α=0.3   D=16 N=1024 K_iter=8  [primary hypothesis]",
     1024, 8, 0.3),
    ("E.  signed α=0.1   D=16 N=1024 K_iter=8  [weaker + scale]",
     1024, 8, 0.1),
    ("F.  signed α=0.5   D=16 N=512  K_iter=8  [stronger coupling]",
     512,  8, 0.5),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}")
    print("Goal: signed coupling × scale (K_iter=8 × N=1024) — do breakthroughs stack?")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    keys    = ["Ref", "A", "B", "C", "D", "E", "F"]
    for key, (label, N, K_iter, alpha) in zip(keys, CONFIGS):
        resonant = make_resonant(N=N, K_iter=K_iter).to(DEVICE)
        model    = resonant if alpha is None else SGNNET_SignedCoupling(resonant, alpha_signed=alpha)
        meta     = {"N": N, "K_iter": K_iter, "D": 16,
                    "alpha_signed": alpha if alpha is not None else 0.0,
                    "mechanism": "signed_full" if alpha is not None else "baseline"}
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = Path("results/train_step23_signed_scale.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref18 = 0.4015   # step18 config A best
    ref   = results.get("Ref", {}).get("top1_best", 0.2856)
    print(f"\n-- Signed coupling × scale  (ref_baseline={ref:.4f}  ref_step18A={ref18:.4f}) ---")
    print("  %-54s  %5s  %6s  %6s  %9s  %+8s  %8s  %6s" % (
        "Config", "N", "K_it", "alpha", "top1_best", "vs_18A", "ep_frac", "t(s)"))
    print("  " + "-"*105)
    for k, r in results.items():
        d = r["top1_best"] - ref18
        print("  %-54s  %5d  %6d  %6.1f  %9.4f  %+8.4f  %7.1f%%  %6.0f" % (
            r["label"][:54], r.get("N", 0), r.get("K_iter", 0),
            r.get("alpha_signed", 0.0), r["top1_best"], d,
            r.get("best_epoch_frac", 0) * 100, r["elapsed_s"]))
