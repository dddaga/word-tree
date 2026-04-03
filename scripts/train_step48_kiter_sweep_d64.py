"""Step 48: K_iter scaling sweep at D=64.

CONTEXT
=======
At D=16 (step13), K_iter=8 was optimal; K_iter=12 degraded (-7pp, over-smoothing on S^15):
  K_iter=1:  22.90%
  K_iter=2:  25.81%
  K_iter=5:  33.99%
  K_iter=8:  36.69%  ← peak at D=16
  K_iter=12: 29.53%  ← over-smoothed at D=16

At D=64 (step29), the K_iter=3→8 gap is much larger than at D=16:
  K_iter=3 D=64:  40.00%  (Ref0, step29)
  K_iter=8 D=64:  56.28%  (+16pp)
  K_iter=3→8 D=16: only +5pp
  K_iter=3→8 D=64: +16pp  ← 3× larger scaling

This suggests D=64's richer encoding enables deeper refinement before over-smoothing.
Three reasons K_iter>8 may help at D=64 but not D=16:
  1. S^63 supports far more diverse directions than S^15 — over-smoothing threshold shifts up
  2. AntiHebb inhibition (confirmed winner at D=64) actively prevents directional collapse
  3. N=1024 small-world graph diameter ≈ log(1024)/log(K_random) ≈ 5-8 hops,
     so K_iter=8 may already be near-optimal for propagation — but iterative REFINEMENT
     (not propagation) could benefit from more steps on S^63

KEY QUESTION: Is there continued K_iter scaling at D=64, and does AntiHebb inhibition
prevent over-smoothing at high K_iter where D=16 would collapse?

CONFIGS (D=64 N=1024, alpha_reflect=0.5 from step22b calibration):
  Ref    K_iter=8   no AntiHebb  [replicates step22E ~56.28%]
  A      K_iter=12
  B      K_iter=16
  C      K_iter=24
  D      K_iter=32
  E      K_iter=16 + AntiHebb α=0.5  [compound: does inhibition prevent oversmoothing at depth?]
  F      K_iter=8  + AntiHebb α=0.5  [anchor: replicates step29 Config A = 70.14%]
  G      K_iter=32 + AntiHebb α=0.5  [extreme depth + inhibition]

EXPECTED OUTCOMES:
  If A > Ref: K_iter>8 helps at D=64 (include in Gen4 sweep)
  If C,D < B: over-smoothing threshold found between 16 and 24
  If E > F:   depth × AntiHebb compound — K_iter=16 + AntiHebb beats K_iter=8 + AntiHebb
  If E ≈ F:   K_iter is saturated even with inhibition by K_iter=8

To reproduce:
    python -u scripts/train_step48_kiter_sweep_d64.py --device mps
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


def make_resonant(K_iter: int, alpha_reflect: float = 0.5) -> SGNNET_Resonant:
    """Build D=64 N=1024 model with variable K_iter.

    alpha_reflect=0.5 from step22b calibration (confirmed winner over default 0.3).
    """
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
        theta_init=0.1, alpha_reflect=alpha_reflect, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )


class SGNNET_AntiHebb(nn.Module):
    """AntiHebbian inhibition wrapper (confirmed step29 winner: +13.86pp at D=64 K_iter=8).

    Z_inh = alpha * sum_{j in wpos_neighbors} cos(Z_h, Z_j) * Z_j
    Z_new = normalize(Z_struct - Z_inh + alpha_turing * Z_inhibit_phase)
    """

    def __init__(self, base_model: SGNNET_Resonant, alpha_ahebb: float = 0.5):
        super().__init__()
        self.m           = base_model
        self.alpha_ahebb = alpha_ahebb

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
        W_pos_h   = self.m.W_pos[:N]

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            # Anti-Hebbian: suppress similar W_pos neighbors
            W_pos_nb = W_pos_h[conn_hh]                                    # [N, K_hh, D]
            pos_sim  = F.cosine_similarity(W_pos_h.unsqueeze(1),
                                           W_pos_nb, dim=-1)               # [N, K_hh]
            ahebb    = (pos_sim.unsqueeze(0).unsqueeze(-1)
                        * Z_fwd[:, conn_hh, :]).sum(2)                     # [B, N, D]

            Z = F.normalize(
                (Z_struct + self.m.alpha_turing * Z_inh
                 - self.alpha_ahebb * ahebb).clamp(-10, 10),
                dim=-1,
            )

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


# (key, label, K_iter, use_antihebb, alpha_ahebb)
CONFIGS = [
    ("Ref", "Ref   K_iter=8   no-AH  [step22E baseline ~56.28%]",  8,  False, 0.0),
    ("A",   "A     K_iter=12  no-AH",                               12, False, 0.0),
    ("B",   "B     K_iter=16  no-AH",                               16, False, 0.0),
    ("C",   "C     K_iter=24  no-AH",                               24, False, 0.0),
    ("D",   "D     K_iter=32  no-AH",                               32, False, 0.0),
    ("E",   "E     K_iter=16  + AntiHebb α=0.5",                    16, True,  0.5),
    ("F",   "F     K_iter=8   + AntiHebb α=0.5  [step29A ~70.14%]", 8,  True,  0.5),
    ("G",   "G     K_iter=32  + AntiHebb α=0.5  [extreme depth]",   32, True,  0.5),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print("Step 48: K_iter scaling sweep at D=64 N=1024")
    print("D=16 optimum was K_iter=8; K_iter=12 over-smoothed.")
    print("D=64 K_iter=3→8 gave +16pp (3× larger than D=16). Does >8 continue?")
    print("alpha_reflect=0.5 (step22b calibration winner)")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    REF_BASELINE = 0.5628
    results = {}
    for key, label, K_iter, use_ah, alpha_ah in CONFIGS:
        resonant = make_resonant(K_iter=K_iter, alpha_reflect=0.5).to(DEVICE)
        if use_ah:
            model = SGNNET_AntiHebb(resonant, alpha_ahebb=alpha_ah).to(DEVICE)
        else:
            model = resonant
        meta = {"N": N, "D": D, "K_iter": K_iter,
                "antihebb": use_ah, "alpha_ahebb": alpha_ah,
                "alpha_reflect": 0.5}
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step48_kiter_sweep_d64.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    ah_val  = results.get("F",   {}).get("top1_best", 0.7014)
    print(f"\n-- K_iter sweep D=64 (base={ref_val:.4f}  AH_ref={ah_val:.4f}) ---")
    print(f"  {'Config':<55}  {'top1':>6}  {'vs_Ref':>8}  {'vs_AH':>8}  {'t(s)':>6}")
    print("  " + "-"*95)
    for k, r in results.items():
        d_ref = r["top1_best"] - ref_val
        d_ah  = r["top1_best"] - ah_val
        print(f"  {r['label'][:55]:<55}  {r['top1_best']:>6.4f}  {d_ref:>+8.4f}"
              f"  {d_ah:>+8.4f}  {r['elapsed_s']:>6.0f}")

    print("\n  Key questions:")
    print("  A/B/C/D > Ref  → K_iter>8 continues to scale at D=64; include in Gen4")
    print("  C,D < B        → over-smoothing threshold found; note optimal K_iter")
    print("  E > F          → deeper routing + AntiHebb compounds; Gen4 base = K_iter=16")
    print("  G vs E         → marginal returns or collapse at K_iter=32 + AntiHebb")
