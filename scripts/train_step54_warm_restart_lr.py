"""Step 54: Cosine annealing with warm restarts (T_0=10) on best config.

Best config so far: D=64 N=1024 K_iter=8 + AntiHebb α=0.5 = 70.14% (step29 Config A).

Question: Does cycling the LR every 10 epochs help the routing network escape local
minima in the Z-on-S^63 attractor landscape, improving final accuracy?

Configs (all share the same model — only LR schedule differs):
  Ref   ReduceLROnPlateau  patience=10 factor=0.5  [current default, ~70.14%]
  A     CosineAnnealingWarmRestarts  T_0=10 T_mult=1  [constant 10ep cycles]
  B     CosineAnnealingWarmRestarts  T_0=10 T_mult=2  [doubling: 10→20→40→80ep cycles]
  C     CosineAnnealingLR  T=150  [single smooth decay — no restarts, for comparison]

Decision rule:
  A or B > Ref → warm restarts help; adopt for all future experiments
  A > B        → constant cycle preferred (landscape benefits from frequent resets)
  B > A        → longer cycles preferred (restarts too disruptive at T_0=10)
  C > A,B      → smooth decay without restarts is sufficient
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import (
    trainer_kwargs, topology_kwargs, run_metadata, scaled_lambda_safety, GA_BEST,
)
from src.training.dataset import make_loaders

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE = os.environ.get("DEVICE", "mps")
if len(sys.argv) > 1 and sys.argv[1] == "--device":
    DEVICE = sys.argv[2]

EPOCHS  = 150
BATCH   = 128
SEED    = 42
DATA    = "data/store.h5"
N, D, K_ITER = 1024, 64, 8
ALPHA_AHEBB  = 0.5   # confirmed best from step29

_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


# ── Model factory ─────────────────────────────────────────────────────────────
def make_model() -> nn.Module:
    """Fresh AntiHebb α=0.5 model — same architecture as step29 Config A."""
    torch.manual_seed(SEED)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_ITER,
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


# ── Run helper ────────────────────────────────────────────────────────────────
def run(label: str, sched_type: str, sched_T0: int = 10,
        sched_T_mult: int = 1) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va = get_loaders()
    model  = make_model()

    base_tk = trainer_kwargs(N, n_epochs=EPOCHS, sched_type=sched_type)
    base_tk["sched_T0"]     = sched_T0
    base_tk["sched_T_mult"] = sched_T_mult

    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **base_tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    best    = max(h["val_top1"] for h in history)
    best_ep = max(range(len(history)), key=lambda i: history[i]["val_top1"]) + 1
    frac    = best_ep / EPOCHS

    # LR trajectory: sample every 10 epochs
    lr_traj = [round(h["lr"], 6) for h in history[::10]]

    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{EPOCHS} ({frac:.0%})"
          f"  t={elapsed:.0f}s")
    print(f"  lr_traj={lr_traj}")
    return {
        "label": label, "top1_best": best, "best_ep": best_ep,
        "ep_frac": frac, "elapsed_s": round(elapsed),
        "sched_type": sched_type, "sched_T0": sched_T0, "sched_T_mult": sched_T_mult,
        "lr_trajectory": lr_traj,
        "top1_history": [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta": run_metadata(__file__, {
            "N": N, "D": D, "K_iter": K_ITER,
            "alpha_ahebb": ALPHA_AHEBB, "epochs": EPOCHS,
        }),
    }


# ── Configs ───────────────────────────────────────────────────────────────────
CONFIGS = [
    ("Ref", "Ref   ReduceLROnPlateau  patience=10 factor=0.5  [current default]",
     "plateau",        10, 1),
    ("A",   "A     CosineWarmRestarts  T_0=10 T_mult=1  [constant 10ep cycles × 15]",
     "warm_restarts",  10, 1),
    ("B",   "B     CosineWarmRestarts  T_0=10 T_mult=2  [doubling: 10→20→40→80ep]",
     "warm_restarts",  10, 2),
    ("C",   "C     CosineAnnealingLR  T=150  [single smooth decay, no restarts]",
     "cosine",         10, 1),
]


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print("Step 54: LR schedule comparison on best config (AntiHebb α=0.5, D=64)")
    print("Base model: ~70.14% (step29 Config A). Question: do warm restarts help?")
    tr, va = get_loaders()
    print(f"Dataset loaded into RAM: train={len(tr.dataset)}  val={len(va.dataset)}")
    print(f"Dataset: train={len(tr.dataset)}  val={len(va.dataset)}")

    results = {}
    ref_top1 = None

    for key, label, sched, T0, T_mult in CONFIGS:
        results[key] = run(label, sched, T0, T_mult)
        if key == "Ref":
            ref_top1 = results[key]["top1_best"]

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n-- LR Schedule Comparison @ D=64 AntiHebb α=0.5 ----------------")
    print(f"  {'Key':<4}  {'top1':>7}  {'vs Ref':>8}  Label")
    for key, label, _, _, _ in CONFIGS:
        r   = results[key]
        top = r["top1_best"]
        vs  = f"{(top - ref_top1)*100:+.2f}pp" if ref_top1 else "—"
        print(f"  {key:<4}  {top:.4f}  {vs:>8}  {label.split('[')[0].strip()}")

    print(f"\n  Interpretation:")
    print(f"  A or B > Ref → warm restarts help; adopt for all future experiments")
    print(f"  A > B        → T_0=10 constant cycles are optimal")
    print(f"  B > A        → longer cycles preferred (less disruption)")
    print(f"  C > A, B     → smooth single decay sufficient, restarts not needed")

    out_path = "results/train_step54_warm_restart_lr.json"
    with open(out_path, "w") as f:
        import json
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")
