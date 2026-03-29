"""Iteration 2: dynamic_z with resonance threshold + geometric position bias.

Iteration 1 findings:
  - dynamic_z (26.5%) beat resonant (25.3%) and dynamic_gate (25.1%) at N=512
  - dynamic_gate barely helps — fixed topology is the bottleneck, not edge weights
  - dynamic_z was STILL CLIMBING at e120 (26.0% last) — not converged

Gaps identified in dynamic_z:
  1. No resonance threshold: sim.clamp(min=0) passes weak connections (sim~0.01)
     → noise from nearly-orthogonal neurons pollutes the inhibitory signal
  2. No geometric position bias: W_pos never enters routing gradient
     → positions only shaped by readout, not by routing quality

This run tests (on CPU, 90 epochs):
  A. dynamic_z          — iteration 1 winner, reproduced as reference
  B. dynamic_z_thresh   — adds resonance_threshold=0.3 (only strong resonance forms connections)
  C. dynamic_z_geo      — adds geometric position penalty, no threshold
  D. dynamic_z_full     — threshold + geometric bias combined (the target)
  E. dynamic_z_N1024    — scale up: dynamic_z_full at N=1024
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.training.trainer import Trainer
from src.training.experiment_config import trainer_kwargs, topology_kwargs

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=90)
parser.add_argument("--skip-n1024", action="store_true")
args = parser.parse_args()

if args.device == "auto":
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else:
    DEVICE = torch.device(args.device)

EPOCHS_512  = args.epochs
EPOCHS_1024 = max(60, int(args.epochs * 0.8))
BATCH       = 128
print(f"Device: {DEVICE}  Epochs 512={EPOCHS_512}  Epochs 1024={EPOCHS_1024}")

# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    with h5py.File("data/store.h5", "r") as f:
        return (
            torch.from_numpy(f["train/features"][:]),
            torch.from_numpy(f["train/soft_labels"][:]),
            torch.from_numpy(f["train/labels"][:]).long(),
            torch.from_numpy(f["val/features"][:]),
            torch.from_numpy(f["val/soft_labels"][:]),
            torch.from_numpy(f["val/labels"][:]).long(),
        )

def make_loaders(data):
    tf, tsl, tl, vf, vsl, vl = data
    tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tf, tsl, tl),
        batch_size=BATCH, shuffle=True, num_workers=0,
    )
    va = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(vf, vsl, vl),
        batch_size=BATCH, num_workers=0,
    )
    return tr, va

# ── Run ───────────────────────────────────────────────────────────────────────

def run(label: str, model: nn.Module, data, n_hidden: int, epochs: int) -> dict:
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    tr_loader, va_loader = make_loaders(data)
    tk = trainer_kwargs(n_hidden)

    trainer = Trainer(
        model=model, train_loader=tr_loader, val_loader=va_loader,
        device=DEVICE, **tk,
    )
    t0 = time.time()
    history = trainer.train(n_epochs=epochs)
    elapsed = time.time() - t0

    best_top1 = max(h.get("val_top1", 0.0) for h in history)
    last5 = history[-5:]
    result = {
        "label":             label,
        "top1_best":         best_top1,
        "top1_last":         history[-1].get("val_top1", 0.0),
        "final_train_loss":  float(np.mean([h["train_loss"] for h in last5])),
        "final_task_loss":   float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "final_safety_loss": float(np.mean([h.get("safety_loss", 0.0) for h in last5])),
        "epochs_run":        len(history),
        "elapsed_s":         round(elapsed, 1),
    }
    print(f"  top1_best={best_top1:.4f}  top1_last={result['top1_last']:.4f}  "
          f"task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")
    return result

# ── Model factory ─────────────────────────────────────────────────────────────

def make_base(n_hidden: int) -> SGNNET_SmallWorld:
    tk = topology_kwargs(n_hidden)
    return SGNNET_SmallWorld(
        N_in=25088, N_hidden=n_hidden, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=tk["K_iter"],
        n_groups=tk["n_groups"], norm_mode="l2",
    )

def make_dynamic(n_hidden: int, mode: str,
                 resonance_threshold: float = 0.0,
                 geo_gamma: float = 1.0) -> SGNNET_Resonant:
    return SGNNET_Resonant(
        base=make_base(n_hidden),
        K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode=mode,
        resonance_threshold=resonance_threshold,
        geo_gamma=geo_gamma,
    )

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_data()
    print(f"Loaded: train={len(data[0])}  val={len(data[3])}")
    results = {}

    # A. dynamic_z — iteration 1 winner, reproduced as reference (same hyperparams)
    results["dynamic_z"] = run(
        "dynamic_z N=512 [iter1 reference]",
        make_dynamic(512, "dynamic_z", resonance_threshold=0.0).to(DEVICE),
        data, 512, EPOCHS_512,
    )

    # B. dynamic_z + resonance threshold only
    # threshold=0.3 means: on l2-normalised Z, only connections where
    # dot(Z_beam[m], Z[n]) > 0.3 form — roughly within 72° angular separation.
    # Neurons more than ~72° apart don't resonate → no pseudo-connection.
    results["dynamic_z_thresh"] = run(
        "dynamic_z N=512 + threshold=0.3",
        make_dynamic(512, "dynamic_z", resonance_threshold=0.3).to(DEVICE),
        data, 512, EPOCHS_512,
    )

    # C. dynamic_z_geo — geometric bias only, no threshold
    # gamma=1.0: position distance penalises score at same scale as feature similarity.
    # W_pos lives in [0,1]^4 so max dist² ≈ 4; penalises cross-box connections by ~4.
    results["dynamic_z_geo"] = run(
        "dynamic_z_geo N=512 [geo bias only, gamma=1.0]",
        make_dynamic(512, "dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0).to(DEVICE),
        data, 512, EPOCHS_512,
    )

    # D. dynamic_z_full — threshold + geometric bias combined
    results["dynamic_z_full"] = run(
        "dynamic_z_geo N=512 + threshold=0.3 + gamma=1.0 [full]",
        make_dynamic(512, "dynamic_z_geo", resonance_threshold=0.3, geo_gamma=1.0).to(DEVICE),
        data, 512, EPOCHS_512,
    )

    # E. Scale-up: dynamic_z_full at N=1024
    if not args.skip_n1024:
        results["dynamic_z_full_N1024"] = run(
            "dynamic_z_geo N=1024 + threshold=0.3 + gamma=1.0",
            make_dynamic(1024, "dynamic_z_geo", resonance_threshold=0.3, geo_gamma=1.0).to(DEVICE),
            data, 1024, EPOCHS_1024,
        )

    # ── Save ─────────────────────────────────────────────────────────────────

    out = Path("results/train_resonant_iter2.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    # ── Summary ──────────────────────────────────────────────────────────────

    print("\n── Iteration 2 summary ───────────────────────────────────────────")
    print(f"  {'Config':<45}  {'top1_best':>9}  {'top1_last':>9}  {'task':>7}")
    print(f"  {'-'*45}  {'-'*9}  {'-'*9}  {'-'*7}")
    for key, r in results.items():
        print(f"  {key:<45}  {r['top1_best']:>9.4f}  {r['top1_last']:>9.4f}  "
              f"{r['final_task_loss']:>7.4f}")

    best = max(results.values(), key=lambda r: r["top1_best"])
    print(f"\n  Best: {best['label']}  top1_best={best['top1_best']:.4f}")
    print(f"\n  Iter1 reference (dynamic_z 120ep MPS): top1_best=0.2652")
