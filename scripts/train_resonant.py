"""Train SGNNET_Resonant — combined best-of model + dynamic connectivity sweep.

Configurations run:
  A. resonant_N512   — baseline SmallWorld (l2, no routing)
  B. resonant_N512   — Resonant: threshold + reflection + turing, static phase graph
  C. dynamic_gate_N512 — Resonant with input-conditioned edge weights (GAT-style)
  D. dynamic_z_N512  — Resonant with input-dependent topology from Z similarity
  E. resonant_N1024  — scale-up: Resonant best mode at N=1024

Runs on MPS if available (auto-detect). 120 epochs for A-D, 100 for E.
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
parser.add_argument("--epochs", type=int, default=120,
                    help="Epochs for N=512 configs (N=1024 uses epochs//1.2)")
parser.add_argument("--skip-n1024", action="store_true",
                    help="Skip the N=1024 scale-up run")
args = parser.parse_args()

if args.device == "auto":
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else:
    DEVICE = torch.device(args.device)

EPOCHS_512  = args.epochs
EPOCHS_1024 = max(60, int(args.epochs / 1.2))
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

# ── Run one config ─────────────────────────────────────────────────────────────

def run(label: str, model: nn.Module, data, n_hidden: int, epochs: int) -> dict:
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    tr_loader, va_loader = make_loaders(data)
    tk = trainer_kwargs(n_hidden)

    trainer = Trainer(
        model=model,
        train_loader=tr_loader,
        val_loader=va_loader,
        device=DEVICE,
        **tk,
    )
    t0 = time.time()
    history = trainer.train(n_epochs=epochs)
    elapsed = time.time() - t0

    # Best val_top1 across all epochs (not just last — accounts for noise)
    best_top1 = max(h.get("val_top1", 0.0) for h in history)
    last5 = history[-5:]
    result = {
        "label":            label,
        "top1_last":        history[-1].get("val_top1", 0.0),
        "top1_best":        best_top1,
        "final_train_loss": float(np.mean([h["train_loss"] for h in last5])),
        "final_task_loss":  float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "final_safety_loss":float(np.mean([h.get("safety_loss", 0.0) for h in last5])),
        "epochs_run":       len(history),
        "elapsed_s":        round(elapsed, 1),
    }
    ratio = result["final_safety_loss"] / max(result["final_task_loss"], 1e-8)
    print(f"  top1_best={best_top1:.4f}  top1_last={result['top1_last']:.4f}  "
          f"task={result['final_task_loss']:.3f}  safety/task={ratio:.3f}  t={elapsed:.0f}s")
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

def make_resonant(n_hidden: int, mode: str) -> SGNNET_Resonant:
    return SGNNET_Resonant(
        base=make_base(n_hidden),
        K_phase=8,
        beam_size=32,
        theta_init=0.1,
        alpha_reflect=0.3,
        alpha_turing=0.3,
        mode=mode,
    )

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_data()
    print(f"Loaded: train={len(data[0])}  val={len(data[3])}")

    results = {}

    # A. Baseline: plain SmallWorld with l2 (no routing mechanism)
    baseline = make_base(512).to(DEVICE)
    results["baseline_N512"] = run(
        "Baseline SmallWorld N=512 (l2, no routing)",
        baseline, data, 512, EPOCHS_512,
    )

    # B. Resonant: static phase graph, all mechanisms stacked
    resonant = make_resonant(512, mode="resonant").to(DEVICE)
    results["resonant_N512"] = run(
        "Resonant N=512 (threshold+reflection+turing, static phase)",
        resonant, data, 512, EPOCHS_512,
    )

    # C. Dynamic gate: fixed graph, input-conditioned edge weights
    dyn_gate = make_resonant(512, mode="dynamic_gate").to(DEVICE)
    results["dynamic_gate_N512"] = run(
        "Resonant N=512 dynamic_gate (GAT-style activation gating)",
        dyn_gate, data, 512, EPOCHS_512,
    )

    # D. Dynamic Z: graph rebuilt from current Z activations each forward pass
    dyn_z = make_resonant(512, mode="dynamic_z").to(DEVICE)
    results["dynamic_z_N512"] = run(
        "Resonant N=512 dynamic_z (input-dependent topology from Z)",
        dyn_z, data, 512, EPOCHS_512,
    )

    # E. Scale-up: best mechanism at N=1024
    if not args.skip_n1024:
        resonant_1024 = make_resonant(1024, mode="resonant").to(DEVICE)
        results["resonant_N1024"] = run(
            "Resonant N=1024 (threshold+reflection+turing, static phase)",
            resonant_1024, data, 1024, EPOCHS_1024,
        )

    # ── Save ──────────────────────────────────────────────────────────────────

    out = Path("results/train_resonant.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    # ── Summary ───────────────────────────────────────────────────────────────

    print("\n── Resonant training summary ─────────────────────────────────────")
    print(f"  {'Config':<35}  {'top1_best':>9}  {'top1_last':>9}  {'task_loss':>9}")
    print(f"  {'-'*35}  {'-'*9}  {'-'*9}  {'-'*9}")
    for key, r in results.items():
        print(f"  {key:<35}  {r['top1_best']:>9.4f}  {r['top1_last']:>9.4f}  "
              f"{r['final_task_loss']:>9.4f}")

    best = max(results.values(), key=lambda r: r["top1_best"])
    print(f"\n  Best: {best['label']}  top1_best={best['top1_best']:.4f}")
