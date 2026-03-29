"""LR schedule comparison: ReduceLROnPlateau vs CosineAnnealingLR.

Identical setup (dynamic_z_geo, N=512, D=4, 150 epochs) — only the LR
schedule differs. Answers: does cosine give a smoother, higher accuracy
curve than the plateau step-cliff?

Configs:
  A. plateau  — current default (ReduceLROnPlateau patience=10 factor=0.5)
  B. cosine   — CosineAnnealingLR T_max=150 eta_min=1e-7

Both use the same random seed so W_pos init and data order are identical.
"""

from __future__ import annotations
import json, sys, time
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=150)
args = parser.parse_args()

if args.device == "auto":
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else:
    DEVICE = torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = 42
print(f"Device: {DEVICE}  Epochs: {EPOCHS}")


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data(path="data/store.h5"):
    with h5py.File(path, "r") as f:
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
    g = torch.Generator().manual_seed(SEED)
    tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tf, tsl, tl),
        batch_size=BATCH, shuffle=True, generator=g, num_workers=0,
    )
    va = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(vf, vsl, vl),
        batch_size=BATCH, num_workers=0,
    )
    return tr, va


# ── Model factory ─────────────────────────────────────────────────────────────

def make_model(seed: int = SEED) -> nn.Module:
    torch.manual_seed(seed)
    tk = topology_kwargs(512)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=512, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=tk["K_iter"],
        n_groups=tk["n_groups"],
        norm_mode="l2",
        D=4,
        encoding_mode="linear",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo",
        resonance_threshold=0.0,
        geo_gamma=1.0,
    )


# ── Run ──────────────────────────────────────────────────────────────────────

def run(label: str, sched_type: str, data) -> dict:
    print(f"\n{'='*62}\n{label}\n{'='*62}")
    tr_loader, va_loader = make_loaders(data)
    model = make_model().to(DEVICE)
    tk = trainer_kwargs(512, n_epochs=EPOCHS, sched_type=sched_type)
    trainer = Trainer(model=model, train_loader=tr_loader, val_loader=va_loader,
                      device=DEVICE, **tk)
    t0 = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    best  = max(h.get("val_top1", 0.0) for h in history)
    last5 = history[-5:]
    result = {
        "label":           label,
        "sched_type":      sched_type,
        "top1_best":       best,
        "top1_last":       history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "epochs_run":      len(history),
        "elapsed_s":       round(elapsed, 1),
        "lr_history":      [h["lr"] for h in history],
        "top1_history":    [round(h.get("val_top1", 0.0), 4) for h in history],
    }
    print(f"  top1_best={best:.4f}  top1_last={result['top1_last']:.4f}  "
          f"task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_data()
    print(f"Loaded: train={len(data[0])}  val={len(data[3])}")

    results = {}

    results["plateau"] = run(
        "A. ReduceLROnPlateau  (patience=10, factor=0.5) — current default",
        sched_type="plateau",
        data=data,
    )

    results["cosine"] = run(
        "B. CosineAnnealingLR (T_max=150, eta_min=1e-7) — smooth decay",
        sched_type="cosine",
        data=data,
    )

    out = Path("results/test_cosine_lr.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    print("\n── LR schedule comparison ───────────────────────────────────────")
    print(f"  {'Schedule':<45}  {'top1_best':>9}  {'top1_last':>9}  {'epochs':>6}  {'t(s)':>6}")
    print(f"  {'-'*45}  {'-'*9}  {'-'*9}  {'-'*6}  {'-'*6}")
    for key, r in results.items():
        print(f"  {key:<45}  {r['top1_best']:>9.4f}  {r['top1_last']:>9.4f}  "
              f"{r['epochs_run']:>6}  {r['elapsed_s']:>6.0f}")

    p = results["plateau"]["top1_best"]
    c = results["cosine"]["top1_best"]
    delta = c - p
    print(f"\n  Delta (cosine - plateau): {delta:+.4f}")
    print(f"  Winner: {'cosine' if delta > 0 else 'plateau'}")
    print(f"\n  Note: lr_history and top1_history saved in JSON for curve analysis.")
