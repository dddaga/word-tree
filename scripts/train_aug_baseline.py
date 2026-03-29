"""Augmented-data baseline: LR schedule comparison + new reference point.

Runs on store_aug.h5 (18,938 train = 2× original) with dynamic_z_full
(iter2 winner). Two configs — only LR schedule differs:

  A. plateau — ReduceLROnPlateau patience=10 factor=0.5
  B. cosine  — CosineAnnealingLR T_max=120 eta_min=1e-7

With 2× data the loss curve should be smoother, so plateau may fire less
aggressively and close the gap with cosine. Whichever wins becomes the
reference for all subsequent ablations (dropout, encoding, interneurons, etc).

Both configs: N=512, D=4, linear encoding, dynamic_z_full, 120 epochs, MPS.
Same seed → identical W_pos init and data order.
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

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--data",   default="data/store_aug.h5")
parser.add_argument("--sched",  default="both", choices=["both", "plateau", "cosine"],
                    help="Run one schedule or both (default: both)")
args = parser.parse_args()

if args.device == "auto":
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else:
    DEVICE = torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = 42
print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Data: {args.data}")


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    with h5py.File(args.data, "r") as f:
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

def make_model() -> nn.Module:
    torch.manual_seed(SEED)
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
    # dynamic_z_full = dynamic_z_geo mode + resonance_threshold=0.3 (iter2 winner)
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo",
        resonance_threshold=0.3,
        geo_gamma=1.0,
    )


# ── Run ───────────────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_data()
    print(f"Loaded: train={len(data[0])}  val={len(data[3])}")

    results = {}

    if args.sched in ("both", "plateau"):
        results["plateau"] = run(
            f"A. plateau  dynamic_z_geo  N=512  D=4  aug  {EPOCHS}ep",
            sched_type="plateau", data=data,
        )
    if args.sched in ("both", "cosine"):
        results["cosine"] = run(
            f"B. cosine   dynamic_z_geo  N=512  D=4  aug  {EPOCHS}ep",
            sched_type="cosine", data=data,
        )

    suffix = f"_{args.sched}" if args.sched != "both" else ""
    out = Path(f"results/train_aug_baseline{suffix}_{EPOCHS}ep.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    print("\n── Aug-data baseline: LR schedule comparison ────────────────────")
    print(f"  {'Config':<50}  {'top1_best':>9}  {'top1_last':>9}  {'t(s)':>6}")
    print(f"  {'-'*50}  {'-'*9}  {'-'*9}  {'-'*6}")
    for key, r in results.items():
        print(f"  {key:<50}  {r['top1_best']:>9.4f}  {r['top1_last']:>9.4f}  {r['elapsed_s']:>6.0f}")

    p = results["plateau"]["top1_best"]
    c = results["cosine"]["top1_best"]
    winner = "cosine" if c > p else "plateau"
    print(f"\n  Delta (cosine - plateau): {c-p:+.4f}")
    print(f"  Winner: {winner}")
    print(f"\n  → This config becomes the reference for all subsequent ablations.")
    print(f"    Use store_aug.h5 + dynamic_z_full + {winner} LR + 120ep as baseline.")
