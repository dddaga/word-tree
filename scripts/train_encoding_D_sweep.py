"""Encoding mode and D dimensionality sweep.

Two back-to-back ablations:

Part 1 — Encoding mode at D=4 (N=512):
  Isolates the effect of switching from linear spatial encoding to Fourier sinusoidal.
  Same architecture, same N, same D — only the seed vector structure changes.
  A. D=4 linear   — current default [c_norm, h_norm, w_norm]
  B. D=4 fourier  — sinusoidal [sin(h), sin(w), sin(c·2π)]
  Question: does richer direction diversity on S³ from sinusoidal encoding help?

Part 2 — D scaling with Fourier encoding (N=512 and N=1024):
  Uses the winner of Part 1 encoding mode if fourier wins, else linear.
  C. D=4   N=512  fourier  (= B above, reference)
  D. D=8   N=512  fourier
  E. D=16  N=512  fourier
  F. D=8   N=1024 fourier  (scale both N and D)
  Question: does richer S^(D-1) direction space improve accuracy and at what cost?

All use dynamic_z_geo (iter1+geo winner) routing on CPU, 90 epochs.
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
parser.add_argument("--epochs", type=int, default=90)
parser.add_argument("--part", choices=["1", "2", "all"], default="all")
args = parser.parse_args()

if args.device == "auto":
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else:
    DEVICE = torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
print(f"Device: {DEVICE}  Epochs: {EPOCHS}")


# ── Data ─────────────────────────────────────────────────────────────────────

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


# ── Run ──────────────────────────────────────────────────────────────────────

def run(label: str, model: nn.Module, data, n_hidden: int, epochs: int) -> dict:
    print(f"\n{'='*62}\n{label}\n{'='*62}")
    tr_loader, va_loader = make_loaders(data)
    tk = trainer_kwargs(n_hidden)
    trainer = Trainer(model=model, train_loader=tr_loader, val_loader=va_loader,
                      device=DEVICE, **tk)
    t0 = time.time()
    history = trainer.train(n_epochs=epochs)
    elapsed = time.time() - t0

    best  = max(h.get("val_top1", 0.0) for h in history)
    last5 = history[-5:]
    result = {
        "label":            label,
        "top1_best":        best,
        "top1_last":        history[-1].get("val_top1", 0.0),
        "final_task_loss":  float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "epochs_run":       len(history),
        "elapsed_s":        round(elapsed, 1),
    }
    print(f"  top1_best={best:.4f}  top1_last={result['top1_last']:.4f}  "
          f"task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")
    return result


# ── Model factory ─────────────────────────────────────────────────────────────

def make_model(n_hidden: int, D: int, encoding: str) -> nn.Module:
    """Build SGNNET_Resonant with dynamic_z_geo routing."""
    tk = topology_kwargs(n_hidden)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=n_hidden, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=tk["K_iter"],
        n_groups=tk["n_groups"],
        norm_mode="l2",
        D=D,
        encoding_mode=encoding,
    )
    # Use dynamic_z_geo with best params from iter2 (no threshold — iter2 showed
    # threshold=0.3 too aggressive; geo effect pending iter2 config C result)
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo",
        resonance_threshold=0.0,
        geo_gamma=1.0,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_data()
    print(f"Loaded: train={len(data[0])}  val={len(data[3])}")
    results = {}

    # ── Part 1: Encoding mode ablation at D=4, N=512 ─────────────────────────
    if args.part in ("1", "all"):
        print("\n\n── PART 1: Encoding mode at D=4, N=512 ──")

        # A. Linear encoding (current default)
        results["linear_D4_N512"] = run(
            "A. Linear encoding  D=4 N=512 [current default]",
            make_model(512, D=4, encoding="linear").to(DEVICE),
            data, 512, EPOCHS,
        )

        # B. Fourier encoding at D=4
        results["fourier_D4_N512"] = run(
            "B. Fourier encoding D=4 N=512 [sinusoidal sin(h),sin(w),sin(c)]",
            make_model(512, D=4, encoding="fourier").to(DEVICE),
            data, 512, EPOCHS,
        )

        linear_top1  = results["linear_D4_N512"]["top1_best"]
        fourier_top1 = results["fourier_D4_N512"]["top1_best"]
        winner_enc   = "fourier" if fourier_top1 > linear_top1 else "linear"
        print(f"\n  Part 1 result: linear={linear_top1:.4f}  fourier={fourier_top1:.4f}")
        print(f"  Winner encoding: {winner_enc}")

    # ── Part 2: D dimensionality scaling with Fourier encoding ────────────────
    if args.part in ("2", "all"):
        print("\n\n── PART 2: D scaling with Fourier encoding ──")

        # C. D=4 fourier N=512 (= B above if part=all, else fresh)
        if "fourier_D4_N512" not in results:
            results["fourier_D4_N512"] = run(
                "C. Fourier D=4 N=512 [reference for D-sweep]",
                make_model(512, D=4, encoding="fourier").to(DEVICE),
                data, 512, EPOCHS,
            )

        # D. D=8 fourier N=512
        results["fourier_D8_N512"] = run(
            "D. Fourier D=8  N=512",
            make_model(512, D=8, encoding="fourier").to(DEVICE),
            data, 512, EPOCHS,
        )

        # E. D=16 fourier N=512
        results["fourier_D16_N512"] = run(
            "E. Fourier D=16 N=512",
            make_model(512, D=16, encoding="fourier").to(DEVICE),
            data, 512, EPOCHS,
        )

        # F. D=8 fourier N=1024 (scale both N and D)
        results["fourier_D8_N1024"] = run(
            "F. Fourier D=8  N=1024",
            make_model(1024, D=8, encoding="fourier").to(DEVICE),
            data, 1024, EPOCHS,
        )

    # ── Save & summary ────────────────────────────────────────────────────────

    out = Path("results/train_encoding_D_sweep.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    print("\n── Encoding + D sweep summary ────────────────────────────────────")
    print(f"  {'Config':<45}  {'top1_best':>9}  {'task':>7}  {'t(s)':>7}")
    print(f"  {'-'*45}  {'-'*9}  {'-'*7}  {'-'*7}")
    for key, r in results.items():
        print(f"  {key:<45}  {r['top1_best']:>9.4f}  "
              f"{r['final_task_loss']:>7.4f}  {r['elapsed_s']:>7.0f}")

    best = max(results.values(), key=lambda r: r["top1_best"])
    print(f"\n  Best: {best['label']}  top1_best={best['top1_best']:.4f}")
    print(f"  Baseline reference (linear D=4 N=512 iter1): ~0.265")
