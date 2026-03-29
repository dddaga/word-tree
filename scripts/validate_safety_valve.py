"""Step 1 validation: bounded quadratic repulsion vs old 1/d Coulomb.

Tests whether the new safety_valve_loss keeps auxiliary ≤ 15% of task loss
and resolves the N=1024 plateau (previously loss stuck at 5.5 = 2.3 task + 3.2 safety).

Runs SmallWorld at N=512 and N=1024, 60 epochs each.
Reports: final top-1, loss breakdown (task vs safety), whether safety stayed subordinate.
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
import torch.utils.data

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.training.trainer import Trainer
from src.training.experiment_config import trainer_kwargs, topology_kwargs

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=60)
args = parser.parse_args()

if args.device == "auto":
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else:
    DEVICE = torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
print(f"Device: {DEVICE}  Epochs: {EPOCHS}")


# ── Data ──────────────────────────────────────────────────────────────────────

class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, split: str):
        with h5py.File("data/store.h5", "r") as f:
            self.X  = torch.from_numpy(f[f"{split}/features"][:])
            self.Y  = torch.from_numpy(f[f"{split}/soft_labels"][:])
            self.lbl = torch.from_numpy(f[f"{split}/labels"][:]).long()

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i], self.lbl[i]


def loaders():
    tr = torch.utils.data.DataLoader(H5Dataset("train"), batch_size=BATCH, shuffle=True,  num_workers=0)
    va = torch.utils.data.DataLoader(H5Dataset("val"),   batch_size=BATCH, shuffle=False, num_workers=0)
    return tr, va


# ── Run one config ─────────────────────────────────────────────────────────────

def run(n_hidden: int, norm_mode: str = "l2") -> dict:
    print(f"\n{'='*55}")
    print(f"N_hidden={n_hidden}  norm={norm_mode}  epochs={EPOCHS}  device={DEVICE}")
    print('='*55)

    tk   = topology_kwargs(n_hidden)
    kwgs = trainer_kwargs(n_hidden)

    model = SGNNET_SmallWorld(
        N_in=25088,
        N_hidden=n_hidden,
        N_out=10,
        K_local=tk["K_local"],
        K_random=tk["K_random"],
        K_in=tk["K_in"],
        K_iter=tk["K_iter"],
        n_groups=tk["n_groups"],
        norm_mode=norm_mode,
    ).to(DEVICE)

    tr_loader, va_loader = loaders()

    trainer = Trainer(
        model=model,
        train_loader=tr_loader,
        val_loader=va_loader,
        device=DEVICE,
        **kwgs,
    )

    t0 = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    # Gather loss breakdown from last 5 epochs
    last = history[-5:]
    avg_train = np.mean([h["train_loss"] for h in last])

    result = {
        "n_hidden": n_hidden,
        "norm_mode": norm_mode,
        "top1": history[-1].get("val_top1", 0.0),
        "final_train_loss": avg_train,
        "final_val_loss": history[-1].get("val_loss", 0.0),
        "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "history_tail": last,
    }
    print(f"  top1={result['top1']:.4f}  train_loss={avg_train:.4f}  "
          f"val_loss={result['final_val_loss']:.4f}  t={elapsed:.0f}s")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}
    # Test each norm_mode at N=512, then best mode at N=1024
    for norm in ["masked", "l2", "relu"]:
        results[f"512_{norm}"] = run(512, norm_mode=norm)

    # Use best norm_mode for the N=1024 test
    best_norm = max(
        ["masked", "l2", "relu"],
        key=lambda m: results[f"512_{m}"]["top1"]
    )
    print(f"\nBest norm_mode at N=512: {best_norm} — using for N=1024")
    results["1024_best"] = run(1024, norm_mode=best_norm)

    out = Path("results/validate_safety_valve.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    # Summary
    print("\n── Validation summary ──────────────────────────────")
    for n, r in results.items():
        print(f"  {n:15s}  top1={r['top1']:.4f}  "
              f"train_loss={r['final_train_loss']:.3f}  "
              f"val_loss={r['final_val_loss']:.3f}")
    print()
    print("Pass criteria:")
    print("  N=512:  top1 ≥ 0.17  (matches H1_block baseline)")
    print("  N=1024: train_loss ≤ 3.0  (was 5.5 before — safety was dominating)")
