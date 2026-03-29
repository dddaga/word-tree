"""Exp 2: SGNNET_SmallWorld scaling sweep.

Sweeps N_hidden over [256, 512, 1024, 2048, 4096, 10000] using the
small-world fixed fan-in architecture (no O(N²) cdist, no dense C_hh matmul).

Key differences from Exp 1:
  - C_hh replaced by conn_hh [N, K_hh] gather (small-world topology)
  - C_input replaced by conn_in [N, K_in] gather (block-local fan-in)
  - cdist removed from forward pass entirely
  - Safety valve disabled for N > 5000 (O(N²) guard still in losses.py)

Output:
  results/exp2_sw_{N}.json        per-run metrics
  results/exp2_smallworld.json    summary table
"""

from __future__ import annotations

import json
import os
import time

import h5py
import torch
import torch.utils.data

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.training.trainer import Trainer
from src.training.experiment_config import trainer_kwargs, topology_kwargs, GA_BEST
from src.utils.metrics import compute_all_metrics

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

CLASS_NAMES = [
    "tench", "english_springer", "cassette_player", "chain_saw", "church",
    "french_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute",
]
VGG16_FC_PARAMS = 123_642_856
FULL_EPOCHS = 100   # shorter than exp1 — establish scaling trend
N_HIDDEN_SWEEP = [256, 512, 1024, 2048, 4096, 10000, 20000]

# Shared hyperparameters (from Exp 1 GA best)
LR_WPOS = GA_BEST["lr_Wpos"]
BATCH_SIZE = GA_BEST["batch_size"]


def load_data() -> tuple:
    with h5py.File("data/store.h5", "r") as f:
        return (
            torch.tensor(f["train/features"][:]),
            torch.tensor(f["train/soft_labels"][:]),
            torch.tensor(f["train/labels"][:]),
            torch.tensor(f["val/features"][:]),
            torch.tensor(f["val/soft_labels"][:]),
            torch.tensor(f["val/labels"][:]),
        )


def run_one(n_hidden: int, data: tuple, device: str) -> dict:
    train_f, train_sl, train_l, val_f, val_sl, val_l = data

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_f, train_sl, train_l),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_f, val_sl, val_l),
        batch_size=BATCH_SIZE, num_workers=0,
    )

    topo = topology_kwargs(n_hidden)
    model = SGNNET_SmallWorld(
        N_hidden=n_hidden,
        N_in=train_f.shape[1],
        K_in=topo["K_in"],
        K_local=topo["K_local"],
        K_random=topo["K_random"],
        n_groups=topo["n_groups"],
        K_iter=topo["K_iter"],
        sparsity=0.90,
    )
    total_params = sum(p.numel() for p in model.parameters())

    tkwargs = trainer_kwargs(n_hidden, lr_wpos=LR_WPOS)
    trainer = Trainer(model, train_loader, val_loader, device=device, **tkwargs)

    print(f"  N={n_hidden}  params={total_params}  lambda_safety={tkwargs['lambda_safety']:.4f}")

    t_start = time.perf_counter()

    def log(m):
        if m["epoch"] % 25 == 0 or m["epoch"] < 3:
            elapsed = time.perf_counter() - t_start
            print(
                f"  [N={n_hidden}] epoch {m['epoch']:3d}  "
                f"train={m['train_loss']:.4f}  val={m['val_loss']:.4f}  "
                f"t={elapsed:.0f}s"
            )

    history = trainer.train(FULL_EPOCHS, log_fn=log)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/exp2_sw_{n_hidden}_best.pt")

    result = trainer.evaluate()
    metrics = compute_all_metrics(
        result["scores"].numpy(), result["labels"].numpy(), CLASS_NAMES
    )

    elapsed = time.perf_counter() - t_start
    out = {
        "experiment": f"exp2_sw_{n_hidden}",
        "N_hidden": n_hidden,
        "hyperparams": {
            "K_in": 50, "K_local": 4, "K_random": 2, "K_iter": 3,
            "lr_wpos": LR_WPOS, "lambda_safety": tkwargs["lambda_safety"],
        },
        "full_epochs": len(history),
        "elapsed_s": round(elapsed, 1),
        "top1_accuracy": metrics["top1_accuracy"],
        "mAP": metrics["mAP"],
        "params": total_params,
        "percent_of_vgg16_fc": round(total_params / VGG16_FC_PARAMS * 100, 4),
        "training_history": [
            {"epoch": h["epoch"], "train_loss": h["train_loss"], "val_loss": h["val_loss"]}
            for h in history
        ],
    }

    os.makedirs("results", exist_ok=True)
    with open(f"results/exp2_sw_{n_hidden}.json", "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"[N={n_hidden}] top1={metrics['top1_accuracy']:.4f}  "
        f"mAP={metrics['mAP']:.4f}  params={total_params}  t={elapsed:.0f}s"
    )
    return out


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Sweeping N_hidden={N_HIDDEN_SWEEP}  epochs={FULL_EPOCHS}")
    print(f"Architecture: SmallWorld  K_in=50  K_local=4  K_random=2  K_iter=3\n")

    data = load_data()
    print(f"Loaded data: train={data[0].shape[0]}, val={data[3].shape[0]}\n")

    results = []
    for n_hidden in N_HIDDEN_SWEEP:
        print(f"\n--- N_hidden={n_hidden} ---")
        r = run_one(n_hidden, data, device)
        results.append(r)

    print("\n=== SmallWorld Scaling Summary ===")
    print(f"{'N_hidden':>8}  {'params':>8}  {'top-1':>8}  {'mAP':>8}  {'time(s)':>8}")
    for r in results:
        print(
            f"{r['N_hidden']:>8}  {r['params']:>8}  "
            f"{r['top1_accuracy']:>8.4f}  {r['mAP']:>8.4f}  {r['elapsed_s']:>8.0f}"
        )

    summary = {
        "architecture": "SGNNET_SmallWorld",
        "sweep": [
            {"N_hidden": r["N_hidden"], "params": r["params"],
             "top1_accuracy": r["top1_accuracy"], "mAP": r["mAP"],
             "elapsed_s": r["elapsed_s"]}
            for r in results
        ],
    }
    with open("results/exp2_smallworld.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved: results/exp2_smallworld.json")


if __name__ == "__main__":
    main()
