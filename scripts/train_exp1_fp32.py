"""Exp 1 (Stage B) FP32 comparison run.

Reuses the GA best config from the original Exp 1 (exp1_ga_results.json)
and trains for 150 epochs with autocast disabled (FP32 throughout).

Purpose: determine whether FP16 autocast is hurting numerical precision
on this tiny model (1064 params, KL divergence, cdist on MPS).

Output:
  results/exp1_fp32_full.json      -- metrics after 150-epoch FP32 training
  checkpoints/exp1_fp32_best.pt    -- best model checkpoint
"""

from __future__ import annotations

import json
import os

import h5py
import torch
import torch.utils.data

from src.sgnnet.model_wave import SGNNET_Wave
from src.training.trainer import Trainer
from src.utils.metrics import compute_all_metrics

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

CLASS_NAMES = [
    "tench", "english_springer", "cassette_player", "chain_saw", "church",
    "french_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute",
]
VGG16_FC_PARAMS = 123_642_856
FULL_EPOCHS = 150


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}  |  autocast: DISABLED (FP32)")

    # Load GA best config from original Exp 1
    with open("results/exp1_ga_results.json") as f:
        ga_result = json.load(f)
    best = ga_result["best_config"]
    print(f"Reusing GA config: {best}")

    # Load data
    with h5py.File("data/store.h5", "r") as f:
        train_f  = torch.tensor(f["train/features"][:])
        train_sl = torch.tensor(f["train/soft_labels"][:])
        train_l  = torch.tensor(f["train/labels"][:])
        val_f    = torch.tensor(f["val/features"][:])
        val_sl   = torch.tensor(f["val/soft_labels"][:])
        val_l    = torch.tensor(f["val/labels"][:])
    print(f"Loaded data: train={train_f.shape[0]}, val={val_f.shape[0]}")

    bs = best["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_f, train_sl, train_l),
        batch_size=bs, shuffle=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_f, val_sl, val_l),
        batch_size=bs, num_workers=0,
    )

    model = SGNNET_Wave(
        N_hidden=best["N_hidden"], K=best["K"], D=4, N_in=train_f.shape[1],
        sparsity=0.90, use_proximity=True, use_wphase=False,
    )

    trainer = Trainer(
        model, train_loader, val_loader,
        lr_wpos=best["lr_Wpos"],
        lambda_safety=best["lambda_safety"],
        device=device,
        use_amp=False,          # <-- FP32: autocast disabled
    )

    def log(m):
        if m["epoch"] % 10 == 0 or m["epoch"] < 5:
            print(
                f"  epoch {m['epoch']:3d}  "
                f"train_loss={m['train_loss']:.4f}  "
                f"val_loss={m['val_loss']:.4f}"
            )

    history = trainer.train(FULL_EPOCHS, log_fn=log)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/exp1_fp32_best.pt")

    # Evaluate
    result = trainer.evaluate()
    metrics = compute_all_metrics(
        result["scores"].numpy(), result["labels"].numpy(), CLASS_NAMES
    )
    total_params = sum(p.numel() for p in model.parameters())

    out = {
        "experiment": "exp1_fp32",
        "hyperparams": best,
        "use_amp": False,
        "full_epochs": FULL_EPOCHS,
        "top1_accuracy": metrics["top1_accuracy"],
        "mAP": metrics["mAP"],
        "per_class": metrics["per_class"],
        "params": total_params,
        "percent_of_vgg16_fc": round(total_params / VGG16_FC_PARAMS * 100, 4),
        "sparsity": {
            "C_input": float((~model.C_input_mask).float().mean()),
            "C_hh":    float((~model.C_hh_mask).float().mean()),
            "C_ho":    float((~model.C_ho_mask).float().mean()),
        },
        "training_history": [
            {"epoch": i, "train_loss": h["train_loss"], "val_loss": h["val_loss"]}
            for i, h in enumerate(history)
        ],
    }
    os.makedirs("results", exist_ok=True)
    with open("results/exp1_fp32_full.json", "w") as f:
        json.dump(out, f, indent=2)

    # Side-by-side comparison with original FP16 run
    with open("results/exp1_full.json") as f:
        fp16_result = json.load(f)
    fp16_top1 = fp16_result["top1_accuracy"]
    fp16_map  = fp16_result["mAP"]

    print("\n--- FP16 vs FP32 Comparison ---")
    print(f"{'':20s}  {'FP16 (50ep)':>12}  {'FP32 (150ep)':>12}")
    print(f"{'top-1 accuracy':20s}  {fp16_top1:>12.4f}  {metrics['top1_accuracy']:>12.4f}")
    print(f"{'mAP':20s}  {fp16_map:>12.4f}  {metrics['mAP']:>12.4f}")
    print(f"{'params':20s}  {fp16_result['params']:>12d}  {total_params:>12d}")


if __name__ == "__main__":
    main()
