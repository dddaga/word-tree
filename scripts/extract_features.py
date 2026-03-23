"""Run VGG16 feature extraction and write HDF5 tensor store + CSV manifest."""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.data.dataset import IMAGENETTE_CLASSES, get_dataloader
from src.data.extractor import VGGExtractor
from src.data.store import TensorStore


def main() -> None:
    extractor = VGGExtractor()
    print(f"Device: {extractor.device}")

    # MPS does not support pin_memory — use num_workers=0 to avoid spawn issues
    num_workers = 0
    pin_memory = extractor.device == "cpu"

    # --- Train split ---
    train_loader = get_dataloader("train", batch_size=256, num_workers=num_workers)
    train_features, train_soft_labels, train_labels = extractor.extract_all(
        train_loader, desc="Extracting train"
    )
    print(f"Train: {train_features.shape}, {train_soft_labels.shape}, {train_labels.shape}")

    # Flush MPS cache between passes
    if extractor.device == "mps":
        torch.mps.empty_cache()

    # --- Val split ---
    val_loader = get_dataloader("val", batch_size=256, num_workers=num_workers)
    val_features, val_soft_labels, val_labels = extractor.extract_all(
        val_loader, desc="Extracting val"
    )
    print(f"Val: {val_features.shape}, {val_soft_labels.shape}, {val_labels.shape}")

    # --- Write HDF5 ---
    TensorStore.write(
        "data/store.h5",
        (train_features, train_soft_labels, train_labels),
        (val_features, val_soft_labels, val_labels),
    )
    print("store.h5 written")

    # --- Write manifest ---
    TensorStore.write_manifest(
        "data/manifest.csv",
        train_labels.numpy(),
        val_labels.numpy(),
        IMAGENETTE_CLASSES,
    )
    print("manifest.csv written")
    print("Done.")


if __name__ == "__main__":
    main()
