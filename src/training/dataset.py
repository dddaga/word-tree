"""In-memory HDF5 dataset for SGNNET training.

Loads the entire split into RAM tensors on construction.
On Mac Studio (256 GB RAM) this eliminates all per-batch I/O overhead
and removes the need for DataLoader workers.

Expected HDF5 layout:
    /{split}/features    [N, D_feat]  float32
    /{split}/soft_labels [N, C]       float32
    /{split}/labels      [N]          int64

Usage:
    from src.training.dataset import H5Dataset, make_loaders
    tr, va = make_loaders("data/store.h5", batch_size=128, seed=42)
"""

from __future__ import annotations
import torch
import torch.utils.data
import h5py


class H5Dataset(torch.utils.data.Dataset):
    """HDF5 dataset — full split loaded into RAM on construction."""

    def __init__(self, path: str, split: str = "train"):
        with h5py.File(path, "r") as f:
            self.features    = torch.from_numpy(f[f"{split}/features"][:])
            self.soft_labels = torch.from_numpy(f[f"{split}/soft_labels"][:])
            self.labels      = torch.from_numpy(f[f"{split}/labels"][:]).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.soft_labels[idx], self.labels[idx]


def make_loaders(
    path: str,
    batch_size: int = 128,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Return (train_loader, val_loader) with full dataset in RAM.

    Parameters
    ----------
    path        : path to HDF5 file (must have train/ and val/ groups)
    batch_size  : samples per batch
    seed        : generator seed for reproducible shuffle order
    num_workers : kept for API compatibility; defaults to 0 (no I/O benefit
                  once data is in RAM; also avoids MPS/multiprocess issues)
    """
    train_ds = H5Dataset(path, split="train")
    val_ds   = H5Dataset(path, split="val")

    gb = (train_ds.features.numel() + val_ds.features.numel()) * 4 / 1e9
    print(f"Dataset loaded into RAM: train={len(train_ds)}  val={len(val_ds)}"
          f"  ({gb:.2f} GB features)")

    g = torch.Generator().manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader
