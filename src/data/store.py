"""HDF5 tensor store and CSV manifest for Imagenette features.

Persists VGG16 pre-FC features (25088-dim) and soft labels (10-dim) extracted
by VGGExtractor into a single HDF5 file with sample-level chunking for fast
random-access reads during training.

Schema (locked per D-05):
    store.h5
      /train/features    [N_train, 25088]  float32
      /train/soft_labels [N_train, 10]     float32
      /train/labels      [N_train]         int64
      /val/features      [N_val,   25088]  float32
      /val/soft_labels   [N_val,   10]     float32
      /val/labels        [N_val]           int64
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# TensorStore
# ---------------------------------------------------------------------------

class TensorStore:
    """Read/write interface for the HDF5 tensor store.

    Parameters
    ----------
    path : str
        Path to the HDF5 file (default: ``data/store.h5``).
    """

    def __init__(self, path: str = "data/store.h5") -> None:
        self.path = str(path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @staticmethod
    def write(
        path: str,
        train_data: tuple[Tensor, Tensor, Tensor],
        val_data: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        """Write features, soft labels, and labels for both splits.

        Parameters
        ----------
        path : str
            Output HDF5 file path.
        train_data : tuple[Tensor, Tensor, Tensor]
            ``(features [N,25088], soft_labels [N,10], labels [N])`` for train.
        val_data : tuple[Tensor, Tensor, Tensor]
            Same structure for val.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        train_features, train_soft_labels, train_labels = train_data
        val_features, val_soft_labels, val_labels = val_data

        with h5py.File(path, "w") as f:
            # Train split
            f.create_dataset(
                "train/features",
                data=train_features.numpy(),
                dtype="float32",
                chunks=(1, 25088),
            )
            f.create_dataset(
                "train/soft_labels",
                data=train_soft_labels.numpy(),
                dtype="float32",
                chunks=(1, 10),
            )
            f.create_dataset(
                "train/labels",
                data=train_labels.numpy(),
                dtype="int64",
            )
            # Val split
            f.create_dataset(
                "val/features",
                data=val_features.numpy(),
                dtype="float32",
                chunks=(1, 25088),
            )
            f.create_dataset(
                "val/soft_labels",
                data=val_soft_labels.numpy(),
                dtype="float32",
                chunks=(1, 10),
            )
            f.create_dataset(
                "val/labels",
                data=val_labels.numpy(),
                dtype="int64",
            )

    @staticmethod
    def write_manifest(
        path: str,
        train_labels: np.ndarray,
        val_labels: np.ndarray,
        class_names: list[str],
    ) -> None:
        """Write CSV manifest mapping every record to its metadata.

        CSV columns: ``index, split, class_name, class_idx, h5_idx``

        Parameters
        ----------
        path : str
            Output CSV file path.
        train_labels : np.ndarray
            Integer class labels for train split (shape [N_train]).
        val_labels : np.ndarray
            Integer class labels for val split (shape [N_val]).
        class_names : list[str]
            Mapping from class index to human-readable name (length 10).
        """
        rows = []
        global_idx = 0

        for h5_idx, label in enumerate(train_labels):
            rows.append({
                "index": global_idx,
                "split": "train",
                "class_name": class_names[int(label)],
                "class_idx": int(label),
                "h5_idx": h5_idx,
            })
            global_idx += 1

        for h5_idx, label in enumerate(val_labels):
            rows.append({
                "index": global_idx,
                "split": "val",
                "class_name": class_names[int(label)],
                "class_idx": int(label),
                "h5_idx": h5_idx,
            })
            global_idx += 1

        df = pd.DataFrame(rows, columns=["index", "split", "class_name", "class_idx", "h5_idx"])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, split: str, idx: int) -> tuple[np.ndarray, np.ndarray, int]:
        """Read a single record from the store.

        Parameters
        ----------
        split : str
            ``"train"`` or ``"val"``.
        idx : int
            Row index within the split.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, int]
            ``(features[25088], soft_labels[10], label)``.
        """
        with h5py.File(self.path, "r") as f:
            features = f[f"{split}/features"][idx]
            soft_labels = f[f"{split}/soft_labels"][idx]
            label = int(f[f"{split}/labels"][idx])
        return features, soft_labels, label

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, int]:
        """Read from train split by index (for ROADMAP verification compat)."""
        return self.read("train", idx)

    def get_split(self, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load an entire split into memory.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(features[N,25088], soft_labels[N,10], labels[N])``.
        """
        with h5py.File(self.path, "r") as f:
            features = f[f"{split}/features"][:]
            soft_labels = f[f"{split}/soft_labels"][:]
            labels = f[f"{split}/labels"][:]
        return features, soft_labels, labels

    def shape(self, split: str) -> dict[str, tuple]:
        """Return shapes of all datasets in a split."""
        with h5py.File(self.path, "r") as f:
            return {
                "features": f[f"{split}/features"].shape,
                "soft_labels": f[f"{split}/soft_labels"].shape,
                "labels": f[f"{split}/labels"].shape,
            }
