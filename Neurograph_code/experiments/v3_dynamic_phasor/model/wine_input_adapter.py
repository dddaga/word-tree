"""
Wine Quality Input Adapter
Custom adapter for UCI Wine Quality (Red) dataset -- 11 features, 6 classes.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import math
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from core.high_res_tables import HighResolutionLookupTables

RED_WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

WINE_FEATURE_NAMES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]


class WineInputAdapter(nn.Module):
    """Deep neural network input adapter for Wine Quality dataset."""

    def __init__(
        self,
        input_dim: int = 11,
        num_input_nodes: int = 22,
        vector_dim: int = 5,
        phase_bins: int = 512,
        mag_bins: int = 1024,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        device: str = "cpu",
        test_size: float = 0.30,
        val_fraction: float = 0.50,
        seed: int = 42,
        data_path: Optional[str] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_input_nodes = num_input_nodes
        self.vector_dim = vector_dim
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.device_name = device
        self.seed = seed

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.output_dim = num_input_nodes * vector_dim * 2  # phase + magnitude

        # Build projection network
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_d, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_d = h
        layers.append(nn.Linear(in_d, self.output_dim))
        layers.append(nn.Tanh())  # bound to [-1, 1]
        self.projection = nn.Sequential(*layers)

        # Load and prepare dataset
        self.scaler = StandardScaler()
        self._load_dataset(test_size, val_fraction, seed, data_path)

        self.to(device)

    # ------------------------------------------------------------------ data
    def _load_dataset(
        self, test_size: float, val_fraction: float, seed: int,
        data_path: Optional[str] = None,
    ) -> None:
        """Load Wine Quality Red, normalize, and split."""
        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path, sep=";")
        else:
            # Try local cache first
            cache_dir = os.path.join(os.path.dirname(__file__), "..", "data")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, "winequality-red.csv")
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path, sep=";")
            else:
                try:
                    df = pd.read_csv(RED_WINE_URL, sep=";")
                    df.to_csv(cache_path, sep=";", index=False)
                except Exception:
                    raise RuntimeError(
                        f"Cannot download Wine dataset. Place winequality-red.csv "
                        f"(semicolon-separated) at {cache_path}"
                    )

        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)

        # Remap quality labels 3-8 -> 0-5
        self.label_map = {v: i for i, v in enumerate(sorted(set(y)))}
        self.inverse_label_map = {i: v for v, i in self.label_map.items()}
        y = np.array([self.label_map[v] for v in y])
        self.num_classes = len(self.label_map)

        # Stratified split: train / (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y,
        )
        # Split temp into val and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_fraction, random_state=seed, stratify=y_temp,
        )

        # Fit scaler on train only
        X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val = self.scaler.transform(X_val).astype(np.float32)
        X_test = self.scaler.transform(X_test).astype(np.float32)

        self.X_train = torch.tensor(X_train, device=self.device_name)
        self.y_train = torch.tensor(y_train, dtype=torch.long, device=self.device_name)
        self.X_val = torch.tensor(X_val, device=self.device_name)
        self.y_val = torch.tensor(y_val, dtype=torch.long, device=self.device_name)
        self.X_test = torch.tensor(X_test, device=self.device_name)
        self.y_test = torch.tensor(y_test, dtype=torch.long, device=self.device_name)

        print(f"Wine Quality dataset loaded:")
        print(f"  Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)}")
        print(f"  Classes: {self.num_classes} (quality {list(self.label_map.keys())})")

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features to [-1, 1] output."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.projection(x).squeeze(0)

    def quantize_to_phase_mag(
        self, projected: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape projected output and quantize to phase/magnitude indices.

        projected: [output_dim] with values in [-1, 1]
        Returns:
            (phase_indices [num_input_nodes, vector_dim],
             mag_indices   [num_input_nodes, vector_dim])
        """
        reshaped = projected.view(self.num_input_nodes, self.vector_dim, 2)
        phase_raw = reshaped[:, :, 0]  # [-1, 1]
        mag_raw = reshaped[:, :, 1]

        # Phase: [-1,1] -> [0, 2*pi) -> [0, N-1]
        phase_cont = (phase_raw + 1.0) / 2.0 * (2 * math.pi)
        phase_indices = torch.floor(
            (phase_cont % (2 * math.pi)) / (2 * math.pi) * self.phase_bins
        ).long()
        phase_indices = torch.clamp(phase_indices, 0, self.phase_bins - 1)

        # Magnitude: [-1,1] -> [-3, 3] -> [0, M-1]
        mag_cont = mag_raw * 3.0  # scale to [-3, 3]
        mag_norm = (mag_cont + 3.0) / 6.0  # [0, 1]
        mag_indices = torch.floor(mag_norm * self.mag_bins).long()
        mag_indices = torch.clamp(mag_indices, 0, self.mag_bins - 1)

        return phase_indices, mag_indices

    def get_input_context(
        self,
        sample_idx: int,
        input_node_ids: list,
        dataset: str = "train",
    ) -> Tuple[Dict, int]:
        """
        Get input context dict for a single sample.

        Returns:
            (input_context {node_id: (phase_idx [D], mag_idx [D])}, target_label)
        """
        if dataset == "train":
            x, y = self.X_train[sample_idx], self.y_train[sample_idx].item()
        elif dataset == "val":
            x, y = self.X_val[sample_idx], self.y_val[sample_idx].item()
        else:
            x, y = self.X_test[sample_idx], self.y_test[sample_idx].item()

        projected = self.forward(x)
        phase_indices, mag_indices = self.quantize_to_phase_mag(projected)

        input_context = {}
        for i, node_id in enumerate(input_node_ids):
            input_context[node_id] = (phase_indices[i], mag_indices[i])

        return input_context, y

    def get_dataset_info(self) -> Dict:
        return {
            "dataset_size": len(self.X_train) + len(self.X_val) + len(self.X_test),
            "train_size": len(self.X_train),
            "val_size": len(self.X_val),
            "test_size": len(self.X_test),
            "num_classes": self.num_classes,
            "feature_names": WINE_FEATURE_NAMES,
            "label_map": self.label_map,
        }
