"""
2D FFNN: same as 1D but input is (features, positional_encoding) stacked on a new axis.
Each node has two components (feature-driven and position-driven), summed before activation —
analogous to phase/magnitude with standard activations, for future radiation-resonance training.
Sparse variant applies fixed conduction masks per tower (same idea as ffnn_sparsity_benchmark).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_positional_encoding(feature_dim: int) -> np.ndarray:
    """(feature_dim,) encoding of positions 0 .. feature_dim-1 (same dim as features)."""
    pos = np.arange(feature_dim, dtype=np.float32)
    div = np.exp(np.arange(0, feature_dim, 2, dtype=np.float32) * (-np.log(10000.0) / feature_dim))
    pe = np.zeros((feature_dim,), dtype=np.float32)
    pe[0::2] = np.sin(pos[0::2] * div[: (feature_dim + 1) // 2])
    pe[1::2] = np.cos(pos[1::2] * div[: feature_dim // 2]) if feature_dim > 1 else pe
    return pe


def to_2d_input(X: np.ndarray, pe: np.ndarray) -> np.ndarray:
    """Stack features and broadcast PE: (N, F) -> (N, 2, F)."""
    pe_broadcast = np.broadcast_to(pe, (X.shape[0], pe.size)).astype(np.float32)
    return np.stack([X, pe_broadcast], axis=1)


class FeedForwardNet2D(nn.Module):
    """
    Two parallel FFNNs (feature tower + positional tower), summed at output.
    Input x: (B, 2, F). Each node (hidden and output) has two components; we sum them.
    Enables later radiation-resonance policy by treating the two streams like phase/magnitude.
    """

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.tower_f = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )
        self.tower_p = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, F)
        x_f, x_p = x[:, 0, :], x[:, 1, :]
        return self.tower_f(x_f) + self.tower_p(x_p)


def estimate_macs_2d(input_dim: int, hidden_size: int, output_dim: int) -> int:
    """MACs for one forward pass (two towers, same as 2× single-tower MACs)."""
    one_tower = (input_dim * hidden_size + hidden_size) + (hidden_size * output_dim + output_dim)
    return 2 * one_tower


def make_conduction_mask(
    out_features: int,
    in_features: int,
    sparsity: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Binary mask; each row keeps floor((1-sparsity)*in_features) ones."""
    if sparsity >= 1.0:
        return torch.zeros(out_features, in_features)
    n_active = max(1, int(round((1.0 - sparsity) * in_features)))
    mask = torch.zeros(out_features, in_features)
    for i in range(out_features):
        idxs = rng.choice(in_features, size=n_active, replace=False)
        mask[i, idxs] = 1.0
    return mask


class SparseFeedForwardNet2D(nn.Module):
    """2D FFNN with fixed conduction masks on all four weight matrices (both towers)."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        sparsity: float,
        seed: int = 42,
    ):
        super().__init__()
        self.fc1_f = nn.Linear(input_dim, hidden_size)
        self.fc2_f = nn.Linear(hidden_size, output_dim)
        self.fc1_p = nn.Linear(input_dim, hidden_size)
        self.fc2_p = nn.Linear(hidden_size, output_dim)

        rng = np.random.default_rng(seed)
        self.register_buffer(
            "mask1_f", make_conduction_mask(hidden_size, input_dim, sparsity, rng)
        )
        self.register_buffer(
            "mask2_f", make_conduction_mask(output_dim, hidden_size, sparsity, rng)
        )
        self.register_buffer(
            "mask1_p", make_conduction_mask(hidden_size, input_dim, sparsity, rng)
        )
        self.register_buffer(
            "mask2_p", make_conduction_mask(output_dim, hidden_size, sparsity, rng)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f, x_p = x[:, 0, :], x[:, 1, :]
        h_f = F.relu(F.linear(x_f, self.fc1_f.weight * self.mask1_f, self.fc1_f.bias))
        h_p = F.relu(F.linear(x_p, self.fc1_p.weight * self.mask1_p, self.fc1_p.bias))
        out_f = F.linear(h_f, self.fc2_f.weight * self.mask2_f, self.fc2_f.bias)
        out_p = F.linear(h_p, self.fc2_p.weight * self.mask2_p, self.fc2_p.bias)
        return out_f + out_p


def count_active_params_2d(model: SparseFeedForwardNet2D) -> int:
    """Active weight connections + all biases (both towers)."""
    w = (
        int(model.mask1_f.sum().item()) + int(model.mask2_f.sum().item())
        + int(model.mask1_p.sum().item()) + int(model.mask2_p.sum().item())
    )
    b = (
        model.fc1_f.bias.numel() + model.fc2_f.bias.numel()
        + model.fc1_p.bias.numel() + model.fc2_p.bias.numel()
    )
    return w + b


def count_active_macs_2d(model: SparseFeedForwardNet2D) -> int:
    """Active MACs (unmasked weights only, both towers)."""
    return (
        int(model.mask1_f.sum().item()) + int(model.mask2_f.sum().item())
        + int(model.mask1_p.sum().item()) + int(model.mask2_p.sum().item())
    )
