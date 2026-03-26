"""Masked normalization for SGNNET_Wave activations.

Only neurons with |Z_j| > eps participate in mean/var computation.
Inactive neurons (near-zero magnitude) are zeroed in the output.
Works for both real activations (Stage A) and phasor (Stage B/C).
"""

from __future__ import annotations

import torch


def masked_normalize(
    x: torch.Tensor,
    eps_active: float = 1e-6,
    eps_norm: float = 1e-5,
) -> torch.Tensor:
    """Normalize x over active neurons only, zeroing inactive ones.

    Parameters
    ----------
    x : [batch, N, D] tensor (real activations or Z_re / Z_im)
    eps_active : magnitude threshold for considering a neuron active
    eps_norm : epsilon added to std for numerical stability

    Returns
    -------
    normalized : [batch, N, D] with inactive neurons zeroed
    """
    # Magnitude per neuron: [batch, N]
    mag = x.norm(dim=-1)

    # Active mask: [batch, N] bool -> [batch, N, 1] float for broadcasting
    active = (mag > eps_active)
    active_float = active.unsqueeze(-1).float()  # [batch, N, 1]

    # Count of active neurons per batch element: [batch, 1, 1]
    count = active_float.sum(dim=1, keepdim=True).clamp(min=1)

    # Mean over active neurons only: [batch, 1, D]
    mean = (x * active_float).sum(dim=1, keepdim=True) / count

    # Variance over active neurons only: [batch, 1, D]
    var = ((x - mean) ** 2 * active_float).sum(dim=1, keepdim=True) / count

    # Normalize and zero out inactive neurons
    normalized = (x - mean) / (var.sqrt() + eps_norm)
    return normalized * active_float
