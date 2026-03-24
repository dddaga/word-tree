"""Geometric primitives for SGNNET: r* and dynamic connectivity.

Provides personal_volume_radius (r*), hidden-hidden dynamic connectivity,
and hidden-to-output dynamic connectivity using batched cdist.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Personal volume radius
# ---------------------------------------------------------------------------

def personal_volume_radius(N: int, D: int, box_size: float = 1.0) -> float:
    """Compute r* = (box_size / 2) / N^(1/D).

    Properties:
      - r* <= box_size/2 always (guaranteed within confined space)
      - Expected neighbors per neuron = 1 in any dimension
      - No free hyperparameter: determined entirely by N and D
    """
    R = box_size / 2.0
    return R / (N ** (1.0 / D))


# ---------------------------------------------------------------------------
# Hidden-hidden dynamic connectivity
# ---------------------------------------------------------------------------

def dynamic_connectivity_hh(
    A_hidden: torch.Tensor,
    W_hidden: torch.Tensor,
    N_hidden: int,
    D: int,
    box_size: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic connectivity among hidden neurons.

    Parameters
    ----------
    A_hidden : [batch, N_hidden, D] current hidden activations
    W_hidden : [N_hidden, D] hidden neuron weight positions
    N_hidden : number of hidden neurons
    D        : geometric dimensionality
    box_size : confining hypercube side length

    Returns
    -------
    contribution : [batch, N_hidden, D] activation contribution
    gate         : [batch, N_hidden, N_hidden] proximity mask
    """
    r_star = personal_volume_radius(N_hidden, D, box_size)
    batch = A_hidden.shape[0]

    # Pairwise distances: [batch, N_hidden, N_hidden]
    W_exp = W_hidden.unsqueeze(0).expand(batch, -1, -1)
    dists = torch.cdist(A_hidden, W_exp)

    # Gaussian kernel + hard gate
    strength = torch.exp(-dists ** 2 / (r_star ** 2 + 1e-8))
    gate = (dists < r_star).float()
    strength = strength * gate

    # Zero diagonal: no self-connections
    eye = torch.eye(N_hidden, device=A_hidden.device).unsqueeze(0)
    strength = strength * (1.0 - eye)
    gate = gate * (1.0 - eye)

    # Normalize per target neuron (column-wise, dim=1 = source)
    strength = strength / (strength.sum(dim=1, keepdim=True) + 1e-8)

    # Propagate: weighted sum of source activations for each target
    contribution = torch.einsum("bij,bjd->bid", strength, A_hidden)

    return contribution, gate


# ---------------------------------------------------------------------------
# Hidden-to-output dynamic connectivity
# ---------------------------------------------------------------------------

def dynamic_connectivity_ho(
    A_hidden: torch.Tensor,
    W_out: torch.Tensor,
    N_hidden: int,
    D: int,
    box_size: float = 1.0,
) -> torch.Tensor:
    """Dynamic connectivity from hidden neurons to output positions.

    Parameters
    ----------
    A_hidden : [batch, N_hidden, D] current hidden activations
    W_out    : [N_out, D] output neuron weight positions
    N_hidden : number of hidden neurons (used for r* computation)
    D        : geometric dimensionality
    box_size : confining hypercube side length

    Returns
    -------
    contribution : [batch, N_out, D] activation contribution to outputs
    """
    r_star = personal_volume_radius(N_hidden, D, box_size)
    batch = A_hidden.shape[0]

    # Distances: [batch, N_hidden, N_out]
    W_exp = W_out.unsqueeze(0).expand(batch, -1, -1)
    dists = torch.cdist(A_hidden, W_exp)

    # Gaussian kernel + hard gate
    strength = torch.exp(-dists ** 2 / (r_star ** 2 + 1e-8))
    gate = (dists < r_star).float()
    strength = strength * gate

    # Normalize per output neuron (dim=1 = source hidden neurons)
    strength = strength / (strength.sum(dim=1, keepdim=True) + 1e-8)

    # Propagate: hidden activations -> output positions
    contribution = torch.einsum("bio,bid->bod", strength, A_hidden)

    return contribution
