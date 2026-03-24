"""Loss functions for SGNNET training.

Three components:
  - safety_valve_loss: dead-zone Coulomb repulsion (Section 7.2)
  - load_balance_loss: variance penalty on selection frequency (Section 7.3)
  - total_loss: KL-div task + safety + load balance
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import personal_volume_radius


# -------------------------------------------------------------------
# Safety valve loss (dead-zone Coulomb repulsion)
# -------------------------------------------------------------------

def safety_valve_loss(
    W: torch.Tensor,
    box_size: float = 1.0,
    N: int | None = None,
    D: int | None = None,
) -> torch.Tensor:
    """Dead-zone Coulomb repulsion for neuron positions.

    Returns exactly 0 when all neurons are well-separated and away
    from walls. Activates steeply when neurons collide or approach
    a boundary within r_repel = r* / 2.

    Parameters
    ----------
    W        : [N_total, D] neuron positions (hidden + output)
    box_size : confining hypercube side length
    N        : number of neurons (inferred from W.shape[0] if None)
    D        : dimensionality (inferred from W.shape[1] if None)
    """
    if N is None:
        N = W.shape[0]
    if D is None:
        D = W.shape[1]

    r_star = personal_volume_radius(N, D, box_size)
    r_repel = r_star / 2.0

    # --- Mutual repulsion ---
    if W.shape[0] > 1:
        dists = torch.cdist(W, W)  # [N_total, N_total]
        mask = ~torch.eye(W.shape[0], dtype=torch.bool, device=W.device)
        d_pairs = dists[mask].clamp(min=1e-8)
        mutual = F.relu(1.0 / d_pairs - 1.0 / r_repel).mean()
    else:
        mutual = torch.tensor(0.0, device=W.device)

    # --- Boundary repulsion ---
    dist_lower = W.clamp(min=1e-8)
    dist_upper = (box_size - W).clamp(min=1e-8)
    d_wall = torch.minimum(dist_lower, dist_upper)
    boundary = F.relu(1.0 / d_wall - 1.0 / r_repel).mean()

    return mutual + boundary


# -------------------------------------------------------------------
# Load balance loss
# -------------------------------------------------------------------

def load_balance_loss(selection_counts: torch.Tensor) -> torch.Tensor:
    """Penalize variance in neuron selection frequency.

    Parameters
    ----------
    selection_counts : [N_hidden] how often each neuron was selected
    """
    freq = selection_counts.float() / (selection_counts.sum() + 1e-8)
    return freq.var()


# -------------------------------------------------------------------
# Total loss
# -------------------------------------------------------------------

def total_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    W: torch.Tensor,
    gate: torch.Tensor | None = None,
    box_size: float = 1.0,
    N: int | None = None,
    D: int | None = None,
    lambda_safety: float = 0.5,
    lambda_lb: float = 0.01,
) -> torch.Tensor:
    """Combined training loss: KL-div + safety valve + load balance.

    Parameters
    ----------
    scores       : [batch, N_out] model output logits
    targets      : [batch, N_out] soft probability targets (from VGG16)
    W            : [N_total, D] neuron positions
    gate         : [batch, N_hidden, N_hidden] proximity mask (optional)
    box_size     : confining hypercube side length
    N, D         : passed to safety_valve_loss
    lambda_safety: weight for safety valve loss
    lambda_lb    : weight for load balance loss
    """
    # Task loss: KL divergence for distillation
    task = F.kl_div(
        F.log_softmax(scores, dim=-1),
        targets,
        reduction="batchmean",
    )

    # Safety valve
    safety = safety_valve_loss(W, box_size, N, D)

    # Load balance
    if gate is not None:
        counts = gate.sum(dim=(0, 1))  # [N_hidden]
        lb = load_balance_loss(counts)
    else:
        lb = torch.tensor(0.0, device=W.device)

    return task + lambda_safety * safety + lambda_lb * lb
