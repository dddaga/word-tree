"""Loss functions for SGNNET training.

Three components:
  - safety_valve_loss: bounded quadratic repulsion (auxiliary, must not dominate task)
  - load_balance_loss: variance penalty on selection frequency
  - total_loss: KL-div task + safety + load balance
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import personal_volume_radius


# -------------------------------------------------------------------
# Safety valve loss (bounded quadratic repulsion)
# -------------------------------------------------------------------

def safety_valve_loss(
    W: torch.Tensor,
    box_size: float = 1.0,
    N: int | None = None,
    D: int | None = None,
    task_loss: torch.Tensor | None = None,
    clip_frac: float = 0.15,
) -> torch.Tensor:
    """Bounded quadratic repulsion keeping the safety valve as an auxiliary loss.

    Design goal: safety contribution ≤ clip_frac * task_loss in steady state.
    At well-separated configurations: returns exactly 0.
    At collision: bounded at [0, 1] per pair (no 1/d divergence).

    Parameters
    ----------
    W          : [N_total, D] neuron positions (hidden + output), or joint
                 [N_total, 2D] when W_pos and W_phase are concatenated
    box_size   : confining hypercube side length
    N, D       : inferred from W.shape if None
    task_loss  : current task loss (detached). When provided, safety is soft-capped
                 at clip_frac * task_loss so the auxiliary never dominates.
    clip_frac  : maximum fraction of task_loss the safety valve may contribute
    """
    if N is None:
        N = W.shape[0]
    if D is None:
        D = W.shape[1]

    r_star = personal_volume_radius(N, D, box_size)
    r_repel = r_star / 2.0

    # --- Mutual repulsion (quadratic, bounded at [0,1] per pair) ---
    # Replaces 1/d Coulomb which diverges to infinity at collision.
    # margin = how far inside the danger zone: 0 when d >= r_repel, r_repel when d=0
    # repulsion = (margin / r_repel)^2: bounded in [0,1], gradient well-behaved
    if W.shape[0] > 1 and W.shape[0] <= 5000:
        dists = torch.cdist(W, W)                         # [N, N]
        eye = torch.eye(W.shape[0], device=W.device)
        off_diag = 1.0 - eye
        # Set diagonal to r_repel so margin=0 (no self-repulsion)
        d_safe = dists * off_diag + eye * r_repel
        margin = F.relu(r_repel - d_safe) * off_diag      # [N, N], zero for well-separated
        repulsion = (margin / r_repel).pow(2)
        n = W.shape[0]
        mutual = repulsion.sum() / (n * (n - 1))
    else:
        mutual = torch.tensor(0.0, device=W.device)

    # --- Boundary repulsion (same quadratic form) ---
    dist_lower = W.clamp(min=0.0)
    dist_upper = (box_size - W).clamp(min=0.0)
    d_wall = torch.minimum(dist_lower, dist_upper)
    wall_margin = F.relu(r_repel - d_wall)
    boundary = (wall_margin / r_repel).pow(2).mean()

    raw = mutual + boundary

    # --- Soft cap: safety never exceeds clip_frac of task loss ---
    # Prevents the repulsion from overpowering the classification objective.
    # .detach() ensures no gradient flows back through the cap threshold.
    if task_loss is not None:
        cap = clip_frac * task_loss.detach().clamp(min=1e-6)
        raw = raw.clamp(max=cap)

    return raw


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
    W            : [N_total, D] neuron positions (or joint W_pos||W_phase)
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

    # Safety valve — passes task loss so repulsion is capped at 15% of task
    safety = safety_valve_loss(W, box_size, N, D, task_loss=task)

    # Load balance
    if gate is not None:
        counts = gate.sum(dim=(0, 1))  # [N_hidden]
        lb = load_balance_loss(counts)
    else:
        lb = torch.tensor(0.0, device=W.device)

    return task + lambda_safety * safety + lambda_lb * lb
