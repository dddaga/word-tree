"""Initialization utilities for SGNNET.

Phase 3: random uniform initialization for W positions.
Phase 4 may add K-means initialization if convergence is slow (D-09).
"""

from __future__ import annotations

import torch


def initialize_sgnnet(model) -> None:
    """Initialize SGNNET neuron positions to random uniform in [0, box_size].

    C matrices are already initialized in the constructor (_make_sparse_c),
    so only W needs re-initialization here.

    Parameters
    ----------
    model : SGNNET instance
    """
    with torch.no_grad():
        model.W.uniform_(0, model.box_size)
