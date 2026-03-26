"""Phasor proximity routing with path-length phase for SGNNET_Wave.

Implements Gaussian-weighted phasor propagation between neurons based on
their geometric positions. Phase rotation is determined by path length
relative to wavelength lambda = r*/2 (D-08).
"""

from __future__ import annotations

import math

import torch

from .geometry import personal_volume_radius


def phasor_proximity_routing(
    Z_re: torch.Tensor,
    Z_im: torch.Tensor,
    W_pos: torch.Tensor,
    N_hidden: int,
    D: int,
    box_size: float,
    use_wphase: bool = False,
    W_phase: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Phasor proximity routing among neurons.

    Parameters
    ----------
    Z_re, Z_im : [batch, N_hidden, D] real and imaginary activations
    W_pos       : [N_hidden, D] neuron weight positions
    N_hidden    : number of neurons (for r* computation)
    D           : geometric dimensionality
    box_size    : confining hypercube side length
    use_wphase  : whether to apply learned W_phase rotation (Exp 2)
    W_phase     : [N_hidden, D] per-neuron learned phase (if use_wphase)

    Returns
    -------
    Z_out_re, Z_out_im : [batch, N_hidden, D] routed phasor activations
    """
    r_star = personal_volume_radius(N_hidden, D, box_size)
    lam = r_star / 2.0

    # Pairwise distances: [N_hidden, N_hidden]
    dists = torch.cdist(W_pos.unsqueeze(0), W_pos.unsqueeze(0)).squeeze(0)

    # Gaussian strength with hard gate at r*
    strength = torch.exp(-dists ** 2 / (r_star ** 2 + 1e-8))
    strength = strength * (dists < r_star).float()

    # No self-connections
    strength.fill_diagonal_(0)

    # Normalize per target neuron (sum over source dim=0)
    strength = strength / (strength.sum(dim=0, keepdim=True) + 1e-8)

    # Phase rotation from path length: [N_hidden, N_hidden]
    phase_ij = 2.0 * math.pi * dists / (lam + 1e-8)
    cos_p = torch.cos(phase_ij)
    sin_p = torch.sin(phase_ij)

    # Phasor products via einsum (source i -> target j)
    # strength*cos_p: [i, j], Z_re: [b, i, d] -> [b, j, d]
    sc = strength * cos_p
    ss = strength * sin_p
    Z_prox_re = (
        torch.einsum("ij,bjd->bid", sc, Z_re)
        - torch.einsum("ij,bjd->bid", ss, Z_im)
    )
    Z_prox_im = (
        torch.einsum("ij,bjd->bid", ss, Z_re)
        + torch.einsum("ij,bjd->bid", sc, Z_im)
    )

    # Optional learned phase rotation (Stage C / Exp 2)
    if use_wphase and W_phase is not None:
        cos_w = torch.cos(W_phase).unsqueeze(0)  # [1, N_hidden, D]
        sin_w = torch.sin(W_phase).unsqueeze(0)
        Z_out_re = Z_prox_re * cos_w - Z_prox_im * sin_w
        Z_out_im = Z_prox_re * sin_w + Z_prox_im * cos_w
        return Z_out_re, Z_out_im

    return Z_prox_re, Z_prox_im
