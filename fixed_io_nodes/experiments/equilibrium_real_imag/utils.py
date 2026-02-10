"""Local helpers for equilibrium experiment. No changes to core."""

import torch


def activation_real_imag(phases: torch.Tensor, mags: torch.Tensor, gamma: float = 1.0):
    """
    Complex view: e^(i*theta) with theta = sin(m); magnitude cos(phi).
    real = cos(phase)*cos(sin(mag)), imag = cos(phase)*sin(sin(mag)).
    gamma is ignored (iota = i is the imaginary unit; theta = sin(m) only).
    Sum over last dim.
    """
    phase_cos = torch.cos(phases)
    theta = torch.sin(mags)
    real = (phase_cos * torch.cos(theta)).sum(dim=-1)
    imag = (phase_cos * torch.sin(theta)).sum(dim=-1)
    return real, imag
