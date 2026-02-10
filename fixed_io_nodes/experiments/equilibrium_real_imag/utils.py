"""Local helpers for equilibrium experiment. No changes to core."""

import torch


def activation_real_imag(phases: torch.Tensor, mags: torch.Tensor, gamma: float = 1.0):
    """
    real = cos(phase)*cos(gamma*sin(mag)), imag = cos(phase)*sin(gamma*sin(mag)).
    Sum over last dim. Clamps exponent range for stability.
    """
    phase_cos = torch.cos(phases)
    arg = gamma * torch.sin(mags)
    arg = torch.clamp(arg, min=-10.0, max=10.0)
    real = (phase_cos * torch.cos(arg)).sum(dim=-1)
    imag = (phase_cos * torch.sin(arg)).sum(dim=-1)
    return real, imag
