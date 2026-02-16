"""
Complex activation per cell: a_d = r_d*e^(i*phi_d) with r_d = e^(gamma*sin(m_d)).
Net activation (vector algebra): Net = sum_d a_d = real + i*imag.
Net activation strength for node = |Net| = sqrt(real^2 + imag^2). All sums over vector_dim (last dim).
"""

import torch


def activation_real_imag(phases: torch.Tensor, mags: torch.Tensor, gamma: float = 1.0):
    """
    Per-cell a_d = r_d*e^(i*phase_d), r_d = exp(gamma*sin(mag_d)). Net = sum_d a_d.
    Returns (real, imag) so Net = real + i*imag; net activation strength = sqrt(real^2+imag^2).
    """
    mag_exponent = gamma * torch.sin(mags)
    mag_exponent = torch.clamp(mag_exponent, min=-10.0, max=10.0)
    mag_factor = torch.exp(mag_exponent)
    real = (torch.cos(phases) * mag_factor).sum(dim=-1)
    imag = (torch.sin(phases) * mag_factor).sum(dim=-1)
    return real, imag


def activation_strength_from_real_imag(
    real: torch.Tensor,
    imag: torch.Tensor,
    mags: torch.Tensor = None,
    gamma: float = 1.0,
):
    """
    Magnitude r = sqrt(real^2 + imag^2) (s = r*e^(i*phi), r = e^(gamma*sin(m))).
    Activation strength for beam = magnitude r.
    """
    return torch.sqrt(real * real + imag * imag + 1e-12)


def activation_strength(phases: torch.Tensor, mags: torch.Tensor, gamma: float = 1.0):
    """Activation strength = magnitude r for beam search."""
    real, imag = activation_real_imag(phases, mags, gamma)
    return activation_strength_from_real_imag(real, imag)


def theta_from_real_imag(real: torch.Tensor, imag: torch.Tensor):
    """theta = atan2(imag, real)."""
    return torch.atan2(imag, real + 1e-12)


def conduction_radiation_alignments(phase_effective: torch.Tensor, mag_target: torch.Tensor, gamma: float):
    """
    For a target with effective_phase = target_phase + source_phase, compute alignment for proportion split.
    conduction_align = sum_d cos(phase_effective_d)*exp(gamma*sin(mag_target_d))
    radiation_align = sum_d sin(phase_effective_d)*exp(gamma*sin(mag_target_d))
    """
    mag_exponent = gamma * torch.sin(mag_target)
    mag_exponent = torch.clamp(mag_exponent, min=-10.0, max=10.0)
    mag_factor = torch.exp(mag_exponent)
    conduction_align = (torch.cos(phase_effective) * mag_factor).sum()
    radiation_align = (torch.sin(phase_effective) * mag_factor).sum()
    return conduction_align, radiation_align
