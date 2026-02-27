"""
Issue 1: Complete Complex Activation
Full phasor exp(i*phi + gamma*sin(m)) with both real and imaginary channels.

Current: signal = cos(phi) * exp(sin(m))  (real part only)
New:     signal = exp(gamma*sin(m)) * [cos(phi) + i*sin(phi)]
"""

import torch
import math
from typing import Tuple

from core.high_res_tables import HighResolutionLookupTables


class ComplexSignalComputer:
    """Computes full complex phasor signals using existing lookup tables."""

    def __init__(self, lookup_tables: HighResolutionLookupTables, gamma: float = 1.0):
        self.lookup = lookup_tables
        self.gamma = gamma

    def get_complex_signal(
        self,
        phase_indices: torch.Tensor,
        mag_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute full complex phasor signal.

        Returns:
            (real, imag, envelope) each of shape matching input indices.
            real     = exp(gamma*sin(m)) * cos(phi)
            imag     = exp(gamma*sin(m)) * sin(phi)
            envelope = exp(gamma*sin(m))
        """
        cos_vals = self.lookup.lookup_phase(phase_indices)        # cos(phi)
        sin_vals = self.lookup.lookup_phase_sin(phase_indices)    # sin(phi)
        exp_sin_vals = self.lookup.lookup_magnitude(mag_indices)  # exp(sin(m))

        # Apply gamma scaling: exp(gamma*sin(m)) = exp(sin(m))^gamma
        if self.gamma != 1.0:
            envelope = exp_sin_vals ** self.gamma
        else:
            envelope = exp_sin_vals

        real = envelope * cos_vals
        imag = envelope * sin_vals
        return real, imag, envelope

    def compute_complex_strength(
        self,
        phase_indices: torch.Tensor,
        mag_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar activation strength = sum of envelope across dims (always positive)."""
        _, _, envelope = self.get_complex_signal(phase_indices, mag_indices)
        return torch.sum(envelope)

    def compute_complex_gradients(
        self,
        phase_indices: torch.Tensor,
        mag_indices: torch.Tensor,
        upstream_real: torch.Tensor,
        upstream_imag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gradient of loss through both real and imaginary channels.

        d(real)/d(phi)  = -sin(phi) * envelope
        d(real)/d(m)    =  gamma * cos(m_range) * exp(gamma*sin(m)) * cos(phi)
        d(imag)/d(phi)  =  cos(phi) * envelope
        d(imag)/d(m)    =  gamma * cos(m_range) * exp(gamma*sin(m)) * sin(phi)

        Combined via chain rule with upstream gradients from both channels.
        """
        phase_indices = torch.clamp(phase_indices, 0, self.lookup.N - 1)
        mag_indices = torch.clamp(mag_indices, 0, self.lookup.M - 1)

        cos_vals = self.lookup.lookup_phase(phase_indices)
        sin_vals = self.lookup.lookup_phase_sin(phase_indices)
        exp_sin_vals = self.lookup.lookup_magnitude(mag_indices)

        if self.gamma != 1.0:
            envelope = exp_sin_vals ** self.gamma
        else:
            envelope = exp_sin_vals

        # d(envelope)/d(m) = gamma * cos(m_range) * exp(gamma*sin(m))
        # The mag_exp_sin_grad_table stores cos(m_range)*exp(sin(m))
        # We need to adjust for gamma
        mag_grad_base = self.lookup.mag_exp_sin_grad_table[mag_indices]
        if self.gamma != 1.0:
            # d/dm[exp(sin(m))^gamma] = gamma * exp(sin(m))^(gamma-1) * cos(m)*exp(sin(m))
            #                         = gamma * exp(sin(m))^gamma * cos(m)
            # mag_grad_base = cos(m)*exp(sin(m)), so:
            d_envelope_dm = self.gamma * mag_grad_base * (exp_sin_vals ** (self.gamma - 1.0))
        else:
            d_envelope_dm = mag_grad_base

        # Phase gradients: chain rule through both real and imag
        # d(real)/d(phi) = -sin(phi) * envelope
        # d(imag)/d(phi) = cos(phi) * envelope
        phase_grad = (
            upstream_real * (-sin_vals * envelope) +
            upstream_imag * (cos_vals * envelope)
        )

        # Magnitude gradients: chain rule through both real and imag
        # d(real)/d(m) = d_envelope_dm * cos(phi)
        # d(imag)/d(m) = d_envelope_dm * sin(phi)
        mag_grad = (
            upstream_real * (d_envelope_dm * cos_vals) +
            upstream_imag * (d_envelope_dm * sin_vals)
        )

        # Apply scale factors
        phase_grad = phase_grad * self.lookup.phase_grad_scale
        mag_grad = mag_grad * self.lookup.mag_grad_scale

        return phase_grad, mag_grad
