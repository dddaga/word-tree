"""
V3 Phase Cell
Wraps ModularPhaseCell with complex activation (Issue 1) + normalization (Issue 4).
Same __call__ interface as ModularPhaseCell.
"""

import torch
import torch.nn as nn
from typing import Tuple

from core.high_res_tables import HighResolutionLookupTables
from core.modular_cell import ModularPhaseCell

from experiments.v3_dynamic_phasor.components.complex_signal import ComplexSignalComputer
from experiments.v3_dynamic_phasor.components.signal_normalization import PhasorNormalization


class V3PhaseCell(nn.Module):
    """Phase cell with full complex activation and signal normalization."""

    def __init__(
        self,
        vector_dim: int,
        lookup_tables: HighResolutionLookupTables,
        gamma: float = 1.0,
        enable_complex: bool = True,
        enable_normalization: bool = True,
        mag_bins: int = 1024,
    ):
        super().__init__()

        self.vector_dim = vector_dim
        self.lookup = lookup_tables
        self.phase_bins = lookup_tables.N
        self.mag_bins = mag_bins
        self.enable_complex = enable_complex
        self.enable_normalization = enable_normalization

        # Underlying cell for phase-magnitude transfer logic
        self.base_cell = ModularPhaseCell(vector_dim, lookup_tables)

        # V3 components
        self.complex_signal = ComplexSignalComputer(lookup_tables, gamma=gamma)
        self.normalizer = PhasorNormalization(mag_bins=mag_bins)

    def forward(
        self,
        ctx_phase_idx: torch.Tensor,
        ctx_mag_idx: torch.Tensor,
        self_phase_idx: torch.Tensor,
        self_mag_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with complex activation + normalization.

        Returns same tuple as ModularPhaseCell:
            (phase_out, mag_out, signal, strength, grad_phase, grad_mag)
        """
        # Direct modular transfer (same as base cell)
        phase_out = (ctx_phase_idx + self_phase_idx) % self.phase_bins
        mag_out = (ctx_mag_idx + self_mag_idx) % self.mag_bins

        # Normalization: clamp magnitude indices
        if self.enable_normalization:
            mag_out = self.normalizer.clamp_magnitude_indices(mag_out)

        if self.enable_complex:
            # Full complex phasor
            real, imag, envelope = self.complex_signal.get_complex_signal(phase_out, mag_out)

            # Use real part as the signal vector (matches existing interface)
            signal = real

            # Apply RMSNorm to the signal
            if self.enable_normalization:
                signal = self.normalizer.rmsnorm_signal(signal)

            # Strength from envelope (always positive)
            strength = torch.sum(envelope)

            # Gradients through complex channels
            # For backward compat, approximate upstream_grad as ones
            upstream_real = torch.ones_like(real)
            upstream_imag = torch.zeros_like(imag)
            grad_phase, grad_mag = self.complex_signal.compute_complex_gradients(
                phase_out, mag_out, upstream_real, upstream_imag,
            )
        else:
            # Fallback to base cell behavior
            signal, grad_phase, grad_mag = self.lookup.forward(phase_out, mag_out)
            if self.enable_normalization:
                signal = self.normalizer.rmsnorm_signal(signal)
            strength = torch.sum(signal)

        return phase_out, mag_out, signal, strength, grad_phase, grad_mag

    def compute_routing_strength(
        self, ctx_phase_idx: torch.Tensor, self_phase_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Phase alignment routing strength (delegated to base cell)."""
        return self.base_cell.compute_routing_strength(ctx_phase_idx, self_phase_idx)
