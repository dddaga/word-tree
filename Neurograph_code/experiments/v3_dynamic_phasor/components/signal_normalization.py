"""
Issue 4: Normalization
Two-pronged approach: magnitude index clamping (discrete) + signal RMSNorm (continuous).
"""

import torch
from typing import Optional


class PhasorNormalization:
    """Signal normalization for phasor-based GNN."""

    def __init__(
        self,
        mag_bins: int = 1024,
        clamp_low_frac: float = 0.1,
        clamp_high_frac: float = 0.9,
        eps: float = 1e-8,
    ):
        self.mag_bins = mag_bins
        self.clamp_low = int(mag_bins * clamp_low_frac)   # 102 for 1024
        self.clamp_high = int(mag_bins * clamp_high_frac)  # 921 for 1024
        self.eps = eps

    def clamp_magnitude_indices(self, mag_indices: torch.Tensor) -> torch.Tensor:
        """Clamp magnitude indices to safe range, preventing exp(sin(m)) extremes."""
        return torch.clamp(mag_indices, self.clamp_low, self.clamp_high)

    def rmsnorm_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Root-Mean-Square normalization of signal vectors.
        Preserves direction, controls scale. Output RMS ~ 1.0.
        """
        rms = torch.sqrt(torch.mean(signal ** 2) + self.eps)
        return signal / rms

    def normalize_strengths(self, strengths: torch.Tensor) -> torch.Tensor:
        """Optional per-timestep normalization of node strengths across a set of nodes."""
        if strengths.numel() == 0:
            return strengths
        mean = strengths.mean()
        std = strengths.std() + self.eps
        return (strengths - mean) / std
