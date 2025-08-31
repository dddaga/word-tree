# core/cell.py

import torch
import torch.nn as nn

class PhaseCell(nn.Module):
    def __init__(self, vector_dim: int, lookup_module):
        """
        Discrete phase-magnitude interaction module using lookup tables.

        Args:
            vector_dim (int): Dimensionality of phase/magnitude vectors
            lookup_module (ExtendedLookupTableModule): Includes cos/sin tables
        """
        super().__init__()
        self.D = vector_dim
        self.lookup = lookup_module
        self.N = lookup_module.N
        self.M = lookup_module.M

    def forward(self, ctx_phase_idx, ctx_mag_idx, self_phase_idx, self_mag_idx):
        """
        Args:
            ctx_phase_idx: LongTensor [D] — phase indices from context
            ctx_mag_idx:   LongTensor [D] — mag indices from context
            self_phase_idx: LongTensor [D] — node's own phase indices
            self_mag_idx:   LongTensor [D] — node's own mag indices

        Returns:
            phase_out: LongTensor [D] — summed phase indices (mod N)
            mag_out:   LongTensor [D] — summed mag indices (mod M)
            signal:    FloatTensor [D] — element-wise signal = cos ⊙ exp
            activation_strength: Float — sum(signal)
            grad_phase: FloatTensor [D] — d(signal)/d(phase)
            grad_mag:   FloatTensor [D] — d(signal)/d(mag)
        """
        # Add and wrap indices
        phase_out = (ctx_phase_idx + self_phase_idx) % self.N
        mag_out   = (ctx_mag_idx + self_mag_idx) % self.M

        # Forward values
        cos_vals = self.lookup.lookup_phase(phase_out)         # [D]
        exp_vals = self.lookup.lookup_magnitude(mag_out)       # [D]
        signal   = cos_vals * exp_vals                         # [D]

        # Gradients
        dcos = self.lookup.lookup_phase_grad(phase_out)        # [D]
        dexp = self.lookup.lookup_magnitude_grad(mag_out)      # [D]

        grad_phase = dcos * exp_vals                           # d(signal)/d(phase)
        grad_mag   = cos_vals * dexp                           # d(signal)/d(mag)

        strength = signal.sum()

        return phase_out, mag_out, signal, strength, grad_phase, grad_mag
