"""SGNNET_Wave nn.Module: Wave architecture with phasor activations.

Supports three experimental stages:
  Stage A: static binary C, real activations (use_proximity=False)
  Stage B: + phasor proximity routing with spatial phase (use_proximity=True)
  Stage C: + learned W_phase operator (use_proximity=True, use_wphase=True)

Binary C matrices are registered as buffers (D-05: no learned values).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import compute_spatial_encoding
from .norm_masked import masked_normalize
from .wave_routing import phasor_proximity_routing


# -------------------------------------------------------------------
# Binary C matrix helper (D-05: mask only, no learned values)
# -------------------------------------------------------------------

def _make_binary_c(
    rows: int,
    cols: int,
    sparsity: float,
    zero_diag: bool = False,
) -> torch.Tensor:
    """Create a binary sparse mask tensor.

    Returns a float tensor of 0/1 values. Not an nn.Parameter.
    Every row is guaranteed at least one connection.
    """
    mask = (torch.rand(rows, cols) < (1.0 - sparsity)).float()

    if zero_diag and rows == cols:
        mask.fill_diagonal_(0)

    # Guarantee at least 1 connection per row
    dead_rows = mask.sum(dim=1) == 0
    if dead_rows.any():
        indices = dead_rows.nonzero(as_tuple=True)[0]
        random_cols = torch.randint(0, cols, (indices.shape[0],))
        mask[indices, random_cols] = 1.0

    return mask


# -------------------------------------------------------------------
# SGNNET_Wave Module
# -------------------------------------------------------------------

class SGNNET_Wave(nn.Module):
    """Sparse Geometric Neural Network with wave/phasor architecture.

    Parameters
    ----------
    N_hidden    : number of hidden neurons
    N_out       : number of output neurons (classes)
    D           : geometric dimensionality (fixed at 4)
    N_in        : input dimensionality (VGG16 pool5 flattened)
    sparsity    : fraction of zero entries in C matrices
    K           : total forward pass iterations
    box_size    : confining hypercube side length
    use_proximity : enable phasor proximity routing (Stage B/C)
    use_wphase  : enable learned W_phase operator (Stage C only)
    """

    def __init__(
        self,
        N_hidden: int = 256,
        N_out: int = 10,
        D: int = 4,
        N_in: int = 25088,
        sparsity: float = 0.90,
        K: int = 3,
        box_size: float = 1.0,
        use_proximity: bool = False,
        use_wphase: bool = False,
    ):
        super().__init__()
        self.N_hidden = N_hidden
        self.N_out = N_out
        self.D = D
        self.N_in = N_in
        self.K = K
        self.box_size = box_size
        self.use_proximity = use_proximity
        self.use_wphase = use_wphase

        # Learnable positions: hidden + output (D-01)
        self.W_pos = nn.Parameter(
            torch.rand(N_hidden + N_out, D) * box_size
        )

        # Spatial encoding buffer (precomputed, deterministic)
        self.register_buffer("spatial_coords", compute_spatial_encoding(N_in))

        # Binary C masks as buffers (D-05: immutable, no gradients)
        self.register_buffer("C_input_mask", _make_binary_c(N_in, N_hidden, sparsity))
        self.register_buffer(
            "C_hh_mask", _make_binary_c(N_hidden, N_hidden, sparsity, zero_diag=True)
        )
        self.register_buffer("C_ho_mask", _make_binary_c(N_hidden, N_out, sparsity))

        # Learned phase operator (Stage C / Exp 2 only, D-10)
        if use_wphase:
            self.W_phase = nn.Parameter(torch.zeros(N_hidden + N_out, D))
        else:
            self.W_phase = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Three-phase forward: seed -> iterate -> readout.

        Parameters
        ----------
        x : [batch, N_in] flat feature vector

        Returns
        -------
        scores : [batch, N_out] class scores
        """
        Z_re, Z_im = self._seed(x)
        Z_re, Z_im = self._iterate_hidden(Z_re, Z_im)
        return self._output_readout(Z_re, Z_im)

    # ---------------------------------------------------------------
    # Phase 1: Seeding
    # ---------------------------------------------------------------

    def _seed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        spatial = self.spatial_coords.unsqueeze(0).expand(B, -1, -1)
        A_input = torch.cat([x.unsqueeze(-1), spatial], dim=-1)  # [B, N_in, D=4]

        Z_re = torch.einsum("bid,ih->bhd", A_input, self.C_input_mask)
        Z_re = masked_normalize(Z_re)
        Z_im = torch.zeros_like(Z_re)
        return Z_re, Z_im

    # ---------------------------------------------------------------
    # Phase 2: Hidden iterations
    # ---------------------------------------------------------------

    def _iterate_hidden(
        self, Z_re: torch.Tensor, Z_im: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        W_hidden = self.W_pos[: self.N_hidden]

        for _k in range(self.K - 1):
            # Static path through binary C_hh
            S_re = torch.einsum("bhd,hj->bjd", Z_re, self.C_hh_mask)
            S_im = torch.einsum("bhd,hj->bjd", Z_im, self.C_hh_mask)

            if self.use_proximity:
                w_phase = (
                    self.W_phase[: self.N_hidden]
                    if self.W_phase is not None
                    else None
                )
                P_re, P_im = phasor_proximity_routing(
                    Z_re, Z_im, W_hidden, self.N_hidden,
                    self.D, self.box_size,
                    use_wphase=self.use_wphase, W_phase=w_phase,
                )
                Z_re = masked_normalize(S_re + P_re)
                Z_im = masked_normalize(S_im + P_im)
            else:
                # Stage A: real activations only
                Z_re = masked_normalize(S_re)
                Z_im = torch.zeros_like(Z_re)

        return Z_re, Z_im

    # ---------------------------------------------------------------
    # Phase 3: Output readout
    # ---------------------------------------------------------------

    def _output_readout(
        self, Z_re: torch.Tensor, Z_im: torch.Tensor
    ) -> torch.Tensor:
        W_out = self.W_pos[self.N_hidden:]

        # Static path: hidden -> output via C_ho
        S_re_out = torch.einsum("bhd,ho->bod", Z_re, self.C_ho_mask)

        if self.use_proximity:
            S_im_out = torch.einsum("bhd,ho->bod", Z_im, self.C_ho_mask)
            # Magnitude of phasor output
            A_out = torch.sqrt(S_re_out ** 2 + S_im_out ** 2 + 1e-8)
        else:
            # Stage A: real activations with ReLU
            A_out = F.relu(S_re_out)

        # Self-projection readout
        W_out_norm = F.normalize(W_out, dim=-1)
        scores = (A_out * W_out_norm.unsqueeze(0)).sum(dim=-1)  # [B, N_out]
        return scores
