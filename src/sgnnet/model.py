"""SGNNET nn.Module: Sparse Geometric Neural Network.

Three-phase forward pass (seeding, hidden iterations, output injection)
with self-projection readout. Consumes geometry primitives from geometry.py
and spatial encoding from encoding.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import compute_spatial_encoding
from .geometry import dynamic_connectivity_hh, dynamic_connectivity_ho


# -------------------------------------------------------------------
# Sparse C matrix helpers
# -------------------------------------------------------------------

def _make_sparse_c(
    rows: int,
    cols: int,
    sparsity: float,
    zero_diag: bool = False,
) -> tuple[nn.Parameter, torch.Tensor]:
    """Create a sparse C matrix (values + mask).

    Returns (values_param, mask_buffer). Mask is fixed at init;
    values are learnable. Every row guaranteed >= 1 connection.
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

    values = nn.Parameter(torch.randn(rows, cols) * mask * 0.01)
    return values, mask


# -------------------------------------------------------------------
# SGNNET Module
# -------------------------------------------------------------------

class SGNNET(nn.Module):
    """Sparse Geometric Neural Network module.

    Parameters
    ----------
    N_hidden : int  — number of hidden neurons
    N_out    : int  — number of output neurons (classes)
    D        : int  — geometric dimensionality
    N_in     : int  — input dimensionality (VGG16 pool5 flattened)
    sparsity : float — fraction of zero entries in C matrices
    K        : int  — total forward pass iterations
    box_size : float — confining hypercube side length
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
    ):
        super().__init__()
        self.N_hidden = N_hidden
        self.N_out = N_out
        self.D = D
        self.N_in = N_in
        self.sparsity = sparsity
        self.K = K
        self.box_size = box_size

        # Learnable neuron positions: hidden + output only (D-03)
        self.W = nn.Parameter(torch.rand(N_hidden + N_out, D) * box_size)

        # Spatial encoding buffer (precomputed, deterministic)
        self.register_buffer(
            "spatial_coords", compute_spatial_encoding(N_in)
        )

        # Three sparse C matrices (D-05)
        self.C_input_values, c_in_mask = _make_sparse_c(
            N_in, N_hidden, sparsity
        )
        self.register_buffer("C_input_mask", c_in_mask)

        self.C_hh_values, c_hh_mask = _make_sparse_c(
            N_hidden, N_hidden, sparsity, zero_diag=True
        )
        self.register_buffer("C_hh_mask", c_hh_mask)

        self.C_ho_values, c_ho_mask = _make_sparse_c(
            N_hidden, N_out, sparsity
        )
        self.register_buffer("C_ho_mask", c_ho_mask)

        # Normalization (D-11)
        self.norm = nn.LayerNorm(D)

        # Gate tracking for load balance loss
        self._last_gate: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Three-phase forward pass.

        Parameters
        ----------
        x : [batch, N_in] flat feature vector

        Returns
        -------
        scores : [batch, N_out] class scores
        """
        B = x.shape[0]

        # Phase 1 -- Seeding
        A_hidden = self._seed(x, B)

        # Phase 2 -- Hidden iterations (K-1 steps)
        A_hidden = self._iterate_hidden(A_hidden)

        # Phase 3 -- Output injection + readout
        scores = self._output_readout(A_hidden, B)
        return scores

    # ---------------------------------------------------------------
    # Phase 1: Seeding
    # ---------------------------------------------------------------

    def _seed(self, x: torch.Tensor, B: int) -> torch.Tensor:
        spatial = self.spatial_coords.unsqueeze(0).expand(B, -1, -1)
        A_input = torch.cat([x.unsqueeze(-1), spatial], dim=-1)

        C_in = self.C_input_values * self.C_input_mask
        A_hidden = F.relu(self.norm(
            torch.einsum("bid,ih->bhd", A_input, C_in)
        ))
        return A_hidden

    # ---------------------------------------------------------------
    # Phase 2: Hidden iterations
    # ---------------------------------------------------------------

    def _iterate_hidden(self, A_hidden: torch.Tensor) -> torch.Tensor:
        W_hidden = self.W[: self.N_hidden]
        C_hh = self.C_hh_values * self.C_hh_mask

        self._last_gate = None
        for _k in range(self.K - 1):
            static = torch.einsum("bhd,hj->bjd", A_hidden, C_hh)
            dyn, gate = dynamic_connectivity_hh(
                A_hidden, W_hidden, self.N_hidden, self.D, self.box_size
            )
            self._last_gate = gate
            A_hidden = F.relu(self.norm(static + dyn))
        return A_hidden

    # ---------------------------------------------------------------
    # Phase 3: Output injection + readout
    # ---------------------------------------------------------------

    def _output_readout(
        self, A_hidden: torch.Tensor, B: int
    ) -> torch.Tensor:
        W_out = self.W[self.N_hidden :]
        C_ho = self.C_ho_values * self.C_ho_mask

        static_out = torch.einsum("bhd,ho->bod", A_hidden, C_ho)
        dyn_out = dynamic_connectivity_ho(
            A_hidden, W_out, self.N_hidden, self.D, self.box_size
        )
        A_out = F.relu(static_out + dyn_out)

        # Self-projection readout (D-06, ARCH-03)
        W_norm = F.normalize(W_out, dim=-1)
        scores = (A_out * W_norm.unsqueeze(0)).sum(dim=-1)
        scores = scores + 1e-4  # epsilon safety net (D-07)
        return scores
