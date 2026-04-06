"""
nn.Module wrappers for sparse operations.

SparseLinear  — drop-in replacement for nn.Linear with a sparse weight matrix.
SparseMatMul  — standalone sparse-times-dense (or dense-times-sparse) module.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Union

from .static_sparse import StaticSparseMatrix


class SparseLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear using a sparse weight matrix.

    The weight sparsity pattern is fixed at construction time (static).
    Values are learnable by default (same as nn.Linear).

    Parameters
    ----------
    weight : Tensor | StaticSparseMatrix
        Either a dense (out_features, in_features) weight tensor
        (non-zeros determined by threshold) or a pre-built StaticSparseMatrix.
    bias : bool | Tensor
        If True, learns an additive bias.  Pass a Tensor to use a fixed bias.
    threshold : float
        Entries with |w| <= threshold are treated as structural zeros.
    learnable_values : bool
        If True the sparse values are nn.Parameters and are updated by the
        optimizer.  If False they are treated as a fixed constant (fastest).

    Example
    -------
    weight = torch.randn(256, 512) * mask   # mask is your sparsity structure
    layer  = SparseLinear(weight)
    out    = layer(x)   # x: (batch, 512) -> out: (batch, 256)
    """

    def __init__(
        self,
        weight: Union[Tensor, StaticSparseMatrix],
        bias: Union[bool, Tensor] = True,
        threshold: float = 0.0,
        learnable_values: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        strategy: str = "auto",
        reorder: bool = False,
    ):
        super().__init__()

        if isinstance(weight, StaticSparseMatrix):
            self.sparse_weight = weight
        else:
            if weight.dim() != 2:
                raise ValueError(f"weight must be 2-D (out, in), got {weight.shape}")
            self.sparse_weight = StaticSparseMatrix.from_dense(
                weight, threshold=threshold,
                learnable=learnable_values, device=device,
                strategy=strategy, reorder=reorder,
            )

        self.out_features = self.sparse_weight.M
        self.in_features  = self.sparse_weight.K

        if learnable_values and isinstance(self.sparse_weight.values, nn.Parameter):
            self.register_parameter("sparse_values", self.sparse_weight.values)

        if isinstance(bias, bool):
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.out_features,
                                device=device or
                                self.sparse_weight.row_indices.device)
                )
            else:
                self.bias = None
        elif isinstance(bias, Tensor):
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (..., in_features)  ->  (..., out_features)
        """
        # Flatten batch dims, run sparse matmul, restore shape
        *batch, in_f = x.shape
        if in_f != self.in_features:
            raise ValueError(
                f"Input last dim {in_f} != in_features {self.in_features}"
            )

        if len(batch) == 0:
            # 1-D input (in_features,) -> (out_features,)
            out = self.sparse_weight.mm(x)
        elif len(batch) == 1:
            # 2-D (batch, in) -> (batch, out)  via  sparse(out,in) @ x.T -> out.T
            out = self.sparse_weight.mm(x.t()).t()
        else:
            # Higher-rank: (..., in) -> (..., out)
            flat = x.reshape(-1, in_f)                        # (B, in)
            out_flat = self.sparse_weight.mm(flat.t()).t()     # (B, out)
            out = out_flat.reshape(*batch, self.out_features)

        if self.bias is not None:
            out = out + self.bias

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"nnz={self.sparse_weight.nnz}, "
            f"density={self.sparse_weight.density:.4%}, "
            f"bias={self.bias is not None}"
        )


class SparseMatMul(nn.Module):
    """
    Fixed sparse matrix that can be applied to a dense input each forward pass.

    Useful when the sparse matrix is a pre-computed graph Laplacian,
    adjacency matrix, or projection that never changes.

    mode='left'  : out = sparse @ x   (sparse on left)
    mode='right' : out = x @ sparse   (sparse on right)
    """

    def __init__(
        self,
        sparse: Union[Tensor, StaticSparseMatrix],
        mode: str = "left",
        threshold: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        strategy: str = "auto",
        reorder: bool = False,
    ):
        super().__init__()
        if mode not in ("left", "right"):
            raise ValueError("mode must be 'left' or 'right'")
        self.mode = mode

        if isinstance(sparse, StaticSparseMatrix):
            self._sparse = sparse
        else:
            self._sparse = StaticSparseMatrix.from_dense(
                sparse, threshold=threshold, learnable=False, device=device,
                strategy=strategy, reorder=reorder,
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "left":
            return self._sparse.mm(x)
        else:
            return self._sparse.rmm(x)

    def extra_repr(self) -> str:
        return (
            f"shape={self._sparse.shape}, "
            f"nnz={self._sparse.nnz}, "
            f"density={self._sparse.density:.4%}, "
            f"mode={self.mode!r}"
        )
