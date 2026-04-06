"""
Utility helpers for converting, inspecting, and benchmarking sparse matrices.
"""

from __future__ import annotations
import torch
from torch import Tensor
from typing import Optional, Union

from .static_sparse import StaticSparseMatrix


def to_static_sparse(
    matrix: Tensor,
    threshold: float = 0.0,
    learnable: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> StaticSparseMatrix:
    """
    Convenience: convert any 2-D dense tensor to a StaticSparseMatrix.

    Entries with |value| <= threshold are dropped (treated as structural zeros).
    """
    return StaticSparseMatrix.from_dense(
        matrix, threshold=threshold, learnable=learnable, device=device
    )


def sparsity_info(matrix: Union[Tensor, StaticSparseMatrix]) -> dict:
    """Return a dict with shape, nnz, density, and estimated memory savings."""
    if isinstance(matrix, StaticSparseMatrix):
        M, K = matrix.shape
        nnz  = matrix.nnz
    else:
        if matrix.dim() != 2:
            raise ValueError("matrix must be 2-D")
        M, K = matrix.shape
        nnz  = int((matrix != 0).sum().item())

    density   = nnz / (M * K)
    dense_mem = M * K * 4          # float32 bytes
    sparse_mem = nnz * 4 + nnz * 4 + nnz * 4  # values + row_idx + col_idx (int32)
    saving_pct = max(0.0, 1.0 - sparse_mem / dense_mem) * 100

    return dict(
        shape=(M, K),
        nnz=nnz,
        density=density,
        sparsity=1.0 - density,
        dense_memory_mb=dense_mem / 1e6,
        sparse_memory_mb=sparse_mem / 1e6,
        memory_saving_pct=saving_pct,
    )


def random_sparse_matrix(
    M: int,
    K: int,
    density: float,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> StaticSparseMatrix:
    """
    Create a random StaticSparseMatrix with approximately `density` fill.
    Useful for benchmarking.
    """
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    else:
        gen = None

    nnz = max(1, int(M * K * density))
    row_idx = torch.randint(0, M, (nnz,), generator=gen)
    col_idx = torch.randint(0, K, (nnz,), generator=gen)
    values  = torch.randn(nnz, dtype=dtype)

    # Deduplicate
    pairs = row_idx * K + col_idx
    unique_pairs, inverse = torch.unique(pairs, return_inverse=True)
    # Scatter sum for duplicate positions
    uniq_vals = torch.zeros(unique_pairs.shape[0], dtype=dtype)
    uniq_vals.scatter_add_(0, inverse, values)
    row_idx = (unique_pairs // K).long()
    col_idx = (unique_pairs % K).long()

    return StaticSparseMatrix(
        uniq_vals, row_idx, col_idx, (M, K), learnable=False, device=device
    )


def verify_sparse_mm(
    sparse: StaticSparseMatrix,
    dense: Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> bool:
    """
    Verify sparse_mm output matches dense reference on CPU.
    Returns True if outputs are close.
    """
    dense_cpu = dense.cpu().float()
    ref = sparse.to_dense().cpu().float() @ dense_cpu

    sparse_cpu = StaticSparseMatrix(
        sparse.values.detach().cpu().float() if sparse.learnable else sparse.values.cpu().float(),
        sparse.row_indices.cpu(),
        sparse.col_indices.cpu(),
        sparse.shape,
        learnable=False,
    )
    got = sparse_cpu.mm(dense_cpu)

    match = torch.allclose(ref, got, atol=atol, rtol=rtol)
    if not match:
        max_err = (ref - got).abs().max().item()
        print(f"[verify_sparse_mm] MISMATCH — max abs error: {max_err:.6f}")
    return match
