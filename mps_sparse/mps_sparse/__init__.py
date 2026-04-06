"""
mps_sparse — efficient sparse matrix multiplication for PyTorch on Apple MPS.

Public API
----------
StaticSparseMatrix   core sparse matrix class with pre-computed indices
SparseLinear         nn.Module drop-in for nn.Linear with sparse weights
SparseMatMul         nn.Module for a fixed sparse projection
sparse_mm            functional: sparse @ dense
dense_sparse_mm      functional: dense @ sparse
to_static_sparse     convert a dense tensor to StaticSparseMatrix
sparsity_info        inspect density / memory stats of a matrix
random_sparse_matrix create random sparse matrix (benchmarking)
verify_sparse_mm     numerical correctness check
"""

from .static_sparse import StaticSparseMatrix
from .modules import SparseLinear, SparseMatMul
from .ops import sparse_mm, dense_sparse_mm
from .utils import to_static_sparse, sparsity_info, random_sparse_matrix, verify_sparse_mm

__version__ = "0.1.0"

__all__ = [
    "StaticSparseMatrix",
    "SparseLinear",
    "SparseMatMul",
    "sparse_mm",
    "dense_sparse_mm",
    "to_static_sparse",
    "sparsity_info",
    "random_sparse_matrix",
    "verify_sparse_mm",
]
