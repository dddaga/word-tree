"""
StaticSparseMatrix — sparse matrix with fixed non-zero pattern.

Pre-computes at construction (once):
  - COO indices sorted by row + embedding_bag offsets  (for sparse @ dense)
  - Transposed version on demand (for dense @ sparse)

Values are stored in row-sorted order internally.
"""

from __future__ import annotations
import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Union

from .ops import sparse_mm, scipy_sparse_mm, DENSITY_THRESHOLD, HAS_SCIPY


class StaticSparseMatrix:
    """
    Sparse matrix with pre-computed static structure, optimised for MPS.

    Usage
    -----
    ssm = StaticSparseMatrix.from_dense(weight)   # build from dense tensor
    out = ssm @ x                                  # sparse(M,K) @ x(K,N)
    out = x @ ssm                                  # x(B,M) @ sparse(M,K) → (B,K)
    """

    def __init__(
        self,
        values: Tensor,         # (nnz,) in ANY order — will be re-ordered internally
        row_indices: Tensor,    # (nnz,)
        col_indices: Tensor,    # (nnz,)
        shape: tuple[int, int],
        learnable: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        strategy: str = "auto",
        reorder: bool = False,
    ):
        self.shape = shape
        self.M, self.K = shape
        self.strategy = strategy

        target = torch.device(device) if device is not None else None

        def _mv(t: Tensor, dtype=None) -> Tensor:
            if dtype:
                t = t.to(dtype=dtype)
            return t.to(device=target).contiguous() if target is not None else t.contiguous()

        # ---- Sort everything by row (required for embedding_bag) ----
        row_raw = _mv(row_indices.long())
        col_raw = _mv(col_indices.long())
        val_raw = _mv(values.float())

        perm = torch.argsort(row_raw, stable=True)   # (nnz,) — original → sorted

        row_s = row_raw[perm]      # row-sorted row indices
        col_s = col_raw[perm]      # row-sorted col indices
        val_s = val_raw[perm]      # row-sorted values

        # Store row-sorted COO permanently
        self._row  = row_s                     # (nnz,)
        self._col  = col_s                     # (nnz,)
        self._perm = perm                      # for inverse-sort if needed

        if learnable:
            self._values: Tensor = nn.Parameter(val_s)
        else:
            self._values = val_s

        self.learnable = learnable

        # ---- Embedding-bag offsets (M,) ----
        row_counts  = torch.bincount(row_s, minlength=self.M)
        offsets     = torch.zeros(self.M, dtype=torch.long, device=row_s.device)
        if self.M > 1:
            offsets[1:] = row_counts.cumsum(0)[:-1]
        self._offsets: Tensor = offsets

        # Density
        self._density: float = int(perm.shape[0]) / max(self.M * self.K, 1)

        # For high-density non-learnable matrices, pre-cache the dense form so
        # mm() can use MPS-optimised dense matmul (always faster above ~1.5%).
        self._dense_cache: Optional[Tensor] = None
        if not learnable and self._density > DENSITY_THRESHOLD:
            out = torch.zeros(self.M, self.K, dtype=val_s.dtype,
                              device=val_s.device)
            out[row_s, col_s] = val_s
            self._dense_cache = out

        # Transposed version (built lazily)
        self._transposed: Optional["StaticSparseMatrix"] = None

        # ---- scipy CSR path (strategy="scipy") ----
        # Pre-build scipy CSR matrix on CPU for Apple Accelerate sparse BLAS.
        # Optionally apply Reverse Cuthill-McKee reordering for better cache locality.
        self._scipy_csr = None
        self._rcm_perm: Optional[Tensor] = None
        if HAS_SCIPY and strategy in ("auto", "scipy"):
            import numpy as np
            from scipy.sparse.csgraph import reverse_cuthill_mckee
            row_cpu = row_s.cpu().numpy().astype(np.int32)
            col_cpu = col_s.cpu().numpy().astype(np.int32)
            val_cpu = val_s.detach().cpu().numpy().astype(np.float32)
            if reorder:
                import scipy.sparse as sp_local
                # Build CSR temporarily to compute RCM permutation
                _csr_tmp = sp_local.csr_matrix(
                    (val_cpu, (row_cpu, col_cpu)), shape=(self.M, self.K)
                )
                perm_np = reverse_cuthill_mckee(_csr_tmp)
                self._rcm_perm = torch.from_numpy(perm_np.copy())
                inv_perm = np.argsort(perm_np)
                # Re-map indices through RCM permutation
                row_rcm = inv_perm[row_cpu]
                col_rcm = inv_perm[col_cpu]
                import scipy.sparse as sp2
                self._scipy_csr = sp2.csr_matrix(
                    (val_cpu, (row_rcm, col_rcm)), shape=(self.M, self.K)
                )
            else:
                import scipy.sparse as sp3
                self._scipy_csr = sp3.csr_matrix(
                    (val_cpu, (row_cpu, col_cpu)), shape=(self.M, self.K)
                )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dense(
        cls,
        dense: Tensor,
        threshold: float = 0.0,
        learnable: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        strategy: str = "auto",
        reorder: bool = False,
    ) -> "StaticSparseMatrix":
        if dense.dim() != 2:
            raise ValueError(f"dense must be 2-D, got {tuple(dense.shape)}")
        mask     = dense.abs() > threshold
        row_idx, col_idx = mask.nonzero(as_tuple=True)
        values   = dense[row_idx, col_idx].detach().clone()
        return cls(values, row_idx, col_idx, tuple(dense.shape),  # type: ignore
                   learnable=learnable, device=device or dense.device,
                   strategy=strategy, reorder=reorder)

    @classmethod
    def from_sparse_tensor(
        cls,
        sparse: Tensor,
        learnable: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        strategy: str = "auto",
        reorder: bool = False,
    ) -> "StaticSparseMatrix":
        if sparse.layout == torch.sparse_csr:
            sparse = sparse.to_sparse_coo()
        sparse  = sparse.coalesce().cpu()
        indices = sparse.indices()
        values  = sparse.values()
        shape   = tuple(sparse.shape[:2])  # type: ignore
        return cls(values, indices[0], indices[1], shape,
                   learnable=learnable, device=device,
                   strategy=strategy, reorder=reorder)

    @classmethod
    def from_adjacency(
        cls,
        edge_index: Tensor,
        num_nodes_src: int,
        num_nodes_dst: int,
        edge_weights: Optional[Tensor] = None,
        learnable: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        strategy: str = "auto",
        reorder: bool = False,
    ) -> "StaticSparseMatrix":
        """Build from graph edge_index (PyG convention: shape (2, E))."""
        if edge_index.shape[0] == 2:
            row_idx, col_idx = edge_index[0], edge_index[1]
        else:
            row_idx, col_idx = edge_index[:, 0], edge_index[:, 1]
        if edge_weights is None:
            edge_weights = torch.ones(row_idx.shape[0], dtype=torch.float32)
        return cls(edge_weights, row_idx, col_idx,
                   (num_nodes_src, num_nodes_dst),
                   learnable=learnable, device=device,
                   strategy=strategy, reorder=reorder)

    # ------------------------------------------------------------------
    # Core matmul
    # ------------------------------------------------------------------

    def mm(self, dense: Tensor) -> Tensor:
        """self(M,K) @ dense(K,...) → (M,...)"""
        # scipy CPU path: stays on CPU, no MPS dispatch overhead
        # 37x faster than dense MPS at 0.1% density on Apple Silicon
        if self.strategy == "scipy" and self._scipy_csr is not None:
            vals = self._get_values().cpu()
            return scipy_sparse_mm(
                vals, self._row.cpu(), self._col.cpu(),
                dense.cpu().float(), self.M, self.K,
                self._scipy_csr.data, self._scipy_csr.indices,
                self._scipy_csr.indptr,
            )

        self._move_to(dense.device)
        # High-density non-learnable: use pre-cached dense weight for fast MPS GEMM
        if self._dense_cache is not None:
            w = self._dense_cache.to(device=dense.device, dtype=dense.dtype)
            return w @ dense
        vals = self._get_values()
        return sparse_mm(
            vals, self._row, self._col, dense, self.M,
            sorted_col=self._col,
            offsets=self._offsets,
        )

    def rmm(self, dense: Tensor) -> Tensor:
        """dense(B,M) @ self(M,K) → (B,K)"""
        # Equivalent to (self.T(K,M) @ dense.T(M,B)).T
        return self.t().mm(dense.t()).t()

    def t(self) -> "StaticSparseMatrix":
        """Return (cached) transposed view."""
        if self._transposed is None:
            vals = self._get_values().detach() if not self.learnable else self._get_values().data
            # Rebuild with swapped row/col — __init__ will re-sort by new row
            self._transposed = StaticSparseMatrix(
                vals, self._col, self._row,
                (self.K, self.M),
                learnable=False,          # transposed always non-learnable
                device=self._row.device,
            )
        return self._transposed

    # ------------------------------------------------------------------
    # Operator overloads
    # ------------------------------------------------------------------

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.mm(other)

    def __rmatmul__(self, other: Tensor) -> Tensor:
        return self.rmm(other)

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device: Union[str, torch.device]) -> "StaticSparseMatrix":
        device = torch.device(device)
        self._row     = self._row.to(device)
        self._col     = self._col.to(device)
        self._perm    = self._perm.to(device)
        self._offsets = self._offsets.to(device)
        if self.learnable:
            self._values = nn.Parameter(self._values.data.to(device))  # type: ignore
        else:
            self._values = self._values.to(device)
        if self._dense_cache is not None:
            self._dense_cache = self._dense_cache.to(device)
        if self._transposed is not None:
            self._transposed.to(device)
        return self

    def _move_to(self, device: torch.device) -> None:
        if self._row.device != device:
            self.to(device)

    def _get_values(self) -> Tensor:
        """Return values in row-sorted order (matches self._row / self._col)."""
        return self._values  # always stored in row-sorted order

    # ------------------------------------------------------------------
    # Properties & utilities
    # ------------------------------------------------------------------

    @property
    def nnz(self) -> int:
        return int(self._row.shape[0])

    @property
    def density(self) -> float:
        return self._density

    @property
    def values(self) -> Tensor:
        return self._values

    @property
    def row_indices(self) -> Tensor:
        return self._row

    @property
    def col_indices(self) -> Tensor:
        return self._col

    def to_dense(self, device: Optional[Union[str, torch.device]] = None) -> Tensor:
        target = torch.device(device) if device else self._row.device
        dtype  = (self._values.data if self.learnable else self._values).dtype
        out    = torch.zeros(self.M, self.K, device=target, dtype=dtype)
        v      = (self._values.data if self.learnable else self._values).to(target)
        out[self._row.to(target), self._col.to(target)] = v
        return out

    def __repr__(self) -> str:
        return (
            f"StaticSparseMatrix(shape={self.shape}, nnz={self.nnz}, "
            f"density={self.density:.4%}, learnable={self.learnable})"
        )
