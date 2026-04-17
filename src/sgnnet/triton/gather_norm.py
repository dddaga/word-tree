"""Triton fused gather+sum+L2-normalize kernel for SmallWorld base routing.

Step530 validated: 1.17-1.18x over torch.compile max-autotune (K_iter=5,
N=2048, D=16, B=32/128, RTX 5060 Ti).

Replaces the two-line hot path in model_smallworld._route():
    Z = Z[:, conn_hh, :].sum(dim=2)
    Z = F.normalize(Z, dim=-1)

with a single kernel that eliminates the [B, N, K_hh, D] intermediate tensor
and fuses normalize into registers.

Only used when: CUDA device + norm_mode="l2" + D is power-of-2 <= 128.
Falls back to eager otherwise.
"""
from __future__ import annotations

TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass

import torch
import torch.nn.functional as F


if TRITON_AVAILABLE:

    @triton.jit
    def _fused_gather_sum_norm_kernel(
        Z_in_ptr,
        Z_out_ptr,
        conn_ptr,
        B:        tl.constexpr,
        N:        tl.constexpr,
        K_HH:     tl.constexpr,
        D:        tl.constexpr,
        BLOCK_B:  tl.constexpr,
        EPS:      tl.constexpr,
    ):
        """Each program handles one neuron × BLOCK_B batch items.

        Grid: (N, ceil(B / BLOCK_B)).
        Gathers K_HH neighbours, sums, L2-normalises — all in registers.
        No [B, N, K_hh, D] intermediate.
        """
        pid_n  = tl.program_id(0)
        pid_b  = tl.program_id(1)
        b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        d_offs = tl.arange(0, D)
        b_mask = b_offs < B

        acc = tl.zeros([BLOCK_B, D], dtype=tl.float32)
        for k in tl.static_range(K_HH):
            nb   = tl.load(conn_ptr + pid_n * K_HH + k)
            ptrs = Z_in_ptr + b_offs[:, None] * (N * D) + nb * D + d_offs[None, :]
            acc += tl.load(ptrs, mask=b_mask[:, None], other=0.0)

        norm_sq  = tl.sum(acc * acc, axis=1, keep_dims=True)
        inv_norm = tl.rsqrt(norm_sq + EPS)
        acc      = acc * inv_norm

        out_ptrs = Z_out_ptr + b_offs[:, None] * (N * D) + pid_n * D + d_offs[None, :]
        tl.store(out_ptrs, acc, mask=b_mask[:, None])


def gather_sum_norm_step(
    Z: "torch.Tensor",
    conn: "torch.Tensor",
    block_b: int = 32,
    eps: float = 1e-12,
) -> "torch.Tensor":
    """Single routing step: gather K_hh neighbours, sum, L2-normalize.

    Args:
        Z    : [B, N, D] fp32 CUDA, contiguous
        conn : [N, K_hh] int64 CUDA, contiguous
        block_b: batch tile size (32 optimal on RTX 5060 Ti)
        eps  : L2-norm epsilon

    Returns [B, N, D] normalised.
    """
    if not TRITON_AVAILABLE or Z.device.type != "cuda":
        Z = Z[:, conn, :].sum(dim=2)
        return F.normalize(Z, dim=-1)

    B_, N_, D_ = Z.shape
    Z_out = torch.empty_like(Z)
    grid = (N_, (B_ + block_b - 1) // block_b)
    _fused_gather_sum_norm_kernel[grid](
        Z, Z_out, conn,
        B=B_, N=N_, K_HH=conn.shape[1], D=D_,
        BLOCK_B=block_b, EPS=eps,
    )
    return Z_out


def route_k_iter(
    Z: "torch.Tensor",
    conn: "torch.Tensor",
    k_iter: int,
    block_b: int = 32,
    eps: float = 1e-12,
) -> "torch.Tensor":
    """K_iter rounds of fused gather+sum+L2-normalize.

    Args:
        Z    : [B, N, D] fp32 CUDA
        conn : [N, K_hh] int64 CUDA
        k_iter: number of routing rounds
    """
    if not TRITON_AVAILABLE or Z.device.type != "cuda":
        for _ in range(k_iter):
            Z = Z[:, conn, :].sum(dim=2)
            Z = F.normalize(Z, dim=-1)
        return Z

    conn_c = conn.contiguous()
    for _ in range(k_iter):
        Z = gather_sum_norm_step(Z, conn_c, block_b=block_b, eps=eps)
    return Z


def is_supported(D: int) -> bool:
    """True when D is a power-of-2 in [1, 128] — required for tl.arange/tl.zeros."""
    return TRITON_AVAILABLE and (D & (D - 1) == 0) and 1 <= D <= 128
