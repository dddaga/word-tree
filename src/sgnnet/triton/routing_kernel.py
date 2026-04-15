"""Triton fused kernel for SGNNET DeltaProj routing step.

Fuses the K_iter hot path from SGNNET_DeltaProj.forward:
    Z_fwd   = relu(Z - theta)              [B, N, D]
    Z_nb    = Z_fwd[:, conn_hh, :]        [B, N, K_hh, D]  -- eliminated
    proj    = (Z_nb * dw_norm).sum(-1)    [B, N, K_hh, 1]
    Z_nb    = Z_nb * |proj|               sign-invariant gate
    Z_struct = Z_nb.sum(dim=2)            [B, N, D]
    Z_rem   = Z_fwd - Z
    Z_refl  = alpha * Z_refl + Z_rem
    Z_new   = normalize(clamp(Z_struct + Z_refl, -10, 10), -1)

Grid: 2D — (B, ceil(N / BLOCK_N)).
  pid_b = program_id(0) → one batch item per row
  pid_n = program_id(1) → BLOCK_N neurons per column

This avoids any Python range(B) loop inside the kernel (SM_120 LLVM issue),
avoids constexpr batch indexing (Triton 3.x constraint), and has no `continue`.
D=16 and K_HH=2 are constexpr — fully unrolled at compile time.

Memory layout:
  Z, Z_reflected: [B, N, D] fp32, stride_b=N*D, stride_n=D, stride_d=1
  conn_hh:        [N, K_hh] int32, stride_n=K_hh, stride_k=1
  dw_norm:        [N, K_hh, D] fp32, stride_n=K_hh*D, stride_k=D, stride_d=1
  theta:          [N] fp32
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

# ---------------------------------------------------------------------------
# Triton kernel — 2D grid (B, ceil(N/BLOCK_N))
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 16},  num_warps=2),
            triton.Config({"BLOCK_N": 32},  num_warps=2),
            triton.Config({"BLOCK_N": 64},  num_warps=4),
            triton.Config({"BLOCK_N": 128}, num_warps=4),
            triton.Config({"BLOCK_N": 256}, num_warps=8),
            triton.Config({"BLOCK_N": 512}, num_warps=8),
        ],
        key=["B", "N", "D", "K_HH"],
    )
    @triton.jit
    def routing_step_proj_kernel(
        Z_ptr,             # [B, N, D] fp32
        theta_ptr,         # [N]       fp32
        conn_ptr,          # [N, K_hh] int32
        dw_ptr,            # [N, K_hh, D] fp32
        Z_refl_ptr,        # [B, N, D] fp32
        Z_out_ptr,         # [B, N, D] fp32
        Z_refl_out_ptr,    # [B, N, D] fp32
        alpha_reflect,     # float
        N,                 # int (runtime)
        D:     tl.constexpr,
        K_HH:  tl.constexpr,
        stride_zb: tl.constexpr,   # N * D
        stride_zn: tl.constexpr,   # D
        BLOCK_N: tl.constexpr,
    ):
        """2D grid: program(b, tile_n) handles one batch item × BLOCK_N neurons.

        No Python range loop. No constexpr tensor indexing. Fully static inner loops.
        """
        b_idx  = tl.program_id(0)   # scalar batch index
        pid_n  = tl.program_id(1)   # neuron tile index

        n_start = pid_n * BLOCK_N
        n_off   = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]
        n_mask  = n_off < N                          # [BLOCK_N]
        d_idx   = tl.arange(0, D)                   # [D]

        # Per-neuron threshold [BLOCK_N]
        theta = tl.load(theta_ptr + n_off, mask=n_mask, other=0.0)

        # Base addresses for Z rows in this batch item
        z_row   = b_idx * stride_zb + n_off * stride_zn   # [BLOCK_N]
        z_addr  = z_row[:, None] + d_idx[None, :]          # [BLOCK_N, D]
        z_mask  = n_mask[:, None]                           # [BLOCK_N, 1] → broadcast

        # Load Z[b, n_tile, :] and Z_reflected[b, n_tile, :]
        Z_cur  = tl.load(Z_ptr      + z_addr, mask=z_mask, other=0.0,
                         eviction_policy="evict_last")
        Z_refl = tl.load(Z_refl_ptr + z_addr, mask=z_mask, other=0.0)

        # Step 1: Z_fwd = relu(Z - theta)
        Z_fwd   = tl.maximum(Z_cur - theta[:, None], 0.0)

        # Steps 2-4: gather senders, apply dw projection gate, sum over K_HH
        Z_struct = tl.zeros([BLOCK_N, D], dtype=tl.float32)

        for k in tl.static_range(K_HH):
            # Sender index: conn[n_off, k]
            conn_addr  = n_off * K_HH + k                  # [BLOCK_N]
            sender_idx = tl.load(conn_ptr + conn_addr,
                                 mask=n_mask, other=0)      # [BLOCK_N] int32

            # dw_norm[n_off, k, :] → [BLOCK_N, D]
            dw_row  = n_off * (K_HH * D) + k * D           # [BLOCK_N]
            dw_addr = dw_row[:, None] + d_idx[None, :]      # [BLOCK_N, D]
            dw_k    = tl.load(dw_ptr + dw_addr, mask=z_mask, other=0.0,
                              eviction_policy="evict_last")  # [BLOCK_N, D]

            # Gather Z at sender positions (sender_idx values always in [0, N))
            send_row  = b_idx * stride_zb + sender_idx * stride_zn  # [BLOCK_N]
            send_addr = send_row[:, None] + d_idx[None, :]           # [BLOCK_N, D]
            Z_raw_s   = tl.load(Z_ptr + send_addr,
                                eviction_policy="evict_last")        # [BLOCK_N, D]

            # Apply relu-threshold with sender's own theta (no mask: indices always valid)
            theta_s = tl.load(theta_ptr + sender_idx)                # [BLOCK_N]
            Z_nb_k  = tl.maximum(Z_raw_s - theta_s[:, None], 0.0)  # [BLOCK_N, D]

            # proj = sum_d(Z_nb_k * dw_k) → [BLOCK_N], gate = |proj|
            proj_k = tl.sum(Z_nb_k * dw_k, axis=1)                  # [BLOCK_N]
            gate_k = tl.abs(proj_k)                                  # [BLOCK_N]

            Z_struct = Z_struct + Z_nb_k * gate_k[:, None]

        # Steps 5-8: reflection + combine + clamp + l2-normalize
        Z_rem      = Z_fwd - Z_cur
        Z_refl_new = alpha_reflect * Z_refl + Z_rem
        Z_combined = tl.clamp(Z_struct + Z_refl_new, -10.0, 10.0)

        norm_sq  = tl.sum(Z_combined * Z_combined, axis=1)   # [BLOCK_N]
        inv_norm = tl.rsqrt(norm_sq + 1e-12)
        Z_new    = Z_combined * inv_norm[:, None]

        # Store outputs
        tl.store(Z_out_ptr      + z_addr, Z_new,      mask=z_mask,
                 eviction_policy="evict_first")
        tl.store(Z_refl_out_ptr + z_addr, Z_refl_new, mask=z_mask,
                 eviction_policy="evict_first")


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def routing_step_proj(
    Z:            "torch.Tensor",
    theta:        "torch.Tensor",
    conn_hh:      "torch.Tensor",
    dw_norm:      "torch.Tensor",
    Z_reflected:  "torch.Tensor",
    alpha_reflect: float,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Fused SGNNET DeltaProj routing step (Triton kernel).

    Args:
        Z            : [B, N, D] fp32, CUDA
        theta        : [N] fp32, CUDA (abs'd threshold)
        conn_hh      : [N, K_hh] int32, CUDA
        dw_norm      : [N, K_hh, D] fp32, CUDA
        Z_reflected  : [B, N, D] fp32, CUDA
        alpha_reflect: float
    Returns (Z_next, Z_reflected_next).
    """
    if not TRITON_AVAILABLE:
        return _routing_step_proj_pytorch(Z, theta, conn_hh, dw_norm, Z_reflected, alpha_reflect)

    B, N, D = Z.shape
    K_HH    = conn_hh.shape[1]

    Z           = Z.contiguous()
    theta       = theta.contiguous()
    conn_hh     = conn_hh.to(torch.int32).contiguous()
    dw_norm     = dw_norm.contiguous()
    Z_reflected = Z_reflected.contiguous()

    Z_out      = torch.empty_like(Z)
    Z_refl_out = torch.empty_like(Z)

    stride_zb = N * D
    stride_zn = D

    # 2D grid: (B, ceil(N / BLOCK_N))
    grid = lambda meta: (B, triton.cdiv(N, meta["BLOCK_N"]))

    routing_step_proj_kernel[grid](
        Z, theta, conn_hh, dw_norm, Z_reflected,
        Z_out, Z_refl_out,
        float(alpha_reflect),
        N,
        D=D, K_HH=K_HH,
        stride_zb=stride_zb, stride_zn=stride_zn,
    )

    return Z_out, Z_refl_out


def _routing_step_proj_pytorch(
    Z:            "torch.Tensor",
    theta:        "torch.Tensor",
    conn_hh:      "torch.Tensor",
    dw_norm:      "torch.Tensor",
    Z_reflected:  "torch.Tensor",
    alpha_reflect: float,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """PyTorch reference — fallback and test oracle.

    dw_norm: [N, K_hh, D] (no batch dim).
    """
    theta_pos = theta.unsqueeze(0).unsqueeze(-1)   # [1, N, 1]
    dw        = dw_norm.unsqueeze(0)               # [1, N, K_hh, D]
    conn64    = conn_hh.to(torch.int64)

    Z_fwd    = F.relu(Z - theta_pos)
    Z_nb     = Z_fwd[:, conn64, :]
    proj     = (Z_nb * dw).sum(-1, keepdim=True)
    Z_nb     = Z_nb * proj.abs()
    Z_struct = Z_nb.sum(dim=2)

    Z_remainder = Z_fwd - Z
    Z_refl_next = alpha_reflect * Z_reflected + Z_remainder

    Z_new = F.normalize(
        (Z_struct + Z_refl_next).clamp(-10.0, 10.0),
        dim=-1,
    )
    return Z_new, Z_refl_next
