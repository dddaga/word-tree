"""Benchmark sparse vs dense matmul strategies for C_input on MPS.

Three approaches for the core operation:
  einsum("bid,ih->bhd", A_input, C_mask)  where C_mask is 90% sparse

1. Dense:   bool->float cast, standard einsum (current approach)
2. Sparse:  torch.sparse_csr_tensor + mm (PyTorch native sparse)
3. Indexed: precompute COO nonzero indices, gather + scatter_add
            (only touches connected pairs — pure index arithmetic)

The indexed approach is the conceptual answer to "bypass the matmul entirely":
  - Store only the (src, dst) pairs that have a connection
  - For each connection: gather A_input[b, src, :] and add to Z[b, dst, :]
  - No multiplication by zero, no memory allocated for zeros
"""

from __future__ import annotations

import time

import torch


# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
B = 64        # batch size
N_IN = 25088  # input neurons
N_H = 256     # hidden neurons
D = 4         # geometric dims
SPARSITY = 0.90
RUNS = 20     # timing repetitions

print(f"Device: {DEVICE}  B={B}  N_in={N_IN}  N_h={N_H}  D={D}  sparsity={SPARSITY}")
print(f"Dense entries: {N_IN*N_H:,}  |  Non-zero: {int(N_IN*N_H*(1-SPARSITY)):,}\n")

# Build bool mask and test input
bool_mask = (torch.rand(N_IN, N_H) > SPARSITY).to(DEVICE)
A_input = torch.randn(B, N_IN, D, device=DEVICE)

# Precompute COO indices (done once at model init, not per forward)
src_idx, dst_idx = bool_mask.nonzero(as_tuple=True)  # [nnz], [nnz]
nnz = src_idx.shape[0]
print(f"Actual non-zeros: {nnz:,}  ({nnz/(N_IN*N_H)*100:.1f}%)\n")


# -------------------------------------------------------------------
# Approach 1: Dense (current)
# -------------------------------------------------------------------

def dense_einsum(A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cast bool->float, standard einsum. No sparsity awareness."""
    return torch.einsum("bid,ih->bhd", A, mask.float())


# -------------------------------------------------------------------
# Approach 2: PyTorch sparse CSR
# -------------------------------------------------------------------

def make_sparse_csr(mask: torch.Tensor) -> torch.Tensor | None:
    """Try to create a CSR sparse tensor. Returns None if unsupported."""
    try:
        csr = mask.float().to_sparse_csr()
        # Test that a matmul works on device
        test = torch.mm(
            csr,
            torch.randn(N_H, D, device=DEVICE)
        )
        _ = test.sum()  # force eval
        return csr
    except Exception as e:
        print(f"  Sparse CSR unavailable on {DEVICE}: {e}")
        return None


def sparse_mm(A: torch.Tensor, csr) -> torch.Tensor:
    """Reshape A for sparse mm, then reshape back.

    einsum "bid,ih->bhd" = for each d: Z[:,h,d] = sum_i A[:,i,d] * C[i,h]
    Equivalent to: Z[b,:,d] = (C.T @ A[b,:,d]) for each b,d
    With sparse: mm(C.T, A.reshape(N_in, B*D)).reshape(N_h, B, D)
    """
    # A: [B, N_in, D] -> [N_in, B*D]
    A_2d = A.permute(1, 0, 2).reshape(N_IN, B * D)
    # sparse mm: [N_in, N_h].T @ [N_in, B*D] = [N_h, B*D]
    out_2d = torch.mm(csr.t(), A_2d)
    # [N_h, B*D] -> [B, N_h, D]
    return out_2d.reshape(N_H, B, D).permute(1, 0, 2)


# -------------------------------------------------------------------
# Approach 3: Index-based gather + scatter_add (no matmul at all)
# -------------------------------------------------------------------

def indexed_gather_scatter(
    A: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    n_hidden: int,
) -> torch.Tensor:
    """Only process connected (src->dst) pairs.

    For each connection k: accumulate A[b, src[k], :] into Z[b, dst[k], :]
    Uses scatter_add — pure index arithmetic, zero multiplications.
    """
    # Gather input activations at connected src neurons: [B, nnz, D]
    gathered = A[:, src, :]  # index select along dim 1

    # Scatter-add into hidden neurons: [B, N_h, D]
    Z = torch.zeros(A.shape[0], n_hidden, D, device=A.device, dtype=A.dtype)
    dst_exp = dst.view(1, -1, 1).expand(A.shape[0], -1, D)
    Z.scatter_add_(1, dst_exp, gathered)
    return Z


# -------------------------------------------------------------------
# Correctness check
# -------------------------------------------------------------------

print("=== Correctness check ===")
ref = dense_einsum(A_input, bool_mask)

idx_out = indexed_gather_scatter(A_input, src_idx, dst_idx, N_H)
max_diff_idx = (ref - idx_out).abs().max().item()
print(f"Indexed vs Dense max diff: {max_diff_idx:.2e}  {'OK' if max_diff_idx < 1e-4 else 'MISMATCH'}")

csr = make_sparse_csr(bool_mask)
if csr is not None:
    sp_out = sparse_mm(A_input, csr)
    max_diff_sp = (ref - sp_out).abs().max().item()
    print(f"Sparse  vs Dense max diff: {max_diff_sp:.2e}  {'OK' if max_diff_sp < 1e-4 else 'MISMATCH'}")
else:
    print("Sparse CSR: skipped (not supported on this device)")

# -------------------------------------------------------------------
# Timing
# -------------------------------------------------------------------

def time_fn(name: str, fn, warmup: int = 3):
    for _ in range(warmup):
        fn()
    if DEVICE == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        fn()
    if DEVICE == "mps":
        torch.mps.synchronize()
    ms = (time.perf_counter() - t0) / RUNS * 1000
    print(f"  {name:<30s} {ms:8.2f} ms/iter")


print(f"\n=== Timing ({RUNS} runs each) ===")
time_fn("1. Dense einsum (current)",
        lambda: dense_einsum(A_input, bool_mask))

time_fn("3. Indexed gather+scatter",
        lambda: indexed_gather_scatter(A_input, src_idx, dst_idx, N_H))

if csr is not None:
    time_fn("2. Sparse CSR mm",
            lambda: sparse_mm(A_input, csr))

print("\n=== Memory (bool mask vs float mask) ===")
print(f"  bool mask  [{N_IN}x{N_H}]: {bool_mask.element_size() * bool_mask.numel() / 1e6:.2f} MB")
float_mask = bool_mask.float()
print(f"  float mask [{N_IN}x{N_H}]: {float_mask.element_size() * float_mask.numel() / 1e6:.2f} MB")
print(f"  COO indices (src+dst):    {(src_idx.element_size() * nnz * 2) / 1e6:.2f} MB")
