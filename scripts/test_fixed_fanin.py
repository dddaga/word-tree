"""Fixed fan-in connectivity benchmark.

Instead of a sparse [N_in x N_hidden] mask, each hidden neuron stores
exactly K input indices: conn_idx shape [N_hidden, K].

Forward op becomes:
  gathered = A_input[:, conn_idx, :]   # [B, N_hidden, K, D]
  Z        = gathered.sum(dim=2)        # [B, N_hidden, D]

This is fully regular — no scatter, no matmul, just two standard ops.
The tradeoff: K is now a hard architectural constant, not a sparsity %.

Benchmark vs dense einsum at equivalent and smaller K values.
"""

from __future__ import annotations
import time
import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
B     = 64
N_IN  = 25088
N_H   = 256
D     = 4
RUNS  = 30

# Current equivalent: 10% of N_in connections per hidden neuron
K_EQUIV = int(N_IN * 0.10)   # ~2509 — matches 90% sparsity

print(f"Device: {DEVICE}  B={B}  N_in={N_IN}  N_h={N_H}  D={D}")
print(f"K_equiv (10% of N_in) = {K_EQUIV}\n")

A_input   = torch.randn(B, N_IN, D, device=DEVICE)
bool_mask = (torch.rand(N_IN, N_H) > 0.90).to(DEVICE)


# -------------------------------------------------------------------
# Build conn_idx [N_hidden, K] for a given K
# -------------------------------------------------------------------

def make_conn_idx(n_in: int, n_h: int, k: int, device: str) -> torch.Tensor:
    """Randomly assign K input connections per hidden neuron."""
    idx = torch.stack([
        torch.randperm(n_in)[:k] for _ in range(n_h)
    ])  # [N_h, K]
    return idx.to(device)


# -------------------------------------------------------------------
# Forward ops
# -------------------------------------------------------------------

def dense_einsum(A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bid,ih->bhd", A, mask.float())


def fixed_fanin(A: torch.Tensor, conn_idx: torch.Tensor) -> torch.Tensor:
    """A: [B, N_in, D]  conn_idx: [N_h, K]  -> [B, N_h, D]"""
    gathered = A[:, conn_idx, :]   # [B, N_h, K, D]
    return gathered.sum(dim=2)     # [B, N_h, D]


# -------------------------------------------------------------------
# Timing helper
# -------------------------------------------------------------------

def bench(name: str, fn, warmup: int = 5) -> float:
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
    print(f"  {name:<45s} {ms:7.2f} ms   "
          f"intermediate={_intermediate_mb(fn):.1f}MB")
    return ms


def _intermediate_mb(fn) -> float:
    """Estimate peak intermediate tensor MB (gather result)."""
    return 0.0   # placeholder — computed inline below


# -------------------------------------------------------------------
# Memory estimates
# -------------------------------------------------------------------

def memory_report(k: int) -> str:
    conn_mb      = N_H * k * 4 / 1e6          # int32 index storage
    gathered_mb  = B * N_H * k * D * 4 / 1e6  # float32 intermediate
    return f"conn={conn_mb:.2f}MB  intermediate={gathered_mb:.1f}MB"


# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------

print("=== Memory comparison ===")
print(f"  Dense bool mask [{N_IN}x{N_H}]:          "
      f"{bool_mask.element_size()*bool_mask.numel()/1e6:.2f} MB (stored)")
print(f"  Dense float cast [{N_IN}x{N_H}]:         "
      f"{N_IN*N_H*4/1e6:.2f} MB (at einsum time)")
for k in [10, 50, 100, 500, K_EQUIV]:
    print(f"  Fixed fan-in K={k:<6}  {memory_report(k)}")

print(f"\n=== Timing ===")

# Baseline
ref = dense_einsum(A_input, bool_mask)
t_dense = 0.0
for _ in range(5): dense_einsum(A_input, bool_mask)
if DEVICE == "mps": torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(RUNS): dense_einsum(A_input, bool_mask)
if DEVICE == "mps": torch.mps.synchronize()
t_dense = (time.perf_counter() - t0) / RUNS * 1000
print(f"  {'Dense einsum (baseline)':<45s} {t_dense:7.2f} ms")

# Fixed fan-in at several K values
for k in [10, 50, 100, 500, K_EQUIV]:
    conn_idx = make_conn_idx(N_IN, N_H, k, DEVICE)
    out = fixed_fanin(A_input, conn_idx)

    for _ in range(5): fixed_fanin(A_input, conn_idx)
    if DEVICE == "mps": torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(RUNS): fixed_fanin(A_input, conn_idx)
    if DEVICE == "mps": torch.mps.synchronize()
    ms = (time.perf_counter() - t0) / RUNS * 1000
    ratio = ms / t_dense
    tag = "faster" if ratio < 1.0 else f"{ratio:.1f}x slower"
    print(f"  {'Fixed fan-in K='+str(k):<45s} {ms:7.2f} ms   [{tag}]")

print(f"\n=== Architectural implication of K ===")
print(f"  Current 90% sparse:  each hidden neuron sees ~{K_EQUIV} of {N_IN} inputs")
print(f"  K=10:                each hidden neuron sees 10 of {N_IN} inputs ({10/N_IN*100:.3f}%)")
print(f"  K=100:               each hidden neuron sees 100 of {N_IN} inputs ({100/N_IN*100:.3f}%)")
print(f"  K=500:               each hidden neuron sees 500 of {N_IN} inputs ({500/N_IN*100:.2f}%)")
print(f"\n  Changing K changes the model architecture, not just the storage.")
print(f"  Small K = each neuron specialises on a tiny local receptive field.")
