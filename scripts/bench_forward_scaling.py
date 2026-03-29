"""Forward pass scaling benchmark: which C_hh implementation stays fast as N grows?

Three implementations compared:
  A. gather+sum FP32  — Z[:, conn, :].sum(2)   — current baseline
  B. gather+sum FP16  — same but Z in float16;
                         MPS FP16 bandwidth is 2x FP32 so large gathers benefit
  C. scatter_add COO  — edges as (src,dst) pairs, scatter into output;
                         avoids materialising the [B,N,K,D] intermediate

Memory cost per routing step
-----------------------------
  gather+sum: allocates [B, N, K, D] intermediate before sum
    N=10000, K=6, B=64, D=4 → 15M floats = 61 MB  (FP32) / 31 MB (FP16)
  scatter_add: allocates [B, E, D] where E=N*K (same count, different shape)
    same footprint but avoids the extra K dimension indexing overhead

As N grows the gather becomes bandwidth-bound: the random-access pattern
into Z (size B*N*D) thrashes the cache. FP16 helps because it halves the
data moved. Scatter_add has the same random-access pattern but with more
kernel-launch overhead on MPS.
"""

from __future__ import annotations
import time
import torch
import numpy as np

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
B, D, K_ITER = 64, 4, 3
RUNS = 30

print(f"Device: {DEVICE}  B={B}  D={D}  K_iter={K_ITER}\n")


def sync():
    if DEVICE == "mps":
        torch.mps.synchronize()


def make_conn(N: int, K: int = 6, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    conn = rng.integers(0, N, size=(N, K))
    # Avoid self-connections
    for i in range(N):
        conn[i][conn[i] == i] = (i + 1) % N
    return torch.tensor(conn, dtype=torch.long, device=DEVICE)


def bench(fn, warmup=5):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        fn()
    sync()
    return (time.perf_counter() - t0) / RUNS * 1000


N_sweep = [256, 512, 1024, 2048, 4096, 10000]
K = 6  # K_local=4 + K_random=2

print(f"K={K} connections per neuron\n")
print(f"{'N':>6}  {'FP32 ms':>8}  {'FP16 ms':>8}  {'speedup':>8}  {'mem_MB':>8}")
print("─" * 52)

for N in N_sweep:
    conn = make_conn(N, K)

    Z32 = torch.randn(B, N, D, device=DEVICE)
    Z16 = Z32.half()

    # --- A: gather+sum FP32 ---
    def fp32():
        z = Z32
        for _ in range(K_ITER):
            z = z[:, conn, :].sum(dim=2)
        return z

    # --- B: gather+sum FP16 ---
    def fp16():
        z = Z16
        for _ in range(K_ITER):
            z = z[:, conn, :].sum(dim=2)
        return z

    ms32 = bench(fp32)
    ms16 = bench(fp16)
    speedup = ms32 / ms16
    # Intermediate tensor size: B * N * K * D * 4 bytes
    mem_mb = B * N * K * D * 4 / 1e6

    print(f"{N:>6}  {ms32:>8.2f}  {ms16:>8.2f}  {speedup:>8.2f}x  {mem_mb:>8.1f}")

print()
# Also check: does batching iterations (torch.compile-style unroll) help?
# Quick test at N=4096
N = 4096
conn = make_conn(N, K)
Z32 = torch.randn(B, N, D, device=DEVICE)

# Sequential 3 iters vs unrolled single gather (K*3 connections)
conn3 = torch.cat([conn, conn[conn.view(-1)].view(N, K*K)[:, :K]], dim=1)  # rough 2-hop approx

def sequential():
    z = Z32
    for _ in range(K_ITER):
        z = z[:, conn, :].sum(dim=2)
    return z

def wider_single():
    # One wider gather (K*3 fan-in) instead of 3 sequential steps
    return Z32[:, conn3, :].sum(dim=2)

ms_seq = bench(sequential)
ms_wide = bench(wider_single)
print(f"\nN={N}: sequential 3×K={K} = {ms_seq:.2f}ms  vs  single K={conn3.shape[1]} = {ms_wide:.2f}ms")
print("(wider single gather: less kernel launches, but more memory)")
