"""Sparse routing profiling — wall-clock comparison of data structures.

Tests whether alternative sparse representations (hashmap, scatter-add, sparse matmul,
JAX) can beat PyTorch's native gather-sum for SGNNET's routing step.

Current implementation: Z[:, conn_hh, :].sum(dim=2)  — O(N·K_hh·D)
Already sparse! Question is wall-clock, not algorithmic complexity.

Implementations tested:
  1. gather_sum    — current PyTorch (baseline)
  2. scatter_add   — torch.scatter_add_ with edge list
  3. sparse_mm     — torch.sparse COO matrix multiply
  4. dict_routing  — Python dict-based (user hypothesis)
  5. loop_gather   — explicit loop over K_hh (CPU-friendly?)
  6. jax_gather    — JAX vmap gather-sum (if JAX available)

Run: python benchmarks/bench_sparse_routing.py [--device cpu|mps] [--jax]
"""
from __future__ import annotations
import argparse, time, sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# ─── Config ───────────────────────────────────────────────────────────
CONFIGS = [
    {"name": "step199_optimal", "N": 2048, "D": 16, "K_hh": 2, "K_iter": 5, "B": 128},
    {"name": "step195_1pct",    "N": 2048, "D": 16, "K_hh": 2, "K_iter": 6, "B": 128},
    {"name": "step185_floor",   "N": 2048, "D": 16, "K_hh": 4, "K_iter": 8, "B": 128},
    {"name": "N4096_record",    "N": 4096, "D": 16, "K_hh": 2, "K_iter": 5, "B": 64},
    {"name": "N8192_large",     "N": 8192, "D": 16, "K_hh": 2, "K_iter": 5, "B": 32},
]

N_WARMUP = 5
N_ITERS = 50


def build_small_world_conn(N, K_hh, n_groups=None, seed=42):
    """Build small-world connectivity: K_local local + K_random random."""
    rng = np.random.RandomState(seed)
    if n_groups is None:
        n_groups = max(8, N // 8)
    K_random = max(1, K_hh // 4)
    K_local = K_hh - K_random
    group_size = N // n_groups
    conn = np.zeros((N, K_hh), dtype=np.int64)
    for i in range(N):
        g = i // group_size
        g_start = g * group_size
        g_end = min(g_start + group_size, N)
        local = rng.choice(range(g_start, g_end), size=K_local, replace=True)
        rand = rng.randint(0, N, size=K_random)
        conn[i] = np.concatenate([local, rand])
    return torch.from_numpy(conn)


# ─── Implementation 1: gather_sum (current) ──────────────────────────
def route_gather_sum(Z, conn_hh, K_iter):
    for _ in range(K_iter):
        Z = Z[:, conn_hh, :].sum(dim=2)
        Z = F.normalize(Z, dim=-1)
    return Z


# ─── Implementation 2: scatter_add ───────────────────────────────────
def build_edge_list(conn_hh):
    """Convert [N, K_hh] index table to (src, dst) edge list for scatter."""
    N, K_hh = conn_hh.shape
    dst = torch.arange(N, device=conn_hh.device).unsqueeze(1).expand(N, K_hh).reshape(-1)
    src = conn_hh.reshape(-1)
    return src, dst

def route_scatter_add(Z, src, dst, N, K_iter):
    B, _, D = Z.shape
    for _ in range(K_iter):
        Z_src = Z[:, src, :]  # [B, E, D]
        Z_new = torch.zeros(B, N, D, device=Z.device, dtype=Z.dtype)
        dst_exp = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
        Z_new.scatter_add_(1, dst_exp, Z_src)
        Z = F.normalize(Z_new, dim=-1)
    return Z


# ─── Implementation 3: sparse matmul ─────────────────────────────────
def build_sparse_adj(conn_hh, N):
    """Build sparse NxN adjacency from conn_hh."""
    K_hh = conn_hh.shape[1]
    rows = torch.arange(N, device=conn_hh.device).unsqueeze(1).expand(N, K_hh).reshape(-1)
    cols = conn_hh.reshape(-1)
    indices = torch.stack([rows, cols])
    values = torch.ones(N * K_hh, device=conn_hh.device, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

def route_sparse_mm(Z, adj_sparse, K_iter):
    B, N, D = Z.shape
    for _ in range(K_iter):
        # Reshape for sparse matmul: [N, N] @ [N, B*D] → [N, B*D]
        Z_flat = Z.permute(1, 0, 2).reshape(N, B * D)  # [N, B*D]
        Z_flat = torch.sparse.mm(adj_sparse, Z_flat)     # [N, B*D]
        Z = Z_flat.reshape(N, B, D).permute(1, 0, 2)     # [B, N, D]
        Z = F.normalize(Z, dim=-1)
    return Z


# ─── Implementation 4: dict-based routing ─────────────────────────────
def build_routing_dict(conn_hh):
    """Store connectivity as Python dict: {dst: [src1, src2, ...]}."""
    N, K_hh = conn_hh.shape
    routing = {}
    for i in range(N):
        routing[i] = conn_hh[i].tolist()
    return routing

def route_dict(Z, routing_dict, K_iter):
    """Route using dict lookups — tests user's hashmap hypothesis."""
    B, N, D = Z.shape
    for _ in range(K_iter):
        Z_new = torch.zeros_like(Z)
        for dst, srcs in routing_dict.items():
            Z_new[:, dst, :] = Z[:, srcs, :].sum(dim=1)
        Z = F.normalize(Z_new, dim=-1)
    return Z


# ─── Implementation 5: explicit loop over K_hh ───────────────────────
def route_loop(Z, conn_hh, K_iter):
    K_hh = conn_hh.shape[1]
    for _ in range(K_iter):
        Z_sum = torch.zeros_like(Z)
        for k in range(K_hh):
            Z_sum += Z[:, conn_hh[:, k], :]
        Z = F.normalize(Z_sum, dim=-1)
    return Z


# ─── Benchmark runner ─────────────────────────────────────────────────
def time_fn(fn, *args, n_warmup=N_WARMUP, n_iters=N_ITERS, device="cpu"):
    """Time a function, return median ms."""
    for _ in range(n_warmup):
        fn(*args)
    if device == "mps":
        torch.mps.synchronize()

    times = []
    for _ in range(n_iters):
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        if device == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times), np.std(times)


def run_config(cfg, device_str):
    N, D, K_hh, K_iter, B = cfg["N"], cfg["D"], cfg["K_hh"], cfg["K_iter"], cfg["B"]
    device = torch.device(device_str)

    print(f"\n{'─'*60}")
    print(f"Config: {cfg['name']}  N={N} D={D} K_hh={K_hh} K_iter={K_iter} B={B}")
    print(f"FLOPs/sample: {3*N*K_hh*D*K_iter:,} ({3*N*K_hh*D*K_iter/1e6:.2f}M)")
    print(f"Sparsity: {K_hh/N*100:.3f}% connected ({100-K_hh/N*100:.2f}% sparse)")
    print(f"Device: {device_str}")
    print(f"{'─'*60}")

    # Build data
    conn_hh = build_small_world_conn(N, K_hh).to(device)
    Z = F.normalize(torch.randn(B, N, D, device=device), dim=-1)

    results = {}

    # 1. gather_sum (baseline)
    med, std = time_fn(route_gather_sum, Z, conn_hh, K_iter, device=device_str)
    results["gather_sum"] = med
    print(f"  gather_sum (current):  {med:7.2f} ms  (±{std:.2f})")

    # 2. scatter_add
    src, dst = build_edge_list(conn_hh)
    med, std = time_fn(route_scatter_add, Z, src, dst, N, K_iter, device=device_str)
    results["scatter_add"] = med
    speedup = results["gather_sum"] / med
    print(f"  scatter_add:           {med:7.2f} ms  (±{std:.2f})  {speedup:.2f}x")

    # 3. sparse_mm
    adj = build_sparse_adj(conn_hh, N).to(device)
    try:
        med, std = time_fn(route_sparse_mm, Z, adj, K_iter, device=device_str)
        results["sparse_mm"] = med
        speedup = results["gather_sum"] / med
        print(f"  sparse_mm:             {med:7.2f} ms  (±{std:.2f})  {speedup:.2f}x")
    except Exception as e:
        print(f"  sparse_mm:             FAILED ({e})")

    # 4. dict routing (only small N — too slow otherwise)
    if N <= 2048:
        routing_dict = build_routing_dict(conn_hh.cpu())
        Z_cpu = Z.cpu()
        med, std = time_fn(route_dict, Z_cpu, routing_dict, K_iter, device="cpu",
                           n_warmup=2, n_iters=5)
        results["dict_routing"] = med
        speedup = results["gather_sum"] / med
        print(f"  dict_routing (CPU):    {med:7.2f} ms  (±{std:.2f})  {speedup:.2f}x")
    else:
        print(f"  dict_routing:          SKIPPED (N={N} too large)")

    # 5. loop over K_hh
    med, std = time_fn(route_loop, Z, conn_hh, K_iter, device=device_str)
    results["loop_K_hh"] = med
    speedup = results["gather_sum"] / med
    print(f"  loop_K_hh:             {med:7.2f} ms  (±{std:.2f})  {speedup:.2f}x")

    return results


def try_jax(configs):
    """Optional JAX benchmark if available."""
    try:
        import jax
        import jax.numpy as jnp
        from jax import vmap
        print(f"\n{'='*60}")
        print("JAX BENCHMARK")
        print(f"{'='*60}")
        print(f"JAX version: {jax.__version__}, devices: {jax.devices()}")
    except ImportError:
        print("\nJAX not installed — skipping JAX benchmarks.")
        print("Install with: pip install jax jaxlib")
        return

    for cfg in configs[:2]:  # Only first 2 configs for JAX
        N, D, K_hh, K_iter, B = cfg["N"], cfg["D"], cfg["K_hh"], cfg["K_iter"], cfg["B"]
        print(f"\nJAX Config: {cfg['name']}  N={N} D={D} K_hh={K_hh} K_iter={K_iter} B={B}")

        conn_hh_np = build_small_world_conn(N, K_hh).numpy()
        conn_hh_jax = jnp.array(conn_hh_np)
        Z_jax = jax.random.normal(jax.random.PRNGKey(42), (B, N, D))
        Z_jax = Z_jax / jnp.linalg.norm(Z_jax, axis=-1, keepdims=True)

        def jax_route_step(Z, conn):
            Z_nb = Z[:, conn, :]  # [B, N, K_hh, D]
            Z_new = Z_nb.sum(axis=2)  # [B, N, D]
            return Z_new / jnp.linalg.norm(Z_new, axis=-1, keepdims=True)

        def jax_route(Z, conn, K_iter):
            for _ in range(K_iter):
                Z = jax_route_step(Z, conn)
            return Z

        # JIT compile
        jax_route_jit = jax.jit(jax_route, static_argnums=(2,))

        # Warmup
        for _ in range(3):
            _ = jax_route_jit(Z_jax, conn_hh_jax, K_iter).block_until_ready()

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            _ = jax_route_jit(Z_jax, conn_hh_jax, K_iter).block_until_ready()
            times.append((time.perf_counter() - t0) * 1000)

        med = np.median(times)
        std = np.std(times)
        print(f"  jax_jit_gather:        {med:7.2f} ms  (±{std:.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--jax", action="store_true", help="Also run JAX benchmarks")
    args = parser.parse_args()

    print("=" * 60)
    print("SGNNET Sparse Routing Profiler")
    print("=" * 60)
    print(f"Question: Can alternative data structures beat gather-sum?")
    print(f"Current impl: Z[:, conn_hh, :].sum(dim=2) — already O(N·K·D)")
    print(f"Testing: scatter_add, sparse_mm, dict/hashmap, K_hh loop, JAX")

    all_results = {}
    for cfg in CONFIGS:
        all_results[cfg["name"]] = run_config(cfg, args.device)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY — Speedup vs gather_sum baseline (>1.0 = faster)")
    print(f"{'='*60}")
    methods = ["scatter_add", "sparse_mm", "loop_K_hh", "dict_routing"]
    header = f"{'Config':<20}" + "".join(f"{m:<16}" for m in methods)
    print(header)
    print("─" * len(header))
    for cfg in CONFIGS:
        name = cfg["name"]
        r = all_results[name]
        base = r["gather_sum"]
        vals = []
        for m in methods:
            if m in r:
                vals.append(f"{base/r[m]:.2f}x")
            else:
                vals.append("—")
        print(f"{name:<20}" + "".join(f"{v:<16}" for v in vals))

    print(f"\nBaseline times (gather_sum):")
    for cfg in CONFIGS:
        r = all_results[cfg["name"]]
        print(f"  {cfg['name']:<20} {r['gather_sum']:.2f} ms")

    if args.jax:
        try_jax(CONFIGS)

    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    print("If all speedups are ≤1.0x: gather_sum is already optimal.")
    print("FLOPs reduction comes from architecture (D, K_hh, K_iter),")
    print("not from data structure changes.")


if __name__ == "__main__":
    main()
