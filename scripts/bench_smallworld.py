"""Small-world C_hh architecture benchmark.

Tests different (N, K_local, K_random) configurations and reports:
  - Forward pass speed (ms/batch)
  - Graph diameter — average shortest path between random pairs;
    low diameter = information travels across the network quickly
  - Effective fan-in per neuron (K_total = K_local + K_random)
  - Memory footprint

Architecture
------------
C_hh is built as a "small-world" graph (Watts-Strogatz inspired):
  - Each neuron connects to K_local neurons within its spatial block
  - Each neuron also connects to K_random neurons chosen uniformly at random
    (the "long-range shortcuts" that collapse graph diameter)

This is stored as conn_hh [N, K_total] — a fixed fan-in index table —
so the forward op is just:  Z_next = Z[:, conn_hh, :].sum(dim=2)
A rectangular index table: cache-friendly, no sparse kernels needed.

Graph diameter intuition
------------------------
In a purely local graph (K_random=0), neurons in opposite corners of the
box need O(N^(1/D)) hops to communicate. Adding just 1-2 random shortcuts
per neuron collapses this to O(log N) — the Watts-Strogatz result.
We verify this empirically here.
"""

from __future__ import annotations

import time
import math
import collections

import torch
import numpy as np

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
B = 64       # batch size (match training)
D = 4        # geometric dims
N_IN = 25088 # input features (VGG16)
RUNS = 20    # timing repeats

print(f"Device: {DEVICE}  B={B}  D={D}\n")


# ── Graph diameter via BFS on random sample ──────────────────────────────────

def bfs_avg_distance(adj: list[list[int]], n_sample: int = 50, seed: int = 0) -> float:
    """Average shortest-path length via BFS from n_sample random source nodes.

    This is the key metric for small-world graphs: we want low average distance
    (fast global communication) without needing full all-pairs BFS (which is O(N²)).
    """
    rng = np.random.default_rng(seed)
    N = len(adj)
    sources = rng.choice(N, size=min(n_sample, N), replace=False)
    total, count = 0, 0
    for src in sources:
        dist = [-1] * N
        dist[src] = 0
        q = collections.deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        reachable = [d for d in dist if d > 0]
        if reachable:
            total += sum(reachable)
            count += len(reachable)
    return total / count if count else float("inf")


def build_smallworld_conn(N: int, G: int, K_local: int, K_random: int,
                          seed: int = 0) -> tuple[torch.Tensor, list[list[int]]]:
    """Build [N, K_local+K_random] connection index table.

    G      : number of spatial groups  (block size = N//G)
    K_local: connections within the same group
    K_random: long-range random connections to any other neuron
    """
    rng = np.random.default_rng(seed)
    group_size = N // G
    K_total = K_local + K_random
    conn = np.zeros((N, K_total), dtype=np.int64)
    adj = [[] for _ in range(N)]

    for h in range(N):
        g = h // group_size
        # Local: K_local neighbours within same group (excluding self)
        local_pool = [i for i in range(g * group_size, (g + 1) * group_size) if i != h]
        local_pool = local_pool or [h]  # fallback if group_size=1
        local_chosen = rng.choice(local_pool,
                                  size=min(K_local, len(local_pool)),
                                  replace=len(local_pool) < K_local)
        # Random: K_random long-range shortcuts
        other_pool = [i for i in range(N) if i != h]
        rand_chosen = rng.choice(other_pool, size=K_random, replace=K_random > len(other_pool))

        nbrs = np.concatenate([local_chosen, rand_chosen])
        conn[h] = nbrs
        adj[h] = nbrs.tolist()

    return torch.tensor(conn, dtype=torch.long, device=DEVICE), adj


# ── Forward pass timing ───────────────────────────────────────────────────────

def bench_forward(conn_hh: torch.Tensor, N: int, K_iter: int = 3) -> float:
    """Time K_iter rounds of hidden→hidden propagation.

    Each round: Z = Z[:, conn_hh, :].sum(dim=2)
    conn_hh shape: [N, K_total]

    The gather+sum is the core op for every routing step.
    """
    Z = torch.randn(B, N, D, device=DEVICE)

    # Warmup
    for _ in range(5):
        for _ in range(K_iter):
            Z = Z[:, conn_hh, :].sum(dim=2)
    if DEVICE == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(RUNS):
        Zt = Z
        for _ in range(K_iter):
            Zt = Zt[:, conn_hh, :].sum(dim=2)
    if DEVICE == "mps":
        torch.mps.synchronize()
    ms = (time.perf_counter() - t0) / RUNS * 1000
    return ms


# ── Sweep ─────────────────────────────────────────────────────────────────────

configs = [
    # (N,    G,   K_local, K_random)
    # Small N — vary K_random to show diameter collapse
    (256,   32,   4,  0),   # pure local (no shortcuts)
    (256,   32,   4,  1),   # 1 shortcut
    (256,   32,   4,  2),   # 2 shortcuts
    (256,   32,   4,  4),   # 4 shortcuts
    # Medium N
    (1024,  64,   4,  2),
    (1024,  64,   8,  2),
    (2048,  64,   4,  2),
    (2048,  128,  4,  2),
    # Large N
    (4096,  128,  4,  2),
    (10000, 200,  4,  2),
    (10000, 200,  4,  4),
]

print(f"{'N':>6}  {'G':>4}  {'K_loc':>5}  {'K_rnd':>5}  "
      f"{'K_tot':>5}  {'ms/batch':>9}  {'avg_dist':>9}  {'connected':>9}")
print("─" * 78)

for (N, G, K_local, K_random) in configs:
    conn_hh, adj = build_smallworld_conn(N, G, K_local, K_random)
    ms = bench_forward(conn_hh, N, K_iter=3)
    avg_dist = bfs_avg_distance(adj, n_sample=50)
    K_total = K_local + K_random
    # Quick connectivity check: is graph weakly connected?
    # (Use BFS from node 0 and see fraction reached)
    visited = set()
    q = collections.deque([0])
    visited.add(0)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                q.append(v)
    frac_reached = len(visited) / N

    print(f"{N:>6}  {G:>4}  {K_local:>5}  {K_random:>5}  "
          f"{K_total:>5}  {ms:>9.2f}  {avg_dist:>9.2f}  {frac_reached:>9.1%}")

print()
print("Notes:")
print("  avg_dist: average hops between random pairs — lower = faster information flow")
print("  connected: fraction of neurons reachable from node 0 (want 100%)")
print("  ms/batch: 3 routing iterations, batch=64")
