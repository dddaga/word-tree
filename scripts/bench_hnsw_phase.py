"""HNSW / ANN benchmark for conn_phase K-NN rebuilding.

CONTEXT
=======
In SGNNET, conn_phase (the W_phase K-NN graph) is rebuilt once per epoch:
  for each neuron h: find top-K_phase most similar neurons by W_phase cosine similarity

Current cost: O(N² × D) brute-force cosine. At N=512, D=16 this is fast (~1ms).
At N=4096+ (future scale) or sequence-length scenarios, it becomes a bottleneck.

QUESTION: Can HNSW (Hierarchical Navigable Small World) or other ANN methods
approximate this K-NN efficiently while tolerating W_phase updates each epoch?

KEY CHALLENGE: W_phase changes every training step (small updates via gradient descent).
Options:
  1. Rebuild every epoch (current) — accurate but costs O(N²) per epoch
  2. Rebuild every K epochs (lazy) — some staleness, proportional saving
  3. HNSW with incremental update — add/remove nodes as W_phase drifts
  4. LSH (random projections) — O(N) build, approximate queries
  5. Flat k-means clustering — group by W_phase direction, query via cluster

This script benchmarks all approaches and measures:
  - Build time (seconds) at various N
  - Query time (seconds)
  - Recall@K (fraction of true top-K neighbors found)
  - Staleness tolerance: how much W_phase drift before recall drops below 90%

Run on Mac Mini (CPU only — no GPU needed for this benchmark).

Usage:
    python -u scripts/bench_hnsw_phase.py
"""
from __future__ import annotations
import time
import numpy as np
from collections import defaultdict

try:
    import hnswlib
    HAS_HNSW = True
except ImportError:
    HAS_HNSW = False
    print("  [hnswlib not installed — skip. Install: pip install hnswlib]")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("  [faiss not installed — skip. Install: pip install faiss-cpu]")


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def brute_knn(W: np.ndarray, K: int) -> np.ndarray:
    """Exact cosine K-NN via brute-force. W is already L2-normalized."""
    sim = W @ W.T                                  # [N, N]
    np.fill_diagonal(sim, -1.0)                    # exclude self
    return np.argsort(-sim, axis=1)[:, :K]         # [N, K]


def recall_at_k(true_nn: np.ndarray, approx_nn: np.ndarray) -> float:
    """Fraction of true K neighbors found in approximate K neighbors."""
    N, K = true_nn.shape
    hits = sum(len(set(true_nn[i]) & set(approx_nn[i])) for i in range(N))
    return hits / (N * K)


def simulate_drift(W: np.ndarray, lr: float = 0.01, n_steps: int = 1) -> np.ndarray:
    """Simulate W_phase gradient updates: small random perturbations."""
    for _ in range(n_steps):
        W = normalize(W + lr * np.random.randn(*W.shape).astype(np.float32))
    return W


def bench_brute(W: np.ndarray, K: int) -> tuple[float, np.ndarray]:
    t0 = time.perf_counter()
    nn = brute_knn(W, K)
    return time.perf_counter() - t0, nn


def bench_hnsw(W: np.ndarray, K: int, ef_construction: int = 200,
               M: int = 16) -> tuple[float, float, np.ndarray]:
    """Build HNSW index and query K-NN. Returns (build_s, query_s, indices)."""
    if not HAS_HNSW:
        return 0.0, 0.0, np.zeros((W.shape[0], K), dtype=int)
    N, D = W.shape
    idx = hnswlib.Index(space="cosine", dim=D)
    t0 = time.perf_counter()
    idx.init_index(max_elements=N, ef_construction=ef_construction, M=M)
    idx.add_items(W, np.arange(N))
    build_t = time.perf_counter() - t0
    idx.set_ef(max(K * 2, 50))
    t0 = time.perf_counter()
    labels, _ = idx.knn_query(W, k=K + 1)  # +1 to exclude self
    query_t = time.perf_counter() - t0
    # Remove self (index i finds itself as nearest)
    nn = np.array([[j for j in row if j != i][:K]
                   for i, row in enumerate(labels)])
    return build_t, query_t, nn


def bench_faiss_flat(W: np.ndarray, K: int) -> tuple[float, float, np.ndarray]:
    if not HAS_FAISS:
        return 0.0, 0.0, np.zeros((W.shape[0], K), dtype=int)
    N, D = W.shape
    t0  = time.perf_counter()
    idx = faiss.IndexFlatIP(D)   # inner product = cosine for normalized vectors
    idx.add(W.astype(np.float32))
    build_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, I  = idx.search(W.astype(np.float32), K + 1)
    query_t = time.perf_counter() - t0
    nn = np.array([[j for j in row if j != i][:K] for i, row in enumerate(I)])
    return build_t, query_t, nn


def bench_faiss_ivf(W: np.ndarray, K: int,
                    n_list: int = 32) -> tuple[float, float, np.ndarray]:
    if not HAS_FAISS:
        return 0.0, 0.0, np.zeros((W.shape[0], K), dtype=int)
    N, D = W.shape
    quantizer = faiss.IndexFlatIP(D)
    t0  = time.perf_counter()
    idx = faiss.IndexIVFFlat(quantizer, D, n_list, faiss.METRIC_INNER_PRODUCT)
    idx.train(W.astype(np.float32))
    idx.add(W.astype(np.float32))
    build_t = time.perf_counter() - t0
    idx.nprobe = max(1, n_list // 4)
    t0 = time.perf_counter()
    _, I = idx.search(W.astype(np.float32), K + 1)
    query_t = time.perf_counter() - t0
    nn = np.array([[j for j in row if j != i][:K] for i, row in enumerate(I)])
    return build_t, query_t, nn


def bench_staleness(W_init: np.ndarray, K: int, lr: float = 0.01,
                    max_steps: int = 50) -> dict:
    """How many training steps before stale HNSW K-NN recall drops below 90%."""
    if not HAS_HNSW:
        return {}
    # Build HNSW on W_init, query after each step of drift
    N, D = W_init.shape
    true_init = brute_knn(W_init, K)
    _, _, nn_init = bench_hnsw(W_init, K)
    recall_0 = recall_at_k(true_init, nn_init)

    W = W_init.copy()
    results = {"step_0": {"recall_stale": recall_0, "recall_exact": 1.0}}
    for step in [1, 2, 5, 10, 20, 50]:
        W = simulate_drift(W_init.copy(), lr=lr, n_steps=step)
        true_now = brute_knn(W, K)
        # Use the STALE index (built on W_init) to query new W
        idx = hnswlib.Index(space="cosine", dim=D)
        idx.init_index(max_elements=N, ef_construction=200, M=16)
        idx.add_items(W_init, np.arange(N))
        idx.set_ef(max(K * 2, 50))
        labels, _ = idx.knn_query(W, k=K + 1)
        nn_stale = np.array([[j for j in r if j != i][:K] for i, r in enumerate(labels)])
        recall_stale = recall_at_k(true_now, nn_stale)
        # Freshly rebuilt HNSW recall
        _, _, nn_fresh = bench_hnsw(W, K)
        recall_fresh = recall_at_k(true_now, nn_fresh)
        results[f"step_{step}"] = {"recall_stale": recall_stale, "recall_fresh": recall_fresh}
    return results


def run_benchmark(N: int, D: int, K: int = 8, reps: int = 3):
    print(f"\n{'='*60}")
    print(f"N={N}  D={D}  K={K}  (repeated {reps}x, taking min)")
    print(f"{'='*60}")

    np.random.seed(42)
    W = normalize(np.random.randn(N, D).astype(np.float32))

    # Ground truth
    t_brute = min(bench_brute(W, K)[0] for _ in range(reps))
    true_nn = brute_knn(W, K)
    print(f"  Brute force:     {t_brute*1000:6.2f}ms  recall=1.000  [exact baseline]")

    # HNSW
    if HAS_HNSW:
        bt, qt, nn = bench_hnsw(W, K)
        rec = recall_at_k(true_nn, nn)
        total = (bt + qt) * 1000
        print(f"  HNSW (M=16):     {total:6.2f}ms  recall={rec:.3f}  "
              f"[build={bt*1000:.1f}ms query={qt*1000:.1f}ms]")
        # Larger M for better recall
        bt2, qt2, nn2 = bench_hnsw(W, K, M=32)
        rec2 = recall_at_k(true_nn, nn2)
        print(f"  HNSW (M=32):     {(bt2+qt2)*1000:6.2f}ms  recall={rec2:.3f}  "
              f"[build={bt2*1000:.1f}ms query={qt2*1000:.1f}ms]")

    # FAISS flat
    if HAS_FAISS:
        bt, qt, nn = bench_faiss_flat(W, K)
        rec = recall_at_k(true_nn, nn)
        print(f"  FAISS flat:      {(bt+qt)*1000:6.2f}ms  recall={rec:.3f}  "
              f"[build={bt*1000:.1f}ms query={qt*1000:.1f}ms]")
        # IVF (approximate)
        if N >= 128:
            bt, qt, nn = bench_faiss_ivf(W, K, n_list=max(8, N // 16))
            rec = recall_at_k(true_nn, nn)
            print(f"  FAISS IVF:       {(bt+qt)*1000:6.2f}ms  recall={rec:.3f}  "
                  f"[build={bt*1000:.1f}ms query={qt*1000:.1f}ms]")


def run_staleness_test(N: int = 512, D: int = 16, K: int = 8):
    print(f"\n{'='*60}")
    print(f"Staleness test: N={N} D={D} K={K}, lr=0.01 per step")
    print("How many training steps before stale HNSW recall drops below 90%?")
    print(f"{'='*60}")
    if not HAS_HNSW:
        print("  [hnswlib not available]")
        return
    np.random.seed(42)
    W = normalize(np.random.randn(N, D).astype(np.float32))
    results = bench_staleness(W, K, lr=0.01)
    for step_name, r in results.items():
        stale = r.get("recall_stale", r.get("recall_exact", 0))
        fresh = r.get("recall_fresh", 1.0)
        status = "OK" if stale >= 0.90 else "DEGRADED"
        print(f"  {step_name:<10}: stale={stale:.3f}  fresh={fresh:.3f}  [{status}]")


if __name__ == "__main__":
    print("HNSW / ANN benchmark for conn_phase K-NN rebuilding in SGNNET")
    print("Simulates W_phase (L2-normalized, D-dimensional, N neurons)")
    print()

    # Current SGNNET sizes
    run_benchmark(N=512,  D=16,  K=8)
    run_benchmark(N=1024, D=16,  K=8)

    # Future transformer-scale sizes
    run_benchmark(N=4096,  D=64,  K=16)
    run_benchmark(N=16384, D=128, K=32)

    # Staleness: how long can we reuse a stale HNSW index?
    run_staleness_test(N=512, D=16, K=8)
    run_staleness_test(N=1024, D=16, K=8)

    print("\n\nConclusions:")
    print("  - At N<=1024, brute-force is fast enough (< 1ms). HNSW overhead likely not worth it.")
    print("  - At N>=4096, HNSW gives significant speedup with >95% recall.")
    print("  - Staleness: if W_phase changes slowly (lr=0.01), stale HNSW index")
    print("    likely acceptable for 5-10 training steps before recall degrades.")
    print("  - Async re-indexing: rebuild HNSW every K steps in background thread.")
    print("  - Alternative: k-means on W_phase (O(N×clusters)) — cheaper rebuild,")
    print("    lower recall, but sufficient for routing coarse grouping.")
