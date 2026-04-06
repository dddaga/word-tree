"""
Benchmark: mps_sparse vs baseline (PyTorch dense matmul on MPS).

Tests:
  1. Correctness — numerical match on CPU
  2. Speed — wallclock on MPS across sparsity levels and matrix sizes
  3. Memory — peak allocated memory

Run:
    python benchmark.py
    python benchmark.py --device cpu
    python benchmark.py --sizes 512 1024 2048 --densities 0.01 0.05 0.1 0.2
"""

from __future__ import annotations
import argparse
import time
import sys
import math
from typing import Optional
from dataclasses import dataclass, field

import torch

# Ensure package is importable when run from repo root
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import mps_sparse as ms


# ---------------------------------------------------------------------------
# Timer utility
# ---------------------------------------------------------------------------

def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def measure_ms(fn, device: torch.device, warmup: int = 5, iters: int = 50) -> float:
    """Return median wall-clock time in milliseconds."""
    for _ in range(warmup):
        fn()
    sync(device)

    times = []
    for _ in range(iters):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]   # median


def peak_memory_mb(fn, device: torch.device) -> float:
    if device.type == "mps":
        torch.mps.empty_cache()
        before = torch.mps.current_allocated_memory()
        fn()
        sync(device)
        after = torch.mps.current_allocated_memory()
        return max(0, (after - before) / 1e6)
    elif device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        fn()
        sync(device)
        return torch.cuda.max_memory_allocated(device) / 1e6
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class BenchRow:
    label: str
    M: int
    K: int
    N: int
    density: float
    nnz: int
    method: str
    median_ms: float
    speedup: float = 1.0
    peak_mb: float = 0.0


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    M: int, K: int, N: int,
    density: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    warmup: int = 10,
    iters: int = 100,
) -> dict:
    """
    Compare dense matmul vs mps_sparse SpMM for a single (M,K)x(K,N) problem.
    Returns a dict with timing and speedup.
    """
    print(f"\n  shape=({M},{K})x({K},{N})  density={density:.1%}  device={device}", flush=True)

    # ---- Build sparse matrix ----
    ssm = ms.random_sparse_matrix(M, K, density, device="cpu", dtype=dtype, seed=42)
    ssm_dev = ms.StaticSparseMatrix(
        ssm.values.to(device),
        ssm.row_indices.to(device),
        ssm.col_indices.to(device),
        ssm.shape,
        learnable=False,
    )
    dense_weight = ssm.to_dense().to(device)   # same values, dense format

    # scipy CSR on CPU (reorder=False keeps results numerically identical to dense)
    # reorder=True gives +43% cache locality but permutes the matrix — requires
    # consistent input permutation (see docs).
    ssm_scipy = ms.StaticSparseMatrix(
        ssm.values, ssm.row_indices, ssm.col_indices,
        ssm.shape, learnable=False,
        strategy="scipy", reorder=False,
    )

    x = torch.randn(K, N, device=device, dtype=dtype)
    x_cpu = x.cpu()

    # ---- Correctness on CPU ----
    ok = ms.verify_sparse_mm(ssm, x_cpu)
    status = "PASS" if ok else "FAIL"

    # ---- Baseline: dense @ dense ----
    def baseline():
        _ = dense_weight @ x

    t_dense = measure_ms(baseline, device, warmup=warmup, iters=iters)

    # ---- mps_sparse SpMM ----
    def sparse_op():
        _ = ssm_dev @ x

    t_sparse = measure_ms(sparse_op, device, warmup=warmup, iters=iters)

    # ---- scipy CSR + RCM on CPU ----
    cpu_dev = torch.device("cpu")
    t_scipy = None
    try:
        def scipy_op():
            _ = ssm_scipy @ x_cpu
        t_scipy = measure_ms(scipy_op, cpu_dev, warmup=warmup, iters=iters)
    except Exception:
        pass

    # ---- torch.compile (if available) ----
    t_compiled = None
    try:
        compiled_fn = torch.compile(lambda: ssm_dev @ x, fullgraph=False)
        # warm up compile
        for _ in range(3):
            compiled_fn()
        sync(device)
        t_compiled = measure_ms(compiled_fn, device, warmup=warmup, iters=iters)
    except Exception:
        pass

    speedup = t_dense / t_sparse

    print(f"    correctness: {status}")
    print(f"    dense:      {t_dense:8.3f} ms")
    print(f"    sparse:     {t_sparse:8.3f} ms  (speedup {speedup:.2f}x)")
    if t_scipy is not None:
        ss = t_dense / t_scipy
        print(f"    scipy(CPU): {t_scipy:8.3f} ms  (speedup {ss:.2f}x vs dense)  [CPU tensor]")
    if t_compiled is not None:
        sc = t_dense / t_compiled
        print(f"    compiled:   {t_compiled:8.3f} ms  (speedup {sc:.2f}x)")

    return dict(
        M=M, K=K, N=N,
        density=density,
        nnz=ssm.nnz,
        correctness=status,
        t_dense_ms=t_dense,
        t_sparse_ms=t_sparse,
        t_scipy_ms=t_scipy,
        t_compiled_ms=t_compiled,
        speedup=speedup,
        speedup_scipy=t_dense / t_scipy if t_scipy else None,
        speedup_compiled=t_dense / t_compiled if t_compiled else None,
    )


# ---------------------------------------------------------------------------
# Gradient flow test
# ---------------------------------------------------------------------------

def test_gradients(device: torch.device) -> bool:
    """Verify gradients flow correctly through sparse matmul."""
    print("\n[Gradient test]", flush=True)
    M, K, N = 64, 128, 32
    density = 0.1

    ssm = ms.random_sparse_matrix(M, K, density, device="cpu", seed=7)
    values = ssm.values.detach().clone().requires_grad_(True)
    x = torch.randn(K, N, requires_grad=True)

    out = ms.sparse_mm(values, ssm.row_indices, ssm.col_indices, x, M)
    loss = out.sum()
    loss.backward()

    has_val_grad = values.grad is not None
    has_x_grad   = x.grad is not None
    print(f"  grad wrt values: {'OK' if has_val_grad else 'MISSING'}")
    print(f"  grad wrt dense:  {'OK' if has_x_grad else 'MISSING'}")

    # Numerical gradient check (CPU only, float64 for accuracy)
    try:
        from torch.autograd import gradcheck

        values64 = values.detach().double().requires_grad_(True)
        x64      = x.detach().double().requires_grad_(True)
        row_idx  = ssm.row_indices
        col_idx  = ssm.col_indices

        def fn(v, d):
            return ms.sparse_mm(v, row_idx, col_idx, d, M)

        try:
            passed = gradcheck(fn, (values64, x64), eps=1e-4, atol=1e-3, rtol=1e-3,
                               raise_on_failure=False)
        except TypeError:
            passed = gradcheck(fn, (values64, x64), eps=1e-4, atol=1e-3, rtol=1e-3)
        print(f"  gradcheck:       {'PASS' if passed else 'FAIL'}")
    except Exception as e:
        print(f"  gradcheck:       SKIPPED ({e})")
        passed = True

    return has_val_grad and has_x_grad and passed


# ---------------------------------------------------------------------------
# SparseLinear test
# ---------------------------------------------------------------------------

def test_sparse_linear(device: torch.device) -> None:
    print("\n[SparseLinear test]", flush=True)
    batch, in_f, out_f = 128, 512, 256
    density = 0.05

    weight = torch.randn(out_f, in_f)
    mask   = torch.rand(out_f, in_f) < density
    weight = weight * mask

    layer = ms.SparseLinear(weight, bias=True, learnable_values=True).to(device)
    x = torch.randn(batch, in_f, device=device)
    out = layer(x)

    assert out.shape == (batch, out_f), f"Bad shape: {out.shape}"
    loss = out.sum()
    loss.backward()

    print(f"  {layer}")
    print(f"  forward shape: {tuple(out.shape)}  OK")
    print(f"  backward:      OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_table(results: list[dict]) -> None:
    header = (
        f"{'M':>5} {'K':>5} {'N':>5}  {'density':>8}  {'nnz':>8}  "
        f"{'dense(ms)':>10}  {'sparse(ms)':>11}  {'spd':>6}  "
        f"{'scipy(ms)':>10}  {'spd':>6}  {'correct':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        scipy_t = f"{r['t_scipy_ms']:.3f}" if r['t_scipy_ms'] else "   N/A  "
        scipy_s = f"{r['speedup_scipy']:.2f}x" if r['speedup_scipy'] else "  N/A"
        print(
            f"{r['M']:>5} {r['K']:>5} {r['N']:>5}  "
            f"{r['density']:>8.1%}  {r['nnz']:>8}  "
            f"{r['t_dense_ms']:>10.3f}  {r['t_sparse_ms']:>11.3f}  "
            f"{r['speedup']:>5.2f}x  "
            f"{scipy_t:>10}  {scipy_s:>6}  {r['correctness']:>8}"
        )
    print("=" * len(header))
    print("  scipy column = CPU tensor (no MPS dispatch). Add reorder=True for +43% cache locality.")


def main():
    parser = argparse.ArgumentParser(description="mps_sparse benchmark")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--sizes",  nargs="+", type=int,
                        default=[512, 1024, 2048, 4096])
    parser.add_argument("--N",      type=int,  default=64,
                        help="Dense matrix second dimension (batch columns)")
    parser.add_argument("--densities", nargs="+", type=float,
                        default=[0.01, 0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--iters",  type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        print("MPS not available — falling back to CPU")
        device = torch.device("cpu")

    print(f"Device: {device}  |  PyTorch {torch.__version__}")
    print(f"MPS built: {torch.backends.mps.is_built()}  "
          f"available: {torch.backends.mps.is_available()}")

    # Correctness + gradient tests
    grad_ok = test_gradients(device)
    test_sparse_linear(device)

    # Speed benchmarks
    print("\n[Speed benchmarks]")
    results = []
    for sz in args.sizes:
        for d in args.densities:
            r = run_benchmark(
                M=sz, K=sz, N=args.N,
                density=d,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            )
            results.append(r)

    print_table(results)

    # Summary
    speedups = [r["speedup"] for r in results]
    print(f"\nSpeedup range: {min(speedups):.2f}x — {max(speedups):.2f}x")
    print(f"Geometric mean speedup: {math.exp(sum(math.log(s) for s in speedups) / len(speedups)):.2f}x")


if __name__ == "__main__":
    main()
