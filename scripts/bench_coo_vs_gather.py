"""Benchmark: COO/CSR sparse matmul vs dense gather for SGNNET routing.

SGNNET uses Z[:, conn_hh, :] (gather) for neighbor aggregation.
This script compares wall-time and memory against torch.sparse equivalents.

Tests at multiple N values with fixed K_hh=2, D=16.

Usage:
    python scripts/bench_coo_vs_gather.py --device cuda
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--iters", type=int, default=200)
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

D = 16
K_HH = 2
BATCH_SIZES = [1, 32, 128]
N_VALUES = [2048, 4096, 8192, 16384]
WARMUP = args.warmup
ITERS = args.iters


def build_conn_hh(N, K):
    """Random fixed-degree connectivity (mimics small-world)."""
    conn = torch.stack([torch.randperm(N)[:K] for _ in range(N)])
    return conn


def build_sparse_adj(N, conn_hh, device):
    """Build sparse adjacency from conn_hh [N, K] index tensor."""
    K = conn_hh.shape[1]
    rows = torch.arange(N, device=device).unsqueeze(1).expand(-1, K).reshape(-1)
    cols = conn_hh.reshape(-1).to(device)
    indices = torch.stack([rows, cols])
    values = torch.ones(N * K, device=device)
    adj_coo = torch.sparse_coo_tensor(indices, values, (N, N))
    adj_csr = adj_coo.to_sparse_csr()
    return adj_coo, adj_csr


def bench_gather(Z, conn_hh, iters):
    """Current SGNNET approach: Z[:, conn_hh, :] gather + sum."""
    if Z.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        Z_nb = Z[:, conn_hh, :]       # [B, N, K, D]
        out = Z_nb.sum(dim=2)          # [B, N, D]
    if Z.device.type == "cuda":
        torch.cuda.synchronize()
    elif Z.device.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_gather_dw(Z, conn_hh, dw, iters):
    """Current SGNNET ΔW-proj: gather + project + weighted sum."""
    if Z.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        Z_nb = Z[:, conn_hh, :]                           # [B, N, K, D]
        proj = (Z_nb * dw).sum(dim=-1, keepdim=True)      # [B, N, K, 1]
        out = (Z_nb * proj.abs()).sum(dim=2)               # [B, N, D]
    if Z.device.type == "cuda":
        torch.cuda.synchronize()
    elif Z.device.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_sparse_coo(Z, adj_coo, iters):
    """COO sparse matmul: adj @ Z for each batch element."""
    if Z.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        # sparse mm doesn't support batched, so loop or reshape
        B = Z.shape[0]
        if B == 1:
            out = torch.sparse.mm(adj_coo, Z.squeeze(0))
            out = out.unsqueeze(0)
        else:
            out = torch.stack([torch.sparse.mm(adj_coo, Z[b]) for b in range(B)])
    if Z.device.type == "cuda":
        torch.cuda.synchronize()
    elif Z.device.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_sparse_csr(Z, adj_csr, iters):
    """CSR sparse matmul."""
    if Z.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        B = Z.shape[0]
        if B == 1:
            out = torch.sparse.mm(adj_csr, Z.squeeze(0))
            out = out.unsqueeze(0)
        else:
            out = torch.stack([torch.sparse.mm(adj_csr, Z[b]) for b in range(B)])
    if Z.device.type == "cuda":
        torch.cuda.synchronize()
    elif Z.device.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_dense_mm(Z, adj_dense, iters):
    """Dense matmul baseline (adj @ Z) for reference."""
    if Z.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = torch.bmm(adj_dense.unsqueeze(0).expand(Z.shape[0], -1, -1), Z)
    if Z.device.type == "cuda":
        torch.cuda.synchronize()
    elif Z.device.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def measure_memory(fn, *fn_args):
    """Measure peak GPU memory delta for a function call."""
    if DEVICE.type != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.max_memory_allocated()
    fn(*fn_args)
    after = torch.cuda.max_memory_allocated()
    return (after - before) / 1024 / 1024  # MB


def main():
    print(f"{'='*80}")
    print(f"COO/CSR vs Gather Benchmark — device={DEVICE}")
    print(f"D={D}, K_hh={K_HH}, warmup={WARMUP}, iters={ITERS}")
    print(f"{'='*80}\n")

    results = []

    for N in N_VALUES:
        conn_hh = build_conn_hh(N, K_HH).to(DEVICE)
        adj_coo, adj_csr = build_sparse_adj(N, conn_hh, DEVICE)

        # ΔW direction vectors [1, N, K, D]
        W_pos = F.normalize(torch.randn(N, D, device=DEVICE), dim=-1)
        dw = F.normalize(
            W_pos.unsqueeze(1) - W_pos[conn_hh], dim=-1
        ).unsqueeze(0)

        # Dense adj for reference (only if N <= 8192, else OOM)
        adj_dense = None
        if N <= 8192:
            adj_dense = adj_coo.to_dense()

        print(f"--- N={N} ---")
        print(f"  conn_hh: {conn_hh.shape}, nnz={N*K_HH}")
        if adj_dense is not None:
            density = (N * K_HH) / (N * N) * 100
            print(f"  adj density: {density:.4f}%")

        for B in BATCH_SIZES:
            Z = F.normalize(torch.randn(B, N, D, device=DEVICE), dim=-1)

            # Warmup
            for _ in range(WARMUP):
                _ = Z[:, conn_hh, :].sum(dim=2)

            t_gather = bench_gather(Z, conn_hh, ITERS)
            t_gather_dw = bench_gather_dw(Z, conn_hh, dw, ITERS)
            t_coo = bench_sparse_coo(Z, adj_coo, ITERS)
            t_csr = bench_sparse_csr(Z, adj_csr, ITERS)

            t_dense = None
            if adj_dense is not None and B <= 32:
                t_dense = bench_dense_mm(Z, adj_dense, ITERS)

            row = {
                "N": N, "B": B, "K": K_HH, "D": D,
                "gather_ms": round(t_gather * 1000, 3),
                "gather_dw_ms": round(t_gather_dw * 1000, 3),
                "coo_ms": round(t_coo * 1000, 3),
                "csr_ms": round(t_csr * 1000, 3),
                "dense_ms": round(t_dense * 1000, 3) if t_dense else None,
                "speedup_coo_vs_gather": round(t_coo / t_gather, 2) if t_gather > 0 else None,
                "speedup_csr_vs_gather": round(t_csr / t_gather, 2) if t_gather > 0 else None,
            }
            results.append(row)

            coo_ratio = f"{t_coo/t_gather:.2f}×" if t_gather > 0 else "?"
            csr_ratio = f"{t_csr/t_gather:.2f}×" if t_gather > 0 else "?"
            dense_str = f"  dense={t_dense*1000:.3f}ms" if t_dense else ""
            print(f"  B={B:>3}: gather={t_gather*1000:.3f}ms  "
                  f"gather+ΔW={t_gather_dw*1000:.3f}ms  "
                  f"COO={t_coo*1000:.3f}ms({coo_ratio})  "
                  f"CSR={t_csr*1000:.3f}ms({csr_ratio}){dense_str}")

        # Memory measurement at B=32
        if DEVICE.type == "cuda":
            Z32 = F.normalize(torch.randn(32, N, D, device=DEVICE), dim=-1)
            mem_gather = measure_memory(bench_gather, Z32, conn_hh, 1)
            mem_coo = measure_memory(bench_sparse_coo, Z32, adj_coo, 1)
            print(f"  Memory (B=32): gather={mem_gather:.1f}MB  COO={mem_coo:.1f}MB")

        print()

    # Summary
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'N':>6} {'B':>4} {'gather':>10} {'gather+ΔW':>12} {'COO':>10} {'CSR':>10} {'COO/gather':>12}")
    for r in results:
        print(f"{r['N']:>6} {r['B']:>4} {r['gather_ms']:>9.3f}ms {r['gather_dw_ms']:>11.3f}ms "
              f"{r['coo_ms']:>9.3f}ms {r['csr_ms']:>9.3f}ms {r['speedup_coo_vs_gather']:>11.2f}×")

    out_path = ROOT / "results" / "bench_coo_vs_gather.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
