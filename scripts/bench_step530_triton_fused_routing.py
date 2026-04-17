"""Step 530: Triton fused gather+sum+normalize kernel for hidden→hidden routing.

# CUDA-5060ti-validated

TARGET
======
Current routing loop in model_smallworld.py:
    Z = Z[:, self.conn_hh, :].sum(dim=2)   # [B,N,K_hh,D] → [B,N,D]
    Z = F.normalize(Z, dim=-1)

Problem: split into 2 kernels → 2× launch overhead + sync per K_iter step.
  v1 showed Triton gather+sum alone = 4× faster than compiled single-step,
  but full K_iter=5 trails because normalize is a separate PyTorch kernel.

Solution (v2): fuse gather + sum + L2-normalize into ONE kernel.
  With D=16, norm = sqrt(sum(acc²)) computed in registers; no extra memory.
  Eliminates K_iter launch/sync pairs → should win end-to-end.

VARIANTS BENCHMARKED
====================
  eager              : Z[:, conn_hh, :].sum(dim=2) + F.normalize (PyTorch)
  compiled           : torch.compile(max-autotune) on full routing loop
  triton_v1 (no norm): gather+sum only — 4× single-step, trails full loop
  triton_v2_fused    : gather+sum+L2norm fused — target for full-loop win

SUCCESS CRITERION
=================
  triton_v2_fused full K_iter=5 < compiled full.
  If yes → replace _route() in model_smallworld.py.

To run:
    python -u scripts/bench_step530_triton_fused_routing.py
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--N",      type=int, default=2048)
parser.add_argument("--B",      type=int, default=128)
parser.add_argument("--D",      type=int, default=16)
parser.add_argument("--K_hh",   type=int, default=2)
parser.add_argument("--K_iter", type=int, default=5)
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--reps",   type=int, default=500)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

if not torch.cuda.is_available():
    print("ERROR: CUDA required for Triton benchmark."); sys.exit(1)

import triton
import triton.language as tl

DEVICE = torch.device(args.device)
N, B, D, K_hh, K_iter = args.N, args.B, args.D, args.K_hh, args.K_iter

SLOT = "5060ti_cuda"
OUT_PATH = ROOT / "results" / f"bench_step530_triton_routing__{SLOT}.json"


# ---------------------------------------------------------------------------
# Triton kernel: fused gather+sum, eliminates [B,N,K_hh,D] intermediate
# ---------------------------------------------------------------------------

@triton.jit
def fused_gather_sum_kernel(
    Z_in_ptr,   # [B, N, D]  contiguous
    Z_out_ptr,  # [B, N, D]  contiguous
    conn_ptr,   # [N, K_hh]  int64
    B: tl.constexpr,
    N: tl.constexpr,
    K_HH: tl.constexpr,
    D: tl.constexpr,      # = 16
    BLOCK_B: tl.constexpr,
):
    """Each program handles one neuron (pid_n) × BLOCK_B batch elements."""
    pid_n = tl.program_id(0)          # neuron index in [0, N)
    pid_b = tl.program_id(1)          # batch block index

    b_start = pid_b * BLOCK_B
    b_offs  = b_start + tl.arange(0, BLOCK_B)   # [BLOCK_B]
    d_offs  = tl.arange(0, D)                    # [D] = [16]
    b_mask  = b_offs < B

    # Accumulator in registers: [BLOCK_B, D]
    acc = tl.zeros([BLOCK_B, D], dtype=tl.float32)

    # Loop over K_hh neighbours (small, static — unrolled by Triton)
    for k in tl.static_range(K_HH):
        nb = tl.load(conn_ptr + pid_n * K_HH + k)   # scalar neighbour index
        # Load Z[b_offs, nb, :] — shape [BLOCK_B, D]
        ptrs = (Z_in_ptr
                + b_offs[:, None] * (N * D)
                + nb * D
                + d_offs[None, :])
        acc += tl.load(ptrs, mask=b_mask[:, None], other=0.0)

    # Store Z_out[b_offs, pid_n, :]
    out_ptrs = (Z_out_ptr
                + b_offs[:, None] * (N * D)
                + pid_n * D
                + d_offs[None, :])
    tl.store(out_ptrs, acc, mask=b_mask[:, None])


def triton_gather_sum(Z: torch.Tensor, conn: torch.Tensor,
                      block_b: int = 32) -> torch.Tensor:
    """Single routing step via Triton kernel."""
    B_, N_, D_ = Z.shape
    K_ = conn.shape[1]
    Z_out = torch.empty_like(Z)
    grid = (N_, (B_ + block_b - 1) // block_b)
    fused_gather_sum_kernel[grid](
        Z, Z_out, conn,
        B=B_, N=N_, K_HH=K_, D=D_,
        BLOCK_B=block_b,
    )
    return Z_out


def triton_routing(Z: torch.Tensor, conn: torch.Tensor,
                   k_iter: int, block_b: int = 32) -> torch.Tensor:
    """Full K_iter routing pass via Triton (unfused norm — v1 baseline)."""
    for _ in range(k_iter):
        Z = triton_gather_sum(Z, conn, block_b=block_b)
        Z = F.normalize(Z, dim=-1)
    return Z


# ---------------------------------------------------------------------------
# Triton v2: fused gather+sum+L2normalize in one kernel
# ---------------------------------------------------------------------------

@triton.jit
def fused_gather_sum_norm_kernel(
    Z_in_ptr,
    Z_out_ptr,
    conn_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    K_HH: tl.constexpr,
    D: tl.constexpr,
    BLOCK_B: tl.constexpr,
    EPS: tl.constexpr,
):
    """gather + sum + L2-normalize, all in registers. No intermediate tensor."""
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

    # L2 normalize across D dimension (D=16, fits in registers)
    norm_sq  = tl.sum(acc * acc, axis=1, keep_dims=True)   # [BLOCK_B, 1]
    inv_norm = tl.rsqrt(norm_sq + EPS)
    acc      = acc * inv_norm

    out_ptrs = Z_out_ptr + b_offs[:, None] * (N * D) + pid_n * D + d_offs[None, :]
    tl.store(out_ptrs, acc, mask=b_mask[:, None])


def triton_gather_sum_norm(Z: torch.Tensor, conn: torch.Tensor,
                           block_b: int = 32, eps: float = 1e-12) -> torch.Tensor:
    B_, N_, D_ = Z.shape
    Z_out = torch.empty_like(Z)
    grid  = (N_, (B_ + block_b - 1) // block_b)
    fused_gather_sum_norm_kernel[grid](
        Z, Z_out, conn,
        B=B_, N=N_, K_HH=conn.shape[1], D=D_,
        BLOCK_B=block_b, EPS=eps,
    )
    return Z_out


def triton_routing_fused(Z: torch.Tensor, conn: torch.Tensor,
                         k_iter: int, block_b: int = 32) -> torch.Tensor:
    """Full K_iter routing — ONE kernel per step, normalize fused."""
    for _ in range(k_iter):
        Z = triton_gather_sum_norm(Z, conn, block_b=block_b)
    return Z


# ---------------------------------------------------------------------------
# Baseline: eager and compiled
# ---------------------------------------------------------------------------

def eager_routing(Z: torch.Tensor, conn: torch.Tensor, k_iter: int) -> torch.Tensor:
    for _ in range(k_iter):
        Z = Z[:, conn, :].sum(dim=2)
        Z = F.normalize(Z, dim=-1)
    return Z


def bench(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3  # ms


def main():
    print(f"Step 530 — Triton fused gather+sum routing kernel")
    print(f"  N={N}  B={B}  D={D}  K_hh={K_hh}  K_iter={K_iter}")
    print(f"  Triton {triton.__version__}  CUDA {torch.version.cuda}")
    print(f"  Intermediate tensor eliminated: {B*N*K_hh*D*4/1e6:.1f} MB/step")
    print()

    # Setup
    torch.manual_seed(42)
    Z = torch.randn(B, N, D, device=DEVICE)
    # Build small-world conn_hh (simple ring + random for bench)
    conn = torch.stack([
        torch.arange(N, device=DEVICE),                           # ring: n→n-1
        torch.randperm(N, device=DEVICE),                         # random
    ], dim=1)[:, :K_hh].contiguous()

    # Correctness check: single step
    Z_ref = Z[:, conn, :].sum(dim=2)
    Z_tri = triton_gather_sum(Z, conn, block_b=32)
    max_diff = (Z_ref - Z_tri).abs().max().item()
    print(f"  Correctness check (single step): max_diff={max_diff:.2e}  {'PASS' if max_diff < 1e-4 else 'FAIL'}")
    if max_diff >= 1e-4:
        print("  ABORT: kernel incorrect"); sys.exit(1)

    # Full routing correctness — v1 (unfused norm)
    Z_ref_full  = eager_routing(Z, conn, K_iter)
    Z_tri_full  = triton_routing(Z, conn, K_iter, block_b=32)
    max_full    = (Z_ref_full - Z_tri_full).abs().max().item()
    print(f"  Correctness check (v1 unfused, K_iter={K_iter}): max_diff={max_full:.2e}  {'PASS' if max_full < 1e-4 else 'FAIL'}")

    # v2 fused normalize correctness
    Z_tri_v2    = triton_routing_fused(Z, conn, K_iter, block_b=32)
    max_v2      = (Z_ref_full - Z_tri_v2).abs().max().item()
    print(f"  Correctness check (v2 fused norm, K_iter={K_iter}): max_diff={max_v2:.2e}  {'PASS' if max_v2 < 1e-4 else 'FAIL'}")
    if max_v2 >= 1e-4:
        print("  ABORT: v2 kernel incorrect"); sys.exit(1)
    print()

    # Compile baseline
    print("  Compiling baseline (max-autotune)...", flush=True)
    compiled_step  = torch.compile(lambda z: z[:, conn, :].sum(dim=2), mode="max-autotune")
    compiled_route = torch.compile(lambda z: eager_routing(z, conn, K_iter), mode="max-autotune")
    _ = compiled_step(Z); _ = compiled_route(Z)
    torch.cuda.synchronize()
    print("  Done.\n")

    W, R = args.warmup, args.reps

    t_eager_step   = bench(lambda: Z[:, conn, :].sum(dim=2),              W, R)
    t_comp_step    = bench(lambda: compiled_step(Z),                       W, R)
    t_v1_step      = bench(lambda: triton_gather_sum(Z, conn, 32),         W, R)
    t_v2_step      = bench(lambda: triton_gather_sum_norm(Z, conn, 32),    W, R)

    t_eager_full   = bench(lambda: eager_routing(Z, conn, K_iter),               W, R)
    t_comp_full    = bench(lambda: compiled_route(Z),                             W, R)
    t_v1_full      = bench(lambda: triton_routing(Z, conn, K_iter, 32),          W, R)
    t_v2_full      = bench(lambda: triton_routing_fused(Z, conn, K_iter, 32),    W, R)

    print(f"  {'Variant':<24} {'single-step ms':>15}  {'vs comp':>8}  {'K={K_iter} full ms':>15}  {'vs comp':>8}")
    print(f"  {'-'*76}")
    rows = [
        ("eager",              t_eager_step,  t_eager_full),
        ("compiled",           t_comp_step,   t_comp_full),
        ("triton_v1 (no norm)", t_v1_step,    t_v1_full),
        ("triton_v2 (fused)",  t_v2_step,     t_v2_full),
    ]
    for name, ts, tf in rows:
        vs_s = f"{t_comp_step/ts:.2f}×" if "comp" not in name else "—"
        vs_f = f"{t_comp_full/tf:.2f}×" if "comp" not in name else "—"
        print(f"  {name:<24} {ts:>15.4f}  {vs_s:>8}  {tf:>15.4f}  {vs_f:>8}")

    v2_wins = t_v2_full < t_comp_full
    verdict = "BEATS compiled — production candidate" if v2_wins else "trails compiled"
    print(f"\n  triton_v2 fused K_iter={K_iter}: {verdict}")
    if v2_wins:
        print(f"  → {t_comp_full/t_v2_full:.2f}× speedup — replace _route() in model_smallworld.py")
    else:
        print(f"  → torch.compile max-autotune already fuses normalize; Triton not needed")

    results = {
        "config":  {"N": N, "B": B, "D": D, "K_hh": K_hh, "K_iter": K_iter,
                    "triton": triton.__version__, "cuda": torch.version.cuda},
        "correctness": {"v1_max_diff": max_full, "v2_max_diff": max_v2,
                        "pass": max_v2 < 1e-4},
        "single_step_ms": {"eager": round(t_eager_step,5), "compiled": round(t_comp_step,5),
                           "triton_v1": round(t_v1_step,5), "triton_v2_fused": round(t_v2_step,5)},
        "full_routing_ms": {"eager": round(t_eager_full,5), "compiled": round(t_comp_full,5),
                            "triton_v1": round(t_v1_full,5), "triton_v2_fused": round(t_v2_full,5)},
        "verdict": {"full_routing": verdict,
                    "triton_v2_vs_compiled": round(t_comp_full/t_v2_full, 3)},
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
