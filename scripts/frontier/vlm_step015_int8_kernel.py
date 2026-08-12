"""vlm_step015 -- can a real int8 GEMM capture the 2x that vlm_step014 proved is physically there?

step014 established three things at the SmolVLM `lm_head` shape ([T,576] @ [576,49280]): the fp16
GEMM is memory-bound at ~86% of peak bandwidth, the card moves int8 at the SAME GB/s as fp16 (so
half the bytes take half the time -- the 2x is available), and `torch._int_mm` captures NONE of it
(0.96-1.00x). That indicts one kernel. It does NOT establish that the 2x is unreachable on sm_120,
which is what the storage-only caveat on every §16/§21 quantization claim currently rests on.

This script asks the question with a kernel we control. Triton ships inside torch, so there is no
new dependency and nothing is installed into the teammates' shared venv.

THE CONTROL IS THE POINT. Timing a Triton int8 kernel against cuBLAS fp16 confounds two variables:
int8-vs-fp16 AND triton-vs-cuBLAS. So a STRUCTURALLY IDENTICAL fp16 Triton kernel is benchmarked
too -- same tiling, same loop, same autotune grid, only the dtype differs. `triton_int8 vs
triton_fp16` is then a clean one-variable comparison, and `triton_fp16 vs cuBLAS` separately prices
how much is lost by not being cuBLAS. Reading only the int8-vs-cuBLAS number would let a mediocre
Triton kernel masquerade as "int8 doesn't help on this card".

Correctness gate before any timing is reported: the Triton int8 output must be BIT-EXACT against
`torch._int_mm` (both are exact integer arithmetic -- there is no tolerance to argue about, so a
mismatch means the kernel is wrong and its timings are meaningless).
"""
from __future__ import annotations
import argparse
import time

import torch
import triton
import triton.language as tl

# (BM, BN, BK, num_stages, num_warps). M is tiny (T <= 576) and N is huge (49280), so the useful
# configs are small-BM/large-BN; a square tiling wastes most of its rows on padding.
CFGS = [(16, 128, 64, 4, 4), (16, 256, 64, 4, 8), (32, 128, 64, 4, 4), (32, 256, 64, 4, 8),
        (64, 128, 64, 4, 4), (64, 256, 64, 4, 8), (64, 128, 128, 3, 4), (128, 128, 64, 4, 8)]


@triton.jit
def _gemm(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ACC: tl.constexpr):
    """One kernel for both dtypes. ACC picks the accumulator (tl.int32 for int8 -> IMMA,
    tl.float32 for fp16 -> HMMA); everything else is identical by construction, which is what makes
    the int8-vs-fp16 comparison one-variable."""
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptr = A + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptr = B + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=ACC)
    for k in range(0, K, BK):
        ka = offs_k[None, :] + k
        kb = offs_k[:, None] + k
        a = tl.load(a_ptr, mask=(offs_m[:, None] < M) & (ka < K), other=0)
        b = tl.load(b_ptr, mask=(kb < K) & (offs_n[None, :] < N), other=0)
        acc += tl.dot(a, b, out_dtype=ACC)
        a_ptr += BK * sak
        b_ptr += BK * sbk
    c = C + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(a, b, cfg, acc_dtype):
    BM, BN, BK, stages, warps = cfg
    M, K = a.shape
    N = b.shape[1]
    out = torch.empty((M, N), dtype=torch.int32 if acc_dtype is tl.int32 else torch.float32,
                      device=a.device)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _gemm[grid](a, b, out, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                out.stride(0), out.stride(1), BM=BM, BN=BN, BK=BK, ACC=acc_dtype,
                num_stages=stages, num_warps=warps)
    return out


def bench(fn, iters=50, warmup=10):
    """Median-of-iters wall time in ms. Median, not mean: one stray scheduler hiccup on a 0.15 ms
    kernel moves a mean by tens of percent, which is the size of the effect being measured."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def best_cfg(a, b, acc_dtype, iters):
    """Smallest median over CFGS. Configs that fail to compile or run are skipped, not fatal --
    a shape/warp combination being illegal on sm_120 is information about that config, not about
    the dtype, and killing the run would lose every other measurement."""
    best = (float("inf"), None)
    for cfg in CFGS:
        try:
            run(a, b, cfg, acc_dtype)
            t = bench(lambda: run(a, b, cfg, acc_dtype), iters=iters)
        except Exception as e:                       # noqa: BLE001 -- see docstring
            print(f"      cfg {cfg}: SKIP ({type(e).__name__})")
            continue
        print(f"      cfg {cfg}: {t:.4f} ms")
        if t < best[0]:
            best = (t, cfg)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=576)        # SmolVLM-256M hidden size
    ap.add_argument("--N", type=int, default=49280)      # vocab -> lm_head output
    ap.add_argument("--T", type=int, nargs="+", default=[64, 128, 576])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    a_ = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(a_.seed)
    print("=" * 78)
    print(f"vlm_step015 int8 kernel  torch={torch.__version__}  triton={triton.__version__}  "
          f"cap={torch.cuda.get_device_capability()}  {torch.cuda.get_device_name()}")
    print(f"  shape [T,{a_.K}] @ [{a_.K},{a_.N}]   weight bytes: fp16 "
          f"{a_.K * a_.N * 2 / 1e6:.1f} MB  int8 {a_.K * a_.N / 1e6:.1f} MB")

    w8 = torch.randint(-127, 127, (a_.K, a_.N), dtype=torch.int8, device=dev)
    w16 = torch.randn(a_.K, a_.N, dtype=torch.float16, device=dev)
    rows = []
    for T in a_.T:
        print(f"\n  --- T={T} ---")
        x8 = torch.randint(-127, 127, (T, a_.K), dtype=torch.int8, device=dev)
        x16 = torch.randn(T, a_.K, dtype=torch.float16, device=dev)

        cu16 = bench(lambda: torch.mm(x16, w16), iters=a_.iters)
        try:
            im = bench(lambda: torch._int_mm(x8, w8), iters=a_.iters)
        except Exception as e:                       # noqa: BLE001
            im = float("nan")
            print(f"    torch._int_mm unavailable: {type(e).__name__}")
        print(f"    cuBLAS fp16 {cu16:.4f} ms   torch._int_mm {im:.4f} ms")

        print("    triton fp16:")
        t16, c16 = best_cfg(x16, w16, tl.float32, a_.iters)
        print("    triton int8:")
        t8, c8 = best_cfg(x8, w8, tl.int32, a_.iters)

        # Correctness gate -- exact integer arithmetic on both sides, so equality is the right test.
        gate = "n/a"
        if c8 is not None:
            ref = torch._int_mm(x8, w8)
            gate = "EXACT" if torch.equal(run(x8, w8, c8, tl.int32), ref) else "MISMATCH"
            print(f"    int8 correctness vs torch._int_mm: {gate}")

        rows.append((T, cu16, im, t16, t8, c16, c8, gate))

    print("\n" + "=" * 78)
    print(f"{'T':>5} {'cuBLAS16':>9} {'_int_mm':>9} {'tri16':>9} {'tri8':>9} "
          f"{'8/16 tri':>9} {'8/cuBLAS':>9} {'gate':>9}")
    for T, cu16, im, t16, t8, c16, c8, gate in rows:
        print(f"{T:>5} {cu16:>9.4f} {im:>9.4f} {t16:>9.4f} {t8:>9.4f} "
              f"{t16 / t8:>9.2f}x {cu16 / t8:>9.2f}x {gate:>9}")
    print("  8/16 tri = one-variable dtype effect (identical kernel).  "
          "8/cuBLAS = what a deployment would actually gain.")
    print("  step014 showed the memory system offers 2.00x. Anything well under that in the "
          "'8/16 tri' column is kernel/occupancy, not bandwidth.")


if __name__ == "__main__":
    main()
