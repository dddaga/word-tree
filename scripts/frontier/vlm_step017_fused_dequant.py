"""vlm_step017 -- does fusing the dequant scale into the epilogue recover the traffic step015 paid?

step015 got 1.17-1.98x on the SmolVLM `lm_head` shape with a Triton int8 GEMM, but it reported its
own number as a FLOOR and named why: the kernel writes an **int32** output, 4 bytes/element against
the fp16 baseline's 2. At T=576 that is 113.5 MB written where fp16 writes 56.7 MB, so the int8 arm
wins 1.55x while moving MORE total bytes than the thing it beats. A deployment does not do that --
it multiplies by the per-token and per-column scales inside the kernel and writes fp16.

This measures that. The claim under test is narrow and falsifiable: fusing the dequant should save
~56.7 MB of write traffic at T=576 and therefore beat the int32-output kernel, on a shape step014
showed is memory-bound.

THE CONTROL IS AGAIN THE POINT, and there are two of them.
  1. `_gemm_i32` is step015's kernel verbatim -- same tiling, same loop, same autotune grid. The
     fused kernel differs ONLY in its epilogue, so `deq vs i32` is a one-variable measurement of
     fusion and nothing else. It also has to reproduce step015's numbers, which is a free drift gate
     across two separate sessions on the same card.
  2. `i32 + separate dequant` is the honest unfused baseline. Comparing a fused kernel against a
     GEMM that never dequantizes at all would credit fusion with work the unfused path also has to
     do; the comparison that decides deployment is fused-total vs unfused-total.

Correctness gate before any timing is read: the fused fp16 output must match a reference built as
`torch._int_mm(...).float() * s_a * s_w` cast to fp16. Both sides accumulate exactly in int32 and
scale in fp32, so the only admissible difference is fp16 rounding of the final store; the gate
reports max abs and max rel error rather than asserting a tolerance nobody derived.
"""
from __future__ import annotations
import argparse
import time

import torch
import triton
import triton.language as tl

# Identical to vlm_step015's list. M is tiny (T <= 576) and N is huge (49280), so useful configs are
# small-BM/large-BN. Kept the same so the i32 control is comparable cell-for-cell with step015.
CFGS = [(16, 128, 64, 4, 4), (16, 256, 64, 4, 8), (32, 128, 64, 4, 4), (32, 256, 64, 4, 8),
        (64, 128, 64, 4, 4), (64, 256, 64, 4, 8), (64, 128, 128, 3, 4), (128, 128, 64, 4, 8)]
nan = float("nan")


@triton.jit
def _gemm(A, B, C, SA, SW, M, N, K, sam, sak, sbk, sbn, scm, scn,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, FUSE: tl.constexpr):
    """ONE kernel for both arms; FUSE switches only the epilogue, so the main loop is not merely
    equivalent between arms but literally the same code. FUSE=0 is step015's int32 store. FUSE=1
    multiplies by the PER-TOKEN activation scale SA (one per row of A) and the PER-ROW weight scale
    SW -- which lands on the COLUMNS of B here, B being the transposed `lm_head` weight. That is
    exactly `vlm_quant.ARMS['w8a8_row']` as measured in step016, so this kernel and that accuracy
    number describe one scheme rather than two."""
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptr = A + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptr = B + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0)
        b = tl.load(b_ptr, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0)
        acc += tl.dot(a, b, out_dtype=tl.int32)
        a_ptr += BK * sak
        b_ptr += BK * sbk
    c = C + offs_m[:, None] * scm + offs_n[None, :] * scn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if FUSE:
        sa = tl.load(SA + offs_m, mask=offs_m < M, other=0.0)
        sw = tl.load(SW + offs_n, mask=offs_n < N, other=0.0)
        tl.store(c, (acc.to(tl.float32) * sa[:, None] * sw[None, :]).to(tl.float16), mask=mask)
    else:
        tl.store(c, acc, mask=mask)


def run(a, b, sa, sw, cfg, fuse):
    BM, BN, BK, stages, warps = cfg
    M, K = a.shape
    N = b.shape[1]
    out = torch.empty((M, N), dtype=torch.float16 if fuse else torch.int32, device=a.device)
    _gemm[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a, b, out, sa, sw, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), BM=BM, BN=BN, BK=BK, FUSE=fuse,
        num_stages=stages, num_warps=warps)
    return out


def bench(fn, iters=50, warmup=10):
    """Median-of-iters wall time in ms -- median for the same reason as step015: one scheduler
    hiccup on a 0.15 ms kernel moves a mean by the size of the effect being measured."""
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


def best_cfg(call, iters):
    """Smallest median over CFGS. A config that fails to compile is skipped, not fatal: an illegal
    shape/warp combination on sm_120 is information about that config, not about the epilogue."""
    best = (float("inf"), None)
    for cfg in CFGS:
        try:
            call(cfg)
            t = bench(lambda: call(cfg), iters=iters)
        except Exception as e:                        # noqa: BLE001 -- see docstring
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
    ap.add_argument("--device", default="cuda")   # launch_slot.sh always passes this
    a_ = ap.parse_args()

    dev = torch.device(a_.device)
    torch.manual_seed(a_.seed)
    torch.backends.cuda.matmul.allow_tf32 = False   # reference exactness must not depend on a flag
    print("=" * 82)
    print(f"vlm_step017 fused dequant  torch={torch.__version__}  triton={triton.__version__}  "
          f"cap={torch.cuda.get_device_capability()}  {torch.cuda.get_device_name()}")
    print(f"  shape [T,{a_.K}] @ [{a_.K},{a_.N}]  out bytes/elem: i32 4  fp16 2")

    w8 = torch.randint(-127, 127, (a_.K, a_.N), dtype=torch.int8, device=dev)
    w16 = torch.randn(a_.K, a_.N, dtype=torch.float16, device=dev)
    sw = (torch.rand(a_.N, device=dev) * 0.01 + 0.001).float()
    rows = []
    for T in a_.T:
        print(f"\n  --- T={T} ---")
        x8 = torch.randint(-127, 127, (T, a_.K), dtype=torch.int8, device=dev)
        x16 = torch.randn(T, a_.K, dtype=torch.float16, device=dev)
        sa = (torch.rand(T, device=dev) * 0.01 + 0.001).float()

        cu16 = bench(lambda: torch.mm(x16, w16), iters=a_.iters)
        print(f"    cuBLAS fp16 {cu16:.4f} ms")
        print("    triton int8 -> int32 (step015 control):")
        ti, ci = best_cfg(lambda c: run(x8, w8, sa, sw, c, 0), a_.iters)
        print("    triton int8 -> fp16 (fused dequant):")
        td, cd = best_cfg(lambda c: run(x8, w8, sa, sw, c, 1), a_.iters)

        # Unfused TOTAL: the GEMM plus the separate scale pass a deployment would otherwise run.
        unf = bench(lambda: (run(x8, w8, sa, sw, ci, 0).float()
                             * sa[:, None] * sw[None, :]).half(), iters=a_.iters) if ci else nan
        gate = "n/a"
        if cd is not None:
            # Reference accumulates in fp32, NOT via torch._int_mm, which refuses M <= 16 and so
            # cannot gate the T=1 case the eval harness actually runs. This is not a loosened
            # tolerance: |sum| <= 127*127*K, so with K=576 every partial sum is an integer below
            # 2**24 and therefore exactly representable in fp32 regardless of accumulation order.
            # The bound is asserted rather than assumed, so a larger K fails loudly instead of
            # silently degrading the gate. One path for every M -- no threshold keyed to the shape
            # that happened to break.
            assert 127 * 127 * a_.K < 2 ** 24, f"K={a_.K}: fp32 reference no longer exact"
            ref = ((x8.float() @ w8.float()) * sa[:, None] * sw[None, :]).half()
            got = run(x8, w8, sa, sw, cd, 1)
            d = (got.float() - ref.float()).abs()
            mx = d.max().item()
            rel = (d / ref.float().abs().clamp(min=1e-6)).max().item()
            gate = "EXACT" if mx == 0.0 else f"max_abs={mx:.3e} max_rel={rel:.3e}"
            print(f"    fused correctness vs _int_mm+scale: {gate}")
        rows.append((T, cu16, ti, unf, td, ci, cd, gate))

    print("\n" + "=" * 82)
    print(f"{'T':>5} {'cuBLAS16':>9} {'tri_i32':>9} {'unfused':>9} {'fused':>9} "
          f"{'fus/i32':>8} {'fus/unf':>8} {'fus/cuB':>8}")
    for T, cu16, ti, unf, td, ci, cd, gate in rows:
        print(f"{T:>5} {cu16:>9.4f} {ti:>9.4f} {unf:>9.4f} {td:>9.4f} "
              f"{ti / td:>7.2f}x {unf / td:>7.2f}x {cu16 / td:>7.2f}x")
    for T, cu16, ti, unf, td, ci, cd, gate in rows:
        print(f"  T={T:<4} best cfg i32={ci} deq={cd}  gate: {gate}")
    print("  fus/i32 = the one-variable effect of the epilogue (identical main loop).")
    print("  fus/unf = what a deployment gains by fusing rather than scaling in a second pass.")
    print("  fus/cuB = the number a deployment quotes: int8+dequant vs the fp16 GEMM it replaces.")
    print("  tri_i32 must reproduce step015's column (0.1012 / 0.1519 / 0.4458 at T=64/128/576) --")
    print("  a free cross-session drift gate on the same card, same kernel, same configs.")


if __name__ == "__main__":
    main()
