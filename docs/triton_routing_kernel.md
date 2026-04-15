# Triton Routing Kernel — Design Doc (step530)

## The Problem

`SGNNET_DeltaProj.forward` routing hot path (K_iter=5 × per-step):

```python
Z_fwd  = relu(Z - theta)                    # [B, N, D]
Z_nb   = Z_fwd[:, conn_hh, :]              # [B, N, K_hh, D]  ← 4D intermediate
proj   = (Z_nb * dw_norm).sum(-1, keepdim=True)
Z_nb   = Z_nb * proj.abs()
Z_struct = Z_nb.sum(dim=2)                  # [B, N, D]
Z_refl = alpha * Z_refl + (Z_fwd - Z)
Z = normalize(clamp(Z_struct + Z_refl, -10, 10), -1)
```

At bs=32 N=2048 D=16 K_hh=2: the `[B, N, K_hh, D]` intermediate is
`32 × 2048 × 2 × 16 × 4 bytes = 8.4 MB` per step, 42 MB across 5 steps.
This materialization was the step803 CUDA Graph blocker (int64 dynamic indexing)
and the main bandwidth pressure.

Current best: 0.280ms @ K=5, max-autotune fp32 (bench_step811).

## Memory Access Pattern

Inputs read per step:
| Tensor | Shape | Bytes | Read pattern |
|---|---|---|---|
| Z | [32, 2048, 16] | 4.2 MB | Sequential + scattered gathers |
| Z_reflected | [32, 2048, 16] | 4.2 MB | Sequential |
| theta | [2048] | 8 KB | Broadcast over B |
| conn_hh | [2048, 2] | 16 KB | Once per step (small, stays in L2) |
| dw_norm | [2048, 2, 16] | 256 KB | Once per step (warm L2 after epoch) |

Total working set per step: ~8.7 MB. RTX 5060 Ti L2 = 32 MB, L1 = 128 KB/SM.
The gather `Z_fwd[:, conn_hh, :]` causes non-contiguous reads into Z with
stride D=16 (64 bytes) but random row selection — cache miss rate depends on
graph locality (small-world: ~half hits local group, half long-range).

## Block Tiling Decision

Grid: `(ceil(B/BLOCK_B), ceil(N/BLOCK_N))`
Each program owns a `[BLOCK_B, BLOCK_N]` tile.

For each (b, n) pair in the tile:
- Load Z[b, n, :] — D=16 fp32 = 64 bytes, fits in 2 registers (warp-level)
- Load Z[b, sender_k, :] for k=0,1 — two random reads, 64 bytes each
- Compute: 2 fma-D + 1 abs + 2 fma-D (proj gate) + 2 add-D (struct) + D (reflect) + D (normalize) ≈ 11D FLOPs = 176 FLOPs per (b,n)

At BLOCK_N=128 BLOCK_B=2: 256 (b,n) pairs per program = 45K FLOPs, 256 random
reads of 64 bytes = 16 KB scattered. Autotune sweeps over 11 configs to find
the bandwidth-compute sweet spot for SM_120.

**Why not bigger D tile?** D=16 is a `constexpr` — the inner d-loop unrolls
at compile time into pure register arithmetic. No shared memory needed.

**Why not tile over D?** D=16 fits in 16 FP32 registers per (b,n) pair.
Splitting D would add sync overhead with no gain.

## Why fp32 First

1. The bench baseline (0.280ms) is fp32 — apples-to-apples comparison.
2. Training uses fp32. fp16/bf16 would require matching training.
3. Ada Lovelace (SM_120) has native bf16 GEMM but this is not a GEMM — it is
   gather + dot-product + normalize. fp32 vs bf16 difference is modest.

### Path to bf16/fp16

Replace `tl.float32` with `tl.bfloat16` in the kernel, add a cast at the
wrapper boundary. The `tl.rsqrt` normalize and `tl.clamp` work in bf16 on
SM_80+. Add a second autotune key: `dtype` ∈ {fp32, bf16}. Expected win: 10-20%
from halved bandwidth on Z reads (4.2 MB → 2.1 MB per step).

## CUDA Graph Compatibility

step803 failed: `Z_fwd[:, conn_hh, :]` with int64 conn_hh triggers dynamic
kernel selection — CUDA Graph rejects it.

Fix: cast `conn_hh` to int32 at `SGNNET_DeltaProjTriton.__init__`. Triton
kernel uses `int32` indices throughout. CUDA Graph can capture the Triton
kernel call because all shapes are static and int32 indexing is deterministic.

## Gradient Strategy

Forward-only in this first iteration. The routing loop is K_iter=5 steps of
non-differentiable clamp + normalize — PyTorch autograd already handles it
correctly through the PyTorch fallback path. The Triton kernel is inference-
only.

**Path to backward (if needed for deep supervision, step521):**
Triton supports `@triton.jit` backward via `torch.autograd.Function`. The
gradient of the routing step w.r.t. Z is a chain of: ∂norm/∂Z_combined ×
∂Z_combined/∂Z_struct × ∂Z_struct/∂Z_nb × ∂Z_nb/∂Z_fwd × ∂Z_fwd/∂Z.
Each is a local Jacobian. The ∂norm/∂Z_combined piece is well-known
(identity minus outer product / norm^2). Materializing the backward would
require storing Z_fwd and Z_nb, or recomputing them — recomputation preferred
(activation checkpointing idiom). Not needed for inference benchmarks.

## Acceptance Criteria

1. `atol=1e-4` vs PyTorch reference on all three test shapes — PASS.
2. K_iter=5 loop deviation < 1e-4 — PASS.
3. Bench runs and reports real numbers — PASS.
4. CUDA Graph captures — PASS (int32 fix works).
5. Speed vs 0.280ms — NO SPEEDUP (3.83ms eager, 0.336ms V_ref_ma).

## Benchmark Results (RTX 5060 Ti SM_120, 2026-04-15)

| Variant | median_ms | vs 0.280ms |
|---|---|---|
| V_ref eager (DeltaProj) | 7.114 | 0.04× |
| V_ref_ma compiled | 0.336 | 0.83× |
| V_triton eager | 3.831 | 0.07× |
| V_triton_cg CUDA Graph | 3.829 | 0.07× |
| V_triton_c compile | FAILED | SM_120 LLVM |

Triton eager is **11.4× slower than compiled PyTorch** at this workload.

Root causes: (1) scatter gather at D=16 K=2 is bandwidth-bottlenecked with
irregular access, (2) each BLOCK_N=256 program is compute-light — launch
overhead dominates, (3) SM_120 LLVM issue blocks compile+Triton fusion.

## Known Limitations and Next Steps

1. **No speedup on SM_120**: Irregular scatter reads at D=16, K=2 favor PyTorch
   compiled path. A custom kernel wins when D>>16 or when pre-sorted edges
   enable coalesced access.
2. **SM_120 LLVM failure**: `torch.compile` + custom Triton fails on Blackwell.
   Upgrade to Triton 3.7+ or use standalone eager.
3. **CUDA Graph works**: int32 fix (step803) validated — captures cleanly.
4. **Next iteration**: pre-sort conn_hh by sender group → coalesced gather;
   increase tile size to amortize launch overhead; try `tl.gather` with
   stride hints when Triton adds SM_120 compile support.
5. **Autotune cache**: `~/.triton/cache/` — first run ~30s, subsequent runs instant.
