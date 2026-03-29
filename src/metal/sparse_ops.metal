/*
 * sparse_ops.metal
 * Sparse seeding kernels for SGNNET_Wave C_input operation.
 *
 * Core operation:
 *   Z[b, h, d] = sum over i where C[i,h]=1  of  A[b, i, d]
 *   A: [B, N_in, D]   conn_idx: [N_h, K]   Z: [B, N_h, D]
 *
 * Three kernels with increasing sophistication:
 *   1. naive_fanin        — one thread per (h, d), serial loop over K
 *   2. simd_fanin         — SIMD-group (warp) cooperatively reduces K
 *   3. tiled_block_fanin  — threadgroup shared memory tiles the K loop
 *
 * Apple Silicon context
 * ─────────────────────
 * SIMD group size  : 32 threads  (equivalent to CUDA warp)
 * Threadgroup mem  : 32 KB       (fast, on-chip)
 * AMX units        : handle dense GEMM natively (what PyTorch uses)
 * GPU cores (M2Max): 38 — each runs many threadgroups concurrently
 *
 * Why custom kernels can win over PyTorch dense GEMM here
 * ────────────────────────────────────────────────────────
 * Dense GEMM reads ALL N_in*N_h entries from DRAM even though 90% are zero.
 * With fixed fan-in K: only K*N_h entries are ever read from A_input.
 * At K=100, N_h=256: 25,600 reads vs 6.4M for dense → 250x less data movement.
 * Bandwidth is the bottleneck on MPS, not compute.
 */

#include <metal_stdlib>
using namespace metal;

// ─── constants baked in at pipeline creation time ───────────────
constant uint B     [[function_constant(0)]];   // batch size
constant uint N_H   [[function_constant(1)]];   // hidden neurons
constant uint K     [[function_constant(2)]];   // connections per neuron
constant uint D     [[function_constant(3)]];   // geometric dims (4)
constant uint N_IN  [[function_constant(4)]];   // input neurons (25088)


// ════════════════════════════════════════════════════════════════
// Kernel 1: Naive fixed fan-in
// ─────────────────────────────
// Grid:  [N_h, D, B]  — one thread per output element
// Each thread owns one (h, d, b) triple, loops over K connections.
//
// Optimization: none beyond avoiding the zero-multiplications.
// Good baseline — simple to reason about, correct.
// ════════════════════════════════════════════════════════════════
kernel void naive_fanin(
    device const float* A          [[buffer(0)]],  // [B, N_in, D]
    device const uint*  conn_idx   [[buffer(1)]],  // [N_h, K]
    device       float* Z          [[buffer(2)]],  // [B, N_h, D]
    uint3 gid [[thread_position_in_grid]]
) {
    uint h = gid.x;   // hidden neuron index
    uint d = gid.y;   // dimension index
    uint b = gid.z;   // batch index

    if (h >= N_H || d >= D || b >= B) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        uint src = conn_idx[h * K + k];          // which input neuron
        acc += A[b * N_IN * D + src * D + d];    // A[b, src, d]
    }
    Z[b * N_H * D + h * D + d] = acc;
}


// ════════════════════════════════════════════════════════════════
// Kernel 2: SIMD-group cooperative fan-in
// ────────────────────────────────────────
// Grid:  [N_h * SIMD_SIZE, D, B]
// Threadgroup: [SIMD_SIZE, 1, 1]  (SIMD_SIZE = 32)
//
// Key idea: one SIMD group (32 threads) handles one hidden neuron h.
// Each thread handles K/32 of the K connections, then simd_sum
// reduces across the group in a single instruction — no atomics.
//
// Optimization: reduces K loop length by 32×.
// simd_sum is a single hardware instruction on Apple Silicon.
// ════════════════════════════════════════════════════════════════
constant uint SIMD_SIZE = 32;

kernel void simd_fanin(
    device const float* A          [[buffer(0)]],
    device const uint*  conn_idx   [[buffer(1)]],
    device       float* Z          [[buffer(2)]],
    uint3 gid  [[thread_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]]
) {
    uint h = gid.x / SIMD_SIZE;   // which hidden neuron
    uint d = gid.y;
    uint b = gid.z;

    if (h >= N_H || d >= D || b >= B) return;

    // Each of 32 lanes handles K/32 connections
    float partial = 0.0f;
    for (uint k = simd_lane; k < K; k += SIMD_SIZE) {
        uint src = conn_idx[h * K + k];
        partial += A[b * N_IN * D + src * D + d];
    }

    // Hardware warp-reduction: single instruction, no shared memory needed
    float total = simd_sum(partial);

    // Only lane 0 writes the result
    if (simd_lane == 0) {
        Z[b * N_H * D + h * D + d] = total;
    }
}


// ════════════════════════════════════════════════════════════════
// Kernel 3: Tiled block fan-in with threadgroup (shared) memory
// ──────────────────────────────────────────────────────────────
// Threadgroup: [TILE_K, D, 1]   — TILE_K threads share one hidden neuron
// Grid:        [N_h * TILE_K, D, B]
//
// Two-level tiling:
//  Outer: one threadgroup per (h, b) — threads share the same conn_idx row
//  Inner: threads split the K connections, load into threadgroup memory,
//         then tree-reduce in shared memory (log2(TILE_K) steps)
//
// Why this helps over simd_fanin:
//  - conn_idx[h, :] is loaded once into registers shared across the group
//  - A_input values are fetched with unit-stride in d (coalesced)
//  - Tree reduction in threadgroup memory avoids register spilling for large K
// ════════════════════════════════════════════════════════════════
constant uint TILE_K = 64;   // threads per hidden neuron in K dimension

kernel void tiled_block_fanin(
    device const float*  A          [[buffer(0)]],
    device const uint*   conn_idx   [[buffer(1)]],
    device       float*  Z          [[buffer(2)]],
    uint3  gid   [[thread_position_in_grid]],
    uint3  tid   [[thread_position_in_threadgroup]],
    uint3  tgs   [[threads_per_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]]  // [TILE_K * D] floats
) {
    uint h      = gid.x / TILE_K;
    uint k_lane = tid.x;     // which slice of K this thread handles
    uint d      = gid.y;
    uint b      = gid.z;

    if (h >= N_H || d >= D || b >= B) return;

    // Phase 1: each thread accumulates its K/TILE_K connections
    float partial = 0.0f;
    for (uint k = k_lane; k < K; k += TILE_K) {
        uint src = conn_idx[h * K + k];
        partial += A[b * N_IN * D + src * D + d];
    }

    // Store partial sum in threadgroup memory (indexed by [k_lane, d])
    scratch[k_lane * D + d] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: tree reduction in shared memory (log2(TILE_K) steps)
    for (uint stride = TILE_K / 2; stride > 0; stride >>= 1) {
        if (k_lane < stride) {
            scratch[k_lane * D + d] += scratch[(k_lane + stride) * D + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes final result
    if (k_lane == 0) {
        Z[b * N_H * D + h * D + d] = scratch[d];
    }
}


// ════════════════════════════════════════════════════════════════
// Kernel 4: Block-sparse seeding (for SGNNET_BlockWave architecture)
// ────────────────────────────────────────────────────────────────
// Exploits block-diagonal C_input: input group g → hidden group g only.
// No conn_idx needed — connectivity is implicit from block structure.
//
// Grid:  [N_h, D, B]
// Each thread (h, d, b):
//  - Determines its input block:  g = h / H_PER_GROUP
//  - Sums over [g * I_PER_GROUP .. (g+1) * I_PER_GROUP] with mask
//
// This is the architecture discussed (block-diagonal + C_hh cross-mixing).
// Completely avoids loading any zero-block inputs.
// ════════════════════════════════════════════════════════════════
constant uint H_PER_GROUP = 8;                // hidden neurons per block
constant uint I_PER_GROUP = N_IN / (N_H / H_PER_GROUP);  // ~784 inputs/block

kernel void block_sparse_seed(
    device const float*  A     [[buffer(0)]],   // [B, N_in, D]
    device const bool*   C_blk [[buffer(1)]],   // [I_PER_GROUP, H_PER_GROUP] per group
    device       float*  Z     [[buffer(2)]],   // [B, N_h, D]
    uint3 gid [[thread_position_in_grid]]
) {
    uint h = gid.x;
    uint d = gid.y;
    uint b = gid.z;

    if (h >= N_H || d >= D || b >= B) return;

    uint g       = h / H_PER_GROUP;              // which block group
    uint h_local = h % H_PER_GROUP;              // position within group
    uint i_start = g * I_PER_GROUP;

    float acc = 0.0f;
    for (uint i = 0; i < I_PER_GROUP; ++i) {
        // C_blk is per-group connectivity [I_PER_GROUP, H_PER_GROUP]
        if (C_blk[i * H_PER_GROUP + h_local]) {
            acc += A[b * N_IN * D + (i_start + i) * D + d];
        }
    }
    Z[b * N_H * D + h * D + d] = acc;
}
