# SGNNET Wall-Time Deep Diagnosis — RTX 5060 Ti

**Date:** 2026-04-15  
**Scope:** Wall-clock inference latency only — NOT FLOPs, NOT params.  
**Hardware:** RTX 5060 Ti (Blackwell SM_120, 16 GB GDDR7, CUDA 12.8, PyTorch 2.11)  
**Note:** Prior Haiku meditation covers the broader efficiency story; this document goes deeper on wall-time mechanics.

---

## 1. Current Wall-Time State

### Validated Numbers (bench_step810/811, 5060ti, bs=32)

| Variant | Median (ms) | vs VGG_FC | vs Linear |
|---|---|---|---|
| VGG_FC (baseline) | 1.570 | 1.00× | 17.6× slower |
| SGNNET_AH K=5 eager fp32 | ~0.53 | ~3× | |
| SGNNET_AH K=5 reduce-overhead fp32 | 0.299 | 5.25× | |
| **SGNNET_AH K=5 max-autotune fp32 (V2)** | **0.280** | **5.60×** | **3.15× slower** |
| SGNNET_AH K=5 CUDA Graph (V5) | 2.212 | 0.71× SLOWER | — |
| Linear (25088→10) | 0.089 | 17.6× faster | 1.00× |
| V3 fp16 | FAILED | mixed-dtype internal | — |
| V4 bf16 | FAILED | mixed-dtype internal | — |

**K=4 vs K=5 (bench_step830, ran today):**  
- K4_ma = 0.263ms, K5_ma = 0.278ms → K=4 is only **5.6% faster** (not 20% projected)
- Paper claim must be revised to 5.6%, not 20%.

**GPU utilization anomaly (step802 diagnostic):**
- fp32 compile: ~99% GPU utilization — compute-bound
- bf16/fp16: ~12.7% GPU utilization — memory-bandwidth-bound (kernel-launch overhead dominates)

**CUDA Graph (V5) catastrophic failure:** 2.212ms — 8× SLOWER than V2. Root cause: the fancy-index `Z[:, conn_hh, :]` (conn_hh int64 [N, K_hh]) is a non-capturable dynamic-indexing op that forces CPU–GPU sync on every replay, destroying the entire point of graph capture.

**The gap to target:** Linear is 0.089ms. SGNNET is 0.280ms. To match Linear we need 3.15× improvement. Realistic target: **0.100–0.150ms** (2–3× improvement via Triton kernel).

---

## 2. Theoretical Floor Analysis

### Hardware Constants (RTX 5060 Ti, Blackwell, GDDR7 128-bit)
- Memory bandwidth: **448 GB/s** (GDDR7 at 28 Gbps × 128-bit / 8)
- Peak fp32 TFLOPS: ~**23.2 TFLOPS** (4,608 CUDA cores × 2 × 2.52 GHz)
- Peak bf16/fp16 Tensor Core TFLOPS: ~**185 TFLOPS** (5th-gen tensor cores, dense)
- Kernel launch overhead (CUDA): ~2–5 µs per kernel on modern drivers

### Data Volume Per K_iter Step (fp32, bs=32, N=2048, D=16, K_hh=2)

**Routing hot path per iteration:**
```
Z_fwd = F.relu(Z - theta)               # reads Z[B,N,D]: 32×2048×16×4B = 4.19 MB
Z_nb  = Z_fwd[:, conn_hh, :]            # reads Z_fwd again + conn_hh: ~4.19 + 0.016 MB
proj  = (Z_nb * dw_norm).sum(-1, keepdim=True)  # reads Z_nb[B,N,K,D]: 8.39 MB
Z_nb *= proj.abs()                      # read+write Z_nb: 8.39 MB
Z_struct = Z_nb.sum(dim=2)              # reduction: 8.39 MB → 4.19 MB written
Z_reflected update + clamp + F.normalize # ~12.57 MB read + write
```

Rough total per-iteration data movement: ~**50 MB** (including all reads + writes of intermediates)  
Across K_iter=5: ~**250 MB** per forward pass  
Plus seed gather (_seed): conn_in gather [B,N,K_in,D] = 32×2048×25×16×4B ≈ **105 MB**  
Readout: negligible  
**Total estimate: ~355 MB per forward pass**

**Memory-bandwidth floor:**  
355 MB ÷ 448 GB/s = **0.79 ms** theoretical minimum IF purely memory-bound and perfectly sequential.

**But:** Modern GPUs overlap reads/writes and have ~900 GB/s effective bandwidth for cached patterns (L2 cache reuse). Assuming 2× effective BW from streaming:  
355 MB ÷ 896 GB/s ≈ **0.40 ms** theoretical floor for current tensor decomposition.

**Current 0.280ms is below this estimate.** This means either:
1. The actual data volume is smaller (intermediates reused in cache between ops)
2. torch.compile's max-autotune is already kernel-fusing some of these
3. D=16 is tiny enough that Z fits in L2 (Z size: 32×2048×16×4B = 4.19 MB; L2 cache RTX 5060 Ti ≈ 40 MB → **Z fits entirely in L2 cache**)

**Revised floor with L2-cached Z:**  
If Z[B,N,D] stays in L2, per-iteration cost is dominated by:
- conn_hh gather: 2048×2×4B (int32) = 0.016 MB (negligible)
- dw_norm read: 2048×2×16×4B = 0.26 MB (fits in L1)
- Actual compute: 2048×2×16×2 ops × 5 iter × 32 batch = 5.24M ops

**Compute floor:** 5.24M × 32 ops per fwd ÷ 23.2 TFLOPS = **0.007 µs** (vanishingly small)  
**Real bottleneck:** Not compute, not memory bandwidth — it's **kernel launch overhead + Python dispatch overhead**.

### Why K=4 Is Only 5.6% Faster (NOT 20%)

At 0.280ms, each K_iter step contributes ~0.280/5 = 0.056ms if perfectly linear. K=4 saves 1 step → theoretical 20% = 0.056ms savings → target 0.224ms.

**Actual savings: 0.015ms (0.280 → 0.263ms). Why?**

**Hypothesis ranking (H1 = most likely):**

| H | Hypothesis | Evidence | Falsifier |
|---|---|---|---|
| **H1** | Non-routing overhead is ~0.150ms (seed gather + readout + Python) and routing overhead is only ~0.130ms; removing 1/5 of routing saves ~0.026ms but compile-time loop unroll partially amortizes it → 0.015ms net savings. | Consistent with L2-cached Z and 5 kernel launches + dispatch. | Profile per-op with ncu; measure seed-only time. |
| **H2** | max-autotune fuses the K_iter loop partially across iterations → loop unrolling means K=4 and K=5 share the same kernel for most steps, with 1 redundant step masked out rather than eliminated. | max-autotune creates a fused loop kernel. K=4 vs K=5 measures show ratio 0.944 not 0.8 → fusion boundary effect. | Re-run bench_step830 with reduce-overhead (no fusion) and compare K=4/K=5 ratio. |
| **H3** | Kernel launch overhead dominates. At ~3µs per kernel × 15 kernels per K_iter step × 5 steps = 225µs = 0.225ms just in launches. | 5060ti shows 99% GPU util → GPU not idle, but CPU dispatch overhead still real. | Add explicit CUDA event timing per kernel. |
| H4 | readout (einsum C_ho) is unexpectedly expensive at N=2048. | C_ho [2048,10] dense bool mask is materialized as float each fwd. | Profile readout separately. |

**Best explanation (H1 + H2 combined):** The K_iter loop accounts for <40% of wall-time. The remaining 60% (seed gather + normalize + readout + all dispatch overhead) is fixed and does not scale with K_iter.

---

## 3. Patterns from Memory + Literature

### From Graphiti
- **V5 CUDA Graph = 2.212ms** (CONFIRMED failure; Graphiti node "V5 CUDA Graph" has this fact)
- **V3 fp16, V4 bf16**: both FAILED due to mixed-dtype internals (conn_hh/conn_in are int64 buffers; they don't get cast with `.to(dtype=bf16)`)
- **step530**: Triton Fused Gather-Multiply-Sum Kernel — planned but NOT yet implemented
- **bench_step830**: K=4 latency measured today — 0.263ms
- **INT8 QAT speedup**: 1.5–2× on tensor cores (Graphiti memory from today's session)
- No prior Triton kernel implementation exists anywhere in memory

### From Literature

**CUDA Graph + fancy index (int64) blocker:**  
PyTorch GitHub Issue #155682 (2025) confirms: CUDA graph capture raises `RuntimeError: operation not permitted when stream is capturing` for dynamic indexing where index tensor values are runtime-determined. The constraint is NOT about int32 vs int64 per se — it's that the op internally syncs to device to get index values.  
Source: [pytorch/pytorch#155682](https://github.com/pytorch/pytorch/issues/155682)

**CUDA Graph workaround — static buffer pattern:**  
NVIDIA's best-practices guide confirms: pre-allocate static placeholder tensors before capture; update with `.copy_()` at replay time (not the index tensor, the *input data* tensor). For index-by-value patterns, the recommended path is to **pre-compute all indexed results outside the graph**, pass them in as static inputs, or convert to a kernel that is inherently static (Triton with compile-time K_hh).  
Source: [NVIDIA CUDA Graph Best Practice](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/handling-dynamic-patterns.html)

**PyGraph (March 2025, arXiv:2503.19779):**  
A compiler framework built atop PyTorch 2.4+ that automatically handles some CUDA-graph-breaking patterns. Requires PyTorch 2.4 (we have 2.11). Achieves 29% improvement over PyTorch2-CG on average, up to 3.36× on XLNET-inference. Does NOT handle input-dependent control flow. **May handle static conn_hh fancy index if treated as a buffer** — but requires PyTorch 2.4+.  
Source: [arXiv:2503.19779](https://arxiv.org/html/2503.19779v2)

**FTC-GNN / Sparse GNN Tensor Core (arXiv:2412.12218):**  
Tested at **D=16 hidden dims** (identical to our config). Speedup vs PyG: **7.10× (GCN)**. Technique: convert sparse adjacency to dense blocks for tensor core GEMM via SGT (Sparse Graph Transformation), fuse into single CUDA+TC kernel. Not directly applicable (different sparsity regime — large graphs with many edges, not N=2048 K_hh=2), but proves D=16 is a valid tensor-core target.  
Source: [arXiv:2412.12218v2](https://arxiv.org/html/2412.12218v2)

**Triton GNN message passing (GitHub triton#472):**  
Core pattern: for each edge, `base_offset = id_source * D`, then load D-dim feature, process, atomic_add to target. The challenge is multi-dimensional pointer arithmetic. For D=16 (small), the recommended approach is to load all 16 dimensions into registers in one block (16 × 4B = 64B — fits in a single cache line).

**torch_scatter vs torch.compile:**  
PyG's own documentation confirms torch.compile **disables** torch_scatter usage (not yet compiler-optimizable). From PyG 2.3+, torch.compile works by substituting pure-PyTorch implementations of scatter_add. This means bench_step832 (V_scat_c = scatter_add + max-autotune) may actually use a different codepath than expected — the custom CUDA kernel gets bypassed.  
Source: [PyG compiled GNN docs](https://pytorch-geometric.readthedocs.io/en/2.4.0/advanced/compile.html)

**RTX 5060 Ti actual specs (confirmed):**  
Blackwell architecture (NOT Ada Lovelace — the task description has an error here). SM_120, GDDR7, **448 GB/s BW**, 23.22 TFLOPS fp32, 5th-gen tensor cores (FP4/FP8/BF16/FP16). Key for wall-time: **BF16 tensor core throughput >> fp32 CUDA-core throughput** (~8× ratio for dense matmul). For small K_hh=2, K_iter=5 scatter operations, tensor cores require explicit tiling to activate.  
Source: [Tom's Hardware RTX 5060 Ti review](https://www.tomshardware.com/pc-components/gpus/nvidia-geforce-rtx-5060-ti-16gb-review)

**INT8 for GNN (QGTC, arXiv:2111.09547; confirmed by Graphiti):**  
INT8 tensor cores require min 16×16 tile sizes for activation. D=16 is exactly at the minimum — beneficial IF the gather + mul + sum can be structured as a 16-wide operation. Expected speedup: 1.5–2× per our earlier analysis. Risk: `scatter_add` does not map to tensor-core INT8 instructions in PyTorch's current backend.

---

## 4. Top 5 Ranked Levers

| Rank | Lever | Current | Target | Effort | Risk | Realistic Gain |
|------|-------|---------|--------|--------|------|----------------|
| **1** | **Triton fused gather+mul+sum kernel (step530)** | 0.280ms | 0.100–0.140ms | High (3–5 days) | Medium | **2–3× (0.140ms)** |
| **2** | **torch_scatter + CUDA Graph (bench_step832 V_scat_cg)** | 0.280ms | 0.150–0.200ms | Low (1 day) | Medium | **1.4–1.9× (0.150ms)** |
| **3** | **fp16/bf16 inference with buffer cast fix** | FAILED | 0.200–0.250ms | Low (½ day) | Low | **1.1–1.4× (0.200ms)** |
| **4** | **INT8 QAT inference on tensor cores (step527)** | 0.280ms | 0.140–0.190ms | Medium (running) | Medium | **1.5–2× IF TC path activated** |
| **5** | **CUDA Graph via int32 index + persistent kernel pattern** | 2.212ms (broken) | 0.100–0.150ms | High (3–4 days) | High | **2–3× IF capture succeeds** |

### Lever 1 Deep-Dive: Triton Fused Gather+Mul+Sum

**What it does:** Replaces the PyTorch chain:
```python
Z_nb  = Z_fwd[:, conn_hh, :]              # [B,N,K_hh,D] — allocates 8.39 MB
proj  = (Z_nb * dw_norm).sum(-1, keepdim=True)
Z_nb *= proj.abs()
Z_struct = Z_nb.sum(dim=2)                # [B,N,D]
```
with a single kernel that loads `Z_fwd[b, conn_hh[n,k], :]` directly into registers, multiplies by `dw_norm[n,k,:]`, accumulates projection coefficient, applies abs-gate, and adds into output `Z_struct[b,n,:]` — never materializing the [B,N,K_hh,D] intermediate.

**Pseudocode for the kernel (D=16, K_hh=2, register-tiled):**
```python
# Grid: [B, N] blocks; each block handles 1 neuron × 1 batch item
# Block size: 32 threads (handle 16 dims with 2 threads each, or 16 threads × 2 K_hh)
@triton.jit
def fused_routing_step(
    Z_fwd_ptr,     # [B, N, D] fp32
    conn_hh_ptr,   # [N, K_hh] int32  ← MUST be int32, not int64
    dw_norm_ptr,   # [N, K_hh, D] fp32
    Z_struct_ptr,  # [B, N, D] fp32 output
    B, N, K_hh, D,
    BLOCK_D: tl.constexpr,  # = 16
    BLOCK_K: tl.constexpr,  # = 2
):
    bid = tl.program_id(0)  # batch index
    nid = tl.program_id(1)  # neuron index
    d_offs = tl.arange(0, BLOCK_D)    # [0..15]
    
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    for k in range(BLOCK_K):                           # unrolled at compile time
        nb_idx = tl.load(conn_hh_ptr + nid * K_hh + k)  # scalar int32
        # Gather: Z_fwd[bid, nb_idx, :]
        z_nb = tl.load(Z_fwd_ptr + bid*N*D + nb_idx*D + d_offs)  # [D] in registers
        # dw_norm[nid, k, :]
        dw = tl.load(dw_norm_ptr + nid*K_hh*D + k*D + d_offs)   # [D] in registers
        # projection coefficient (scalar)
        proj = tl.sum(z_nb * dw, axis=0)                          # scalar
        # gated contribution
        acc += z_nb * tl.abs(proj)                                 # [D]
    
    # Write result
    tl.store(Z_struct_ptr + bid*N*D + nid*D + d_offs, acc)
```
- Grid: B×N = 32×2048 = 65,536 blocks — good occupancy for RTX 5060 Ti (36 SMs)
- Registers: ~5 float[16] arrays = 80 registers per thread — fits within register budget
- Memory: No intermediate [B,N,K_hh,D] tensor; saves 8.39 MB alloc per iteration
- **Critical: conn_hh must be int32** (not int64) for CUDA graph compatibility and for Triton pointer arithmetic

**Why 2–3× is realistic:** The current PyTorch path launches ~5 separate kernels for the mul+sum chain. Each launch costs ~3µs on 5060ti. Fusing to 1 kernel saves ~12µs per iteration × 5 iterations = 60µs = 0.060ms. Plus: eliminating the [B,N,K_hh,D] intermediate cuts memory allocation overhead. Combined: 0.280ms → ~0.120–0.150ms.

**What could go wrong:**
- Atomic race if multiple neurons share a neighbor (Z_struct accumulation); our routing writes to unique Z_struct[n] per block, so no race — this is safe
- Triton compiler on SM_120 (Blackwell) may not have full optimization — NVIDIA's Triton support for SM_120 was added in late 2025; test with `triton.compiler.compile(..., target='cuda:12.8')`
- The loop wrapping K_iter (5 calls to this kernel) still costs 5 kernel launches; to eliminate, fuse all K_iter into one mega-kernel using grid synchronization (`triton.language.sync_threads` not sufficient across blocks → need separate approach)

### Lever 2 Deep-Dive: torch_scatter + CUDA Graph

The scatter_add path in bench_step832 restructures routing as edge-list [E, D] (E=N×K_hh=4096 edges). Key facts:
1. **torch.compile disables torch_scatter** per PyG docs — V_scat_c will silently fall back to PyTorch's own scatter implementation
2. **V_scat_cg (CUDA Graph)** may succeed where V5 (step811) failed, because scatter_add uses *a static integer index tensor* (`edge_recv`, precomputed int64) that doesn't change across forward passes — unlike the original fancy index which involves dynamic index computation
3. **If CUDA Graph capture succeeds on scatter_add**: eliminate kernel launch overhead per K_iter → estimated 0.060ms saved → target ~0.200ms
4. **Risk**: bench_step832 not yet run. May also fail with `cudaErrorStreamCaptureInvalidated`

**int32 conversion as CUDA-graph enabler:**  
The static buffer pattern (NVIDIA docs) requires that ALL tensors inside the captured region have stable memory addresses. `edge_send` and `edge_recv` are int64 but their *values* don't change — the issue is whether CUDA graph replays them correctly. Converting to int32 reduces tensor size by 2× (edge_recv: 2048×2×4B = 16 KB), which is below L1 cache on 5060ti (128 KB per SM). This is the most promising CUDA-graph path.

### Lever 3: fp16/bf16 with Buffer Cast Fix

The failure mode (Graphiti confirmed): `V3 fp16` and `V4 bf16` failed due to mixed-dtype internals. `conn_hh` and `conn_in` are int64 buffers; they don't participate in dtype cast, but the model has fp32 buffers (spatial_coords, C_ho_mask) that need explicit casting.

Fix: add `model = model.to(dtype=torch.float16)` then manually cast all remaining fp32 buffers: `model.base.spatial_coords = model.base.spatial_coords.half()`. Expected speedup: modest (~1.2×) at D=16 because D=16 < 32 (minimum SIMD width for fp16 on tensor cores). The GPU util anomaly at bf16 (12.7%) is a red herring — it was measured during TRAINING (GradScaler overhead), not inference-only.

**IMPORTANT:** At D=16 with K_hh=2 and bs=32, the actual matmul shapes are tiny. Tensor cores require ≥16×16 tiles. Our effective matmul per K_iter step has inner dim D=16 (exactly at boundary). fp16 may not activate tensor cores at all for K_hh=2 sparse scatter — it'll just be fp16 CUDA cores, giving ~1.2–1.4× at best.

---

## 5. Concrete Experiment Proposals

### step840: Triton Fused Routing Kernel (HIGHEST PRIORITY)
- **Config:** Implement Triton kernel per pseudocode above (D=16, K_hh=2, register-tiled)
  - Start from minimal implementation: fused_routing_step as one @triton.jit kernel
  - Replace `SGNNET_SmallWorld._route()` hot path only (K_iter loop body)
  - Keep seed gather and readout in PyTorch
  - Benchmark vs V2 max-autotune (0.280ms) across K_iter ∈ {4, 5}
- **Hypothesis:** Eliminating [B,N,K_hh,D] intermediate + reducing from ~5 kernels to 1 per iter → 0.100–0.140ms
- **Slot:** 5060ti:cuda (CUDA kernel dev requires CUDA)
- **Smoke-test:** `python3 -c "import triton; print(triton.__version__)"` then `python3 scripts/bench_step840_triton_routing.py --help`
- **Criterion:** ≤0.160ms median = significant win; ≤0.120ms = paper-level result
- **Risk:** Triton support for SM_120 (Blackwell); int32 index requirement may need conn_hh dtype change
- **Implementation note:** conn_hh must be cast to int32 BEFORE passing to kernel; add `self.conn_hh = conn_hh.to(torch.int32)` in init

### step841: bf16/fp16 Inference with Buffer Cast Fix
- **Config:** Fix mixed-dtype issue: cast ALL model buffers to half before compile
  - Variant A: fp16 reduce-overhead
  - Variant B: fp16 max-autotune
  - Variant C: bf16 reduce-overhead
  - Variant D: bf16 max-autotune
  - Validate output logits match fp32 within 0.01 tolerance
- **Hypothesis:** Even at D=16, fp16 cuts memory traffic 2× → 1.2–1.4× speedup
- **Slot:** 5060ti:cuda (after bench_step832)
- **Smoke-test:** `python3 scripts/bench_step841_half_inference.py --help`
- **Criterion:** ≤0.230ms = win (>1.2×)
- **Risk:** Normalization (F.normalize) at fp16 with D=16 may have precision issues

### Pending — bench_step832 (already scripted)
- **Config:** torch_scatter scatter_add vs fancy-index, including V_scat_cg CUDA Graph variant
- **Key test:** Does V_scat_cg (scatter_add + CUDA Graph) capture succeed?
- **Slot:** 5060ti:cuda (next available after bench_step830 analysis)
- **Criterion:** V_scat_cg < 0.150ms = CUDA graph path is viable → also enables step840 CUDA-graph variant
- **Risk:** scatter_add still fails CUDA graph capture with int64 edge_recv

### step842: int32 Index Conversion + CUDA Graph Retry
- **Config:** Convert conn_hh and conn_in to int32 (not int64) at model init; retry CUDA graph capture
  - Test that torch.index_select / fancy index with int32 is CUDA-graph-capturable
  - If yes, measure V5_int32 latency
- **Hypothesis:** The CUDA graph capture failure was dtype-specific; int32 static index may be capturable
- **Slot:** 5060ti:cuda
- **Smoke-test:** `python3 -c "import torch; t=torch.randn(10,16).cuda(); idx=torch.tensor([0,1],dtype=torch.int32).cuda(); print(t[idx])"` then verify CUDA graph capture
- **Criterion:** CUDA graph succeeds in capture → any latency < 0.200ms = win
- **Risk:** PyTorch may convert int32 index to int64 internally anyway; check with torch.jit.trace

### step843: Persistent K_iter Kernel (Batched mega-kernel)
- **Config:** Combine all K_iter=5 routing steps into a single Triton kernel using inter-block barriers
  - Use CUDA grid synchronization (`__grid_sync()` or Triton's tl.group_barrier) between routing steps
  - Eliminates 4 of 5 kernel launch gaps (saves ~16µs)
  - More complex: requires Z state to be passed via global memory between "steps" (since no full grid barrier in Triton)
- **Realistic approach:** Not a single Triton block, but a single Python call with explicit K_iter loop inside the kernel using tl.debug_barrier() for intra-block sync only — different neurons don't need inter-block sync since each neuron's Z_struct reads from PREVIOUS iteration's Z (no data hazard at compile time)
- **Hypothesis:** With K_iter fused (no Python loop overhead), saves ~0.020ms
- **Slot:** 5060ti:cuda (after step840 proves the base Triton kernel works)
- **Criterion:** <0.120ms median = step840 + loop fusion working

---

## 6. Risks / Things That Won't Work

### Definitively Won't Work
1. **CUDA Graph on current model (step811 V5 = 2.212ms):** The int64 fancy index forces CPU sync every replay. This is NOT fixable without kernel rewrite.
2. **K_iter distillation for wall-time reduction:** Knowledge distillation KILLS accuracy (step196 confirmed). K_iter is not reducible below 4 without architecture redesign.
3. **bf16 for training speedup:** step801 confirmed bf16+GradScaler HURTS training accuracy. Wall-time BF16 inference might work but training does not.
4. **Max-autotune alone for further speedup:** We're already at V2 max-autotune (0.280ms). Further compiler tuning has <5% room.
5. **N reduction for wall-time:** N=2048 is already optimal for accuracy/efficiency. Halving N to N=1024 saves ~40% routing time but costs accuracy (confirmed by N-scaling experiments).

### High Risk (May Not Work)
6. **PyGraph (arXiv:2503.19779) for CUDA graph:** Requires PyTorch 2.4+. We have PyTorch 2.11. Not compatible without upgrade.
7. **Tensor-core INT8 for scatter:** D=16 is at the minimum tile boundary for INT8 tensor cores (need ≥16 inner dim). May activate, may not. Only viable with explicit tiled GEMM formulation (i.e., inside Triton kernel, not via torch.nn.functional).
8. **torch_scatter V_scat_c with torch.compile:** PyG docs confirm torch.compile DISABLES torch_scatter extension — falls back to pure PyTorch. Net effect vs V2 max-autotune: likely neutral or worse.
9. **Stream parallelism across K_iter steps:** Each step depends on output of previous step (Z). Cannot parallelize across steps without approximate routing. Would require fundamental algorithm change.

### Literature Contradiction
**Claim in efficiency_meditation**: "GPU utilization 99% = compute-bound." **But** the theoretical floor analysis shows we're running at 6.6 GFLOP/s against 23.2 TFLOPS peak — **0.03% of peak compute**. High GPU utilization at such low GFLOP/s = the GPU is busy launching/synchronizing kernels, NOT doing useful math. We are overhead-bound, not compute-bound. This is consistent with the CUDA graph failure being so catastrophic (8×) — every replay caused a sync.

---

## 7. Top 3 Highest-Impact This Week

### 1. Run bench_step832 → Determine if CUDA Graph Path Is Alive
**Why this week:** bench_step832 is already written and queued. If `V_scat_cg` (scatter_add + CUDA Graph) captures successfully, it immediately gives 0.150ms target AND proves the Triton kernel with static indices can do the same. This single result either opens or closes the CUDA-graph path permanently.
- **Action:** Launch bench_step832 on 5060ti after any current job completes
- **Gate:** CUDA graph capture success = pursue step842 (int32 conversion) + step840 (Triton)

### 2. Start step840 (Triton Fused Routing Kernel) — Highest Ceiling
**Why this week:** The Triton kernel is the only lever with a credible path to ≤0.150ms. The pseudocode is now clear (see Section 4). D=16 fits in a cache line. K_hh=2 fits in registers. The implementation is ~100 lines of Triton code.
- **Action:** Implement `fused_routing_step` kernel, validate output matches PyTorch reference (atol=1e-4), then benchmark
- **Immediate prerequisite:** Verify Triton version supports SM_120: `python3 -c "import triton; print(triton.__version__)"` on 5060ti
- **Gate:** <0.160ms = paper-level result; >0.200ms = reassess tiling strategy

### 3. bf16/fp16 Buffer Cast Fix (step841) — Quick Win
**Why this week:** Half-day effort, code change is trivial. Even at 1.2× (0.233ms), it moves the bar. More importantly, if fp16 actually does activate tensor cores at D=16 → 0.140ms is possible and Triton becomes secondary.
- **Action:** Script bench_step841 (copy bench_step811, add dtype cast to all buffers before compile)
- **Gate:** <0.230ms = include in paper; >0.240ms = skip, pursue only as Triton supplement

---

**Theoretical reachable floor (Triton kernel + CUDA Graph + bf16):**  
~0.050–0.080ms (within 2× of Linear at 0.089ms)  
**Without CUDA Graph (Triton kernel alone):**  
~0.100–0.140ms (3.15× improvement from 0.280ms)

*File path: `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/.planning/walltime_meditation_2026-04-15.md`*
