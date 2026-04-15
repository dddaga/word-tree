# SGNNET Wall-Time Breakthrough — 20+ Candidates to Break 0.280 ms

**Date:** 2026-04-15
**Author:** Claude Opus 4.6 (1M ctx), research strategist
**Mandate:** Generate ≥20 well-reasoned candidates to break the 0.280 ms wall on RTX 5060 Ti; rewrite from scratch is allowed.
**Scope:** Research + strategy only. No code changes, no experiment launches.
**Hardware target:** RTX 5060 Ti (Blackwell SM_120, 16 GB GDDR7, 448 GB/s, 23.2 TFLOPS fp32, 5th-gen tensor cores with FP4/FP8/BF16/FP16).

Complementary reads (do not re-derive):
- `walltime_meditation_2026-04-15.md` — Sonnet wall-time diagnosis (Triton + CUDA-Graph focus)
- `efficiency_meditation_2026-04-15.md` — Haiku broader efficiency view
- `learnings/LEARNINGS_design_2026_04_15.md` — today's design log (K-scaling, dynamic-topology closure)
- `learnings/paper/MANUSCRIPT_DRAFT_*.md` — the paper narrative we're defending

---

## 1. Restated Problem — Are We Solving the Right Thing?

SGNNET currently runs at **0.280 ms** (K=5 max-autotune fp32) on RTX 5060 Ti with N=2048, D=16, K_hh=2, K_iter=5. Linear-head is 0.089 ms. VGG_FC baseline is 1.570 ms. **We win vs VGG_FC (5.6×) but lose to Linear (3.15× slower).** The paper claim of "efficient FC replacement" is secure *at the VGG_FC reference*; the claim of "approaching a bare Linear head" is not.

**The one-line diagnosis (today):** we are kernel-launch-overhead bound, not compute or bandwidth bound. Effective compute is **6.6 GFLOP/s vs 23.2 TFLOPS peak = 0.03% peak**. Hundreds of micro-ops (K_iter × ~15 PyTorch ops) each cost ~3 µs of CPU→GPU dispatch, and that sum is the wall. The Z tensor (4.19 MB) fits entirely in L2 (~40 MB); we are neither memory-bound nor compute-bound — we are *dispatch-bound*.

**But is wall-time even the right metric?** Four possible reframings deserve honest assessment before we spend weeks on Triton:

1. **Energy-per-inference** is arguably the real story for "efficient classification heads." A 5.6× wall-time win on a 50 W Blackwell chip vs a 1.6 ms VGG_FC spinning the same chip hotter is a bigger energy win than latency suggests. Paper already hedges: claim is "energy/FLOPs efficiency," not "lowest latency."
2. **Throughput at larger batch** may be dramatically better than latency at batch=32 suggests. If the K_iter loop amortizes well with B=256, SGNNET could win *per-sample* by more than 5.6× while absolute latency stays flat.
3. **Parameter efficiency** (34,976 vs VGG_FC 119.59M = 0.029%) is *already* a 3400× win on params — a meaningful story even if wall-time stalls at 0.280 ms.
4. **The routing mechanism's value is "why" not "how fast."** If SGNNET is a *discovery machine* that reveals structure which can then be distilled into a Linear-head-shaped student, the paper becomes about *what* was learned, not the deployment artifact.

**What does "breakthrough" mean here?** I take it to mean: either (a) ≤0.150 ms on 5060ti (within 1.7× of Linear at same accuracy), OR (b) a principled reframing that makes wall-time a secondary metric. Both deserve exploration below.

---

## 2. State of the Art (Validated)

### Numbers the whole mediation rests on

| Quantity | Value | Source |
|---|---|---|
| Current best latency | 0.280 ms (K=5 max-autotune fp32) | bench_step811 V2 |
| K=4 latency | 0.263 ms (5.6% faster) | bench_step830 today |
| Linear head | 0.089 ms | bench_step810 |
| VGG_FC baseline | 1.570 ms | bench_step810 |
| Params | 34,976 | reconciled today (2× doc error fixed) |
| True per-sample FLOPs (ncu) | 1.85 M | step800 |
| MLP_37 matched-FLOPs | 97.71% @ 1.86 M FLOPs, 928K params | step403b |
| GCN / GAT on same graph | 48.9% / 47.0% | step404 |
| Best SGNNET @ efficiency config | 95.52% (K=5) / 97.30% (K=5 rot+aug) / 97.71% (N=4096 K=5 rot+aug) | step199 / step235 / step266 |
| Effective compute | 6.6 GFLOP/s = 0.03% of peak | derived |
| Z data volume | 4.19 MB (fits L2 40 MB) | derived |

### Key structural findings (today)

- **K_iter is not the whole wall.** 60% of wall-time (~0.168 ms) is non-routing overhead (seed gather, normalize, readout, dispatch) that does NOT scale with K_iter. Dropping K 5→4 saves only 5.6% because only ~40% of time scales with K.
- **CUDA Graph killed** (step803) by `Z[:, conn_hh, :]` where conn_hh is int64 — internal device-to-host sync on each replay. V5 runs 2.212 ms (8× slower than eager).
- **Deep supervision killed** (step521): trajectory value is only in the final state.
- **Dynamic topology nearly closed** (step511–525 all KILLED except 524-S1 β-scalar, untested).
- **GCN/GAT on the same small-world graph decisively lose** (47–49% vs 95%+). The *routing rule* is load-bearing, not the topology.
- **MLP_37 at matched FLOPs beats SGNNET by 2.19pp** at the efficiency config (step403b, a paper blocker). At N=4096 SGNNET catches up (97.71% both), but still not a dominance story.

### Hardware realities (Blackwell SM_120)

Triton 3.x fully supports SM_120 ([NVIDIA Blackwell + Triton blog](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/)). Tensor cores natively handle FP4/FP6/FP8/BF16/FP16. FP4 matmul is up to **4.6× FP8 throughput** on Blackwell ([NVFP4 blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)). CUTLASS 4.0 adds SM_120 kernels ([CUTLASS Blackwell docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)). **But all tensor-core paths require ≥16×16 tiles, and our inner-K (K_hh=2) is far below that** — tensor cores won't activate without tiling reformulation.

---

## 3. Patterns from Graphiti + Literature (Cited)

### From Graphiti (group_id="dhiraj")

- **step196 / K_iter distillation**: "K_iter distillation negatively affects and degrades performance during K_iter reduction" — the student fails to absorb routing dynamics. Teacher-weight-init is sufficient for mild K-compression but full K=1 students break (2026-04-10).
- **Trajectory does NOT matter at inference** (step521 deep supervision, 2026-04-10 fact: "student needs to match the routing trajectory, not just the final output" — but deep supervision killed anyway, meaning the trajectory is implicit, not explicit).
- **Over-smoothing increases with N at fixed D=16** (2026-04-10 fact) — implies shorter K_iter is *better* at large N, which is why step265 finds K=4 > K=5 at N=4096 for ΔW proj (+0.97 pp).
- **K-scaling law is mechanism-specific**: ΔW proj prefers K=4; ΔW rot+aug prefers K=5. Over-smoothing applies to *gating* (which attenuates magnitude) not rotation (which preserves it).
- **INT8 QAT wrap rate = 0 at D=16** with scale=100 (step527). D=16 bounds activations to ±0.25×100=±25, inside int8 ±127 range. **INT8 tensor cores → 1.5–2× projected speedup**, conditional on kernel path.
- **Dynamic connectivity is closed**: step511-514, step523, step525 all killed; only step524-S1 (β on frozen edges) is untested.
- **Topology stability is load-bearing**: W_pos is co-adapted to the fixed topology; ANY perturbation — learned or random — hurts (step525 T3 collapsed to 54%).

### From literature (2024–2026)

**Kernel-launch-bound small workloads are a well-known problem.**
- "Look Ma, No Bubbles! Megakernel for Llama-1B" (Hazy Research, May 2025): ~100 kernels fused into one on-GPU-interpreter kernel, achieving **<1 ms on H100** for a 1B model and **<680 µs on B200**. Uses shared-memory paging, counter-based sync, and an on-SM instruction-dispatch interpreter. Fits exactly the "dispatch-bound tiny workload" pattern SGNNET is in. Does NOT handle dynamic control flow, but SGNNET's control flow IS static (K_iter is a fixed int, conn_hh is a static buffer). [Source](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- "Characterizing LLM Inference Workloads" (arXiv 2504.11750, 2025): explicitly identifies the "CPU-bound region" at small batch where latency = Σ(launch_overhead), and argues fusion is the primary lever there. [Source](https://arxiv.org/pdf/2504.11750)
- "Launch of many small kernels 10x slower compared to one kernel" (NVIDIA devforum 2025): quantifies the per-launch cost at ~2–5 µs on modern drivers. [Source](https://forums.developer.nvidia.com/t/launch-of-many-small-kernels-10x-slower-compared-to-one-kernel/350194)
- **PyGraph (arXiv 2503.19779, Mar 2025)**: robust CUDA-graph-in-torch.compile, 29% avg / 3.36× peak over PyTorch2-CG. Requires PyTorch 2.4+; we're on 2.11 so compatible. [Source](https://arxiv.org/html/2503.19779v1)

**CUDA Graphs + fancy-indexing is a known open bug.**
- PyTorch Issue #155682: int32/int64 dynamic index inside captured region raises "operation not permitted when stream is capturing" because the tensor-indexing op internally syncs to read the index values. Confirmed workaround: use `torch.cuda.make_graphed_callables` with a static buffer pattern or rewrite the op in Triton with compile-time-known K_hh. [Source](https://github.com/pytorch/pytorch/issues/155682)
- NVIDIA CUDA-Graph-in-PyTorch guide ([docs.nvidia.com](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/handling-dynamic-patterns.html)): pre-allocate static tensors, use `.copy_()` at replay. Explicitly recommends moving index-dependent ops outside the captured region OR into a static Triton kernel.

**ThunderKittens / FlashInfer / Chipmunk — the "fast kernels for everything" wave.**
- ThunderKittens (arXiv 2410.20399): tile-based abstraction with 16×16 register tiles. [Sparse attention extension](https://github.com/HazyResearch/ThunderKittens/pull/137) gets 9.3× over dense at 93% sparsity. Our D=16 K_hh=2 is far smaller than their tiles but the *grid-level launch-cost hiding* pattern translates.
- FlashInfer (arXiv 2501.01005): JIT template for attention + MoE with user-supplied functors; 29–69% inter-token-latency reduction vs compiler backends; now embedded in vLLM/SGLang. Supports grouped-GEMM + MoE-style dispatch directly. [Source](https://arxiv.org/pdf/2501.01005)
- Chipmunk (2025): warp-specialized persistent grids with training-aware sparsity. Their [192,1] sparse attention pattern could inspire a K_hh-sparse formulation of SGNNET routing. [Source](https://sandyresearch.github.io/chipmunk-part-III/)

**GLNN / GNN→MLP distillation is a real, mature pattern.**
- Graph-Less Neural Networks (ICLR 2022 Zhang et al., arXiv 2110.08727): "distilled MLPs infer 146×–273× faster than GNNs" with competitive accuracy; matches GNN on 6 of 7 datasets. The mature pattern for "use the graph at training, drop it at inference." [Source](https://arxiv.org/abs/2110.08727)
- "Teach Harder, Learn Poorer" (arXiv 2407.14768, 2024) and SA-MLP (arXiv 2210.09609): refinements showing single-shot student MLP can match teacher GNN with the right distillation loss (soft labels + structure-aware features).

**LayerSkip / self-speculative decoding** (Meta, arXiv 2404.16710): LLM self-speculation where early-exit serves as draft and full model serves as verifier. Achieves 2.0–2.16× speedup on real tasks. The SGNNET analogue: use K=1 or K=2 as a *draft*, K=5 as a *verifier*, and re-run only on disagreement. [Source](https://arxiv.org/abs/2404.16710)

**Deep Equilibrium Models (DEQ) + Consistency-DEQ** (arXiv 2602.03024, 2024): iterative fixed-point nets can be compressed to "one-step era" via consistency distillation preserving solver-informed ODE trajectory. Directly analogous to SGNNET's iterative routing. Unlike naive knowledge distillation (step196 killed), **consistency distillation distills the SOLVER behavior not the output** — a materially different loss function. [Source](https://arxiv.org/html/2602.03024)

**AOTInductor** (PyTorch docs, [2025 writeup](https://johal.in/torch-compile-aotinductor-python-ahead-of-time-compilation-workflows-2025/)): ahead-of-time compile to a `.so`, eliminates JIT warmup and some dispatch layers; 2–5× reported on general inference; 13–40% on UNet/UViT. Untried by us.

**Mamba / SSM kernels** (arXiv 2312.00752; VMeanba 2412.16602): selective-scan kernels with fused D-dim reductions. Not architecturally a fit for SGNNET but their *kernel design patterns* (persistent state in registers, warp-level tiling over D) are the template we'd want for a fused K_iter megakernel.

### Conflicts / tension points

1. **"GPU util 99%" vs "0.03% of peak":** both are true. The GPU is never idle (dispatch + normalize + tiny compute fills every slot) but the *useful math* is trivial. GPU-util is a misleading metric at this scale.
2. **Graphiti says K_iter distillation FAILED (step196).** Literature says consistency-DEQ distillation SUCCEEDS. Why the gap? Our step196 used output-matching KD; consistency-DEQ matches the solver trajectory. **This is a different experiment, and the "K=1 student" direction is not actually closed.**
3. **Graphiti says K=4 is as good as K=5 at N=4096** but saves only 5.6% wall-time. Implication: further K_iter reduction has diminishing wall-time returns; we need to attack the *non-K_iter* 60% of the wall.

---

## 4. Self-Reflection — Are We Asking the Right Question?

I want to be honest with Dhiraj. The user said "I am okay with rewriting everything from scratch." Here are the uncomfortable reframings I owe him.

### Reframe #1: The K_iter loop might not need to exist at inference.

Graphiti's step196 "K_iter distillation killed" is load-bearing in our self-story, but it tested **output-matching KD**, not **consistency distillation of the trajectory**. Deep supervision (step521) killed the hypothesis that the trajectory carries supervised value — but that's precisely *why* a consistency distillation should work: if only the fixed point matters, a student that learns the fixed-point map directly should succeed. **The "K_iter=1 student" experiment has not actually been run with the right loss.** This is a major open door.

If a K=1 student lands within 1pp of K=5 teacher, wall-time drops from 0.280 ms to ~0.17 ms trivially (non-routing overhead dominates; we skip 4 routing launches). Combined with Triton + CUDA Graph: plausibly 0.060–0.090 ms. That would *beat* Linear.

### Reframe #2: Is SGNNET a model or a feature extractor for a simpler model?

MLP_37 hits 97.71% at matched FLOPs. That's a higher accuracy than SGNNET at the same scale. A principled read: **the routing computation is discovering a low-rank feature transform that a 37-hidden-unit MLP can express.** If that's true, then:
- The paper's primary artifact could be the *method* (routing reveals the structure), not the *runtime* (the routing network itself).
- The deployed model could be an MLP_37 distilled from an SGNNET teacher — at 0.088 ms (MLP_64 benchmark) this would be indistinguishable from Linear.
- The story becomes "SGNNET is a compute-efficient way to *train* an FC replacement; at inference, the student is an MLP_N."

This is radical but aligns exactly with GLNN (146–273× speedup via distillation). It's been the standard pattern in the graph literature for four years.

### Reframe #3: Is N=2048 the right scale?

N=512 with a smarter mechanism might win overall. We stopped N-scaling down because N=512 plateaued at ~92%. But we never tried N=512 with the ΔW rot+aug mechanism at full K=5 and 150 ep with the step266 augmentation recipe. **N=512 with the current mechanism could halve the gather work in `_seed()` and shave 20–30% of wall-time.** This is a quick experiment that has never been done.

### Reframe #4: Fp32 is throwing away the hardware.

Tensor cores want FP4/FP6/FP8/BF16/FP16. Our fp32 path uses 0.03% of peak. Even if tensor cores don't fully activate at D=16 × K_hh=2 (inner dim too small), **FP8 or FP4 on a 16-wide vector has half the L2 bandwidth footprint of fp32 and enables wider SIMD across Z**. The step527 INT8 QAT wrap-rate=0 finding strongly suggests precision is there. The failed V3/V4 were mixed-dtype bugs, not a fundamental limit.

### Reframe #5: Replace routing with retrieval.

If after training, `dw_norm` is fixed and `conn_hh` is fixed, then for any input Z the per-neuron routing is a deterministic function of Z plus a static tensor. A learned codebook + nearest-neighbor lookup could replace the iterative routing with a single top-K retrieval, then an FC. At N=2048 D=16 this is a tiny index lookup and a tiny FC — microseconds total on GPU.

### Reframe #6: The small-world graph might be cargo cult.

GCN/GAT failed at 47% on the same graph while SGNNET hits 95%. Everyone has concluded "the mechanism is load-bearing, the graph isn't." But we haven't tested whether SGNNET-mechanism on a **different** topology (e.g., random fully-connected sample, or learned from data, or even a dense `conn_hh = torch.arange(N).repeat(K_hh, 1).T`) also hits 95%. If topology is actually a free parameter, we could pick one that maps cleanly onto tensor cores — e.g., block-banded conn_hh where K_hh neighbors form contiguous blocks of 16.

### Reframe #7: What does "training one model at 95%" prove anyway?

The paper claims SGNNET as a general architecture. But we've trained it on Imagenette VGG features + (pending) CIFAR-10 + (pending) SST-2. If *the whole stack* is reducible to an MLP_37 after distillation, the "general architecture" story collapses and we're left with "a better way to train an MLP." That might be a fine paper (GLNN was!), but it's a different paper.

---

## 5. The 20+ Candidates

Scoring convention: **gain** = optimistic realistic-worst range; **effort** = person-days assuming the right skill; **tier** = T0 scout / T1 calibration / T2 validation. Uncertainty marked explicitly. I've grouped into 6 classes and ranked within each by (expected gain × confidence) / effort.

### Class A — Micro-optimization of the current kernel path (low-risk, incremental)

| # | Name | Hypothesis | Mechanism | Projected ms | Effort | Risk | Deps | Falsification | Tier |
|---|---|---|---|---|---|---|---|---|---|
| **A1** | **int32 buffer conversion + CUDA-Graph retry** | Converting conn_hh/conn_in from int64 → int32 removes the internal sync point that killed V5 CUDA Graph. | Register both as int32 at init; verify via `torch.index_select` + graph capture toy test; re-run V5 benchmark. Also audit all `.long()` casts inside `_route`. | 0.280 → 0.150–0.200 ms (0.090 if CG fully works). High variance. | 1–2 d | Medium — int32 may get upcast inside PyTorch's gather op anyway; capture may still fail. | — | One benchmark run: does `torch.cuda.CUDAGraph().replay()` succeed and give <0.250 ms? | T0 |
| **A2** | **bf16 inference with full buffer cast** | V3/V4 failed from mixed-dtype internals. Correcting the cast path should recover 1.2–1.4× from halved memory traffic even without tensor cores activating. | Add `model = model.half(); model.spatial_coords = model.spatial_coords.half(); model.C_ho_mask = model.C_ho_mask.half()`; verify numerical match at eval (atol 1e-3); benchmark both bf16 and fp16. | 0.280 → 0.220–0.260 ms | 0.5 d | Low — known fix for known bug. | — | Median ≤ 0.240 ms. | T0 |
| **A3** | **AOTInductor AOT compile + .so packaging** | Replaces JIT with AOT; empirically 2–5× inference speedup reported across many model classes (30–40% on UNet). Removes torch.compile warmup + dispatch layer residual. | `torch.export.export` → `torch._inductor.aoti_compile_and_package`; load in bench. | 0.280 → 0.180–0.240 ms | 1 d | Low-Medium — AOTI may not support fancy-index + small ops any better than max-autotune. | A1 helpful | Median ≤ 0.240 ms. | T0 |
| **A4** | **torch.compile fullgraph=True audit** | If any graph-break is left in max-autotune path, it drops a full kernel launch. Running fullgraph=True will error on breaks and surface them. | Add `fullgraph=True`; fix any errors by rewriting ops; re-benchmark. | 0.280 → 0.250–0.280 ms (only a polish) | 0.5 d | Low — likely already clean, worst case silent. | — | Same or better latency, no errors. | T0 |
| **A5** | **Batch up `C_ho_mask.float()` out of hot path** | In `_readout`, the cast happens every forward. Pre-cast at init (already float), store as buffer. Trivial but every µs counts. | Code inspection + single edit. | 0.280 → 0.275–0.280 ms | 0.1 d | None. | — | Δ ≥ 0.003 ms. | n/a (just do it) |

### Class B — Algorithmic (change what we compute, not how)

| # | Name | Hypothesis | Mechanism | Projected ms | Effort | Risk | Deps | Falsification | Tier |
|---|---|---|---|---|---|---|---|---|---|
| **B1** | **Consistency-distillation K=1 student** ⭐ | step196 failed on output-KD. The literature (C-DEQ, arXiv 2602.03024) shows that *consistency* distillation — matching intermediate fixed-point states with solver-informed loss — succeeds where output KD fails. SGNNET's K_iter is structurally a DEQ solver. **A K=1 student distilled with consistency loss from a K=5 teacher is a genuinely new experiment.** | Train teacher at K=5 (our current best). Define student at K=1 with a bigger per-neuron transform f_θ. Consistency loss: force student's output to match teacher's intermediate state at *every* K-step simultaneously. Optional: Anderson-acceleration pre-init. | 0.280 → 0.140–0.180 ms (skip 4 of 5 routing steps). Possibly 0.090 ms if combined with A1+A2. **Accuracy gain if student outperforms teacher's K=5 (C-DEQ reports this).** | 3–5 d | **Medium-High** — a new distillation loss that hasn't been tried; requires implementing Anderson acceleration or equivalent. If it works, the paper changes dramatically. | — | T0: 20ep student at K=1 within 1pp of K=5 teacher = viable; 20ep within 3pp = keep exploring; >5pp gap = abandon. | T0 → T1 |
| **B2** | **GLNN-style MLP_N student distilled from SGNNET teacher** ⭐⭐ (HERETICAL) | GLNN (arXiv 2110.08727) achieves 146–273× speedup by distilling GNN→MLP. If SGNNET's routing discovers features that an MLP can express, the deployed artifact is a plain MLP and latency drops to 0.088 ms (MLP_64 benchmark). **Paper reframes as "routing as training-time teacher."** | Train SGNNET teacher at full K=5 config. Distill into MLP_37 (h=37 matches FLOPs of SGNNET) using soft logits + optional structure-aware feature matching (SA-MLP style). Benchmark MLP student latency and accuracy. | **0.280 → 0.088 ms** if student matches. **3.2× win, landing inside Linear territory.** | 3 d | **High — paper-narrative risk.** If MLP_37 student hits 97.71%+ (matching teacher), we are admitting SGNNET itself is not the deployed model. But that's what the MLP_37 result already suggests. | step403b (done) | Within 0.3pp of teacher on held-out data = paper pivot. | T0 → T1 |
| **B3** | **Single-kernel fused `seed + K_iter routing`** | Seed gather (K_in=25) is a fixed conn_in buffer; same shape as conn_hh. Fold seed and all K_iter into one graph execution, eliminating 4–5 launches between phases. | Rewrite `forward()` as one torch.compile'd function with no intermediate Python variables; verify graph has no breaks. | 0.280 → 0.220–0.250 ms (saves 1–2 launches × ~5 µs plus fusion wins) | 1–2 d | Low — incremental. | A4 | Median ≤ 0.240 ms. | T0 |
| **B4** | **Learned codebook + nearest-neighbor routing** (REPLACE iterative routing) | After training, Z → routed-Z is a fixed (input-conditional) mapping. Replace K_iter=5 iteration with: learn a codebook C ∈ ℝ^[M, D] and a residual map f:Z→Z_out via 1-NN over C with M=256. One tiny gather + one tiny FC. | Train teacher. Extract {Z_in, Z_out_routed} pairs across training set. Cluster into M codes; train a small FC from code-index + residual to Z_out. At inference: dot-product with C (256×D=4k flops), argmax, gather, FC. | 0.280 → 0.030–0.060 ms if feasible | 3–4 d | High — Z_out may not be compressible to 256 codes (test with reconstruction error first). | — | Reconstruction ≥ 99% per-sample cosine of teacher Z_out = worth full training. <90% = abandon. | T0 (cheap probe first) |
| **B5** | **K_iter=1 student trained from scratch (no distillation)** | Maybe we were always wrong that K>1 is necessary. Full retrain at K=1 with a wider per-step transform (ΔW proj with learned 1-to-N fan-out) at the efficiency config. | Modify model to use K=1 with per-neuron D×D W_proj matrix; train 150 ep. | 0.280 → 0.140–0.180 ms. Accuracy risk: 90–95%. | 2 d | Medium — may underperform teacher by 2–3pp. | — | Within 1pp of Ref K=5 = big win. >3pp = abandon. | T0 |

### Class C — Architecture rewrite (change what the model IS)

| # | Name | Hypothesis | Mechanism | Projected ms | Effort | Risk | Deps | Falsification | Tier |
|---|---|---|---|---|---|---|---|---|---|
| **C1** | **Block-banded conn_hh for tensor-core activation** | Current conn_hh is random small-world → irregular gathers. If conn_hh is structured so K_hh neighbors form a contiguous 16-element block aligned to D, the gather becomes a dense 16×16 matmul that *activates tensor cores*. K_hh increases to 16 (from 2) but tensor cores make 16× less costly per element. | Build conn_hh as N/16 blocks, each neuron's K_hh=16 neighbors = its block. Train 150ep. Bench with BF16 tensor-core path. | 0.280 → 0.120–0.180 ms; if tensor cores fully activate 0.060 ms. | 2–3 d | Medium — block structure may hurt accuracy (less small-world). Small-world is cited as important. | A2 | Accuracy within 1pp of Ref AND latency ≤ 0.180 ms. | T0 (1 block-size sweep) |
| **C2** | **D=32 tensor-core aligned** | D=16 is below the 16-element tensor-core boundary for some paths and doubling the register load from 64B to 128B is trivial (still single cache line). Accuracy should go up (more geometric capacity) and tensor cores could activate. | Rerun best config with D=32, same N, K_hh, K_iter. | 0.280 → 0.220–0.290 ms (accuracy may improve; latency may be flat or slightly slower without tensor-core activation). But with Triton + FP8 could hit 0.080 ms. | 1 d + 1 d train | Low accuracy risk (D=64 already tested: 97.86% at step89). Latency risk: D=32 could be slower at fp32 without tensor cores. | — | Accuracy ≥ 95%, latency ≤ 0.260 ms. | T0 |
| **C3** | **Hopfield-style single-shot attention replacing K_iter** | Modern Hopfield networks (arXiv 2008.02217) provably equal multi-iteration fixed-point dynamics in a single softmax-attention step. SGNNET's iterative routing may compress to one Hopfield attention over a fixed pattern set. | Replace `_route` with `softmax(Q K^T / τ) V` where K,V are learned fixed patterns [M, D] with M=256. Train. | 0.280 → 0.150–0.200 ms; a single attention is faster than 5 gather-sums if M is small. | 3–4 d | Medium-High — may break the efficiency story (attention has its own params); unclear accuracy. | — | ≥ 94% accuracy AND ≤ 0.200 ms. | T0 |
| **C4** | **Speculative-routing (LayerSkip analogue)** | Use K=2 as "draft" prediction + K=5 as "verifier"; only run the last 3 iterations if draft disagrees with K=5 final expected confidence. LayerSkip shows 2.0× inference speedup on LLMs with this pattern. | Train a K=2 early-exit head alongside the K=5 model. At inference: compute K=2, check confidence; if high, return; else continue to K=5. | Average 0.200 ms (20% of samples need full K=5, 80% done at K=2). High variance per-sample. | 2–3 d | Medium — "confidence" calibration is tricky on 10-class. Batch behavior breaks early-exit (different samples exit at different K). | B1 helpful | Average latency ≤ 0.200 ms with accuracy drop ≤ 0.3pp. | T1 |
| **C5** | **Eliminate K_iter via N=8192 single-step** | At larger N, the graph has lower diameter per routing step; maybe K=1 at N=8192 ≈ K=5 at N=2048 (step209 shows D=16 ceiling at N=8192). Then N=8192 K=1 replaces N=2048 K=5. | Retrain at N=8192, K_iter=1, same D=16. | Latency: seed gather 4× bigger but only 1 routing step. Could land at 0.15–0.25 ms. | 2 d | Medium — may over-smooth; step209 showed D=16 ceiling but didn't test K=1. | — | Accuracy ≥ 96% AND latency ≤ 0.250 ms. | T0 |

### Class D — Hardware / kernel-level (the "go low-level" axis)

| # | Name | Hypothesis | Mechanism | Projected ms | Effort | Risk | Deps | Falsification | Tier |
|---|---|---|---|---|---|---|---|---|---|
| **D1** | **Triton fused routing kernel** (step530 — already in flight) | Eliminates [B,N,K_hh,D] intermediate; fuses 5 PyTorch ops into 1 kernel; uses int32 indices for CUDA-graph friendliness. | See walltime_meditation_2026-04-15.md §4 for pseudocode. Register-tiled D=16, K_hh=2, grid=[B,N]. Already scaffolded in `src/sgnnet/triton/routing_kernel.py`. | 0.280 → 0.100–0.140 ms (2–2.8×) | In flight | Medium — SM_120 Triton newly supported; may require compiler tweaks. | — | ≤ 0.160 ms. | T0 (in flight) |
| **D2** | **Megakernel: fuse ALL of forward() into one Triton launch** (Hazy-Research pattern) | At K=5 + N=2048 D=16, all forward ops fit easily in SM shared memory + registers per block. An on-SM interpreter pattern with counter-sync between "instructions" can eliminate **every** launch except one. Hazy Research achieved <680 µs on B200 for an entire 1B Llama. Our entire model has 34,976 params and 1.85M FLOPs — laughably smaller. | Port forward() into one Triton @triton.jit function: seed gather → K_iter × (gather, mul, sum, normalize) → readout. Use shared memory for Z (4.19 MB — fits per-SM). Use CUDA cooperative groups for inter-block sync between K_iter steps OR (better) run 1 SM per (B, N/n_sms) tile with intra-block sync only. | **0.280 → 0.050–0.090 ms** (2× Linear — within the paper's ideal target) | 7–10 d (requires Triton/CUDA expertise) | High — ambitious; may not beat D1 alone; cooperative-groups semantics are fragile on Blackwell. | D1 | ≤ 0.100 ms. | T1 after D1 proves base kernel |
| **D3** | **FP8 or FP4 routing with block scaling** | Blackwell's 5th-gen tensor cores support FP8/FP6/FP4 natively; FP4 is 4.6× FP8 throughput. D=16 aligns to FP4 block-scaling tile size. Routing is numerically robust (INT8 wrap-rate=0 already proved). | Quantize W_pos, Z, dw_norm to FP8 (or FP4 with block scaling); write fused routing kernel in FP8; use BF16 accumulator. | 0.280 → 0.070–0.140 ms if TC activates | 5–7 d | Medium — FP4 precision at D=16 may lose too much information per scalar projection. FP8 is safer. | D1 | Accuracy within 1pp AND latency ≤ 0.160 ms. | T1 |
| **D4** | **Persistent kernel with warp specialization (Chipmunk pattern)** | A single persistent kernel with producer/consumer warps. Producer warps fetch Z via TMA; consumer warps perform the routing compute; epilogue warp writes. Hides memory and launch latency simultaneously. | Chipmunk-inspired: producer warps stage Z_fwd into shared memory; consumer warps do the per-neuron K_hh gather + multiply + sum; epilogue normalizes and writes. Use warp-specialized double-buffer pipeline. | 0.280 → 0.080–0.120 ms | 7–10 d | High complexity; rare skill. | D1 | ≤ 0.140 ms. | T2 |
| **D5** | **CUDA Graph via `make_graphed_callables` + static-buffer replay** | Wrap the Triton kernel path in `torch.cuda.make_graphed_callables` with pre-allocated static input/output tensors; use `.copy_()` at replay. This is the blessed NVIDIA pattern and should work once fancy-index is in Triton. | Standard pattern per NVIDIA docs. | 0.280 → 0.060–0.110 ms when stacked with D1 | 2 d | Low (well-documented). | D1 | Captures successfully AND ≤ 0.120 ms. | T0 after D1 |
| **D6** | **tinygrad port as reality check** | tinygrad compiles kernel-per-op with extreme shape specialization; "for any very small models, tinygrad crushes everything with its better runtimes, even on NVIDIA." Porting SGNNET there is a cheap sanity check: does a different compiler also hit 0.280 ms, or was torch.compile leaving money? | Port the small-world module to tinygrad; benchmark on 5060 Ti. | Unknown — 0.10–0.30 ms range | 2–3 d | Medium (tinygrad lacks torch-compatible training infra, inference-only port). | — | Latency ≤ 0.200 ms = learn something useful. | T0 |

### Class E — Batching / throughput axis (different question, maybe better answer)

| # | Name | Hypothesis | Mechanism | Projected ms | Effort | Risk | Deps | Falsification | Tier |
|---|---|---|---|---|---|---|---|---|---|
| **E1** | **Benchmark at B=256 and B=1024** | At bs=32 we are latency-bound per-sample. At larger batches, the GPU actually saturates and per-sample latency drops. **Paper's real story may be throughput, not latency.** | Re-run bench_step811 at B ∈ {32, 128, 256, 1024, 2048}. Plot latency per sample. | Per-sample ms at B=1024 may be 0.010–0.050 ms → 20× win framed per-sample. | 0.25 d | None — pure measurement. | — | Per-sample latency scales sublinearly with B. | T0 (trivial) |
| **E2** | **Paper narrative: pivot from latency to throughput** | Literature note: VGG_FC is heavily batch-optimized (big matmul reuses weights); SGNNET per-sample cost is bounded by dispatch which amortizes perfectly with batch. **Per-sample throughput at B=1024 may be the best metric.** | Rewrite §6 of manuscript around B=1024 throughput comparison. | Paper gets stronger, no code change. | 1 d writing | Low. | E1 | Reviewers accept. | n/a |

### Class F — Research-level reframes (rewrite-everything territory)

| # | Name | Hypothesis | Mechanism | Projected ms | Effort | Risk | Deps | Falsification | Tier |
|---|---|---|---|---|---|---|---|---|---|
| **F1** | **SGNNET as post-hoc explainer for MLP_37** (heretical) | If MLP_37 beats SGNNET at matched FLOPs, the routing wasn't *necessary*; it was the *discovery process* that taught us MLP_37 is enough. Paper reframes as: "routing discovers that FC can be replaced by a hidden-37 MLP." | No training needed. Extract learned W_pos vectors; show they align with MLP_37's hidden-layer features via CCA or linear probe. Write a "discovery" paper. | n/a — latency becomes MLP_37's 0.088 ms. | 1 week of writing + analysis | Medium — needs rigorous CCA experiments to support the claim. | step403b | Linear probe shows W_pos linearly decodable from MLP_37 hidden layer. | T2 writing |
| **F2** | **Replace VGG16 feature extractor + SGNNET with a joint-trained tiny convnet** | The 25088-dim VGG features are themselves the bottleneck (seed gather 1.26M MACs, 68% of total). If we joint-train a tiny 256-channel conv + small SGNNET, the whole "FC replacement" argument becomes "sparse classifier beats FC at frozen-feature-eval" — maybe a stronger paper. | Adds a 4-layer tiny conv (128 params/layer) to replace VGG's last layer; co-train with SGNNET. | Unknown — probably worse latency (conv) but better accuracy/efficiency story. | 5–7 d | Medium — changes paper scope significantly. | — | Joint-trained ≥ 96% with < 50k params end-to-end. | T1 |
| **F3** | **Cross-modal story: SGNNET as universal FC replacement** | Paper's strength is claim-general not claim-specific. If SGNNET (same config) matches MLP on vision / text / audio with 1% FLOPs across the board, *wall-time becomes secondary*. | Run step405 (SST-2) + AudioSet-small + LLM-probe on same N=2048 D=16 K=5 config. | Latency is moot; win is breadth of claim. | 1–2 weeks (needs datasets) | Medium — cross-modal may need config tuning. | — | ≥ 3 of 4 modalities match baseline. | T2 |
| **F4** | **Energy-per-inference benchmark (pivot from ms to J)** | Use nvidia-smi energy counters to measure J/sample. On Blackwell, small workloads idle many SMs; SGNNET may use *half* the energy VGG_FC uses even at 5.6× lower latency. 5060ti power is SM_120 + GDDR7 — DVFS-friendly. | `nvidia-smi --query-gpu=power.draw --format=csv` during bench; compute J/sample. | Not latency; a new metric that may favor SGNNET by 10–30×. | 0.5 d | Low. | — | J/sample ≤ 0.3× VGG_FC. | T0 |

### Summary table (ranked by expected-value / effort)

| Rank | # | Name | Expected ms | Confidence | Effort | EV/effort |
|---|---|---|---|---|---|---|
| 1 | D1 | Triton fused routing (in flight) | 0.120 | High | 0 (running) | ∞ |
| 2 | A1 | int32 + CUDA Graph retry | 0.180 | Medium | 1 d | High |
| 3 | A2 | bf16 buffer cast fix | 0.240 | High | 0.5 d | High |
| 4 | E1 | Batch-scaling benchmark | n/a (reframe) | High | 0.25 d | Very High |
| 5 | B2 | GLNN MLP student distillation | 0.088 | Medium-High | 3 d | Very High |
| 6 | B1 | Consistency-DEQ K=1 student | 0.140 | Medium | 4 d | High |
| 7 | D5 | CUDA Graph via make_graphed_callables | 0.100 (stacked) | Medium | 2 d | High |
| 8 | F4 | Energy-per-inference | reframe | High | 0.5 d | High |
| 9 | B3 | Single-kernel fused forward | 0.230 | Medium | 1 d | Medium |
| 10 | C1 | Block-banded conn_hh | 0.150 | Medium | 2 d | Medium |
| 11 | A3 | AOTInductor | 0.220 | Medium | 1 d | Medium |
| 12 | D2 | Megakernel (Hazy-Research) | 0.070 | Low-Medium | 10 d | Medium (high upside) |
| 13 | C5 | N=8192 K=1 | 0.220 | Low-Medium | 2 d | Medium |
| 14 | B4 | Codebook + 1-NN routing | 0.050 | Low | 4 d | Medium |
| 15 | D3 | FP8/FP4 kernel | 0.100 | Medium | 6 d | Medium |
| 16 | B5 | K=1 from scratch | 0.160 | Low-Medium | 2 d | Medium |
| 17 | C2 | D=32 with tensor cores | 0.230 | Medium | 2 d | Low-Medium |
| 18 | C3 | Hopfield-attention replacement | 0.180 | Low | 4 d | Low |
| 19 | C4 | Speculative routing (LayerSkip) | 0.200 avg | Medium | 3 d | Low |
| 20 | D4 | Warp-specialized persistent | 0.100 | Low | 10 d | Low |
| 21 | D6 | tinygrad port | 0.150 | Low | 3 d | Low |
| 22 | A4 | fullgraph audit | 0.270 | High | 0.5 d | Low (polish) |
| 23 | A5 | C_ho_mask pre-cast | 0.278 | High | 0.1 d | Polish |
| 24 | F1 | SGNNET-as-explainer paper | n/a | n/a | 1 wk | Paper pivot |
| 25 | F2 | Joint-trained conv + SGNNET | unknown | Low | 7 d | Paper-scope |
| 26 | F3 | Cross-modal expansion | reframe | Medium | 2 wk | Paper claim |

---

## 6. Top 5 — Deeper Analysis

### #1 — B1: Consistency-Distillation K=1 Student

**Why this is the deepest single lever.** The whole K_iter loop is suspect. step521 killed deep supervision — trajectory supervision doesn't help. step196 killed output-matching distillation — a naive student can't learn from logits alone. **But nobody has tried a consistency loss that matches the fixed-point map itself.** Consistency-DEQ (arXiv 2602.03024, 2024) is the exact algorithmic template, and the authors achieved "one-step era" inference on DEQ models with accuracy *equal to or better than* the teacher. Our K_iter is structurally a DEQ solver. We've never run this experiment.

**Mechanism.** Train an SGNNET K=1 student f_s with a per-neuron transform richer than the teacher's one-step (e.g., D×D W_proj per neuron). Loss = λ₁·CE(f_s(x), y) + λ₂·|f_s(x) − T(x)|² + λ₃·Σ_k |f_s(x; f_s(x), …) − T_k(x)|² where T_k is teacher's state after k iterations. The third term is the *consistency* term — force the student's self-application to equal teacher's intermediate state at every k.

**Realistic vs optimistic.** Optimistic: student hits teacher's 95.52% (or 97.71% at N=4096) and runs at 0.140 ms (skipping 4 of 5 routing launches). Realistic: 1–2pp loss vs teacher, wall-time 0.180 ms. Pessimistic: consistency loss oscillates, student collapses to degenerate mapping. **The C-DEQ paper reports student > teacher; the DEQ literature is broadly positive.**

**The killer experiment (<1 day).** T0 scout: train student at K=1 with only the output-KD loss (λ₁=1, λ₂=1, λ₃=0); 20ep; one seed. If this already lands within 2pp of teacher (despite step196 saying it shouldn't), the direction is alive. If it underperforms as step196 predicted, add the consistency term and redo — this is the novel variable.

**Failure mode.** Consistency loss may require Anderson acceleration or careful hyperparameter tuning; 2–4 days of calibration before it stabilizes. But even a failed attempt has research value because it closes the door on "K_iter can be compressed."

---

### #2 — D1: Triton Fused Routing Kernel (already in flight)

**Why this is the safest high-upside lever.** It's a direct implementation of a known pattern (fused gather-mul-sum) on a well-understood shape (D=16, K_hh=2). The pseudocode is already written; another agent is implementing it. **Projected 2–3× alone; stacks multiplicatively with D5 (CUDA Graph).**

**Realistic vs optimistic.** Optimistic: 0.100 ms (2.8×). Realistic: 0.140 ms (2.0×). Pessimistic: 0.200 ms (1.4×) if Triton on SM_120 still has compiler warts and the gather path isn't fully register-resident.

**The killer experiment.** Already underway. When complete, check: (a) latency, (b) accuracy match (atol 1e-4), (c) CUDA-graph compatibility (capture replay test). Result of (c) is the fork point: if yes, D5 stacks; if no, pursue A1.

**Failure mode.** Triton compiler on SM_120 may not fully optimize fp32 tiles; bf16 tensor-core path may require CUTLASS template (beyond Triton). Workaround: pure fp32 scalar path in Triton still saves the intermediate allocation and a launch.

---

### #3 — B2: GLNN MLP Student (Heretical but possibly decisive)

**Why this could be the ACTUAL paper.** MLP_37 already beats SGNNET by 2.19pp at matched FLOPs (step403b). If we can distill SGNNET-teacher → MLP_37-student and the student *preserves teacher accuracy* (the GLNN 146× pattern), then **the deployed artifact is MLP_37 at 0.088 ms and the paper reframes as "sparse routing is a better way to train an MLP than direct supervised learning."** That's exactly the GLNN storyline, and it's an ICLR/NeurIPS-grade contribution.

**Mechanism.** Train SGNNET-teacher at K=5 (already have step199, step266 checkpoints). Distill into MLP_37 with soft-label KD + (optionally) structure-aware feature matching (SA-MLP style). Benchmark MLP_37 accuracy on Imagenette test.

**Realistic vs optimistic.** Optimistic: MLP_37 student hits 97.71%+ (matching teacher) and runs at 0.088 ms → 3.2× latency win AND +2.19pp accuracy vs unassisted MLP_37. Realistic: student hits 96.5% (between unassisted MLP_37's 97.71% and SGNNET's 95.52%). Pessimistic: student can't exceed unassisted MLP_37 because MLP_37's capacity IS the bottleneck, not the training signal.

**The killer experiment (<1 day).** Distill for 30 epochs with α=0.7 soft, β=0.3 hard. If MLP_37-student on held-out Imagenette ≥ MLP_37-unassisted's 97.71%, the direction is alive. If it's ≤ 97.71%, we've learned that SGNNET isn't actually teaching MLP_37 anything it can't learn alone — a neutral result, but still informative.

**Failure mode.** MLP_37 unassisted already hits 97.71% without any SGNNET guidance. This suggests the teacher signal may not help. But GLNN's 12.36% improvement over stand-alone MLP was on datasets where MLP struggled without graph signal — we'd need Imagenette to be one of those datasets. Unclear whether VGG features + MLP_37 is "MLP-friendly" or "MLP-bottlenecked."

---

### #4 — E1 + F4: Batch-Scaling + Energy-per-Inference (the reframe)

**Why this is the cheapest and potentially most valuable.** Half a day total. Could materially change the paper. The entire wall-time discussion assumes bs=32 is the right benchmark. **But nobody actually uses bs=32 for production inference.** Per-sample latency at bs=1024 is likely 0.010–0.020 ms (28× faster per-sample than current) because dispatch amortizes perfectly over the batch.

**Mechanism.** Re-run bench_step811 at B ∈ {1, 32, 128, 256, 1024, 2048, 4096}. Plot per-sample ms vs B. Measure energy (J/sample) via `nvidia-smi --query-gpu=power.draw`. Compare same plot for VGG_FC.

**Realistic vs optimistic.** I'd bet with 70% confidence that at B=1024, SGNNET's per-sample ms is ≤ 10% of VGG_FC's per-sample ms — a **10×+ throughput win framed per-sample**, vs the current 5.6× latency win. Similarly for energy. The 5.6× might be an *undersell*.

**The killer experiment (<1 day).** Run it tomorrow on 5060ti. Takes 30 minutes.

**Failure mode.** None, really. Even a negative result is a paper contribution: "batch-scaling surprisingly does NOT help SGNNET because of X."

---

### #5 — A1 + D5: int32 indices + CUDA Graph (stacked micro-opts)

**Why this is the cheapest path to 0.100 ms.** If Triton (D1) succeeds and we can re-run CUDA-Graph with int32 indices + static buffers, we likely eliminate the remaining dispatch overhead entirely. NVIDIA's best-practice for this exact pattern is documented and proven.

**Mechanism.** (a) Change `conn_hh = conn_hh.to(torch.int32)` in model init. (b) Verify fancy-index `Z[:, conn_hh_i32, :]` still works. (c) Wrap forward in `torch.cuda.make_graphed_callables` with static input tensors. (d) Benchmark.

**Realistic vs optimistic.** Optimistic: 0.070 ms (capture + 1 replay = 1 kernel launch total). Realistic: 0.120 ms (some paths still sync). Pessimistic: still 0.280 ms if gather op internally upcasts int32→int64.

**The killer experiment (<1 day).** Toy test first: `torch.index_select` on a 2D tensor with int32 index inside `CUDAGraph()` context. If this toy succeeds, the real retry is high-confidence. If the toy fails, we know the path is still closed.

**Failure mode.** PyTorch may silently upcast int32 → int64 inside `__getitem__`. The reliable fix is to replace fancy-indexing with `torch.gather(Z.expand(-1, -1, -1), 1, idx.expand(B, -1, -1, D))` or do the gather inside the Triton kernel.

---

## 7. Heretical Proposals (Rewrite Everything)

The user explicitly asked for heretical ideas. Here are 5.

### H1 — Ship MLP_37 as the artifact. Frame SGNNET as the discovery.

"We studied the space of sparse classifier heads and discovered that a small MLP is sufficient, because [W_pos clustering analysis, routing dynamics → feature similarity argument]. The SGNNET sparse graph architecture is our *method* for discovering this — a training-time scaffold that reveals structure. At deployment, the student MLP_37 inherits 97.71% accuracy with 0.088 ms latency." This is the GLNN storyline applied to FC replacement. It may be the strongest ICML/NeurIPS paper available from this project.

### H2 — Drop K_iter to 1 entirely.

step196 is not a closed door — it was a bad loss function. Re-run the K=1 student with consistency-DEQ loss (B1) or with a richer per-neuron transform (B5). Accept a 1–2pp accuracy loss for a 50% wall-time win. The "iterative routing" story is *not* load-bearing in the paper — what's load-bearing is sparse-topology + geometric-routing.

### H3 — D=2 or D=4.

We never tested D=2 with the current ΔW-proj mechanism. D=2 means 2× smaller memory traffic, possibly tensor-cores-for-different-reasons (D=2 groups per 16-wide fp4 vec), 50% less arithmetic. Accuracy may crater, but if it holds at 92–93%, the efficiency claim becomes 8× tighter than the current 0.79%. "97.86% at D=64" is the *ceiling*. Most of the paper's claims are *at* D=16. Going D=2 changes the efficiency denominator.

### H4 — Abandon small-world topology; train with dense K_hh=16 blocks.

GCN/GAT failed on small-world → "mechanism is load-bearing." But we've never tested SGNNET's mechanism on a *different* topology. If the routing mechanism works equally on block-banded (C1) or even fully dense small-K graphs, we can pick topology to be tensor-core-aligned. Wall-time could drop 3×, and the paper becomes "SGNNET's routing is topology-agnostic, not small-world-specific."

### H5 — The whole paper becomes "Per-sample throughput at batch size B=1024."

Run E1. If at B=1024 SGNNET beats VGG_FC per-sample by 10×+ or by energy 20×+, the paper pivots entirely: wall-time at bs=32 is a deceptive metric; throughput-per-Joule at real-production-batch is the right metric; SGNNET wins by an order of magnitude. **This is the least destructive pivot — all existing SGNNET results stay valid, we just reframe the evaluation axis.**

---

## 8. Recommended 2-Week Sprint Plan

Assumes the Triton kernel (D1) continues in flight and finishes this week. Assumes 5 slots available.

### Week 1

**Monday–Tuesday**

1. **E1 + F4 — Batch-scaling + energy benchmark.** 5060ti, 1 day. Trivial. Tells us whether the paper should pivot to throughput. (Slot: 5060ti:cuda when D1 finishes or on a parallel compile.)
2. **A2 — bf16 buffer-cast fix.** 5060ti, 0.5 d. Quick win likely; 1.2×.
3. **A1 — int32 + CUDA-Graph toy test.** 5060ti, 0.5 d. Forks the rest: if capture works, prioritize D5. If not, focus on D1+A2.

**Wednesday–Thursday**

4. **B2 — GLNN MLP_37 student distillation.** mini:mps (no CUDA needed for MLP distill) OR 5060ti. 2 d. **This is the potentially-decisive experiment for the paper narrative.** Run it this week — the downside is 2 days, the upside is the paper pivots entirely.
5. **B1 Scout — K=1 consistency-DEQ student.** studio:mps, 1 d. 20ep T0 with consistency loss. Forks the trajectory of the second week.

**Friday**

6. Analyze D1 + A1 + A2 + B2 + B1 T0 results. Decide on week-2 focus.

### Week 2

Pick one of two paths based on week-1 results:

**Path α (breakthrough achieved via B2):** if MLP_37 student matches SGNNET, spend week 2 writing the "SGNNET as teacher" paper pivot. Run cross-dataset validation (CIFAR-10, SST-2) with the student. Skip further kernel optimization.

**Path β (kernel path still the best bet):** combine D1 + D5 (Triton + CUDA Graph). Run D2 (megakernel) scoping — 2 days of investigation (not implementation). Run B1 T1 (consistency distillation 75ep). Run A3 (AOTInductor) as safety net.

**Always run:** C1 (block-banded conn_hh) on a CPU slot — it's a free accuracy experiment and if it doesn't hurt accuracy, it enables tensor cores in the long run.

### Slots assignment (illustrative)

| Slot | Week 1 | Week 2 path α | Week 2 path β |
|---|---|---|---|
| 5060ti:cuda | E1, A2, A1, D1 (finish) | MLP student cross-dataset | D5 stacking, AOTInductor |
| mini:mps | B2 distill T0 | MLP student T1 | B1 T1 (consistency) |
| mini:cpu | ongoing step527 | new baselines | new baselines |
| studio:mps | B1 T0, 524-S1 | analysis | D2 scoping |
| studio:cpu | ongoing step268 | step268 continued | C1 block-banded T0 |

---

## 9. What We Need From the User

Decisions only Dhiraj can make:

1. **Paper pivot tolerance.** Are you willing to reframe the paper as "SGNNET teaches MLP_37" (GLNN-style) if B2 works? This is the single biggest decision — it rewrites the abstract.
2. **K_iter-distillation revisit.** step196 was *output-matching* KD. Are you willing to re-open the K_iter=1 student direction with a *consistency* loss? This directly contradicts a prior `CONFIRMED` finding — but I claim the prior finding was underspecified.
3. **Wall-time target precision.** Is 0.150 ms enough to call victory, or do we need ≤0.100 ms? The difference determines whether D1 alone is sufficient or whether we need D2 (megakernel, 10-day investment).
4. **Triton team capacity.** D2 (megakernel) requires 1 skilled Triton developer for 7–10 days. If that's not available, we cap at D1 + D5 stacked, target ~0.100 ms.
5. **Scope of "paper direction change."** If E1 shows 10× throughput win per-sample at bs=1024, do we reframe to throughput? This is a gentler pivot than B2 and arguably more honest about how the model is used.
6. **Am I right that step196 is not conclusive?** You know the experiment better than I do. The test: did step196 match output logits only, or did it also match intermediate Z_k states? If just logits, B1 is new territory. If also Z_k states, I'm wrong and B1 is a rerun of a dead direction.

---

## 10. Sources

- [OpenAI Triton on NVIDIA Blackwell Boosts AI Performance](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/)
- [NVFP4 for Efficient Low-Precision Inference (NVIDIA, 2025)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [CUTLASS Blackwell docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
- [Look Ma, No Bubbles — Megakernel for Llama-1B (Hazy Research, May 2025)](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- [PyGraph: Robust Compiler Support for CUDA Graphs in PyTorch (arXiv 2503.19779, 2025)](https://arxiv.org/html/2503.19779v1)
- [PyTorch Issue #155682: CUDA Graph capture fails with dynamic indexing](https://github.com/pytorch/pytorch/issues/155682)
- [CUDA Graph Best Practice: Handling Dynamic Patterns (NVIDIA docs)](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/handling-dynamic-patterns.html)
- [Characterizing and Optimizing LLM Inference Workloads (arXiv 2504.11750)](https://arxiv.org/html/2504.11750v1)
- [Launch of many small kernels 10× slower (NVIDIA devforum)](https://forums.developer.nvidia.com/t/launch-of-many-small-kernels-10x-slower-compared-to-one-kernel/350194)
- [ThunderKittens: Simple, Fast, Adorable AI Kernels (arXiv 2410.20399)](https://arxiv.org/abs/2410.20399)
- [FlashInfer: Efficient and Customizable Attention Engine (arXiv 2501.01005)](https://arxiv.org/pdf/2501.01005)
- [Chipmunk: GPU Kernel Optimizations (2025)](https://sandyresearch.github.io/chipmunk-part-III/)
- [Graph-less Neural Networks (GLNN, ICLR 2022, arXiv 2110.08727)](https://arxiv.org/abs/2110.08727)
- [Consistency Deep Equilibrium Models (C-DEQ, 2024)](https://arxiv.org/html/2602.03024)
- [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding (Meta, ACL 2024, arXiv 2404.16710)](https://arxiv.org/abs/2404.16710)
- [Deep Equilibrium Models (arXiv 1909.01377)](https://arxiv.org/abs/1909.01377)
- [AOTInductor Documentation (PyTorch 2.11)](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_aot_inductor.html)
- [AOTInductor Practical Guide (2025)](https://johal.in/torch-compile-aotinductor-python-ahead-of-time-compilation-workflows-2025/)
- [State of torch.compile August 2025 (ezyang blog)](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)
- [PyTorch CUDAGraph Trees documentation](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)
- [tinygrad (extreme shape specialization for tiny models)](https://tinygrad.org/)
- [Mamba: Linear-Time Sequence Modeling (arXiv 2312.00752)](https://arxiv.org/pdf/2312.00752)
- [FTC-GNN: Sparse GNN Tensor Core (arXiv 2412.12218)](https://arxiv.org/html/2412.12218v2)
- [SA-MLP: Structure-Aware MLP Distillation (arXiv 2210.09609)](https://arxiv.org/pdf/2210.09609)

---

*File: `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/.planning/walltime_breakthrough_2026-04-15.md` — generated 2026-04-15 by Opus 4.6 (1M).*
