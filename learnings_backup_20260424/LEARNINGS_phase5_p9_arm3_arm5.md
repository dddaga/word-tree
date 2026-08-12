# Phase 5 Part 9: ARM 3 Dynamic Connectivity Closure + ARM 5 Input Architecture

**Date:** 2026-04-03
**Status:** In progress -- step36 complete; step49/50/51/55/PCA queued for dispatch
**Scope:** Close ARM 3 (dynamic connectivity viability) and ARM 5 (best input mechanism)

---

## ARM 3: Dynamic Connectivity at D=64 -- PARTIAL RESULTS

### Question
Can O(N*K) input-dependent dynamic topology match what O(N^2) signed coupling achieved at D=16?

Reference points:
- **Static SmallWorld ceiling:** 56.28% (step22E, no dynamic connectivity)
- **AntiHebb best:** 75.24% (step29 Config C, alpha=0.7, wpos -- all-time best)
- **N-squared recovery target:** 10.93pp gap (56.28% to hypothetical 67.21% full N^2 equivalent)

### Completed: step36 -- Input-Gated Adjacency

**Mechanism:** Modulate static conn_hh edge weights by input similarity. Soft gate: `w_ij = sigmoid((x_i . x_j) / tau)`. Hard gate: `w_ij = 1 if (x_i . x_j) > theta else 0`.

| Config | Mechanism | top1_best | vs Ref | vs Ceiling | N^2 Recovery |
|--------|-----------|-----------|--------|------------|--------------|
| Ref | Static SmallWorld (no gate) | 56.79% | -- | +0.51pp | 4.7% |
| **A** | **Soft gate tau=1.0 theta=0.0** | **58.62%** | **+1.83pp** | **+2.34pp** | **21.4%** |
| B | Soft gate tau=0.5 theta=0.0 | 57.99% | +1.20pp | +1.71pp | 15.6% |
| C | Soft gate tau=2.0 theta=0.0 | 57.66% | +0.87pp | +1.38pp | 12.6% |
| D | Hard gate theta=0.0 | 41.07% | -15.72pp | -15.21pp | -139.2% |
| E | Hard gate theta=0.3 | 11.21% | -45.58pp | -45.07pp | -412.3% |

**Key findings:**
1. **Soft gating works:** Config A adds +2.34pp over static ceiling (58.62%). This is the strongest O(N*K) dynamic connectivity signal at D=64 so far.
2. **Temperature sweet spot:** tau=1.0 (default) is best. Sharper (tau=0.5) or smoother (tau=2.0) both degrade.
3. **Hard gating is catastrophic:** Binary 0/1 edge masking destroys the routing signal. At D=64, cosine similarities on S^63 are near-zero at init -- hard thresholding zeroes most edges immediately.
4. **Differentiability matters:** The gap between soft A (58.62%) and hard D (41.07%) proves that gradient flow through gate values is critical.

### Completed: step31 -- Dynamic Z-KNN (activation-based K-NN)

| Config | Mechanism | top1_best | vs Ceiling |
|--------|-----------|-----------|------------|
| Ref | Static SmallWorld | 56.79% | +0.51pp |
| A | Z-KNN K=8, per step, alpha=1.0 | 46.24% | -10.04pp |
| B | Z-KNN K=16, per step, alpha=1.0 | 47.24% | -9.04pp |
| C | Z-KNN K=32, per step, alpha=1.0 | 47.85% | -8.43pp |
| D | Z-KNN K=8, first step only | 47.67% | -8.61pp |
| E | Z-KNN K=8, per step, alpha=0.3 (additive) | 52.74% | -3.54pp |

**Key finding:** Per-step dynamic Z-KNN hurts badly (-8 to -10pp). Even additive blending (E, alpha=0.3) loses 3.5pp. Activation-based topology is fundamentally noisy at D=64 -- the routing signal hasn't converged enough at any step to serve as a reliable connectivity basis.

### Queued: step49 -- Signed Coupling x K_iter Threshold

**Question:** Is there a K_iter sweet spot (4-7) where signed coupling works at D=64? K_iter=3 = noise (cos-sim on S^63 too small), K_iter=8 = power-iteration collapse.

**Design:** 40-epoch calibration per K_iter={3,4,5,6,7,8} at alpha=0.3. Decision: any config > Ref at 40ep (~42%) = viable.

**Status:** Script ready (`scripts/train_step49_signed_kiter_threshold.py`). Dispatch blocked: 3 experiments running on Mac Studio (cap = 2).

### Queued: step50 -- Spatial W_pos K-NN Dynamic Connectivity

**Question:** Does W_pos-based topology evolution (rebuild conn_hh from W_pos distances every epoch) beat static SmallWorld?

**Design:** 5 configs (K=6, K=8, +interneurons, +AntiHebb, slow evolution) vs static Ref. 150 epochs.

**Status:** Script ready (`scripts/train_step50_spatial_dynamic_conn.py`). Dispatch blocked.

### Queued: step51 -- W_pos K-NN + W_phase Strength Gating

**Question:** Does two-level connectivity (W_pos topology + W_phase strength) beat spatial-only or static?

**Design:** 5 configs including key ablation E (static conn + phase gate only). 150 epochs.

**Status:** Script ready (`scripts/train_step51_spatial_phase_gating.py`). Dispatch blocked.

### ARM 3 Preliminary Verdict (based on step36 + step31)

**PARTIAL-GO (provisional)**

Evidence:
- **FOR dynamic connectivity:** step36 Config A (soft input-gated) achieves 58.62%, beating static ceiling by +2.34pp. This proves O(N*K) input-dependent topology CAN add value.
- **AGAINST:** The gain is modest (21.4% of N^2 recovery) and falls far short of AntiHebb (75.24%).
- **Critical gap:** step36 was tested WITHOUT AntiHebb. The compounding question (does input-gating + AntiHebb > AntiHebb alone?) is unanswered.

**Next steps to finalize verdict:**
1. Run step49: If signed coupling has a K_iter sweet spot, that changes the analysis
2. Run step50/51: If spatial dynamics + phase gating beats input-gating, the best dynamic mechanism may shift
3. Test compounding: step36 Config A mechanism + AntiHebb alpha=0.7 on same base
4. Final verdict after all 4 experiments complete

---

## ARM 5: Input Architecture -- EXPERIMENT QUEUE

### Current Baseline
- **Random K_in=50 sparse gather** from N_in=25088, n_groups=128 flat blocks
- With AntiHebb alpha=0.7: **75.24%** (step29 Config C)
- No learned projection; fixed uniform aggregation per group

### Queued: step55 -- Spatial Grouped Learned Input Projection

**Question:** Does learned per-group spatial projection beat random K_in=50?

| Config | Mechanism | Groups | Overlap |
|--------|-----------|--------|---------|
| Ref | Random K_in=50 (current best) | 128 flat | 100% global |
| A | 7 rows, separate proj, 0% overlap | 7 (VGG spatial) | 0% (pure exclusive) |
| B | 7 rows, separate proj, 20% overlap | 7 | 20% bridge neurons |
| C | 7 rows, separate proj, 40% overlap | 7 | 40% bridge neurons |
| D | 49 positions, separate proj, 20% | 49 | 20% bridge |
| E | 8 flat groups, separate proj, 20% | 8 | 20% bridge |

**Status:** Script + model ready. Dispatch blocked (Mac Studio concurrency).

### Queued: PCA Input Sweep

**Question:** Does PCA-compressed input (k=256, 512, 1024) beat random K_in=50?

| k | N_in | K_in | Expected cumvar |
|---|------|------|-----------------|
| 256 | 256 | 50 | ~0.85-0.90 |
| 512 | 512 | 50 | ~0.92-0.95 |
| 1024 | 1024 | 50 | ~0.97-0.99 |

**Status:** Script written (`scripts/pca_input_sweep.py`). Dispatch blocked.

### ARM 5 Verdict: PENDING

All three input mechanisms (random K_in=50, spatial grouped, PCA) need results before a winner can be declared.

---

## Cross-ARM Analysis: PENDING

**Question:** Does the best dynamic topology (ARM 3 winner) compound with the best input mechanism (ARM 5 winner)?

This requires:
1. ARM 3 verdict (best dynamic topology mechanism)
2. ARM 5 verdict (best input mechanism)
3. A compounding experiment testing both together

---

## Summary Table (all dynamic connectivity experiments at D=64)

| Step | Mechanism | Complexity | Best top1 | vs Ceiling | Status |
|------|-----------|-----------|-----------|------------|--------|
| step31 | Z-KNN (activation K-NN) | O(N*K*D) per step | 52.74% (E) | -3.54pp | DONE -- hurts |
| **step36** | **Input-gated adjacency (soft)** | **O(N*K) amortized** | **58.62% (A)** | **+2.34pp** | **DONE -- helps** |
| step49 | Signed coupling K_iter sweep | O(N^2*D) per step | -- | -- | QUEUED |
| step50 | W_pos K-NN per epoch | O(N^2) per epoch | -- | -- | QUEUED |
| step51 | W_pos K-NN + W_phase gate | O(N^2) per epoch | -- | -- | QUEUED |

**Note:** step49/50/51 use O(N^2) operations for the topology rebuild, but the per-step routing remains O(N*K). The N^2 cost is amortized over the epoch (rebuild once, use K_iter times per batch). For N=1024, this is acceptable; for N=10000+ it would need approximation.

---

*Updated: 2026-04-03 -- step36 complete; step49/50/51/55/PCA queued for dispatch when Mac Studio slots open*
