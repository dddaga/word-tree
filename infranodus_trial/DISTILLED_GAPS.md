# SGNNET Distilled Research Gaps

**Generated:** 2026-04-08  
**Method:** Three-way analysis — V1 (co-occurrence), V2 (TF-IDF), V3 (semantic reading)  
**Corpus:** 25 learnings files + 47 result JSONs (293,887 chars)

---

## Cross-Method Validation

| Gap | V1 | V2 | V3 | Priority |
|-----|----|----|-----|---------|
| G1: K_iter=12 + turing=0.0 at N=4096, full scale | — | — | ✓ | HIGH |
| G2: Group topology (n_groups=8) at N=4096 | — | — | ✓ | HIGH |
| G3: Full N-scaling curve on patched arch | — | — | ✓ | HIGH |
| G4: Softmax/temp routing at N=4096 | — | — | ✓ | HIGH |
| G5: Sparse attention literature ↔ SGNNET empirics | ✓ | ✓ | ✓ | MEDIUM |
| G6: Excitatory/resonance ↔ aux losses | — | ✓ | ✓ | MEDIUM |
| G7: Backward pass efficiency | — | ✓ | ✓ | MEDIUM |
| G8: Redistribution routing + group topology compound | — | — | ✓ | MEDIUM |
| G9: Alpha calibration on patched arch at N=4096 | — | — | ✓ | MEDIUM |
| G10: Coherence evolution across K_iter steps | — | ✓ | ✓ | LOW |
| G11: Stacked parallel SGNNET on patched arch | — | — | ✓ | LOW |
| G12: Safety valve recalibration on patched arch | — | — | ✓ | LOW |

---

## HIGH Priority — Blocks Next Generation

### G1: K_iter=12 + turing=0.0 never combined at full scale

**Problem:** Two confirmed wins exist independently: K_iter=12 at N=4096 (+0.79pp, step71) and turing=0.0 (+0.18pp, step70). Neither was tested together at 100%/150ep.

**What's missing:** A single full-scale run with K_iter=12, turing=0.0, N=4096, 150ep. Current best (97.38%) uses K_iter=8 + turing=0.0. Combining both could push 97.5%+.

**Specific experiment:** step72 or a new step: N=4096, K_iter=12, turing=0.0, AH=1.0, reflect=0.5, 100%/150ep.

**V1/V2 signal:** Not detected — these are result-layer gaps, not text co-occurrence gaps. Requires semantic reading.

---

### G2: Group topology (n_groups=8) at N=4096

**Problem:** step82 confirmed +3.01pp for n_groups=8 at N=1024 (85.63% vs 82.62%). Never tested at N=4096 where the base is 95.87%.

**What's missing:** Does group topology compound with N-scaling? If yes, could push past 97.38% via topology alone with zero new params.

**Specific experiment:** step82 at N=4096 — same script, just change N. 50%/75ep first to gauge, then 100%/150ep if promising.

**Expected direction:** Likely positive — group specialisation effect should strengthen with larger N (more neurons per group).

---

### G3: Full N-scaling curve on patched arch

**Problem:** step56 N-scaling curve is on buggy arch (both bugs fixed in step69 = +9.83pp). The patched arch has a completely different ceiling. step80 has partial data (N=512: 72.79%, N=2048: 92.74%) but no N=8192 or N=16384. The non-monotonicity above N=4096 was on buggy arch — unknown on patched arch.

**What's missing:** N={512,1024,2048,4096,8192} at 100%/150ep on patched arch. Script needed (step72 in queue, no script yet).

**Specific experiment:** step72 — N-scaling on patched arch, 5 configs. Most critical missing data point for the N-scaling law hypothesis.

---

### G4: Redistribution routing at N=4096

**Problem:** step73 (softmax routing, +1.78pp) and step75 (temperature routing, +3.98pp) were both run at N=1024. Never tested at N=4096 where static AH already achieves 95.87%.

**What's missing:** Does dynamic routing add value when the static base is already 95%+? The gate-death theorem is scale-independent, but the marginal value of redistribution routing may decrease at large N where the dense small-world graph already provides rich signal paths.

**Specific experiment:** step75 Config D (best: learned W_temp, tau_0=0.3) at N=4096, 50%/75ep first.

**Expected direction:** Unknown — could add +1-2pp or be marginal. Critical for understanding whether redistribution routing generalizes across scales.

---

## MEDIUM Priority — Opportunity or Untested Combination

### G5: Sparse attention literature disconnected from SGNNET empirics

**V1 signal:** Cluster [3] sgnnet/attention ↔ [5] arch/scaling (density=0.016) — structural gap between the sparse attention research files and the architecture scaling results.  
**V2 signal:** Cluster [5] active/sparsity/FFN/experts exists in isolation.

**Problem:** Three learnings files cover sparse attention research (`LEARNINGS_sparse_attention_*.md`) but are never referenced in experimental design or result analysis. The FFN-replacement goal exists as a stated objective but never informs experiment selection.

**What's missing:** A structured mapping: "which SGNNET mechanism corresponds to which transformer FFN component?" Concretely:
- SGNNET's K_in fan-in = sparse input projection (like SparseGPT)
- AH suppression = attention sparsification (like Longformer local attention)
- K_iter routing = depth/recurrence analog
- Group topology = MoE routing

**Specific experiment:** No new training needed. Write a comparison document mapping SGNNET components to sparse attention literature. Identify which external techniques could be adapted to SGNNET (e.g., dynamic K_in sparsification similar to Mixture-of-Experts routing).

---

### G6: Excitatory/resonance mechanisms never tested with aux losses

**V2 signal:** Cluster [7] excitatory/inhibition/resonance ↔ [11] loss/aux (density=0.002) — two isolated knowledge islands.

**Problem:** All aux loss experiments (step79: sparsity, diversity, phase coherence) were run on static AH routing. The resonance/excitatory mechanisms (steps 58, 61) were killed purely on accuracy, not analyzed for what loss signal they generate. If excitatory mechanisms produce sparser activations, pairing them with a sparsity reward aux loss might stabilize training.

**What's missing:** Test: static AH + sparsity aux loss (step79 winner D, +0.21pp) vs resonance gate + sparsity aux loss. The aux loss may compensate for the activation collapse that killed step58.

**Specific experiment:** One config: step58 resonance gate + sparsity aux λ=0.001, N=1024, 50%/75ep.

**Expected direction:** Speculative — could partially rescue resonance at low cost. Worth one config.

---

### G7: Backward pass efficiency never analyzed

**V2 signal:** Cluster [5] FFN/sparsity ↔ [16] backward/cdist/shape (density=0.004).

**Problem:** EXPERIMENT_REPORT.md counts FLOPs for the forward pass only. For the FFN-replacement claim to hold, backward pass efficiency matters equally for training. SGNNET has K_iter=8 sequential steps of gather+normalize — the backward pass must backpropagate through all 8 steps, which could be expensive due to the normalization Jacobians.

**What's missing:** Estimate backward pass FLOPs. Check: does AH's wpos suppression create gradient issues through the normalization? Are the fixed buffers (conn_in, conn_hh) causing any backward pass fragmentation?

**Specific experiment:** Profile forward + backward on a single batch. Add to EXPERIMENT_REPORT.md.

---

### G8: Redistribution routing + group topology never combined

**Problem:** step73/75 (softmax/temp routing) and step82 (group topology) were designed independently and tested sequentially. The gate-death theorem says redistribution is the viable routing paradigm. Group topology says random groups improve specialization. These could compound.

**What's missing:** Group topology (n_groups=8) + softmax redistribution routing in one model. No interaction between them was tested.

**Specific experiment:** step82 Config A topology (n_groups=8) + step73 Config D routing (softmax, tau=0.3), N=1024, 50%/75ep. 2 configs: just group topology (Ref) vs group topology + redistribution routing.

**Expected direction:** Positive if mechanisms are orthogonal. Risk: softmax routing may interact poorly with within-group AH suppression (same double-sparsity concern as wave-1, but milder since redistribution not gating).

---

### G9: AH alpha never recalibrated on patched arch at N=4096

**Problem:** The alpha=1.0 winner came from step29c (N=1024, buggy arch). On patched arch (step69 +9.83pp), at N=4096, alpha=1.0 was carried forward without recalibration. Could alpha > 1.0 add value at larger N? Or does the patched arch change the optimal alpha?

**What's missing:** Quick alpha sweep at N=4096, patched arch: alpha ∈ {0.5, 1.0, 1.5, 2.0}, 50%/40ep.

**Specific experiment:** 4 configs, 40ep calibration sweep. Low cost for potentially important finding.

---

## LOW Priority — Research/Exploratory

### G10: Phase coherence evolution across K_iter never profiled

**V2 signal:** Cluster [8] coherence/freq/mixing ↔ [13] forward/iter1/pass (density=0.006).

**Problem:** Phase routing was killed at the end-result level (final accuracy), but we never studied how phase coherence evolves step-by-step across K_iter. If coherence collapses early (by step 2-3), later steps are running on noise. This could explain why step60 was killed even when using softmax weights.

**What's missing:** Add diagnostic logging to a phase routing variant: log mean coherence at each K_iter step. Does coherence grow, stabilize, or collapse?

**Specific experiment:** Diagnostic run — not a full training job. Add logging to step73 or step80 variant.

---

### G11: Stacked parallel SGNNET on patched arch

**Problem:** step64 Config F (2-parallel concat-project) showed +1.52pp on buggy arch, best_ep=75/75 (still converging). Never re-tested on patched arch where the base is 10pp higher.

**What's missing:** step85 (queued) — same script, patched arch. N=1024, 50%/75ep.

**Specific experiment:** 3 configs: Ref, A (2-parallel concat-project), B (same + K_iter=12 per branch). Script needed.

---

### G12: Safety valve (lambda_safety) never recalibrated on patched arch

**Problem:** step67 (lambda ablation) ran on buggy arch. With +9.83pp base shift, the optimal safety valve strength may have changed. step74 is queued but has no script.

**What's missing:** step74 — lambda sweep on patched arch. Low urgency since safety valve is not in the critical path.

---

## Proposed Experiment Priority Order

```
GENERATION ADVANCE (validates N-scaling law):
  step72: N-scaling curve, patched arch, 5 configs          ← SCRIPT NEEDED

COMBINE CONFIRMED WINS:
  G1:  K_iter=12 + turing=0.0 at N=4096, 100%/150ep        ← 1 config, no script
  G2:  Group topology at N=4096, 50%/75ep                   ← use step82 script, change N

SCALE TEST (validate redistribution routing generalizes):
  G4:  step75 Config D at N=4096, 50%/75ep                  ← use step75 script, change N

COMPOUND TEST:
  G8:  Group topology + redistribution routing, N=1024      ← 2 configs

CALIBRATION:
  G9:  AH alpha sweep on patched arch at N=4096             ← 4 configs, 40ep

LITERATURE:
  G5:  SGNNET ↔ sparse attention mapping document           ← no training
```

---

## What V1/V2 Found That Didn't Survive V3 Filtering

| V1/V2 Gap | Why Filtered |
|---|---|
| [10] mixing/interneurons ↔ [11] bugs (V1) | Noise — bugs cluster is meta-text, not a research topic |
| [15] interneurons/hub ↔ [16] shape/backward (V2) | Hub interneurons killed (step61); shape/backward is ops noise |
| [12] reflection/Oja/Hopfield ↔ [13] forward/iter1 (V2) | Oja's KILLED (step41); Hopfield not implemented; reflection is working and doesn't need reconnection to forward-pass internals |
| [7] dynamic/Fourier ↔ most clusters (V1) | Already surfaced as G4 (redistribution at N=4096) and G10 (coherence profiling) |
