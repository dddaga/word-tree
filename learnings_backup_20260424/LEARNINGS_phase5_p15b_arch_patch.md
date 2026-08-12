# Phase 5 Part 15b: Arch Patch Results (Steps 66, 69, 70)

**Date:** 2026-04-06 to 2026-04-07
**Parent:** LEARNINGS_phase5_p15_post_wave1.md (TOC)
**Context:** Discovery of two silent bugs in pre-Gen4 codebase. +9.83pp total gain from fixes.

---

## The +9.83pp Code Fix Story

Before 2026-04-06, ALL phase 5 experiments (steps 57-68) ran on code with two silent bugs:

**Bug 1 — Input coverage gap (+6.29pp fix):**
`_build_fanin_conn` used random per-neuron sampling. At N=1024 with K_in=50 and N_in=25088,
~13% of VGG16 features NEVER reached any hidden neuron. The round-robin coverage guarantee
ensures every input feature reaches at least one neuron in its group. This is the larger
contributor (64% of total gain). The coverage gap grows with N — at N=4096 the gap was
proportionally larger. All N-scaling results from step56 are underestimates.

**Bug 2 — alpha_reflect silenced (+3.54pp fix):**
`SGNNET_AntiHebbian.forward()` reimplemented the routing loop but omitted the `Z_reflected`
accumulator. `alpha_reflect=0.5` was stored in `self.m` but never applied across K_iter.
Reflection is a form of residual connection through time — each routing step remembers what
it suppressed, giving the network a way to "reconsider" activations below threshold.

**The two fixes are complementary:** Input coverage ensures the signal is complete at entry;
alpha_reflect ensures suppressed signal has a second chance across iterations.
- input_coverage fix alone: +6.29pp (64% of gain)
- alpha_reflect fix alone: +3.54pp (36% of gain)
- Both combined: +9.83pp total

---

## Step 66: Phase-Target Routing (Local Plasticity) — COMPLETE

**Script:** train_step66_phase_target.py (Mac Mini MPS, 50%/75ep)
**Note:** Script crashed on first launch (resonant_mode= kwarg bug), fixed and relaunched.

| Config | Description | top1_best | vs Ref (83.18%) |
|--------|-------------|-----------|-----------------|
| Ref | Static AH=1.0 (phase-target disabled) | **83.18%** | — |
| A | PhaseTarget routing only — no AH | **67.87%** | −15.31pp |
| B | PhaseTarget + plasticity (lr=0.01) | **57.61%** | −25.57pp |
| C | PhaseTarget + plasticity + diversity (β=0.1) | **61.48%** | −21.70pp |
| D | PhaseTarget + plasticity + AH (ah_alpha=1.0) | **40.33%** | −42.85pp |

**Note on Ref = 83.18%:** Anomalously high vs expected ~73.5%. This is the patched architecture
(input_coverage + alpha_reflect fix). The +9-10pp gap confirms these scripts use the patched base.

**STEP66 VERDICT: ALL phase-target routing variants KILLED.** Static AH is the optimal fixed point.
- A: AH is load-bearing — removing it costs −15pp even without plasticity
- B: Plasticity makes it worse (57.61% vs 67.87%) — gate-death pattern
- C: Diversity penalty partially recovers but −21pp vs Ref — insufficient
- D: AH + phase-target = worst combination (−42pp). They are antagonistic:
  AH suppresses exactly the phase activity that phase-target depends on.

Local plasticity hypothesis falsified. CLOSED.

---

## Step 69: Corrected REF_BASELINE on Patched Architecture — COMPLETE

**Script:** train_step69_corrected_baseline.py (Mac Studio MPS, 50%/75ep)
**Fixes applied:** input coverage guarantee + alpha_reflect restored in AntiHebbian

| Config | Description | top1_best | vs old REF (73.53%) |
|--------|-------------|-----------|---------------------|
| Ref | Gen4 patched (turing=0.0, reflect=0.5, AH=1.0) | **83.36%** | **+9.83pp** |
| A | step57-exact patched (turing=0.3, reflect=0.5, AH=1.0) | **85.04%** | **+11.51pp** |
| B | Gen4 patched, reflect=0.0 (ablate reflection) | **79.82%** | **+6.29pp** |

**NEW REF_BASELINE_v2 = 83.36%** (patched arch, no turing).
**NEW GEN4+ CANDIDATE = 85.04%** (patched arch, turing=0.3).

**alpha_turing finding:** A (turing=0.3) = 85.04% → +1.68pp vs Ref.
Gen4 should adopt turing=0.3 at N=1024. The Turing fraction gain was masked by buggy code.

---

## Step 47: Interneurons D=64 — COMPLETE (buggy arch)

| Config | Description | top1_best | vs OLD_REF (73.53%) |
|--------|-------------|-----------|---------------------|
| Ref | baseline, no interneurons | 56.79% | −16.74pp |
| F | marginal | **57.55%** | **−15.98pp** |
| A | 25% interneurons, readout=all | 55.62% | −17.91pp |
| G | training too short | 54.17% | — |
| B | 50% interneurons | 43.24% | catastrophic |
| C/D/E | gate-dead | 33-38% | — |

Interneurons don't compound with AH at D=64. Degradation accelerates with fraction. KILLED.

---

## Step 53: Low-Rank Dimension Mixing — COMPLETE (buggy arch)

| Config | Description | top1_best | vs Ref (70.17%) |
|--------|-------------|-----------|-----------------|
| Ref | D=64 N=1024 AH α=0.5 | 70.17% | — |
| A | + low-rank UV^T mixing | 69.15% | −1.02pp |
| B | + frequency-pair 2×2 mixing | 67.95% | −2.22pp |
| C | + group 8×8 mixing | 71.06% | +0.89pp |
| D | + group 8×8×8 mixing α=0.1 | 70.62% | +0.45pp |
| E | + freq-pair + AH α=1.0 | 69.78% | −0.39pp |

All below OLD_REF 73.53%. Low-rank mixing disrupts the Fourier layout. KILLED.

---

## Buggy-Arch Backlog: What Results Tell Us

**Stacking (step64):** Harmful on buggy arch → likely also harmful on patched arch.
The geometric reason (same W_pos space, over-smoothing) is architecture-independent.
CLOSED permanently — EXCEPT Config F (parallel concat-project, +1.52pp) → test on patched arch (step85).

**Low-rank mixing (step53):** Neutral/harmful → likely the same on patched arch. CLOSED.

**Interneurons (step47):** Harmful at D=64 (vs helpful at D=16). CLOSED for naive implementation.

**W_phase reconnect (step46):** On patched arch, W_phase may interact differently with the
restored reflection accumulator. The step46 result is informative but not conclusive.
**Consider re-testing on patched arch.**

---

## New Configuration Hierarchy (Patched Arch)

| Config | 50%/75ep | 100%/150ep est. |
|--------|---------|-----------------|
| Gen4 base (turing=0.0, reflect=0.5, AH=1.0, K_iter=8) | 83.36% | ~84%+ |
| Gen4+ (turing=0.3, reflect=0.5, AH=1.0, K_iter=8) | 85.04% | ~86%+ |
| Gen4+ K12 (turing=0.3, K_iter=12, N=4096) | ? | likely >97.32% |

**Critical unknowns (2026-04-06):**
1. K_iter optimal on patched arch at N=4096 → step71
2. N-scaling curve on patched arch → step80
3. Safety valve λ on patched arch → step76
4. turing=0.3 × K_iter interaction → may need joint calibration

---

## Session Synthesis: 2026-04-06 Full Backlog Review

| Step | Description | Machine | Arch | Verdict |
|------|-------------|---------|------|---------|
| step69 | Corrected REF_BASELINE | Mac Studio MPS | PATCHED | 83.36% (NEW BASELINE) |
| step66 | Phase-Target routing | Mac Mini MPS | PATCHED | ALL KILLED (40-68%) |
| step64 | Stacked SGNNET | Mac Mini CPU | BUGGY | ALL KILLED except Config F |
| step53 | Low-rank mixing | Mac Studio CPU | BUGGY | ALL KILLED (69-71%) |
| step47 | Interneurons D=64+AH | Mac Studio MPS | BUGGY | ALL KILLED (43-56%) |

---

## Step 70: Full-Scale Gen4+ Corrected Run — NEW PROJECT BEST (2026-04-07)

**Config:** N=4096, D=64, K_iter=8, 150ep, 100% data. Both bugs fixed.

| Config | top1_best | best_ep | vs old best |
|--------|-----------|---------|-------------|
| **Ref** (turing=0.3 reflect=0.5 AH=1.0) | **97.20%** | 126/150 | **+12.84pp** |
| **B** (turing=0.0 reflect=0.5 AH=1.0) | **97.32%** | **90**/150 | **+13.08pp** (WAS PROJECT BEST) |

**Note:** Current project best is 97.38% (step70-B on corrected run — see LEARNINGS_phase5_p15d).

**Key finding: turing=0.0 > turing=0.3 at N=4096.** Turing mechanism is slightly harmful at
full scale. Contrast with N=1024 (step69): turing=0.3 gave +1.68pp. Turing contribution is N-dependent.

New Gen4+ optimum at N=4096: turing=0.0, reflect=0.5, AH=1.0.

---

## step37 — Phase matrix bank (2026-04-07 COMPLETE)

Best: **56.79%** at e120/150. Matches Ref baseline, no gain. **KILLED.** CLOSED.

---

## step32 — Gen4 compound stacking (2026-04-07)

| Config | Mechanism stack | top1_best |
|--------|-----------------|-----------|
| Ref | calibrated base only (no AH) | 65.04% |
| A | + AH α=1.0 | **73.76%** |
| B–G | + various compounds | 57–72% |

**AH alone wins. Compounding kills.** A(AH alone)=73.76% ≈ REF_BASELINE 73.53%.
Adding any mechanism to AH reduces accuracy. Consistent with step29c. NOTE: buggy arch — relative finding valid.

---

## step46 — W_phase reconnect (2026-04-07 COMPLETE)

| Config | Mechanism | top1_best | best_ep |
|--------|-----------|-----------|---------|
| Ref | standard dynamic_z_geo | 56.43% | 120 |
| A | W_phase → phase gate (additive) | 58.90% | 85 |
| B | W_phase → phase scale (multiplicative) | 58.70% | 119 |
| C | W_phase → routing score (learned) | **59.62%** | 137 |
| D | W_phase → inhibition mask | 58.29% | 148 |

**BORDERLINE.** Best = C (learned routing score, +3.19pp). Below 60% Gen4 re-test threshold.
W_phase adds marginal value when disconnected routing is already working. CLOSED at pre-Gen4 base.
