# SGNNET Distilled Research Gaps

Generated: 2026-04-01
Sources: V1 (co-occurrence), V2 (TF-IDF weighted), V3 (full-context LLM reading)
Method: Three-way parallel analysis, cross-validated and deduplicated

---

## TIER 1 — HIGH PRIORITY (block next generation if unresolved)

### G1. Signed Coupling x High-D Calibration
**Sources:** V3 Gap 1, V3 Tension 1, V3 Tension 5
**The problem:** Signed coupling (+10.93pp at D=16) and D=64 (+19.54pp) are the two strongest axes. Combined they regress to 32.15% (step28) vs 56.28% base. Power iteration converges to dominant eigenvector at K_iter>=8 — hard constraint, not tunable.
**What's missing:** Alpha/tau calibration of signed coupling at D=64. The cosine landscape on S^63 is fundamentally different from S^15. step22b (base routing calibration) is running but signed coupling calibration is not scheduled.
**Experiment:** D=64 K_iter=3 signed coupling alpha sweep {0.01, 0.05, 0.1, 0.3} + tau sweep. 40ep calibration runs.
**Blocks:** Gen4 compound design.

### G2. Anti-Hebbian + Signed Coupling Compound
**Sources:** V3 Gap 4, V3 Combination 1
**The problem:** The two largest individual gains (+8pp anti-Hebbian, +10.93pp signed) have never been tested together at any scale. They operate on orthogonal axes: W_pos spatial suppression vs Z activation coupling.
**What's missing:** Combined experiment. Estimated compound at D=16: ~48% (at 70% compounding).
**Experiment:** D=16 K_iter=3: Ref, +AntiHebb(0.5), +Signed(0.3), +AntiHebb+Signed. 150ep.

### G3. Fast Weights x K_iter=8
**Sources:** V3 Gap 3, V3 Combination 2
**The problem:** Fast W_phase attention (+1.84pp) was only tested at K_iter=3. The mechanism is explicitly designed to benefit from more iterations (Oja converges in O(K_iter) steps). At K_iter=3 it barely adapts; at K_iter=8 it could fully converge.
**What's missing:** Fast W_phase at K_iter=8.
**Experiment:** D=16 N=1024 K_iter=8: Ref, +fast_W_phase(attention, tau=0.25). 150ep.

### G4. Beam Routing x Signed Coupling (Dynamic Sparsification)
**Sources:** V3 Gap 8, V3 Combination 4
**The problem:** Signed coupling is O(N^2). Beam routing selects route=64 input-dependent active neurons. Applying signed coupling only to the beam subset = O(64^2) instead of O(N^2) = 64x cheaper. Step24's static K-NN sparsification recovered only 37% — but beam selection is input-dependent, fundamentally different.
**What's missing:** Beam-restricted signed coupling.
**Experiment:** D=16 K_iter=3: Ref, +signed(full N^2), +signed(beam=64 only), +signed(beam=128). Compare accuracy and FLOP cost.

### G5. Loss Function x Routing (Entirely Unexplored Axis)
**Sources:** V3 Gap 6, V2 gap [10]↔[11] (dim/stdp ↔ loss/safety)
**The problem:** Every experiment modifies the forward pass but uses identical loss (task + safety). No routing-aware auxiliary losses exist. V2's TF-IDF analysis independently confirmed this: loss/safety cluster is disconnected from mechanism clusters.
**What's missing:** Auxiliary losses that reward routing quality (diversity, sparsity, anti-over-smoothing).
**Experiment:** Step39 (running) is the first to bridge this. Await results before designing follow-ups.

### G6. All Mechanisms on Uncalibrated D=64 Base
**Sources:** V3 Tension 5, V3 Gap 2
**The problem:** step22b (base routing param calibration at D=64) is running. ALL current D=64 mechanism experiments (step29, 29b, 31, 33, 34, 36) run on uncalibrated alpha_turing/alpha_reflect/K_phase/beam_size/geo_gamma. Results may be invalidated by calibration.
**What's missing:** step22b completion → step29c (re-validate ALL mechanisms on calibrated base).
**Blocks:** Everything at D=64.

---

## TIER 2 — MEDIUM PRIORITY (opportunity gaps)

### G7. Safety Valve Ineffective at D=64
**Sources:** V3 Gap 12
**The problem:** Safety valve r* = 0.5/N^(1/D). At D=64: r* ≈ 0.49 — neurons on S^63 are already far apart, repulsion may never activate. The formula was designed for D=4. W_pos may be unconstrained at D=64.
**Check:** Log safety_loss values from step22 D=64 runs. If near-zero, the valve is dead.

### G8. Interneurons x Signed Coupling (on correct base)
**Sources:** V3 Gap 5, V3 Combination 6
**The problem:** Interneurons as "routing attractors" in a signed coupling regime could serve as stable cluster centers (DeepSeekMoE shared experts). Step28 tested this but on the wrong base (signed + K_iter=3 at D=64).
**Experiment:** After G1/G6 resolve, test at D=16 K_iter=3 with properly compounded mechanisms.

### G9. MoD Adaptive Depth x Over-Smoothing
**Sources:** V3 Gap 10, V3 Combination 5
**The problem:** K_iter=12 over-smooths. MoD lets neurons freeze when converged. Refractory inhibition (+4.31pp) also prevents dominant neurons from collapsing across steps. Either could extend useful depth beyond K_iter=8.
**Experiment:** Step34 (running). Also: refractory inhibition at K_iter=12 to test anti-smoothing.

### G10. Cross-Dim Mixing x Fast Weights
**Sources:** V3 Gap 11
**The problem:** W_mix (D x D matrix) is static across inputs. A fast-weight or phase-queried W_mix would be truly input-specific cross-dimensional mixing. Neither step30 nor step37 uses intra-forward adaptation of the mixing matrix.
**Experiment:** After step30/step37 results, combine best mixing variant with fast-weight adaptation.

### G11. W_phase Disconnection in Best Config
**Sources:** V3 Tension 7
**The problem:** In dynamic_z_geo mode, W_phase is normalized and passed to _phase_inhibit() but _inhibit_dynamic_z() IGNORES it. The architecture's namesake mechanism is inactive in all D=16+ experiments. W_phase exists, receives gradients through readout, but its routing role is dead.
**Check:** Verify W_phase is actually unused in routing. If so, either reconnect it or remove the parameter.

---

## TIER 3 — LOWER PRIORITY (research questions)

### G12. Translation Invariance x Fourier Encoding
**Sources:** V3 Gap 7
**What:** SGNNET is explicitly position-dependent via Fourier encoding. Channel-only C_input would make seeding position-invariant. Never tested since Fourier was introduced.

### G13. Matformer Nested Training
**Sources:** V3 Gap 9
**What:** Train one model that subsumes all N variants. Engineering optimization, 3x compute savings.

---

## CROSS-METHOD VALIDATION

| Gap | V1 found? | V2 found? | V3 found? | Actionable? |
|-----|-----------|-----------|-----------|-------------|
| G1 Signed x D=64 | No | No | Yes | Yes |
| G2 AntiHebb + Signed | No | No | Yes | Yes |
| G3 FastW x K_iter=8 | No | No | Yes | Yes |
| G4 Beam x Signed sparse | No | No | Yes | Yes |
| G5 Loss x Routing | No | Partial | Yes | Awaiting step39 |
| G6 Uncalibrated D=64 | No | No | Yes | Yes |
| G7 Safety valve D=64 | No | No | Yes | Quick check |
| G8 Interneurons x Signed | No | No | Yes | After G1/G6 |
| G9 MoD x Over-smoothing | No | No | Yes | Step34 running |
| G10 Cross-dim x Fast | No | No | Yes | After step30/37 |
| G11 W_phase disconnected | No | No | Yes | Quick check |
| G12 Translation invariance | No | No | Yes | Low priority |
| G13 Matformer nested | No | No | Yes | Low priority |

**Observation:** V1/V2 co-occurrence methods found zero actionable research gaps. V2 partially detected G5 (loss ↔ mechanisms disconnection). All 13 actionable gaps came from V3 context reading. Co-occurrence is useful for visualization and cluster structure, not for gap reasoning.

---

## IMMEDIATE NEXT EXPERIMENTS (priority order)

1. **Wait for step22b** (D=64 calibration) — blocks everything at D=64
2. **G2: Anti-Hebbian + Signed compound** at D=16 K_iter=3 — highest expected ROI
3. **G3: Fast W_phase at K_iter=8** — quick test, high theoretical upside
4. **G4: Beam-restricted signed coupling** — if it works, solves O(N^2)
5. **G7: Check safety valve at D=64** — 5-minute log inspection, could explain D=64 mechanism failures
6. **G11: Check W_phase routing path** — 5-minute code inspection
