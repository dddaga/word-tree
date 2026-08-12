# Phase 5 -- Part 8: ARM 1+2 Synthesis and Gen4 Configuration (2026-04-03, updated 2026-04-04)

Reference: D=64 N=1024 K_iter=8 | **Current best: 80.08% (step29c Config A, AntiHebb alpha=1.0, calibrated base)**

---

## ARM 1 -- Generational Compounding (Mechanism Calibration on D=64 Base)

### Step 29c Phase 1: Mechanism Calibration (40ep each, calibrated base from step22b)

All mechanisms tested against the step22b-calibrated D=64 base (alpha_reflect=0.5, optimal routing params).

#### AntiHebb Alpha Sweep (COMPLETE)

| Alpha | top1@40ep | vs baseline (56.28%) |
|-------|-----------|---------------------|
| 0.1   | 48.56%    | -7.72pp             |
| 0.3   | 54.42%    | -1.86pp             |
| 0.5   | 59.82%    | +3.54pp             |
| 0.7   | 64.69%    | +8.41pp             |
| **1.0** | **70.98%** | **+14.70pp**     |

**Finding:** Monotonic scaling. Full suppression (alpha=1.0) is optimal on calibrated base. This exceeds the step29 alpha=0.7 result at 40 epochs. The calibrated base routing params unlock stronger AntiHebb.

**Physical interpretation:** At D=64, the Fourier encoding creates near-orthogonal neuron directions on S^63. AntiHebb suppresses structurally nearby neurons proportional to their W_pos cosine similarity. At alpha=1.0, neurons with identical W_pos directions contribute ZERO to their neighbors -- forcing maximal spatial diversity. This is exactly the Mexican-hat surround suppression from cortical V1.

**Decision:** Adopt AntiHebb alpha=1.0 for Gen4 base (up from 0.7).

#### Phase Excitatory Alpha Sweep (COMPLETE)

| Alpha | top1@40ep | vs baseline |
|-------|-----------|-------------|
| 0.1   | 57.91%    | +1.63pp     |
| 0.3   | 59.34%    | +3.06pp (*)  |
| 0.5   | 45.83%    | -10.45pp    |
| 1.0   | 33.43%    | -22.85pp    |

(*) training_too_short flag -- best at e22/40, model still improving

**Finding:** Phase excitatory teleportation is marginal at D=64. Low alpha (0.1-0.3) gives small gains, but alpha >= 0.5 is catastrophic. The W_phase K-NN excitatory graph interferes with established routing at high strength -- it teleports activations to phase-similar neurons, bypassing the geometric routing that AntiHebb is optimizing.

**Decision:** Include phase_exc alpha=0.1 as optional compound test in Gen4. Not a primary mechanism.

#### Fast W_phase (PARTIAL -- 2 of 4 configs done)

| Config              | top1@40ep | vs baseline |
|---------------------|-----------|-------------|
| alpha=0.1, tau=0.25 | 49.45%    | -6.83pp     |
| alpha=0.3, tau=0.25 | 49.30%    | -6.98pp     |
| alpha=0.1, tau=0.50 | running   | --          |
| alpha=0.1, tau=1.00 | pending   | --          |

**Finding:** Fast W_phase at D=64 is clearly harmful, consistent with step30 (cross-dim W_mix: -15pp) and step37 (phase matrix bank: -5pp). All three share the same failure mode: D x D transformations scramble the Fourier encoding structure.

**Decision:** EXCLUDE fast W_phase from Gen4. D x D cross-dim mixing is architecturally incompatible with Fourier encoding at D=64.

#### Interneurons (PENDING)

Phase 1 interneuron calibration has not started yet (blocked behind fast_W_phase completion). D=16 finding (step20): 50% interneurons + readout=all = +2.83pp. Hypothesis: may compound with AntiHebb at D=64.

### Step 29c Phase 2: Full Compound Runs (NOT STARTED)

Phase 2 (150ep each) will test the compound configurations Ref/A-G once Phase 1 completes. Currently blocked on step29c Phase 1 completion.

---

## ARM 2 -- New Mechanisms

### Step 48: K_iter Scaling at D=64 (PARTIAL -- 2 of 8 configs complete)

| Config | K_iter | AntiHebb | top1@150ep | vs Ref |
|--------|--------|----------|------------|--------|
| Ref    | 8      | No       | 58.24%     | --     |
| A      | 12     | No       | 55.08%     | -3.16pp |
| B      | 16     | No       | ~51.44%*   | ~-6.8pp |

(*) B at e70/150 -- partial result

**Finding:** K_iter > 8 HURTS without AntiHebb at D=64. Over-smoothing occurs even though D=64 provides richer directional space than D=16. The critical test (configs E/F/G with AntiHebb) is pending -- AntiHebb may prevent over-smoothing at higher K_iter.

**Preliminary decision:** Keep K_iter=8 for Gen4 unless E/F/G show compounding at K_iter=16.

### Step 54: LR Schedule Comparison (PARTIAL -- 2 of 4 configs complete)

| Config | Schedule             | top1@150ep | vs Ref |
|--------|----------------------|------------|--------|
| Ref    | Plateau p=10 f=0.5   | 70.78%     | --     |
| A      | WarmRestarts T0=10 T1 | 66.98%    | -3.80pp |
| B      | WarmRestarts T0=10 T2 | running    | --     |
| C      | Cosine T=150         | pending    | --     |

**Finding:** Plateau (70.78%) BEATS CosineWarmRestarts (66.98%) by 3.80pp. Notably, the plateau LR NEVER actually decayed -- patience=10 was never triggered because loss kept improving monotonically. This means the "plateau" schedule is effectively constant-LR training at 2.364e-3.

**Key insight:** The routing optimization on S^63 benefits from a stable learning rate. Periodic LR jumps (warm restarts) knock the model off converging attractor trajectories in the high-dimensional activation landscape. The S^63 manifold is so large that the model is never near a local minimum that needs LR cycling to escape.

**Decision:** KEEP plateau as default LR schedule. It auto-adapts to constant-LR when appropriate.

### Step 52: High-D Routing (NOT YET DISPATCHED)

Script written and synced to Mac Studio. Blocked by concurrency (3 processes running, cap = 2). Tests: Z-subspace gating, W_pos-subspace gating, projection routing, centering diversity.

Will dispatch when slot opens.

### Step 53: Low-Rank Dimension Mixing (NOT YET DISPATCHED)

Script written and synced to Mac Studio. Tests: low-rank (rank 4/8), frequency-pair (2x2), group mixing (8x8). All designed to preserve Fourier encoding structure unlike full D x D mixing.

Will dispatch when slot opens.

---

## Gen4 Compound Configuration (step32)

### Adopted Winners

| Source  | Mechanism                | Previous | Gen4    | Evidence |
|---------|--------------------------|----------|---------|----------|
| step29c | AntiHebb alpha           | 0.7      | **1.0** | 70.98% at 40ep (monotonic) |
| step22b | alpha_reflect            | 0.3      | **0.5** | +2.75pp at 40ep calibration |
| step54  | LR schedule              | plateau  | plateau | 70.78% beats warm restarts |
| step48  | K_iter                   | 8        | 8       | No improvement at K_iter>8 |
| step29c | Fast W_phase             | excluded | excluded | 49.45% < 56.28% baseline |
| step29c | Phase excitatory         | --       | **test alpha=0.1** | 57.91% marginal |

### Dead Ends (Excluded from Gen4)

| Mechanism | Steps tested | Verdict |
|-----------|-------------|---------|
| Fast W_phase | step29c, step30, step37 | D x D interference with Fourier encoding |
| Signed coupling | step18/23/28/42/44 | cos-sim noise on S^63 |
| D=128 encoding | step33, step33b | Near-orthogonal seeds on S^127 |
| MoD adaptive K_iter | step34 | Disrupts iterative refinement |
| Oja's rule | step41 | PCA compression destroys diversity |

### Gen4 Script (step32)

`scripts/train_step32_gen4_compound.py` -- 8 configs (Ref, A-G) testing:
- Ref: AntiHebb alpha=0.7 uncalibrated (replicating 75.24%)
- A: AntiHebb alpha=1.0 on calibrated base (Gen4 primary)
- B-E: Progressive compound stacking with phase_exc and centering
- F: 75.24% reproducibility check
- G: alpha=1.0 + calibrated alpha_reflect=0.5

**Dispatch status:** Script synced to Mac Studio. Blocked by concurrency (3 running). Will dispatch when slot opens (expected within ~6-12 hours as step54 configs complete).

---

## Key Decision Log

1. **AntiHebb alpha 0.7 -> 1.0:** Monotonic scaling confirmed at D=64. Full surround suppression is optimal. All 5 alpha values tested.

2. **Plateau LR retained:** 70.78% vs 66.98% (warm restarts). LR effectively stays constant because loss improves monotonically -- plateau auto-adapts.

3. **K_iter stays at 8:** K_iter=12 and 16 worse without AntiHebb. AH configs pending.

4. **D x D mixing pattern confirmed DEAD:** Fast W_phase joins cross-dim W_mix and phase matrix bank as the 3rd mechanism killed by Fourier encoding interference. Only structured sub-D mixing (step53 designs) remains viable.

5. **Phase excitatory: cautious inclusion at alpha=0.1:** Not a primary driver, but worth compound testing. Higher alphas destructive.

---

---

## Step 29c Phase 2: Full Compound Runs (COMPLETE — 2026-04-04, config G still running)

Base: D=64 N=1024 K_iter=8, calibrated routing from step22b (alpha_reflect=0.5, beam_size=16, geo_gamma=0.5, K_phase=8).

| Config | Description | top1_best | vs Ref |
|--------|-------------|-----------|--------|
| Ref    | calibrated base, no mechanisms | 58.68% | -- |
| **A**  | **+ AntiHebb α=1.0** | **80.08%** | **+21.40pp ← NEW ALL-TIME BEST** |
| B      | + phase_exc α=0.3 | 63.64% | +4.96pp |
| C      | + interneurons 25% | 56.99% | -1.69pp |
| D      | + fast_W_phase α=0.1 τ=1.0 | 61.10% | +2.42pp |
| E      | + AntiHebb α=1.0 + phase_exc α=0.3 | 66.96% | +8.28pp |
| F      | + AntiHebb α=1.0 + phase_exc α=0.3 + interneurons 25% | 66.93% | +8.25pp |
| G      | Full Gen4 (all mechanisms) | still running | -- |

**Critical finding: AntiHebb α=1.0 alone = 80.08%. Adding ANY other mechanism KILLS the gain (80% → 67%).**

The compounding failure is striking:
- E = AntiHebb + phase_exc → 66.96% (−13.12pp vs A alone)
- F = AntiHebb + phase_exc + interneurons → 66.93% (similar to E)
- Adding phase_exc (even α=0.3) wipes out the AntiHebb gain entirely

**Why:** At α=1.0, AntiHebb enforces maximal spatial diversity — neurons with similar W_pos directions contribute zero to each other. Phase_exc then creates an excitatory graph on top of the W_phase K-NN topology, which partially re-introduces proximity coupling that AntiHebb is trying to suppress. The two mechanisms are in direct opposition at full strength.

**Decision:** Gen4 compound = AntiHebb α=1.0 ALONE. No phase_exc, no interneurons, no fast_W_phase.

---

## Updated Open Questions

1. ~~Does AntiHebb alpha=1.0 on calibrated base exceed 75.24%?~~ → **YES. 80.08%. +4.84pp over previous best.**
2. Does K_iter > 8 + AntiHebb compound? (step48 configs E/F/G will answer)
3. ~~Do interneurons compound with AntiHebb at D=64?~~ → **NO. 56.99% (−1.69pp vs Ref, −23.09pp vs AntiHebb alone)**
4. Does structured sub-D mixing add anything? (step53 will answer)
5. Does any high-D routing mode beat uniform+AntiHebb? (step52 will answer)

---

## Timeline and Next Steps

- **step29c:** Phase 2 done (config G still running). Result already known: AntiHebb α=1.0 alone = 80.08%.
- **Next:** dispatch step52 and/or step53 now that step29c is winding down
- **Gen4 base confirmed:** AntiHebb α=1.0 + calibrated routing params (step22b)
- **step32 (Gen4 compound) status:** No longer needed in original form — 80.08% IS the Gen4 result. Revise step32 to test variations around AntiHebb α=1.0 (alpha sweep near 1.0, beam_size sensitivity, etc.)
