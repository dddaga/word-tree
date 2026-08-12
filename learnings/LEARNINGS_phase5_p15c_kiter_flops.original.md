# Phase 5 Part 15c: K_iter Sweep + Routing Experiments (Steps 71-83)

**Date:** 2026-04-07 to 2026-04-08
**Parent:** LEARNINGS_phase5_p15_post_wave1.md (TOC)
**Base:** Patched arch (input_coverage + alpha_reflect fix). N=4096 or N=1024 as noted.

---

## Step 71 — K_iter sweep N=4096 — COMPLETE

**Config:** N=4096, D=64, 50%/75ep, turing=0.0, reflect=0.5, AH=1.0

| Config | K_iter | top1_best | vs Ref (95.87%) |
|--------|--------|-----------|-----------------|
| Ref | 8 | **95.87%** | — |
| A | 4 | **92.82%** | −3.05pp |
| B | 6 | **95.11%** | −0.76pp |
| **C** | **12** | **96.66%** | **+0.79pp WINNER** |
| D | 16 | 96.31% | +0.44pp |

**K_iter=12 is optimal at N=4096.** Non-monotone: 4<6<8<16<12.
Peak at 12, not 16 (step68 at N=1024 found 16). K_iter optimal is N-dependent.
Ref at 50%/75ep = 95.87% vs step70 full = 97.32% (−1.45pp from data reduction).

---

## Step 73 — Softmax Routing — COMPLETE

**Scale:** N=1024, 50%/75ep. First conservative (redistribution) routing test.

| Config | Description | top1_best |
|--------|-------------|-----------|
| Ref | baseline | 84.56% |
| A | softmax routing variant A | 55.77% (collapsed) |
| B | softmax routing variant B | 84.87% |
| C | softmax routing variant C | 85.58% |
| **D** | **softmax routing variant D** | **86.34%** |

**Config D is winner (+1.78pp over Ref 84.56%).** Config A collapsed (gate-death pattern).
First dynamic routing mechanism to beat static AH baseline.

---

## Step 75 — Input-Modulated Temperature Routing — COMPLETE

**Scale:** N=1024, Mac Studio MPS, 50%/75ep

| Config | Description | top1_best |
|--------|-------------|-----------|
| Ref | static AH baseline | ~84% |
| A | fixed τ=1.0 | — |
| B | fixed τ=0.3 | — |
| C | learned τ_0=0.5 | 85.76% |
| **D** | **learned τ_0=0.3** | **87.24%** |

**step75 D = 87.24% — best N=1024 result to date.** +3.98pp over static baseline.
Per-neuron temperature parameter scales softmax attention without attenuating signal.
**Never tested at N=4096.**

---

## Step 76 — Alpha_turing Sweep + W_phase Trained — COMPLETE

**Scale:** N=1024, Mac Mini CPU, 50%/75ep. W_phase now trained (was frozen in all prior experiments).

| Config | alpha_turing | top1_best | vs Ref (83.67%) |
|--------|-------------|-----------|-----------------|
| Ref | 0.0 (frozen W_phase) | 83.67% | — |
| **A** | **0.0, W_phase trained** | **86.55%** | **+2.88pp WINNER** |
| B | 0.3, W_phase trained | 84.41% | +0.74pp |
| C | 0.5, W_phase trained | 83.13% | −0.54pp |
| D | 1.0, W_phase trained | 78.37% | −5.30pp |

Config A is winner (+2.88pp). Higher turing values (C=0.5, D=1.0) hurt badly.
turing=0.0 with trained W_phase wins. Higher turing inhibition is harmful.

---

## Step 77 — Learnable Theta — COMPLETE

**Scale:** N=1024, 50%/75ep. theta is an nn.Parameter but was never added to optimizer.

| Config | Description | top1_best | vs Ref |
|--------|-------------|-----------|--------|
| Ref | fixed theta=0.1 | 85.10% | — |
| A | learnable per-neuron theta, full lr | 83.29% | −1.81pp |
| B | learnable theta, 0.1× lr | 84.92% | −0.18pp |
| C | fixed theta, mean-pool aggr | 62.70% | −22.40pp |

**KILLED. Fixed theta=0.1 is optimal.** Learnable theta either hurts (A,B) or collapses (C).
B nearly matches Ref at −0.18pp — learnable theta at lower lr converges to same ~0.099 mean.

---

## Step 79 — Aux Loss Sweep (Patched Arch) — COMPLETE

**Scale:** N=4096, 50%/75ep, Mac Studio MPS.

| Config | Description | top1_best |
|--------|-------------|-----------|
| Ref | no aux loss | 96.10% |
| A | phase coherence λ=0.01 | 95.85% |
| B | phase coherence λ=0.001 | 95.80% |
| C | phase coherence λ=0.0001 | 95.54% |
| D | sparsity reward λ=0.01 | 96.31% |
| E | sparsity reward λ=0.001 | 96.18% |
| F | routing diversity λ=0.01 | 96.31% |

Phase coherence at all λ scales below Ref (KILLED). Sparsity reward D=96.31% ties F (+0.21pp).
Marginal at 50%/75ep — insufficient to promote. **Aux losses not adopted.**

---

## Step 80 — N-Scaling Patched Arch (Partial) — COMPLETE

| N | Params | top1_best (patched 50%/75ep) | step56 buggy ref | delta |
|---|--------|------------------------------|------------------|-------|
| 512 | ~66K | **72.79%** | 69.58% | **+3.21pp** |
| 2048 | ~265K | **92.74%** | 81.10% | **+11.64pp** |
| 4096 | 529K | 95.87% (step71 Ref) | 84.36% (step56) | — |

Patch gain grows with N: +3.21pp at N=512 vs +11.64pp at N=2048. The input coverage bug is
more damaging at larger N. N-scaling curve is steeper on patched arch.

---

## Step 81 — Hebbian Topology Rewiring — COMPLETE

**Scale:** N=1024, Mac Studio CPU, 50%/75ep

| Config | Description | top1_best |
|--------|-------------|-----------|
| Ref | static conn_hh | ~84% |
| **A** | **5%/neuron rewire every 5ep** | **85.55%** |
| B | 10%/neuron every 5ep | — |
| C | 5%/neuron every 10ep | — |
| D | 5%/neuron every 5ep + group init | — |

**step81 A = 85.55% (+1.12pp over Ref).** Hebbian rewiring helps — dynamic connectivity
without signal destruction (topology change, not per-step gate). Uses Hebb: fire together → wire together.

---

## Step 82 — Group-Structured Hidden Topology — COMPLETE

**Scale:** N=1024, 50%/75ep.

**Mac Mini results:**

| Config | Description | top1_best |
|--------|-------------|-----------|
| Ref | spatial topology n_groups=128 | 82.62% |
| **A** | **random-group n_groups=8** | **85.63% (+3.01pp)** |
| B | random-group n_groups=16 | 83.97% |
| C | random-group n_groups=32 | 83.85% |
| D | n_groups=8 + input-group alignment | 69.40% (DEAD) |

**Mac Studio results (consistent):**
- Ref=83.11%, A(n_g=8)=85.25% (+2.14pp), B(n_g=16)=83.62%, C(n_g=32)=84.05%
- D(input-group align)=70.04% KILLED

**n_groups=8 wins on both machines (+2-3pp vs Ref).** Input-group alignment catastrophically
kills performance. Static group topology change, no new params.
**Never tested at N=4096. Likely compounds with N-scaling.**

---

## Step 83 — Group State + Inter-Group Dynamic Routing — KILLED

**Mac Studio (50%/75ep, n_groups=16):**

| Config | Description | top1_best | Δ vs Ref |
|--------|-------------|-----------|----------|
| Ref | Static AH, n_groups=16 | 84.61% | — |
| A | β=0.5, every routing step | 78.60% | −6.01pp |
| B | β=0.5, final step only | 81.96% | −2.65pp |
| C | β=0.1, every step | 84.00% | −0.61pp |

Dynamic routing on top of static group topology is harmful. Root causes:
1. S_g = mean(Z) too coarse — softmax collapses to uniform early in training
2. Temporal mismatch: AH (epoch timescale) vs router (batch timescale) → adversarial
3. Co-adaptation: W_pos and router converge to degenerate fixed point

**Key contrast:** step82A (static group topology alone) = +3.01pp. The topology is valuable;
dynamic routing on top of it is harmful.

---

## Priority Queue from Step 71 Analysis

**P0:** G1 — K_iter=12 + turing=0.0 + K_hh=4, N=4096, 100%/150ep
**P1:** G2 — Group topology N=4096 with K_hh=4
**P2:** step87 — Pure dynamic connectivity (no static conn_hh)
