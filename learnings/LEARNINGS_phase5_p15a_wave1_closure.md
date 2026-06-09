# Phase 5 Part 15a: Wave-1 Closure (Steps 60-68)

**Date:** 2026-04-06
**Parent:** LEARNINGS_phase5_p15_post_wave1.md (TOC)
**REF_BASELINE:** 73.53% (AntiHebb α=1.0, 50% data, 75ep, step57)

---

## Step 60: Phase-Distance Routing (Complete)

**Script:** train_step60_phase_routing_magnitude.py
**Verdict:** KILLED — all 12 non-Ref configs gate-dead

| Config | Description | top1_best | vs Ref |
|--------|-------------|-----------|--------|
| Ref | AntiHebb α=1.0 static | 73.55% | — |
| A | independent, no AH, freq=capped_exp | 14.27% | −59pp |
| B | coherent dynamic-ref, no AH | 14.27% | −59pp |
| C | independent, AH=1.0, freq=capped_exp | 18.73% | −55pp |
| D | coherent, no AH | 13.30% | −60pp |
| A_wt | independent+mag-weighted, no AH | 13.20% | −60pp |
| B_wt | coherent+mag-weighted, no AH | 14.37% | −59pp |
| B_anchor | coherent AH=1.0, freq=capped_exp | 14.14% | −59pp |
| B_abs | coherent absolute, no AH | 14.22% | −59pp |
| E | (gate-dead) | 12.05% | −61pp |
| F | (gate-dead) | 11.92% | −62pp |
| A_prime | (gate-dead) | 10.80% | −63pp |
| B_prime | (gate-dead) | 10.68% | −63pp |

Even configs WITH AH (C, B_anchor) gate-die at 14-19%. All 12 killed.

---

## Step 63: Activation-Gated Routing / AGR (Complete)

**Script:** train_step63_act_gated_routing.py
**Verdict:** KILLED — all non-Ref configs killed

| Config | Description | top1_best | vs Ref |
|--------|-------------|-----------|--------|
| Ref | AntiHebb α=1.0 static | 73.48% | — |
| A | soft-attn mixed candidates, no AH | 31.11% | −42pp |
| B | soft-attn mixed candidates, AH=1.0 | 55.41% | −18pp |
| C | hard top-K mixed candidates, AH=1.0 | 38.88% | −35pp |
| D | soft-attn mixed, AH=1.0, hop_decay=0.9 | 55.29% | −18pp |
| E | soft-attn phase-only candidates, AH=1.0 | 55.13% | −18pp |

AH partially compensates but −18pp vs static AH. All killed.

---

## Step 65: Distance-Phase Routing (Complete)

**Script:** train_step65_dist_phase_routing.py
**Verdict:** KILLED — all configs ~48-49%

| Config | Description | top1_best | vs step57 REF |
|--------|-------------|-----------|---------------|
| Ref | SmallWorld bare (no AH) | 35.06% | −38pp |
| A | γ=0.5 raw | 48.79% | −25pp |
| B | γ=1.0 raw | 48.79% | −25pp |
| C | γ=2.0 raw | 48.43% | −25pp |
| D | γ=0.5 softmax | 48.43% | −25pp |
| E | γ=1.0 softmax | 48.43% | −25pp |
| F | γ=2.0 softmax | 48.43% | −25pp |

Distance-phase geometry adds no routing signal AH doesn't already provide.

---

## Step 54: LR Schedule Sweep — COMPLETE (Performance Killed)

| Config | Description | top1_best |
|--------|-------------|-----------|
| Ref | Plateau scheduler, AH=0.5 | 70.78% |
| A | CosineWarmRestart T_0=10 | 66.98% |
| B | CosineWarmRestart T_0=25 | 68.05% |
| C | CosineWarmRestart T_0=50 | 67.21% |

Plateau scheduler optimal. CLOSED.

---

## Step 64: Stacked SGNNET — COMPLETE (buggy arch, 75ep, 50% data)

| Config | Description | Params | top1_best | vs Ref (73.38%) |
|--------|-------------|--------|-----------|-----------------|
| Ref | 1-layer baseline AH=1.0 | ~133K | 73.38% | — |
| A | 2-layer series, passthrough, no skip | ~266K | 71.62% | −1.76pp |
| B | 2-layer series, passthrough + skip | ~266K | 72.23% | −1.15pp |
| C | 2-layer series, bridge (linear D→D), no skip | ~270K | 70.75% | −2.63pp |
| D | 3-layer series, passthrough, no skip | ~399K | ~55% | ~−18pp |
| E | 2-parallel sum fusion | ~265K | 69.10% | −4.28pp |
| **F** | **2-parallel concat-project** | 273K | **74.90%** | **+1.52pp WINNER** |

Config F (2-parallel concat-project) = +1.52pp vs Ref. best_ep=75/75 — still converging.
Series stacking closed. Parallel concat-project only variant that adds value.
Follow-up: re-test Config F on patched arch (→ step85).

---

## Step 67: Safety Valve λ Ablation — COMPLETE (pre-patch architecture)

| Config | λ_safety | top1_best | vs Ref (73.68%) |
|--------|----------|-----------|-----------------|
| Ref | 0.489 (default scaled) | 73.68% | — |
| A | 0.0 (safety OFF) | 70.78% | −2.90pp |
| B | 0.05 (minimal) | 71.21% | −2.47pp |
| C | 0.1 (light) | 72.59% | −1.09pp |

Non-linear recovery accelerates at λ=0.1. ⚠️ Pre-patch arch — re-validation needed.

---

## Step 68: K_iter Scaling at Gen4 — COMPLETE

**Script:** train_step68_kiter_gen4.py (50%/75ep, N=1024, D=64, AH=1.0)

| Config | K_iter | top1_best | vs Ref (73.63%) |
|--------|--------|-----------|-----------------|
| Ref | 8 | 73.63% | — |
| A | 10 | 73.58% | −0.05pp |
| B | 12 | 72.94% | −0.69pp |
| **C** | **16** | **74.14%** | **+0.51pp WINNER** |
| D | 24 | 70.06% | −3.57pp |

**K_iter=16 new Gen4 optimal at N=1024.** Non-monotone: peak at 16, cliff at 24.
Note: K_iter optimal N-dependent — at N=4096 (step71), K_iter=12 wins.

---

## Step 48: K_iter Scaling at D=64 — SLOT-KILLED (Partial)

| Config | Description | top1_best |
|--------|-------------|-----------|
| Ref | K_iter=8, no AH | 58.24% |
| A | K_iter=12, no AH | 55.08% |
| B-G | (never ran — slot killed) | — |

K_iter=12 WITHOUT AH hurts (−3pp). B-G (AH configs) never ran → superseded by step68.

---

## Step 56: N-Scaling Final (N=10000) — COMPLETE

| N | Params | top1_best | Notes |
|---|--------|-----------|-------|
| 512 | 66K | 69.58% | — |
| 1024 | 133K | 80.92% | — |
| 2048 | 265K | 81.10% | — |
| 4096 | 529K | **84.36%** | PROJECT BEST (at time); best_ep=146/150 |
| 10000 | 1.29M | 82.37% | REGRESSION −1.99pp |

N-scaling NOT monotonic above N=4096. Regression at N=10000 likely:
- W_pos position space too sparse for K_local at N=10000
- AH suppression too aggressive when N>>K_local

**⚠️ Results on buggy arch — patched arch scaling curve unknown.**

---

## Wave-1 Gate-Death Summary

| Experiment | Mechanism | Best result | Failure mode |
|------------|-----------|-------------|--------------|
| step58 | Resonance excitatory | ~73% | Gate collapses signal |
| step59 | Active beam unified | ~73% | Discrete top-K breaks gradient |
| step60 | Phase routing magnitude | ~73% | Coherence gate → 0 signal |
| step61 | Hub interneurons | ~72% | AH suppresses hub paths |
| step63 | Activation-gated routing | ~72% | Multiplicative gate collapses |
| step65 | Distance-phase routing | ~48% | Phase drift → coherence → 0 |
| step66 | Phase-target plasticity | 40-68% | Gating + AH = double sparsity |
| step51 | W_phase spatial gating | 15-20% | Complete gate-death |

**Theorem:** Every multiplicative gate g∈[0,1] over K routing iterations → g^K signal decay.
At g=0.8, K=8: `0.8^8 ≈ 0.17×`. Static AH survives because suppression applies **before**
gather step and `F.normalize()` restores magnitude after each step.

---

## Backlog Audit (2026-04-06)

**Genuinely killed (do NOT rerun):** step54, step60, step63, step65, step51, step52, step53, ARM4 (steps 39/45/46/47)

**Killed by slot pressure (must rerun):** step48 configs B-G → rewrote as step68 ✓

**Genuinely remaining:** step66 (→ KILLED), step68 (→ COMPLETE above)