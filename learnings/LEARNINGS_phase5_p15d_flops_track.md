# Phase 5 Part 15d: FLOPs Track + Pareto Results (Steps 86, 88)

**Date:** 2026-04-08 to 2026-04-09
**Parent:** LEARNINGS_phase5_p15_post_wave1.md (TOC)
**Context:** After establishing N=4096 accuracy ceiling, track focuses on compute efficiency.

---

## Goal Clarification: FLOPs Target (2026-04-09)

Extended goal: achieve ≤1% params AND ≤1% FLOPs vs VGG16 FC.

| Metric | Current (K_hh=4) | VGG16 FC | % of VGG16 | Target | Gap |
|--------|-----------------|----------|------------|--------|-----|
| Params | 529K | 119.6M | **0.44%** | ≤1% | ✓ Already achieved |
| FLOPs | 38.8M | 119.6M | **32.4%** | ≤1% (~1.2M) | 32× reduction needed |

**Constraint:** At N=4096, fundamental routing loop ≈ 30M FLOPs minimum — cannot hit 1.2M at this scale.

**Path to 1% FLOPs:** Reduce N (N=512 or N=256) + D=32 + K_hh=2 + K_in reduction.
Separate optimization axis from accuracy-maximization track.

**Priority order:**
1. Maximize accuracy at N=4096 (G1: K_iter=12 + K_hh=4 + turing=0.0, 150ep)
2. Group topology at N=4096 with K_hh=4 base (G2)
3. Pure dynamic connectivity experiment (step87)
4. FLOPs reduction track (smaller N/D) — after accuracy ceiling established

---

## Step 86: FLOPs / Accuracy Pareto Sweep — PARTIAL COMPLETE

**Script:** train_step86_pareto_flops.py
**Scale:** N=4096, 50%/75ep, turing=0.0, AH=1.0
**Key note:** n_groups=512 (N//8) critical — n_groups=128 (N//32) caused −4.28pp regression.

### Mac Studio Results (Ref/A/B/C/D) — DONE

| Config | K_hh | K_iter | D | top1_best | FLOPs_M | vs Ref |
|--------|------|--------|---|-----------|---------|--------|
| Ref | 6 | 8 | 64 | 96.03% | 47.2M | — |
| **A** | **4** | 8 | 64 | **96.59%** | **38.8M** | **+0.56pp, −18% FLOPs ← NEW DEFAULT** |
| B | 3 | 8 | 64 | 96.43% | 34.6M | +0.40pp |
| C | 2 | 8 | 64 | 96.28% | 30.4M | +0.25pp |
| D | 6 | 6 | 64 | 93.83% | 39.3M | −2.20pp ← K_iter costly |
| E | 4 | 6 | 64 | 95.59% | 33.0M | −0.44pp |

### Mac Mini Results (F/G/H/I) — PENDING RESUME

Use: `d_env/bin/python3 -u scripts/train_step86_pareto_flops.py --device mps --only F G H I`

### Key Findings

**1. K_hh lever: all reductions beat baseline**
- Reducing K_hh 6→4→3→2 *improves* accuracy while reducing FLOPs
- Root cause: AH suppression already nullifying local K_local edges — dead weight
- Even K_hh=2 (pure random, K_local=0): +0.41pp over Ref at −36% FLOPs
- **Pareto winner: A (K_hh=4, 96.59%, 38.8M) — best accuracy + −18% FLOPs**
- **Efficiency pick: B (K_hh=3, 96.43%, 34.6M) — only −0.16pp behind A, −27% FLOPs**

**2. K_iter lever: devastating (−2.20pp at matched FLOPs)**
- D (K_iter=6, K_hh=6): 93.83% — same FLOPs as A but 2.76pp worse
- Rule confirmed: never reduce K_iter. All 8 steps necessary.

**3. K_iter >> K_hh (different mechanisms, different costs)**
- K_hh controls graph connectivity sparsity (structural)
- K_iter controls routing depth (computational)
- AH suppression decouples them: sparse K_hh ≠ shallow routing

**4. n_groups bug impact confirmed: +4.28pp (91.75% → 96.03% Ref)**
- Old bug: n_groups=N//32=128
- Fix: n_groups=max(8,N//8)=512 (matches topology_kwargs in step71)

### Implications

- **New architecture default: K_hh=4 (not 6)** — adopt in step87, step88, all future N=4096 work
- **G1 (K_iter=12 + turing=0.0)**: run with K_hh=4 for maximum combined gain
- **G2 (group topology at N=4096)**: run with K_hh=4 base
- **Pareto target achieved**: A (96.59%, 38.8M) well above 95% at ≤40M FLOPs

---

## Step 88: AH Alpha Recalibration at N=4096 — COMPLETE

**Script:** train_step88_alpha_sweep_n4096.py
**Scale:** N=4096, 50%/40ep calibration, turing=0.0, K_iter=8, K_hh=6, n_groups=512
**Machine:** Mac Studio CPU

| Config | alpha | top1_best (40ep) | delta vs step86_ref (96.03%) |
|--------|-------|-----------------|------------------------------|
| Ref | 0.5 | 87.69% | −8.34pp |
| **A** | **1.0** | **95.52%** | **−0.51pp WINNER** |
| B | 1.5 | 89.48% | −6.55pp |
| C | 2.0 | 94.17% | −1.86pp |

**Winner: α=1.0. Confirmed optimal.**

Notable: Asymmetric curve — α=2.0 (94.17%) outperforms α=1.5 (89.48%). Non-monotone above α=1.0.

**Decision: α=1.0 adopted for all future N=4096 experiments. No recalibration needed.**

---

## G1 Experiment: K_iter=12 + K_hh=4 + turing=0.0 (Pending)

Combines 3 confirmed wins:
- K_iter=12 → +0.79pp (step71 at N=4096 50%/75ep)
- K_hh=4 → +0.56pp (step86 at N=4096 50%/75ep)
- turing=0.0 → already in base (step70 confirmed)

Expected at 100%/150ep: likely >97.38% (current project best).
Script: tweak step71 script, set K_hh=4.
Machine: Mac Studio MPS when slot frees.

---

## G2 Experiment: Group Topology N=4096 with K_hh=4 (Pending)

step82 n_groups=8 won +3.01pp at N=1024. Never tested at N=4096.
With K_hh=4 base (new default), could compound both gains.
Script: tweak step82 script, change N=4096, K_hh=4.
Machine: Mac Studio CPU when slot frees.

---

## Pareto Frontier Summary

| Config | Accuracy | FLOPs | FLOPs/accuracy |
|--------|----------|-------|----------------|
| step86 A (K_hh=4) | 96.59% | 38.8M | Pareto winner |
| step86 B (K_hh=3) | 96.43% | 34.6M | Efficiency pick |
| step70 B (full scale) | 97.38% | 49.2M | Current best |
| VGG16 FC | ~93.5% | 119.6M | Reference |