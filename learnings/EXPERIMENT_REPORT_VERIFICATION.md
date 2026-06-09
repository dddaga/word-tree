# Experiment Report Verification

**Verified:** 2026-04-08
**Report:** `learnings/EXPERIMENT_REPORT.md`
**Data sources:** Local results (`results/train_step*.json`) + Mac Studio results (`ssh mac-studio`)

---

## Accuracy Check

### Tier 1 (Full-scale, 100%/150ep)

| Step | Config | Report Claims | JSON Actual | Match? |
|------|--------|--------------|-------------|--------|
| step70 | B | 97.38% | 97.38% (Mac Studio) | YES |
| step70 | Ref | 97.20% | 97.20% (Mac Studio) | YES |

### Tier 2 (Half-scale, 50%/75ep, N=4096)

| Step | Config | Report Claims | JSON Actual | Match? |
|------|--------|--------------|-------------|--------|
| step71 | C | 96.66% | 96.66% (local) | YES |
| step71 | D | 96.31% | 96.31% (local) | YES |
| step79 | D | 96.31% | 96.31% (Mac Studio) | YES |
| step79 | F | 96.31% | 96.31% (Mac Studio) | YES |
| step79 | E | 96.18% | 96.18% (Mac Studio) | YES |
| step79 | Ref | 96.10% | 96.10% (Mac Studio) | YES |
| step71 | Ref | 95.87% | 95.87% (local) | YES |
| step71 | B | 95.11% | 95.11% (local) | YES |

### Best N=1024

| Step | Config | Report Claims | JSON Actual | Match? |
|------|--------|--------------|-------------|--------|
| step75 | D | 87.24% | 87.24% (Mac Studio) | YES |
| step76 | A | 86.55% | 86.55% (local) | YES |
| step73 | D | 86.34% | 86.34% (local) | YES |
| step82 | A | 85.63% | 85.63% (local) / 85.25% (Mac Studio) | YES (uses local copy) |
| step81 | A | 85.55% | 85.55% (Mac Studio) | YES |

### step82 Detail

| Config | Report Claims | JSON Actual (local) | Match? |
|--------|--------------|---------------------|--------|
| Ref | 82.62% | 82.62% | YES |
| A | 85.63% | 85.63% | YES |
| B | 83.97% | 83.97% | YES |
| C | 83.85% | 83.85% | YES |
| D | 69.40% | 69.40% | YES |

### N-Scaling Table

| N | Report Acc | JSON Actual | Match? |
|---|-----------|-------------|--------|
| 512 | 72.79% | 72.79% (step80 Mac Studio) | YES |
| 1024 | ~83.36% | 83.36% (step69 Ref Mac Studio) | YES |
| 2048 | 92.74% | 92.74% (step80 Mac Studio) | YES |
| 4096 | 95.87% | 95.87% (step71 Ref local) | YES |

**All individual accuracy numbers correct.**

---

## Parameter Check

### Formula
- W_pos: (N_hidden + N_out) x D
- theta: N_hidden
- W_phase: N_hidden x D
- Total = (N+10) x D + N + N x D

### Computed Values

| N | W_pos | theta | W_phase | Total | Report Claims | Match? |
|---|-------|-------|---------|-------|--------------|--------|
| 512 | 33,408 | 512 | 32,768 | **66,688** | 66,688 | YES |
| 1024 | 66,176 | 1,024 | 65,536 | **132,736** | 132K | YES |
| 2048 | 131,712 | 2,048 | 131,072 | **264,832** | 264,832 | YES |
| 4096 | 262,784 | 4,096 | 262,144 | **529,024** | 529,024 | YES |

### VGG16 FC Params (without biases)
- FC1: 25,088 x 4,096 = 102,760,448. Report: 102,760,448. Match.
- FC2: 4,096 x 4,096 = 16,777,216. Report: 16,777,216. Match.
- FC3: 4,096 x 10 = 40,960. Report: 40,960. Match.
- Total: 119,578,624. Report: 119,578,624. Match.

Note: bias terms (4096+4096+10 = 8,202) omitted. Report acknowledges (~14K stated, actual 8,202 for weights-only biases or ~12,298 counting bias vectors -- minor).

### Param Ratio
529,024 / 119,578,624 = 0.4425% -- report says 0.44%. Match.

**All parameter counts verified.**

---

## Compute Check

### FLOP Verification (N=4096, D=64, K_in=50, K_hh=6, beam_size=16)

| Phase | Report FLOPs | Computed | Match? |
|-------|-------------|----------|--------|
| Seed (N x K_in x D) | 13.1M | 4096 x 50 x 64 = 13,107,200 | YES |
| Gate (N x D) | 262K | 4096 x 64 = 262,144 | YES |
| Structural (N x K_hh x D x 2) | 3.1M | 4096 x 6 x 64 x 2 = 3,145,728 | YES |
| Reflection (N x D) | 262K | 262,144 | YES |
| Phase inhib (M x N x D x 2) | 8.4M | 16 x 4096 x 64 x 2 = 8,388,608 | YES |
| L2 norm (N x D x 2) | 524K | 524,288 | YES |
| Readout (N x N_out x D) | 2.6M | 4096 x 10 x 64 = 2,621,440 | YES |

### Per-step Totals
- turing=0: 262K + 3.1M + 262K + 524K = ~4.15M. Report: ~4.2M. Match (rounding).
- turing=0.3: ~4.15M + 8.4M = ~12.55M. Report: ~12.5M. Match.

### Config Totals
| Config | Report Total | Computed | Match? |
|--------|-------------|----------|--------|
| step70 B (K_iter=8, turing=0) | 49.2M | 13.1 + 8x4.15 + 2.6 = 48.9M | YES (rounding) |
| step70 Ref (K_iter=8, turing=0.3) | 116.1M | 13.1 + 8x12.55 + 2.6 = 116.1M | YES |
| step71 C (K_iter=12, turing=0.3) | 166.3M | 13.1 + 12x12.55 + 2.6 = 166.3M | YES |
| step71 D (K_iter=16, turing=0.3) | 216.5M | 13.1 + 16x12.55 + 2.6 = 216.5M | YES |

### VGG16 FC FLOPs
Report: 119,578,624 (= total FC params, each param = 1 MAC). Correct for dense layers.

**All FLOP estimates verified.**

---

## Ranking Check

### Actual Global Top 10 (combined Mac Studio + local)

| Rank | Accuracy | Step/Config | Source |
|------|----------|-------------|--------|
| 1 | 97.38% | step70 B | Mac Studio |
| 2 | 97.20% | step70 Ref | Mac Studio |
| 3 | 96.66% | step71 C | local |
| 4 | 96.31% | step71 D | local |
| 4 | 96.31% | step79 D | Mac Studio |
| 4 | 96.31% | step79 F | Mac Studio |
| 7 | 96.18% | step79 E | Mac Studio |
| 8 | 96.10% | step79 Ref | Mac Studio |
| 9 | 95.87% | step71 Ref | local |
| **10** | **95.85%** | **step79 A** | **Mac Studio** |
| 11 | 95.80% | step79 B | Mac Studio |
| 12 | 95.54% | step79 C | Mac Studio |
| 13 | 95.11% | step71 B | local |

### Report's Top 10

Report ranks step71 B (95.11%) at position 10. But step79 configs A (95.85%), B (95.80%), C (95.54%) all rank higher. **Report top 10 missing 3 results** (step79 A/B/C), includes step71 B which actually rank 13.

### Best N=1024 Ranking

Actual top 5 at N=1024 (50%/75ep):
1. step75 D: 87.24%
2. step76 A: 86.55%
3. step73 D: 86.34%
4. **step75 C: 85.76%** (MISSING from report)
5. step82 A: 85.63%

Report lists step82 A at rank 4, step81 A (85.55%) at rank 5, but step75 C (85.76%) should be rank 4, pushing step82 A to 5 and step81 A to 6.

---

## Completeness Check

- [x] Mechanisms explained for top configs -- detailed explanations ranks 1-10 and notable N=1024 results
- [x] Dead ends documented -- comprehensive table, 12 entries
- [x] Confirmed laws -- 6 laws with supporting evidence
- [x] VGG16 baseline defined -- correct FC param/FLOP counts
- [x] Currently running experiments noted -- step82 complete, step83 in progress
- [x] Takeaways grounded in data -- all recommendations reference specific steps and measured results
- [x] Parameter efficiency claim verified -- 529K / 119.6M = 0.44% < 1%

---

## Issues Found

### Issue 1: Top 10 Ranking Incomplete (MEDIUM)
Report omits step79 configs A (95.85%), B (95.80%), C (95.54%) from Tier 2 top 10. These 3 rank 10th-12th globally, above step71 B (95.11%) placed at rank 10. Actual top 10 should include step79 A at rank 10 instead of step71 B.

### Issue 2: N=1024 Ranking Missing step75 C (MINOR)
step75 C (85.76%) missing from "Best N=1024 results" table. Ranks 4th among N=1024 results, between step73 D (86.34%) and step82 A (85.63%).

### Issue 3: step82 Data Divergence Between Machines (NOTE)
step82 results differ: Mac Mini (A=85.63%) vs Mac Studio (A=85.25%). Report uses Mac Mini (local) values. May indicate different runs or data versions. Not error, but worth noting for reproducibility.

### Issue 4: CLAUDE.md Project Best Stale (NOTE, not report issue)
CLAUDE.md says project best 97.32% (best_ep=90), but JSON shows 97.38% (best_ep=142). Report correctly uses JSON value. CLAUDE.md needs update.

### Issue 5: VGG16 Bias Count (TRIVIAL)
Report says "Bias terms omitted (~14K additional)". Actual bias count: 4096 + 4096 + 10 = 8,202. ~14K figure incorrect but explicitly noted as omitted, doesn't affect comparisons.

---

## Verdict

**PASS with minor issues.**

All 30+ accuracy numbers match source JSONs exactly. Parameter counts and FLOP estimates mathematically verified. VGG16 baseline accurate. Core claim (SGNNET at 0.44% of VGG16 FC params, exceeding its accuracy) validated.

Two ranking errors: (1) Tier 2 top 10 omits 3 step79 configs ranking higher than listed step71 B, (2) N=1024 table omits step75 C. Neither affects conclusions or identity of top 3 results. Project best 97.38% confirmed.