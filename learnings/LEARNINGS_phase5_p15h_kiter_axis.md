# Phase 5 Part 15h — K_iter Axis + Full Efficiency Criterion Met

Continues from p15g (FLOPs floor / D-reduction + K_hh axis).

---

## FULL EFFICIENCY CRITERION MET (2026-04-11)

**step195 INTERIM: 95.44%@ep90 @ 1.18M FLOPs — ≤1% FLOPs + ≥95% accuracy CONFIRMED**

This is the project's core near-term milestone: SGNNET matches VGG16 FC accuracy at ≤1% of its parameters AND ≤1% of its FLOPs. Config: N=2048 D=16 K_hh=2 K_iter=6, full data 150ep Tier-2. Still running to completion for true best_ep.

---

## K_iter Reduction Axis Results (D=16 K_hh=2 N=2048)

FLOPs formula: 3 × 2048 × 2 × 16 × K_iter

### step194 — K_iter=6 Tier-1 (DONE 2026-04-10)
- Config: 50%/75ep, FLOPs=1,179,648 (~1.18M)
- Result: **94.88%@ep71** — EXTRAORDINARY. Beats K_iter=8 Tier-1 (93.86%) by +1.02pp despite 25% fewer FLOPs.
- Finding: K_iter=6 is NOT worse than K_iter=8 at Tier-1 scale. Possibly better LR schedule fit.
- Decision: advance to Tier-2 (step195)

### step195 — K_iter=6 Tier-2 (DONE 2026-04-11)
- Config: 100%/150ep, FLOPs=1,179,648 (~1.18M), params=67K
- Result: **96.08% best_ep=106** — PHASE EXIT, full efficiency criterion MET
- ≤1% FLOPs (1.18M = 0.96% of VGG16 FC 123.6M) + ≥95% accuracy (96.08%, +1.06pp margin)
- Strongest result in the K_iter axis — K_iter=6 actually outperforms K_iter=8 Tier-2 (95.67%) by +0.41pp at 25% fewer FLOPs

### step196 — K_iter=4 Tier-1 (DONE 2026-04-11)
- Config: 50%/75ep, FLOPs=786,432 (~0.79M) — 34% below ≤1% target
- Result: **92.74%@ep72** — KILLED. −2.14pp vs K_iter=6 ref (94.88%).
- Finding: K_iter=4 too few routing steps. Insufficient convergence at this scale.
- Decision: probe K_iter=5 as gap candidate (step197)

### step197 — K_iter=5 Tier-1 (DONE 2026-04-11)
- Config: 50%/75ep, FLOPs=983,040 (~0.98M)
- Result: **93.96%@ep72** — ≥93% threshold met. Advanced to Tier-2 → step199.

### step199 — K_iter=5 Tier-2 (DONE 2026-04-11)
- Config: 100%/150ep, FLOPs=983,040 (~0.98M), params=67K
- Result: **95.52%@ep136** — **SUB-1% FLOPs PHASE EXIT CONFIRMED**
- 0.98M FLOPs = 0.79% of VGG16 FC 123.6M. New minimum efficiency record.
- Margin: +0.52pp above 95% threshold.

### step200 — K_hh=1 K_iter=8 Tier-1 (DONE 2026-04-11)
- Config: 50%/75ep, FLOPs=786,432 (~0.79M), K_hh=1 (minimum connectivity)
- Result: **90.52%@ep72** — KILLED. −5pp vs K_hh=2 baseline.
- Finding: K_hh=1 breaks connectivity. K_hh=2 is minimum viable for D=16 config.

### N-scaling experiments (2026-04-11)
- step201: N=4096 D=16 K_hh=2 K_iter=6 Tier-1 @ 2.36M → **95.64%@ep66** ✓ PHASE EXIT T1
- step202: N=2048 D=16 K_hh=2 K_iter=3 Tier-1 @ 0.59M → **89.25%@ep74** KILLED
- step203: N=4096 D=16 K_hh=2 K_iter=5 Tier-1 @ 1.97M → **96.08%@ep69** — highest T1 ever at D=16
- step204: N=4096 D=16 K_hh=2 K_iter=6 Tier-2 @ 2.36M → **97.15%@ep71** ✓ PHASE EXIT. vs D=64 record: −0.71pp
- step205: N=4096 D=16 K_hh=2 K_iter=5 Tier-2 @ 1.97M → **97.17%@ep118** ✓ PHASE EXIT. NEW D=16 RECORD. vs D=64 record: −0.69pp
- step206: N=8192 D=16 K_hh=2 K_iter=6 Tier-1 @ 4.72M → **95.11%@ep72** — regression vs N=4096 T1 (−0.53pp). N-scaling law breaks at T1 for N=8192. T2 needed.
- step208: N=8192 D=16 K_hh=2 K_iter=5 Tier-1 @ 3.93M → **95.77%@ep51** — K_iter=5 BEATS K_iter=6 at N=8192 T1 (+0.66pp). Same T1 regression vs N=4096 (−0.31pp). T2 → step209.
- step207: N=8192 D=16 K_hh=2 K_iter=6 Tier-2 @ 4.72M → **96.20%@ep54** — N-SCALING BREAKS. −0.95pp vs N=4096 T2 (97.15%). D=16 bottleneck for K_iter=6 at N=8192.
- step209: N=8192 D=16 K_hh=2 K_iter=5 Tier-2 @ 3.93M → ep120=97.02% — **N-SCALING HOLDS**. Tracking N=4096 K_iter=5 T2 (97.17%@ep118). Running.
- step210: N=8192 D=16 K_hh=2 K_iter=4 Tier-1 @ 3.15M → **95.49%@ep65** — K_iter=4 VIABLE at N=8192 (KILLED at N=2048 at 92.74%!). K_iter axis T1: K6=95.11% < K4=95.49% < K5=95.77%.
- step211: N=8192 D=16 K_hh=2 K_iter=3 Tier-1 @ 2.36M → launched (testing if K_iter floor shifts at N=8192)

---

## FLOPs Frontier — Complete Picture (2026-04-11)

### K_hh axis (D=16 fixed, K_iter=8):
| Config | K_hh | FLOPs | T1 best | T2 best | Status |
|--------|------|-------|---------|---------|--------|
| step185 | 4 | 3.15M | ~94% | 95.87% | ✓ EXIT |
| step191/192 | 3 | 2.36M | 94.68% | 95.90% | ✓ EXIT NEW RECORD |
| step190/193 | 2 | 1.57M | 93.86% | 95.67% | ✓ EXIT NEW RECORD |

### K_iter axis (D=16 K_hh=2, N=2048):
| Config | K_iter | FLOPs | T1 best | T2 best | Status |
|--------|--------|-------|---------|---------|--------|
| step190/193 | 8 | 1.57M | 93.86% | 95.67% | ✓ EXIT |
| step194/195 | 6 | 1.18M | 94.88% | **96.08%** | ✓ EXIT ≤1% CRITERION MET |
| step197/199 | 5 | 0.98M | 93.96% | **95.52%** | ✓ EXIT **SUB-1% NEW MINIMUM** |
| step196 | 4 | 0.79M | 92.74% | KILLED | — |
| step202 | 3 | 0.59M | 89.25% | KILLED | floor |

### N-scaling axis (D=16 K_hh=2 K_iter=6):
| N | FLOPs | T1 best | T2 best | Status |
|---|-------|---------|---------|--------|
| 1024 (step198) | 0.59M | 88.92% | KILLED | N=1024 insufficient |
| 2048 (step195) | 1.18M | 94.88% | **96.08%** | ✓ EXIT ≤1% CRITERION MET |
| 4096 (step201/204) | 2.36M | 95.64% | **97.15%** | ✓ EXIT N-scaling law |
| 8192 (step206/207) | 4.72M | 95.11% | **96.20%@ep54** | N-SCALING BREAKS: −0.95pp vs N=4096 T2; plateau after ep54 |

### N-scaling axis (D=16 K_hh=2 K_iter=5):
| N | FLOPs | T1 best | T2 best | Status |
|---|-------|---------|---------|--------|
| 2048 (step197/199) | 0.98M | 93.96% | **95.52%** | ✓ EXIT sub-1% |
| 4096 (step203/205) | 1.97M | **96.08%** | **97.17%** | ✓ EXIT NEW D=16 RECORD |
| 8192 (step208/209) | 3.93M | **95.77%** | running | K_iter=5 beats K_iter=6 at N=8192 T1; T2 in progress |

### Full frontier (all confirmed exits):
| Step | Config | FLOPs | Accuracy | Notes |
|------|--------|-------|----------|-------|
| step176-A | N=2048 D=32 K_hh=4 K_iter=8 | 6.10M | 96.18% | First exit |
| step181 | N=2048 D=20 K_hh=4 K_iter=8 | 3.93M | 96.03% | |
| step185 | N=2048 D=16 K_hh=4 K_iter=8 | 3.15M | 95.87% | D-reduction floor |
| step192 | N=2048 D=16 K_hh=3 K_iter=8 | 2.36M | 95.90% | K_hh reduction |
| step193 | N=2048 D=16 K_hh=2 K_iter=8 | 1.57M | 95.67% | K_hh=2 min |
| step195 | N=2048 D=16 K_hh=2 K_iter=6 | 1.18M | **96.08%** | **≤1% FLOPs CRITERION MET** |
| step199 | N=2048 D=16 K_hh=2 K_iter=5 | 0.98M | **95.52%** | **SUB-1% MINIMUM** |
| step204 | N=4096 D=16 K_hh=2 K_iter=6 | 2.36M | **97.15%** | N-scaling +1.07pp vs N=2048 T2 |
| step205 | N=4096 D=16 K_hh=2 K_iter=5 | 1.97M | **97.17%** | **D=16 RECORD**, −0.69pp from D=64 |

---

## Other Tier-2 Completions (2026-04-10/11)

### step184 — D=24 K_hh=4 Tier-2 (Mac Mini MPS)
- ep70: 95.03% PHASE EXIT. ep110: 94.01% (post-peak oscillation normal).
- Confirms: D=24 also exits at ~4.72M FLOPs (above current floor).

### step183 — D=28 K_hh=4 Tier-2 (Mac Mini CPU)
- ep130: 94.39%, best 94.57%@ep90.
- Likely NO phase exit at D=28 (~5.51M FLOPs). D=28 sits below D=32 floor.

---

## Critical Findings Summary

1. **D=16 is the binding constraint** (CONFIRMED): at fixed FLOPs, higher D + lower K_hh always dominates lower D + higher K_hh. Floor via D-reduction = D=16.

2. **K_iter reduction opens a new axis below D-floor**: at D=16 K_hh=2, reducing K_iter from 8→6 cuts FLOPs 25% with NO accuracy loss at Tier-1 (actually +1.02pp). This is the unexpected result that enabled hitting ≤1% FLOPs.

3. **Full efficiency criterion met at step195**: ≤1% FLOPs (1.18M) + ≥95% accuracy (96.08%). Core near-term milestone achieved.

4. **K_iter=4 killed**: too few routing steps, −2.14pp regression. Minimum routing depth appears to be K_iter=5-6.

5. **N-scaling law at D=16 (CONFIRMED, 2026-04-11)**: Doubling N from 2048→4096 adds +1.07pp at T2 (K_iter=6) and gives the D=16 record at 97.17% (K_iter=5). D=16 can approach D=64 accuracy (97.86%) purely via N-scaling. Gap is now −0.69pp.

6. **N=8192 T1 regression (HYPOTHESIS)**: step206 N=8192 T1=95.11% is −0.53pp below N=4096 T1. Possible causes: BATCH=64 with 50% data, LR mismatch, or D=16 representation bottleneck at large N. T2 (step207) will clarify — if T2 shows +1.5pp lift, law holds at full training.

7. **New D=16 accuracy records**: step205 = 97.17% @ 1.97M FLOPs (K_iter=5, N=4096). step204 = 97.15% @ 2.36M FLOPs (K_iter=6, N=4096). Both within 0.7pp of all-time D=64 record (97.86%) at 50× fewer FLOPs.
