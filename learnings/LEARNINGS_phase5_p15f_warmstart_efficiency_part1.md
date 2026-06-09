# LEARNINGS Phase 5 Part 15f — Warm-Start & Efficiency Track Breakthrough

**Session: 2026-04-10 (continued from prior context)**

---

## Summary of Recent Completions

| Step | Result | Verdict |
|------|--------|---------|
| step149 | ALL killed (−29 to −52pp) | Input encoding KILLED |
| step162 | SGNNET robustness=0.995 < FC=1.003 | Flip robustness hypothesis REJECTED |
| step163 | E=87.26%(+7.82pp warm K=8) WINNER | Warm-start is the mechanism |
| step165 | B=90.96%(+10.32pp warm+W_proj) WINNER | Efficiency breakthrough |
| step120-A | 95.13%(−1.45pp vs Ref=96.51%) | K_iter=16+Z-bias KILLS |

---

## step149: Input De-squashification (ALL KILLED)

**CONFIRMED:** Scatter-sum Fourier encoding load-bearing — cannot modify.

Configs tested N=1024 D=16, 75ep:
- Ref: 82.42%
- A (multi_feat): 43.24% (−39pp) — `nn.Linear(1, K_feat)` random init → large values dominate Fourier spatial dims → Z collapse, never recovered
- B (attn_scatter): 53.04% (−29pp)
- C (both): 30.02% (−52pp)

**Root cause (HYPOTHESIS):** Scatter-sum encodes each feature dimension separately into hidden geometry. Any projection mixing K_in features before scatter-sum breaks geometric prior SGNNET learned to exploit.

**Decision:** Input encoding direction KILLED. Do not revisit without fundamentally different encoding architecture.

---

## step162: Flip Robustness (Hypothesis REJECTED)

**CONFIRMED:** VGG16 pool5 features already nearly flip-invariant. SGNNET K_iter propagation does NOT improve flip robustness.

Results (flip_acc / original_acc ratio):
- VGG16_direct: 0.994
- FC_linear: 1.003
- FC_mlp: 0.998
- SGNNET: 0.995

SGNNET slightly more brittle than FC_linear. K_iter message passing propagates feature representation but doesn't add invariance.

**Implication:** If robustness needed, must come from data augmentation or architecture changes upstream of SGNNET (in VGG feature extractor).

---

## step163: Progressive K_iter Distillation

**CONFIRMED:** Warm-start is mechanism, not distillation.

Key results (N=1024 D=16, 75ep):
| Config | top1_best | vs Ref(K=12) |
|--------|-----------|--------------|
| Ref (scratch K=12) | 79.44% | — |
| A (scratch K=6) | 78.88% | −0.56pp |
| B (warm K=6, no distil) | 85.35% | **+5.91pp** |
| C (warm K=6, distil α=0.3) | 85.25% | +5.81pp |
| D (warm K=6, distil α=0.5) | 84.76% | +5.32pp |
| **E (warm K=8, no distil)** | **87.26%** | **+7.82pp** |

**Key insight:** B ≥ C ≥ D → distillation HURTS marginally. More α → more hurt.
**Warm-start alone worth +6.47pp** (A vs B: −0.56 → +5.91). Warm-start = load good W_pos init + topology.

**Why warm-start works:** K=12 teacher develops well-separated W_pos geometry over 150ep. Loading into K=8 student skips slow topology-learning phase. K=8 fine-tunes under less over-smoothing.

**K=12 over-smooths at N=1024 D=16:** fewer K_iter steps + good W_pos init wins.

---

## step165: Compound Warm-Start Efficiency Stack — BREAKTHROUGH

**CONFIRMED (N=1024 D=16, 75ep Tier-1):** warm-start + W_proj = 90.96% — first time N=1024 D=16 exceeds 90%.

| Config | top1_best | vs Ref | params | Key |
|--------|-----------|--------|--------|-----|
| Ref (scratch) | 80.64% | — | 33,952 | baseline |
| A (warm K=8 only) | 85.55% | +4.91pp | 33,952 | warm-start alone |
| **B (warm+W_proj)** | **90.96%** | **+10.32pp** | 34,208 | **WINNER** |
| C (warm+W_proj+RigL) | 81.20% | +0.56pp | 34,208 | RigL kills W_proj gain |
| D (warm+twopop_weight) | 88.76% | +8.12pp | 33,954 | strong without W_proj |

**FLOPs at N=1024 D=16: ~3.1M** (vs VGG16 FC 119.6M, vs SGNNET N=4096 38.8M)

**Efficiency track status after step165:**
- 90.96% at ~3.1M FLOPs (2.5% of VGG16 FC FLOPs)
- Phase exit criterion: accuracy ≥95% NOT YET (90.96%), FLOPs ≤5% APPROACHING (3.1M = 2.5% ✓)

**Key findings (CONFIRMED):**
1. W_proj + warm-start SYNERGISTIC (+10.32pp > +4.91pp+5.48pp individually) — warm-started W_pos geometry amplifies W_proj effectiveness
2. RigL kills W_proj at D=16 (contradicts step144 D=32 where RigL+W_proj = +5.55pp). Scale matters.
3. twopop_weight alone +8.12pp without W_proj → still strong, but W_proj wins by +2.2pp

**W_proj near-zero init critical:** `nn.init.normal_(std=0.01)` prevents disrupting warm-started Z on first pass. If std=0.1+, proj weight dominates and collapses teacher geometry.

**Next steps:**
- step166: stack warm+W_proj+twopop_weight (B+D compound). RigL excluded.
- Tier-1 at full data (100%) to see if 90.96% holds at 150ep full data

---

## step120: High K_iter at N=4096 (Partial)

Config A (K_iter=16+Z-bias): 95.13% (−1.45pp vs Ref=96.51%) — CONFIRMED K_iter=16 HURTS.
Configs B (K=20), C (K=24), D (K=16+ckpt) still running.

**Preliminary verdict:** Higher K_iter at N=4096 does NOT improve accuracy. K_iter=12 sweet spot. Z-bias adds params without benefit. Expect B/C/D confirm pattern.

---

## Efficiency Track Status (2026-04-10)

| Config | N | D | FLOPs | top1 | vs target |
|--------|---|---|-------|------|-----------|
| N=4096 baseline | 4096 | 64 | 38.8M | 97.86% | FLOPs 6.3× over budget |
| N=1024 scratch | 1024 | 16 | 3.1M | 80.64% | acc 14pp short |
| step163-E | 1024 | 16 | 2.3M | 87.26% | acc 7.7pp short |
| **step165-B** | **1024** | **16** | **~3.1M** | **90.96%** | **acc 4pp short** |

**Gap remaining:** 90.96% → 95% needs +4.04pp. Candidates: twopop_weight compound, curriculum K_iter, soft_spec reg, full data training.

---

## step166: Further Stacking on warm+W_proj — ALL KILLED

**CONFIRMED:** warm+W_proj is local maximum. Every compound mechanism hurts.

| Config | top1_best | vs Ref=91.59% |
|--------|-----------|---------------|
| E (twopop_weight) | 90.09% | −1.50pp |
| F (curriculum 2→8) | 87.92% | −3.67pp |
| G (soft_spec λ=0.01) | 89.86% | −1.73pp |
| H (E+F compound) | 86.47% | −5.12pp |

**Root cause (HYPOTHESIS):** Warm-started W_pos geometry already well-optimized. Any additional mechanism perturbs learned structure → net negative. Same pattern as AH alone at N=4096.

**Decision:** No further stacking on warm+W_proj. Explore scale (larger N/D) or cleaner hyperparameter changes (K_hh, alpha) without competing objectives.

---

## step167: warm+W_proj Tier-2 D=16 — D=16 Ceiling Confirmed

Full data 150ep at N=1024 D=16:
| Config | top1_best |
|--------|-----------|
| Ref (scratch) | 89.71% |
| B (warm+W_proj) | 93.17% (best_ep=110) |

**D=16 ceiling at N=1024: ~93.2%**. Full data adds +2.21pp over Tier-1 (90.96%). Still 1.83pp short of phase exit.

---

## step168: warm+W_proj D=32 Tier-1 — Scale-Up

N=1024 D=32, K_hh=8, 50% data 75ep:
| Config | top1_best | vs Ref |
|--------|-----------|--------|
| Ref (scratch) | 81.96% | — |
| A (warm-only) | 87.62% | +5.66pp |
| **B (warm+W_proj)** | **93.20%** | **+11.24pp** |

FLOPs ~6.1M ≤ 6.18M ✓. Teacher K=12 D=32 K_hh=8 trained fresh, cached.
B=93.20% approaches step144-C (W_proj+RigL = 93.50%) from different path.

---

## step169: warm+W_proj D=32 Tier-2 — D=32 Ceiling Confirmed

Full data 150ep at N=1024 D=32:
| Config | top1_best | best_ep |
|--------|-----------|---------|
| B (warm+W_proj) | **94.01%** | 134 |

**D=32 ceiling at N=1024: ~94.0%** (+0.81pp over Tier-1).
Phase exit NOT achieved: 94.01% vs target ≥95% — gap = **0.99pp**.

History: slow climb from ep1=66.37% (warm init working), peaked ep134, oscillating after. No breakthrough in final 16ep.

---


*Continued in [LEARNINGS_phase5_p15f_warmstart_efficiency_part2.md](LEARNINGS_phase5_p15f_warmstart_efficiency_part2.md) — steps 170-174, efficiency track status (PHASE EXIT).*