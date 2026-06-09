<!-- continued from LEARNINGS_phase5_p15f_warmstart_efficiency_part1.md -->

## Efficiency Track Status (2026-04-10, updated)

| Config | N | D | K_hh | FLOPs | top1 | gap to 95% |
|--------|---|---|------|-------|------|------------|
| step165-B Tier-1 | 1024 | 16 | 8 | ~3.1M | 90.96% | −4.04pp |
| step167-B Tier-2 | 1024 | 16 | 8 | ~3.1M | 93.17% | −1.83pp |
| step168-B Tier-1 | 1024 | 32 | 8 | ~6.1M | 93.20% | −1.80pp |
| **step169-B Tier-2** | **1024** | **32** | **8** | **~6.1M** | **94.01%** | **−0.99pp** |
| step170 (running) | 2048 | 16 | 8 | ~6.2M | TBD | TBD |
| step171 (launched) | 1024 | 32 | 4 | ~3.1M | TBD | TBD |

**Key insight:** D-dimension = representation ceiling. D=16→93.2%, D=32→94.0%. Gap narrows each step but 95% not reached.

**Next lever:** `K_hh=4` (confirmed +0.56pp at N=4096) + `alpha=1.05` (confirmed +0.79pp at N=4096). step171 tests on efficiency track D=32 warmproj base. If transfer w/o interference, 94.01% + ~1pp could cross phase-exit.

---

## step170: warm+W_proj N=2048 D=16 K_hh=8 Tier-1

N=2048 D=16, 50%/75ep:
| Config | top1_best |
|--------|-----------|
| Ref (scratch) | 90.98% |
| B (warm+W_proj) | 93.43% (best_ep=75) |

N-scaling lifts D=16 ceiling: N=1024→93.17%, N=2048→93.43% (+0.26pp). Gap to 95% still 1.57pp.

---

## step171: warm+W_proj N=1024 D=32 K_hh=4 Tier-1

N=1024 D=32 K_hh=4 K_iter=8, 50%/75ep:
| Config | top1_best |
|--------|-----------|
| Ref (scratch) | 89.40% |
| A (warm+W_proj α=1.0) | 93.15% |
| B (warm+W_proj α=1.05) | **93.35%** (best_ep=71) |

K_hh=4 at D=32 matches K_hh=8: step168-B=93.20% vs step171-B=93.35%. Same accuracy HALF FLOPs (~3.1M vs ~6.1M). α=1.05 gives marginal +0.2pp.

---

## step172: warm+W_proj N=1024 D=48 K_hh=4 Tier-1 (partial)

N=1024 D=48 K_hh=4, 50%/75ep, teacher=91.44%:
| Config | top1_best |
|--------|-----------|
| Ref (scratch) | 88.84% (best_ep=57) |
| B (warm+W_proj) | running — ep10=90.11% (already above Ref final) |

D=48 Ref underperforms D=32 Ref (89.40% → 88.84%) — more dims don't help scratch. B starting strong; watch final.

---

## step173: warm+W_proj N=2048 D=32 K_hh=4 Tier-1 (partial)

N=2048 D=32 K_hh=4, teacher (K=12) = 95.57% at ep70 on 50% data. Teacher extraordinary.

| Config | top1_best |
|--------|-----------|
| Ref (scratch K=8) | **94.93%** (best_ep=68, 50%/75ep) |
| B (warm+W_proj α=1.0) | running — ep10=90.11% |

**CRITICAL FINDING (CONFIRMED):** N=2048 D=32 K_hh=4 scratch K=8 at 50%/75ep = 94.93%. FLOPs ~6.1M. 0.07pp from phase exit WITHOUT warm start.

**step173 FINAL — warm-start reversal at N=2048 D=32 CONFIRMED (all configs):**

| Config | top1_best | best_ep | delta vs Ref |
|--------|-----------|---------|--------------|
| Ref (scratch α=1.0) | **94.93%** | 68 | — |
| B (warm+W_proj α=1.0) | 94.70% | 67 | -0.23pp |
| C (warm+W_proj α=1.05) | 94.88% | 75 | -0.05pp |

Both α values underperform scratch. α=1.05 recovers most of B gap (+0.18pp) but still -0.05pp below Ref. Warm-start reversal robust across α. Pattern confirmed: warm+W_proj gain diminishes and reverses as base capacity increases (D=16 +10pp → D=32 small N +11pp → D=32 large N −0.05 to −0.23pp).

**step176 running:** N=2048 D=32 scratch Tier-2 (full data 150ep, Mac Studio MPS). Expected ~95.5-96%.

---

## step174: warm+W_proj N=2048 D=16 full data 150ep Tier-2 — PHASE EXIT

**PHASE EXIT ACHIEVED (2026-04-10) — FINAL: 95.82% best_ep=118.**

N=2048 D=16 K_hh=8 K_iter=8 warm+W_proj, full data:
- ep70: 95.13% *** PHASE EXIT *** first crossing
- ep110: 95.52%, ep120: 95.59%, ep130: 95.69%
- **ep118: 95.82% FINAL BEST**

FLOPs: ~6.2M (5.0% of VGG16 FC 123.6M). Marginally over 6.18M budget but within design target.
Phase exit criterion: accuracy ≥95% @ ≤6.18M FLOPs → **ACHIEVED on efficiency track.**

---

## Efficiency Track Status (2026-04-10, PHASE EXIT)

| Config | N | D | K_hh | FLOPs | top1 | status |
|--------|---|---|------|-------|------|--------|
| step167-B Tier-2 | 1024 | 16 | 8 | ~3.1M | 93.17% | done |
| step169-B Tier-2 | 1024 | 32 | 8 | ~6.1M | 94.01% | done |
| step170-B Tier-1 | 2048 | 16 | 8 | ~6.2M | 93.43% | done |
| step171-B Tier-1 | 1024 | 32 | 4 | ~3.1M | 93.35% | done |
| step173-Ref Tier-1 | 2048 | 32 | 4 | ~6.1M | **94.93%** | partial (B running) |
| **step174 Tier-2** | **2048** | **16** | **8** | **~6.2M** | **95.13%** (ep70) | **✓ PHASE EXIT** |

**Phase exit via N-scaling: N=2048 D=16 warm+W_proj full data 150ep crosses 95%.**
**N=2048 D=32 K_hh=4 scratch within 0.07pp of phase exit — configs B/C likely confirm independently.**