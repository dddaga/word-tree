# Paper Audit — 2026-04-20

**Draft version audited:** v0.1 (2026-04-11)
**Audited against:** findings_log_part3.md + confirmed results up to step943
**Result:** 12 gaps, 7 HIGH priority, ~15hr estimated writing effort

## Critical Path

GAP 3 (ΔW-proj absent) is load-bearing — unlocks GAPs 1, 2, 4, 12.
Current draft describes the pre-ΔW-proj SmallWorld architecture. Results are incomprehensible without it.

## HIGH Priority Gaps (blocks submission)

| # | Section | Gap | Est. |
|---|---------|-----|------|
| 3 | §3, §5 | ΔW-proj routing mechanism ENTIRELY ABSENT from draft | 4–6hr |
| 1 | Abstract, §1.3, §6.2 | Efficiency champion wrong: step199→step605 (35K params, 0.20M FLOPs, 5.26× wall-time) | 2–3hr |
| 2 | Abstract, §1 | Param count wrong: 67K→34,976 (0.029%) everywhere | 0.5hr |
| 4 | §6 | Multi-seed variance absent: 96.38% ± 0.18pp (step887, 3 seeds) | 1hr |
| 5 | §7.3, §8 | MLP+GNN baselines marked "NOT DONE" but ARE done (steps 891–893, step404) | 1.5hr |
| 6 | §7.3, §9 | CIFAR-10 in "future work" but results ARE done (steps 882, 909, 914) | 2hr |
| 12 | §1, §6 | Headline accuracy: 95.52%→96.38% ± 0.18pp (cascades from GAP 3) | 1hr |

## MEDIUM Priority Gaps

| # | Gap | Est. |
|---|-----|------|
| 7 | Gate-death count: 27→30+ mechanisms | 0.5hr |
| 8 | K_in crossover rule absent (K_in=25 at N≤2048, K_in=15 at N≥4096) | 0.5hr |
| 9 | Audio honest negative absent (ESC-50: −11.5pp structural gap, steps 926/928) | 0.5hr |
| 11 | θ threshold overclaimed as load-bearing; step886 shows neutral (simplifies out) | 1hr |

## LOW Priority

| # | Gap | Est. |
|---|-----|------|
| 10 | Text negative (SST-2/AG News) absent from Limitations | 0.25hr |

**Total HIGH: ~13–15hr. Total all: ~16hr.**
