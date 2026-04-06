# neuro_graph — Learnings Index

Human-readable record of decisions, failures, and hard-won insights.
Written so future contributors (and future AI assistants) don't repeat the same mistakes.

GSD planning artifacts (`.planning/`) contain execution history.
This file is the index — each sub-file is under 250 lines.

---

## Sub-files

| File | Content |
|---|---|
| [LEARNINGS_arch.md](LEARNINGS_arch.md) | Architecture decisions, W_pos/W_phase, safety valve, ResonantSGNNet design |
| [LEARNINGS_ops.md](LEARNINGS_ops.md) | Training stability, failed experiments, MPS specifics, performance benchmarks |
| [LEARNINGS_research.md](LEARNINGS_research.md) | Translation invariance, open hypotheses, ceiling analysis |

## Phase 5 Sub-files

| File | Content |
|---|---|
| [LEARNINGS_phase5.md](LEARNINGS_phase5.md) | Phase 5 index + active session state |
| [LEARNINGS_phase5_p1_ladder.md](LEARNINGS_phase5_p1_ladder.md) | Steps 1-4, Combined Run, Dynamic Connectivity |
| [LEARNINGS_phase5_p2_diagnostics.md](LEARNINGS_phase5_p2_diagnostics.md) | Regression diagnostics, trainer bug, infrastructure |
| [LEARNINGS_phase5_p3_breakthrough.md](LEARNINGS_phase5_p3_breakthrough.md) | Steps 6-10b, D=16 breakthrough, RCA conclusion |
| [LEARNINGS_phase5_p4_d16.md](LEARNINGS_phase5_p4_d16.md) | Steps 11-15, new mechanisms, D=16 era |

## Quick Reference — Current Best

| Metric | Value | Config |
|---|---|---|
| Best accuracy | **29.22%** | D=16 Fourier N=512 dynamic_z_geo 150ep (step9A) |
| Previous best (iter1) | 26.52% | D=4 linear dynamic_z 120ep |
| Active ceiling | ~29% | Being expanded by steps 11-15 |

## Experiment Queue

See `EXPERIMENT_QUEUE.md` for priority-ordered experiment list.
