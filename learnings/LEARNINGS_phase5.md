# Phase 5 Learnings — Index

This file is the index. Each part is a standalone document under 250 lines.

## Parts

| File | Content | Lines |
|---|---|---|
| [p1_ladder.md](LEARNINGS_phase5_p1_ladder.md) | Steps 1-4, Combined Run, Dynamic Connectivity Research | ~235 |
| [p2_diagnostics.md](LEARNINGS_phase5_p2_diagnostics.md) | Regression diagnostics, bug fix, infrastructure notes | ~145 |
| [p3_breakthrough.md](LEARNINGS_phase5_p3_breakthrough.md) | Steps 6-10b, D=16 breakthrough, RCA conclusion | ~170 |
| [p4_d16.md](LEARNINGS_phase5_p4_d16.md) | Steps 11-15, new mechanisms, active experiments | ~210 |

## Quick Reference — Key Numbers

| Milestone | Result | Script |
|---|---|---|
| Iter1 reference (D=4 dynamic_z 120ep) | 26.52% | train_resonant.py |
| Encoding sweep D=16 (90ep) | 27.54% | train_encoding_D_sweep.py |
| Step 9A D=16 geo (150ep) | **29.22%** | train_step9_d16_deep.py |
| Step 10a theta-only no-geo (120ep) | 28.38% | train_step10a_routing_ablation_d16.py |
| Step 10b W_phase static (120ep) | 28.59% | train_step10b_wphase_d16.py |

## Active Sessions (2026-03-29)

| Session | Script | Status |
|---|---|---|
| neuro_f | step11 routing ablation WITH geo | RUNNING |
| neuro_g | step12 N scale D=16 | RUNNING |
| neuro_h | step13 beam sweep | RUNNING |
| neuro_i | step13 depth sweep | RUNNING |
| neuro_j | step14 top-K cond + excitatory radiation | RUNNING |

## Experiment Queue

See `EXPERIMENT_QUEUE.md` for full priority-ordered list.
