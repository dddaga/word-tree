# Research Report: CIFAR-10 Gap (−5.67pp vs Linear) — Improvement Plan
*Sub-agent fan-out 2026-06-10. Condensed from agent report.*

## Primary diagnosis
**Seed-projection quality**, not capacity. Evidence: seed_Fisher dominates
(step951); step962 audio dense-seed crossover; K_in=50/100 HURT on CIFAR-10
— alignment problem, not count problem.

## Ranked experiments
| ID | What | Cost | Expected |
|---|---|---|---|
| E1 | TTA hflip prob-averaging (eval-only; needs test-side flipped features) | tiny | +0.3–0.6pp |
| E2 | Feature standardization/whitening T0 (z-score, PCA-whiten); measure CIFAR-10 feature std first | T0 | +0.5–1.5pp |
| E3 | Dense/structured seed projection port from audio line (PCA→512 variants) to CIFAR-10 | T0 | biggest single lever |
| E4 | Feature-space mixup + label smoothing | T0 | +0.3–0.8pp |
| E5 | 3-seed prob ensemble + 300ep snapshot schedule | T2-cost | +0.5–1.0pp |

## Path estimate
Stacked E1–E5 ≈ 83.8–84.5% at N=2048. Crossing Linear 86.24% likely needs
E3 + N=4096 (K_in=15 helps at N≥4096).

## Context
step982 T2 aug pipeline running (A_aug claim threshold = Ref 80.58 + 0.5 = 81.08%).
