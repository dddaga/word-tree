# Research Report: Audio Gap — Critical Correction + Mechanism
*Sub-agent fan-out 2026-06-10. Condensed from agent report.*

## CORRECTION to "gap structural" framing
Gap DID close twice:
- **step962 dense seed**: 57.25% = **+15.5pp ABOVE linear** (but 82× params)
- **step965 subspace**: +1.75pp above linear at 16× params
- **step964**: K0 (no routing) = 60.75% BEST; routing SUBTRACTS −4.75pp on audio

So: not "SGNNET can't do audio" — seeding mechanism mismatched to feature stats.

## Mechanistic smoking gun (computed in-session, CONFIRMED stats)
| Stat | Whisper features | VGG features |
|---|---|---|
| Participation ratio | 4.1 | 180.8 |
| Kurtosis | 0.3 (Gaussian-like) | 51.6 (heavy-tail) |
| Zeros | 0% | 85% |
| Sign | signed/dense | all non-negative |

Unsigned K_in-mean seeding suffers **sign cancellation** on signed dense
features → seed Z variance ≈ 0. VGG immune (non-negative).

## Diagnostics queue
- **D1 random-rotation test on vision** (killer experiment): rotate VGG
  features by random orthogonal Q. Linear invariant; predict SGNNET collapses.
  Confirms seeding-statistics mechanism for BOTH gaps.
- **D2** seed-stage probe ladder: Rademacher signed vs unsigned mean
- **D3** cross-modality feature-stats table (vision/audio/text/TS)

## Long-shots
- **L1**: frame-level Whisper (32×384=12,288-d) — restores coordinate locality
- **L2**: PCA-rotate + signed Rademacher scatter (zero params)
