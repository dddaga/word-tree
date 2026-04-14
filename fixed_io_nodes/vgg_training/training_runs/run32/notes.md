# run32 — Notes

## Config delta from run29
- `cardinality: 100` (was 200)
- All else identical to run29 (uniform routing, N=4146, V=8, I=5)

## Result
- **val_best: 85.89% @ep40** (still improving — val loss still declining)
- FLOPs/fwd: ~0.48B (2x savings vs run17/run29)
- vs run19 (softmax, C=100): **+5.23pp** (80.66% → 85.89%)
- vs run29 (uniform, C=200): -3.26pp at half the FLOPs
- vs run17 (softmax, C=200): -0.96pp at half the FLOPs — nearly matches best softmax at half FLOPs!

## Val trajectory (selected)
| Epoch | Val Acc | run19 same ep |
|-------|---------|----------------|
| ep1   | 21.50%  | — |
| ep5   | 54.50%  | — |
| ep10  | 69.55%  | 65.04% |
| ep15  | 76.23%  | — |
| ep20  | 80.25%  | 73.96% |
| ep25  | 82.90%  | — |
| ep30  | 84.59%  | 78.01% |
| ep35  | 85.22%  | 78.93% |
| ep40  | 85.89%  | 80.66% |

## Analysis
- **Uniform routing sweep complete:**
  - C=2:   18.39% (-10.65pp vs softmax) — uniform WORSE at extreme sparsity
  - C=50:  79.54% (+7.90pp vs softmax) — peak gain at moderate C
  - C=100: 85.89% (+5.23pp vs softmax) — NEW
  - C=200: 89.15% (+2.30pp vs softmax) — gain saturates at high C
- **Gain curve shape:** Gain vs softmax peaks at C=50 (+7.90pp) then declines monotonically. At C=200, gain is small (+2.30pp); at C=2, uniform fails (-10.65pp).
- **Best per-FLOP operating points:**
  - run31 (C=50): 79.54% @ 0.24B FLOPs = 332pp per GB FLOPs
  - run32 (C=100): 85.89% @ 0.48B FLOPs = 179pp per GB FLOPs
  - run29 (C=200): 89.15% @ 0.96B FLOPs = 93pp per GB FLOPs
  - **Lower C = more accuracy per FLOP.** Even though total accuracy drops, the efficiency improves.
- **run32 at 0.48B FLOPs is nearly as good as softmax run17 at 0.96B FLOPs** (85.89% vs 86.85%). Effectively doubled FLOPs efficiency.
- Still improving at ep40 — likely crosses 86% with more epochs.

## Verdict
DONE. **Pareto frontier significantly improved:**
- Best accuracy: 89.15% @ 0.96B FLOPs (run29)
- Best accuracy per FLOP: 85.89% @ 0.48B FLOPs (run32)
- Very good efficiency: 79.54% @ 0.24B FLOPs (run31)

Next: push accuracy ceiling (more iterations?) or reduce FLOPs further (fewer iterations / smaller D).
