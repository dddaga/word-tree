# run42 — Notes

## Config delta from run40
- `cardinality: 25` (was 50)
- All else identical (uniform routing, N=2048, V=16, I=5)

## Result
- **val_best: 77.71% @ep40** (still improving at final epoch)
- FLOPs/fwd: ~0.12B
- vs run40 (C=50): -5.35pp — 2x cheaper, moderate cost
- vs run41 (C=10): +49.74pp — cliff is sharply between C=10 and C=25
- vs run39 (D=16, C=50, I=3): +2.22pp at same ~0.12B FLOPs — Pareto-dominant over run39!

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep1   | 15.36%  |
| ep5   | 44.56%  |
| ep10  | 61.15%  |
| ep15  | 66.70%  |
| ep20  | 70.80%  |
| ep25  | 73.48%  |
| ep30  | 75.85%  |
| ep35  | 76.64%  |
| ep39  | 77.55%  |
| ep40  | 77.71%  |

## Analysis
- **Cliff confirmed between C=10 and C=25:** C=10→27.97%, C=25→77.71% (+49.74pp jump). The viability threshold is sharply between these values.
- **C=25 is above the cliff:** 77.71% is well above random chance (10%) and well above run41's 27.97%.
- **Still improving at ep40:** Slope ~0.16pp/epoch at ep39-40 — likely would reach 79-80% with more epochs.
- **Pareto at 0.12B tier:** run42 (77.71%) beats run39 (D=16, C=50, I=3: 75.49%) at roughly the same FLOPs. The I=5 boost outweighs the halved cardinality.
- **C-I coupling holds:** C=25+I=5 = 77.71%; C=50+I=3 = 75.49% at same FLOPs. High I compensates for low C.
- **Cliff shape:** 
  - C=10 → 27.97% (below floor)
  - C=25 → 77.71% (viable)
  - C=50 → 83.06% (viable)
  - Cliff is between C=10 and C=25 — likely around C=15-18

## Pareto frontier (D=16, uniform routing) — updated
| FLOPs/fwd | Run | Config | Val Best |
|-----------|-----|--------|----------|
| ~0.12B | **run42** | C=25, I=5 | **77.71%** ← Pareto here |
| ~0.23B | run38 | C=100, I=3 | 85.91% |
| ~0.45B | run36 | C=200, I=3 | 89.55% |
| ~0.90B | run37 | C=200, I=5 | 90.45% |

## Verdict
DONE. **C=25 is viable — above the cliff.** Cliff lies between C=10 and C=25.
Config sweep is now exhausted at this level. The FLOPs/accuracy tradeoff bottoms out at 0.12B for viable accuracy (77%+). Getting to 1M FLOPs requires >100x reduction — not achievable through C/I/D sweeps.
