# run22 — Notes

## Config delta from run17
- `cardinality: 4` (was 200)
- All else identical to run17

## Result
- **val_best: 25.83% @ep40** (barely improving at final epoch)
- FLOPs/fwd: 0.02B (48x savings vs run17's 0.96B)
- Gap from run17: **-61.02pp**
- Gap from run21 (C=25): -38.71pp — catastrophic drop from halving cardinality again

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep1   | 12.36%  |
| ep5   | 16.82%  |
| ep10  | 19.69%  |
| ep15  | 21.10%  |
| ep20  | 22.37%  |
| ep25  | 24.38%  |
| ep30  | 24.43%  |
| ep35  | 25.04%  |
| ep38  | 25.17%  |
| ep39  | 25.50%  |
| ep40  | 25.83%  |

## Analysis
- **Confirmed gradient starvation at C=4.** Val starts at 12.36% (barely above random chance: 10%), crawls to 25.83% over 40 epochs. Never breaks out.
- **Gain rate ~0.3-0.4pp/epoch throughout**, decelerating. Completely unlike the clean sigmoid curve in higher-cardinality runs.
- No early-stop triggered: val stayed above 15% threshold (12.36%→21.10% by ep15). The early-stop rule didn't catch this case — C=4 isn't a hard fail, it's just nearly random performance.
- **Train acc only 23.23% at ep40.** Both train and val stuck near 23-25%: underfitting, not overfitting. The GNN cannot learn the task at this sparsity.
- **Cardinality cliff confirmed:** C=200→86.85%, C=100→80.66%, C=50→71.64%, C=25→64.54%, C=4→25.83%. The trend is nonlinear — C=4 falls off a cliff vs C=25.

## Verdict
DONE. C=4 is confirmed below the cardinality floor for our routing mechanism. 25.83% ≈ 2.6x random chance after 40 epochs.
**Cardinality sweep conclusion:** The minimum viable cardinality for meaningful learning is somewhere between C=25 and C=100. C=25 delivered 64.54% (viable but poor), C=4 delivered 25.83% (non-viable).
Cardinality is NOT a free FLOPs lever. The ~7-9pp cost per halving makes cardinality reduction uncompetitive with beam_width for FLOPs savings.
Next: run24 (beam=1024) — beam sweep continues.
