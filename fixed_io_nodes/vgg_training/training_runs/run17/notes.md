# run17 — Notes

**Config delta from run16:** None to config — re-run under new post-update LayerNorm (commit f11f768).
  - Hardcoded mean-subtraction removed from `update_activations`
  - `nn.LayerNorm` moved from pre-update (source mag before aggregation) to post-update (new_mag after update, act_strength recomputed from normalised mag)
  - γ, β now genuinely learnable, affect stored node state
**Hypothesis:** New LN placement (learnable γ, β) improves over old LN (hardcoded γ=1, β=0).
**Status:** DONE (40/40, val_best=**86.85% @ep38**)
**Comparison targets:**
  - run16 (same config, old LN → 86.55% @ep40)
  - FFN target run6 (85.81%)

## Results

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 5 | 49.31% | 64.28% |
| 10 | 63.97% | 73.99% |
| 15 | 70.73% | 78.14% |
| 20 | 74.54% | 81.50% |
| 25 | 76.85% | 83.41% |
| 30 | 78.34% | 84.87% |
| 35 | 81.09% | 85.66% |
| 36 | 80.77% | 86.27% |
| 37 | 80.83% | 86.19% |
| **38** | 81.20% | **86.85%** ← best |
| 39 | 81.27% | 86.14% |
| 40 | 81.70% | 86.17% |

## Analysis

- **run17 (new LN) beats run16 (old LN): +0.30pp** at best config. 86.85% vs 86.55%.
- **Early epochs slower:** ep1-15 run17 consistently ~2-3pp behind run16 at same epoch. At ep10: run17=73.99% vs run16=76.20%. Gap closes by ep20 and matches/exceeds run16 from ep24 onwards.
- **Convergence pattern:** run17 peaked at ep38 (86.85%) and slightly declined to 86.17% @ep40. run16 was still improving at ep40 (86.55%). Suggests new LN allows faster final-stage convergence but with slight overshoot at the end.
- **Train accuracy:** run17 final train=81.70% vs run16 final=82.28%. New LN has slightly lower train acc but higher val acc — less overfitting, better generalization.
- **Exceeded FFN target by +1.04pp** (vs run16's +0.74pp).
- **Evidence tag (HYPOTHESIS):** Single-variable change (LN placement), but modest +0.30pp gain. Result is within possible run-to-run variance; needs run18 + maybe a repeat to confirm CONFIRMED status.

## Verdict

**HYPOTHESIS:** New post-update LN with learnable γ, β slightly improves over old pre-update LN. +0.30pp at best config. Early epochs run slower but final-stage convergence is faster and generalization is slightly better. Learnable γ, β appear to be contributing (needs diagnose.ipynb inspection to confirm they're not collapsed to 1/0).

run18 will test whether this change also helps at T=1.0 (gradient starvation regime).
