# run3 — Notes

**Config delta from run1:** +layernorm=true. 20ep.
**Result:** val best=81.38% (ep19), final=81.38%. Train acc=90.53%.
**Notes:** LayerNorm is the only change from run1. +5.33pp val improvement (76.05% → 81.38%). Train/val gap narrowed (94%/76% → 91%/81%), suggesting layernorm also regularizes.
**Finding: CONFIRMED — LayerNorm helps. Never disabled after this run.**
