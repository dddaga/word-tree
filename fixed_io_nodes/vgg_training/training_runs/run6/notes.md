# run6 — Notes

**Config delta from run5:** radiation_targets=0 (disabled), +dropout=0.2. 40ep. LR decay kept.
**Result:** val best=85.81% (ep39), final=85.81%. Train acc=81.38%.
**Notes:** Best result with FFN. Last epoch = best epoch — model still improving or just plateaued at 40. Notable: train acc (81%) < val acc (85%) — dropout regularizing very effectively. Two changes from run5 (removed radiation, added dropout), gain not individually attributable.
**Status: BASELINE target for all FFN-free runs (run10+).**
**Finding: HYPOTHESIS — radiation removal OR dropout addition (or both) drove the +15.8pp gain. Needs clean ablation to isolate.**
