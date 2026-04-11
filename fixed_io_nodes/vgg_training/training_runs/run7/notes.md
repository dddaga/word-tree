# run7 — Notes

**Config delta from run6:** +topology="layered", iterations=7 (was 5), temporal_decay=0.8 (was 1.0). Stopped at 16 epochs.
**Result:** val best=23.08% (ep9), final=17.43% (ep15).
**Notes:** Three variables changed simultaneously — not a clean topology test. Val peaked ep9, then declined. HYPOTHESIS: temporal_decay=0.8 may additionally hurt by dampening activations over iterations.
**Finding: HYPOTHESIS — layered underperforms. Confounded by iterations and temporal_decay changes. See run9 for cleaner test.**
