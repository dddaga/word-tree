# run9 — Notes

**Config delta from run6:** +topology="layered". iterations=7 (run6 had 5 — small confound), beam_width=0. 38/40 epochs completed.
**Result:** val best=25.45% (ep29), final=25.12% (ep37).
**Notes:** Closest controlled comparison to run6. 60pp gap (85.81% vs 25.45%). Erratic, non-monotonic curve — very different from run6's steady climb. Val peaked ep29 then declined.
Post-mortem: `diagnose.ipynb` created to visualize edge flow matrices, in-degree distributions, per-iteration activation strength by node group. `_iter_stats_hook` added to `native/layer.py` for diagnostics.
**Finding: HYPOTHESIS — layered topology severely underperforms flat. Likely due to mandatory 2-hop path (no direct input→output edges), starving output nodes early in training. See `concepts/topology.md`.**
