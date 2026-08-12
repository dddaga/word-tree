# SGNNET Experiment Report — Addendum

**Source:** Overflow from EXPERIMENT_REPORT.md (lines 249-296)
**Generated:** 2026-04-08

---

## Status at Time of Report (2026-04-08)

### Running at Report Time

| Machine | Slot | Step | Script | Status | What it tests |
|---------|------|------|--------|--------|---------------|
| Mac Mini MPS | 1 | step82 | train_step82_group_topology.py | **COMPLETE** (results in) | Group-structured hidden topology; n_groups=8 wins (+3pp) |
| Mac Mini CPU | 1 | step83 | train_step83_group_routing.py | **KILLED** (completed) | Inter-group dynamic routing — killed, see step83 results |

Mac Studio results collected: step75 (temp routing), step79 (aux losses), step80 (N-scaling), step81 (Hebbian rewire) — all complete.

### step82 result: n_groups=8 is +3pp over default spatial topology
- Ref (spatial, n_groups=128): 82.62%
- A (random-group, n_groups=8): **85.63%** (+3.01pp)
- B (n_groups=16): 83.97%
- C (n_groups=32): 83.85%
- D (n_groups=8 + input-align): 69.40% (KILLED — input alignment hurts badly)

### step83 final result: Group routing KILLED
- Ref=84.61%, A(β=0.5 every-step)=78.60% (−6.01pp KILLED)
- B(β=0.5 final-only)=81.96% (−2.65pp)
- Root cause: S_g=mean(Z) too coarse; co-adaptation with W_pos

---

## Takeaways for Next Experiments

### What works (as of 2026-04-08)
1. **AntiHebbian alpha=1.0 alone** is the foundation. Every winning config uses it.
2. **N=4096** is the current sweet spot — all top results are at this scale.
3. **turing=0.0** is optimal at N=4096 — saves compute AND improves accuracy.
4. **K_iter=12** at N=4096 gives the best accuracy per compute among ablation runs.
5. **Redistribution routing** (softmax, temperature) is the only viable dynamic routing paradigm.
6. **Random-group topology** (n_groups=8) improves over spatial topology by +3pp at N=1024.
7. **K_hh=4** is the new default — free lunch (+0.56pp, −18% FLOPs) confirmed by step86.

### What to try next (updated as of 2026-04-09)
1. **G1: K_iter=12 + K_hh=4 + turing=0.0, N=4096, 150ep** — combine 3 confirmed wins. Most critical.
2. **G2: Group topology N=4096 with K_hh=4** — step82 showed +3pp at N=1024; untested at N=4096.
3. **step87: Pure proximity architecture** — scripted, ready to launch.
4. **step72: Full N-scaling curve on patched arch** — entire step56 curve invalid; script needed.
5. **G4: Redistribution routing at N=4096** — step75D was +3.98pp at N=1024; scale test needed.
6. **step85: Stacked parallel SGNNET on patched arch** — step64 Config F +1.52pp on buggy arch.

### What to avoid
- Any per-step multiplicative gate (gate-death theorem)
- Compounding mechanisms on top of AH (compound failure law)
- Phase-based per-neuron routing (all variants killed across 8+ steps)
- D > 64 (encoding collapses)
- N > 10000 without understanding the regression mechanism
