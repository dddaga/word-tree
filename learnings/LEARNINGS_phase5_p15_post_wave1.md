# Phase 5 Part 15: Post-Wave-1 Results — INDEX

**Date range:** 2026-04-06 to 2026-04-09
**Status:** COMPLETE (split into sub-files)

Table of contents. All content split into sub-files:

---

## Sub-Files

| File | Contents | Key Steps |
|------|----------|-----------|
| [LEARNINGS_phase5_p15a_wave1_closure.md](LEARNINGS_phase5_p15a_wave1_closure.md) | Steps 60-68 results (wave-1 killed experiments) | step60, step63, step65, step64, step67, step68, step48, step56 |
| [LEARNINGS_phase5_p15b_arch_patch.md](LEARNINGS_phase5_p15b_arch_patch.md) | Step69 patch results, step70 full-scale, buggy-arch backlog | step66, step69, step70, step47, step53, step46, step37, step32 |
| [LEARNINGS_phase5_p15c_kiter_flops.md](LEARNINGS_phase5_p15c_kiter_flops.md) | Step71 K_iter sweep, steps 73-83 routing experiments | step71, step73, step75, step76, step77, step79, step80, step81, step82, step83 |
| [LEARNINGS_phase5_p15d_flops_track.md](LEARNINGS_phase5_p15d_flops_track.md) | Step86, step88 FLOPs/Pareto results; G1/G2 pending | step86, step88, G1, G2 |
| [LEARNINGS_phase5_p15e_dynamic_routing_postmortem.md](LEARNINGS_phase5_p15e_dynamic_routing_postmortem.md) | Dynamic routing post-mortem, gate-death analysis, next directions | steps 58-83 analysis, step87 design, G4 design |

---

## Quick Reference

**Key results:**
- REF_BASELINE_v1 (pre-patch): 73.53% (step57)
- REF_BASELINE_v2 (patched arch, 50%/75ep): 83.36% (step69)
- Gen4+ candidate (N=1024, 50%/75ep): 85.04% (step69 Config A, turing=0.3)
- Project best (N=4096, 100%/150ep): **97.38%** (step70 Config B, turing=0.0)
- K_iter optimal at N=4096: 12 (step71 Config C = 96.66%)
- K_hh default: 4 (step86 Config A = 96.59%, −18% FLOPs)
- alpha optimal: 1.0 (step88 confirmed)

**Wave-1 verdict:** All 8 multiplicative gate mechanisms killed by gate-death theorem.

**Post-wave-1 wins:**
- step73 D: softmax redistribution routing +1.78pp at N=1024
- step75 D: temperature routing +3.98pp at N=1024
- step76 A: alpha_turing=0.0 + W_phase trained +2.88pp at N=1024
- step81 A: Hebbian topology rewiring +1.12pp at N=1024
- step82 A: random group topology (n_groups=8) +3.01pp at N=1024

**Next critical experiments:**
- G1: K_iter=12 + K_hh=4 + turing=0.0, N=4096, 150ep (pending)
- G2: Group topology N=4096, K_hh=4 (pending)
- step86 Mac Mini: resume configs F/G/H/I
- step87: Pure proximity architecture (scripted: train_step87_proximity_routing.py)