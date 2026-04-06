# Phase 5 Part 15: Post-Wave-1 Results (Steps 60, 63-67)

**Date:** 2026-04-06
**Status:** IN PROGRESS — step64/step67/step68 running; step60 active; step66 queued (waiting on step60)
**REF_BASELINE:** 73.53% (AntiHebb α=1.0, 50% data, 75ep, step57)

---

## Summary: Wave-1 closure + post-wave-1 experiments

Wave-1 (steps 58-63) completely killed. Post-wave-1 experiments continuing to test orthogonal
hypotheses: stacked architecture depth (step64), local plasticity routing (step66),
safety valve redundancy (step67), and distance-phase routing (step65).

---

## Step 60: Phase-Distance Routing (Complete)

**Script:** train_step60_phase_routing_magnitude.py
**Date:** 2026-04-06
**Verdict:** KILLED — all 12 non-Ref configs gate-dead

| Config | Description | top1_best | vs Ref |
|--------|-------------|-----------|--------|
| Ref | AntiHebb α=1.0 static | 73.55% | — |
| A | independent, no AH, freq=capped_exp | 14.27% | −59pp |
| B | coherent dynamic-ref, no AH | 14.27% | −59pp |
| C | independent, AH=1.0, freq=capped_exp | 18.73% | −55pp |
| D | coherent, no AH | 13.30% | −60pp |
| A_wt | independent+mag-weighted, no AH | 13.20% | −60pp |
| B_wt | coherent+mag-weighted, no AH | 14.37% | −59pp |
| B_anchor | coherent AH=1.0, freq=capped_exp | 14.14% | −59pp |
| B_abs | coherent absolute, no AH | 14.22% | −59pp |
| E | (gate-dead) | 12.05% | −61pp |
| F | (gate-dead) | 11.92% | −62pp |
| A_prime | (gate-dead) | 10.80% | −63pp |
| B_prime | (gate-dead) | 10.68% | −63pp |

**Key insight:** Even configs WITH AH (C, B_anchor) still gate-die at 14-19%. The phase-distance
routing modification is incompatible with stable AH routing. All 12 variants of magnitude
weighting, coherent/independent, frequency encoding — all killed. Comprehensive closure.

---

## Step 63: Activation-Gated Routing / AGR (Complete)

**Script:** train_step63_act_gated_routing.py
**Date:** 2026-04-05 (completed before step60)
**Verdict:** KILLED — all non-Ref configs killed

| Config | Description | top1_best | vs Ref |
|--------|-------------|-----------|--------|
| Ref | AntiHebb α=1.0 static | 73.48% | — |
| A | soft-attn mixed candidates, no AH | 31.11% | −42pp |
| B | soft-attn mixed candidates, AH=1.0 | 55.41% | −18pp |
| C | hard top-K mixed candidates, AH=1.0 | 38.88% | −35pp |
| D | soft-attn mixed, AH=1.0, hop_decay=0.9 | 55.29% | −18pp |
| E | soft-attn phase-only candidates, AH=1.0 | 55.13% | −18pp |

AH partially compensates (B vs A: +24pp) but −18pp vs static AH. Routing redistribution
is inherently destabilizing regardless of soft-attn formulation.

---

## Step 65: Distance-Phase Routing (Complete)

**Script:** train_step65_dist_phase_routing.py
**Date:** 2026-04-05 (completed on Mac Studio CPU)
**Verdict:** KILLED — all configs ~48-49%, no AH Ref at 35.06%

| Config | Description | top1_best | vs step57 REF |
|--------|-------------|-----------|---------------|
| Ref | SmallWorld bare (no AH) | 35.06% | −38pp |
| A | γ=0.5 raw | 48.79% | −25pp |
| B | γ=1.0 raw | 48.79% | −25pp |
| C | γ=2.0 raw | 48.43% | −25pp |
| D | γ=0.5 softmax | 48.43% | −25pp |
| E | γ=1.0 softmax | 48.43% | −25pp |
| F | γ=2.0 softmax | 48.43% | −25pp |

Note: Ref in script = bare SmallWorld (no AH). Configs A-F add distance-phase weighting
exp(-γ·d_norm) on edges. All land at ~48-49% regardless of γ or softmax/raw. The
distance-phase geometry adds no routing signal that AH doesn't already provide better.

---

## Step 48: K_iter Scaling at D=64 — SLOT-KILLED (Partial)

**Script:** train_step48_kiter_sweep_d64.py (ran at 150ep full data, AH=0.5 base)
**Date:** ~2026-04-04 (partial run, killed by slot pressure)
**Status:** PARTIAL — only Ref and A completed; B-G never ran

| Config | Description | top1_best |
|--------|-------------|-----------|
| Ref | K_iter=8, no AH | 58.24% |
| A | K_iter=12, no AH | 55.08% |
| B-G | (never ran — slot killed) | — |

**Finding:** K_iter=12 WITHOUT AH hurts (−3pp). Over-smoothing confirmed without inhibition.
**Gap:** Key question unanswered — does K_iter>8 WITH AH=1.0 help? B-G (AH configs) never ran.
**Action:** Rewrote as step68 with Gen4 base (AH=1.0, 50%/75ep).

---

## Step 54: LR Schedule Sweep — COMPLETE (Performance Killed)

**Script:** train_step54_warm_restart_lr.py (ran at 150ep full data, AH=0.5 base)
**Date:** ~2026-04-04 (complete run)
**Status:** COMPLETE — all configs below Ref, plateau wins

| Config | Description | top1_best |
|--------|-------------|-----------|
| Ref | Plateau scheduler, AH=0.5 | 70.78% |
| A | CosineWarmRestart T_0=10 | 66.98% |
| B | CosineWarmRestart T_0=25 | 68.05% |
| C | CosineWarmRestart T_0=50 | 67.21% |

**Finding:** Cosine warm-restarts hurt (−2 to −4pp vs plateau). Plateau scheduler is optimal.
This question is closed — no need to retest at Gen4 since step56 N=4096=84.36% uses plateau.

---

## Step 64: Stacked SGNNET — IN PROGRESS

**Script:** train_step64_stacked_sgnnet.py (Mac Mini CPU, 50%/75ep)
**Status as of 2026-04-06:** Ref DONE 73.38%, Config A DONE 71.62%, Config B (2-layer+skip) running ~e20

| Config | Description | top1_best | vs step64 Ref (73.38%) | vs REF_BASELINE (73.53%) |
|--------|-------------|-----------|------------------------|--------------------------|
| Ref | 1-layer baseline AH=1.0 | 73.38% | — | −0.15pp |
| A | 2-layer series, passthrough, no skip | 71.62% | −1.76pp | −1.91pp |
| B | 2-layer series, passthrough + skip | running ~e20 | ? | ? |
| C | 2-layer series, bridge (linear D→D) | queued | ? | ? |
| D | 3-layer series, passthrough | queued | ? | ? |
| E | 2-parallel sum fusion | queued | ? | ? |
| F | 2-parallel concat-project fusion | queued | ? | ? |

**Finding so far (2026-04-06):** Config A (series stacking, no skip) is −1.76pp vs 1-layer baseline.
Stacking without skip connections loses 1.76pp. Skip connection (Config B) is the critical test —
if B recovers to Ref or above, stacking with residual paths is viable. A being only −1.76pp
(vs 73.38%) is NOT catastrophic like wave-1 failures (those were −18 to −63pp), suggesting
stacked architecture preserves learned structure but adds optimization difficulty without skip.

---

## Step 67: Safety Valve λ Ablation — PARTIAL (Ref + A done, B starting)

**Script:** train_step67_safety_ablation.py (Mac Studio CPU, 50%/75ep)
**Status as of 2026-04-06:** Ref + Config A DONE (JSON saved), Config B (λ=0.05) now running

| Config | λ_safety | top1_best | vs Ref (73.68%) | delta vs REF_BASELINE (73.53%) |
|--------|----------|-----------|-----------------|--------------------------------|
| Ref | 0.489 (default scaled) | 73.68% | — | +0.15pp |
| A | 0.0 (safety OFF) | **70.78%** | −2.90pp | −2.75pp |
| B | 0.05 (minimal) | running | ? | ? |
| C | 0.1 (light) | queued | ? | ? |

**Finding so far:** Removing safety valve (λ=0) costs −2.90pp vs default. Safety valve IS doing
useful work — AH alone does NOT fully substitute for position-space repulsion. The 2.9pp gap
means O(N²) cdist overhead cannot be dropped. If B/C recover (λ=0.05-0.1 vs 0.489), there
may be room to reduce λ without full penalty, potentially cheaper computation.

**Hypothesis:** AH enforces W_pos diversity via cosine suppression, making position-space
repulsion redundant. If Config A matches Ref, we can drop the O(N²) cdist overhead
in all future experiments.

---

## Step 56: N-Scaling Final (N=10000) — COMPLETE

**Script:** train_step56_n_scaling.py (Mac Studio MPS, 150ep full data)
**Date:** 2026-04-06
**Verdict:** REGRESSES — N=10000 worse than N=4096; N-scaling law peaks at N=4096

| N | Params | top1_best | Notes |
|---|--------|-----------|-------|
| 512 | 66K | 69.58% | — |
| 1024 | 133K | 80.92% | — |
| 2048 | 265K | 81.10% | — |
| 4096 | 529K | **84.36%** | ← PROJECT BEST; best_ep=146/150 |
| 10000 | 1.29M | 82.37% | best_ep=145/150 — REGRESSION |

**Key finding:** N-scaling law is NOT monotonic above N=4096. At N=10000, accuracy drops 2pp.
This reversal likely reflects: (a) W_pos position space in [0,1]^D becomes too sparse for K_local
to form meaningful local neighborhoods at N=10000, OR (b) AH suppression is too aggressive
when N>>K_local, collapsing routing diversity. N=4096 remains the project best (84.36%).

**Action:** Do not pursue N>4096 without understanding the reversal mechanism first.
N-scaling track (step56) CLOSED.

---

## Backlog Audit (2026-04-06)

### Genuinely killed by performance (do NOT rerun):
- step54: plateau wins, cosine warm-restarts hurt
- step60: all 12 phase-routing variants gate-dead
- step63: AGR all killed −18pp+
- step65: distance-phase all killed −25pp
- step51: wave-1 verdict applies (routing gate modification)
- step52: gate-dead at ~45% (high-D subspace routing)
- step53: low-rank mixing killed 69-70%
- ARM 4 (steps 39, 45, 46, 47): all killed 45-54%
- step32: step29c showed AH alone wins, nothing to compound
- step38: step31 killed at ~44%, nothing to distill

### Killed by slot pressure (key configs never ran — must rerun):
- step48 configs B-G: AH compound K_iter configs never ran → **step68 rewrites this**

### Remaining backlog (genuine):
- step66: phase-target local plasticity (QUEUED, waiting for step60 slot)
- step68: K_iter scaling at Gen4 (LAUNCHED 2026-04-06, Mac Studio MPS)
- step57 Gen4: beam routing efficiency (needs scripting)

---

## Step 56: N-Scaling Law — COMPLETE (2026-04-06)

**Script:** train_step56_n_scaling.py (Mac Studio MPS, 100% data, 150ep)
**Status:** COMPLETE — all N configs finished including N=10000

| N | params | top1_best | best_ep | wall_min |
|---|--------|-----------|---------|----------|
| 512 | 66,688 | 69.58% | 134/150 | 31.2 |
| 1024 | 132,736 | 80.92% | 147/150 | 100.2 |
| 2048 | 264,832 | 81.10% | 137/150 | 222.6 |
| 4096 | 529,024 | **84.36%** | 146/150 | 238.1 |
| 10000 | 1,290,640 | 82.37% | 145/150 | 622.3 |

**Key finding:** N=10000 REGRESSES vs N=4096 (82.37% vs 84.36%, −1.99pp). N=4096 remains
the current best. The N-scaling law is NOT monotone: there is a peak at N=4096 and
performance drops with further N increase. Possible causes:
- Overparameterized routing (1.29M params, ~10x N=1024 winner)
- AH anti-Hebbian pressure saturates at large N — too many competing inhibitory signals
- Optimization landscape gets harder (more saddle points) without K_iter compensation

**Action:** step68 (K_iter sweep) is now the next lever — does K_iter>8 at N=4096 recover/extend?
N=10000 with K_iter>8 may be a follow-on if step68 shows K_iter gain.

**Note:** step56 session GONE from Mac Studio → step68 auto-launched (MPS slot freed).

---

## Step 68: K_iter Scaling at Gen4 — PARTIAL (Ref + A done, B+ running)

**Script:** train_step68_kiter_gen4.py (Mac Studio MPS, 50%/75ep, N=1024, D=64, AH=1.0)
**Status as of 2026-04-06:** Ref + Config A DONE (JSON saved), Config B (K_iter=12) running

| Config | K_iter | top1_best | vs Ref (73.63%) | vs REF_BASELINE (73.53%) |
|--------|--------|-----------|-----------------|--------------------------|
| Ref | 8 | 73.63% | — | +0.10pp |
| A | 10 | **73.58%** | −0.05pp | +0.05pp |
| B | 12 | running | ? | ? |
| C | 6 | queued | ? | ? |
| D | 4 | queued | ? | ? |

**Finding so far (2026-04-06):** K_iter=10 is essentially flat vs K_iter=8 (−0.05pp, noise).
More routing iterations do NOT help at N=1024. This is the 50%/75ep diagnostic run —
the full-scale test at N=4096/150ep still valid if B/C/D show signal. But first indication
is K_iter is not the bottleneck: AH routing at K_iter=8 is already near-optimal.

**Hypothesis check:** "K_iter bottleneck at larger N" — K_iter=10 at N=1024 gives no gain.
Must see B (K_iter=12) and potentially rerun at N=4096 to confirm/deny for large-scale.

---

## Step 67: Safety Valve λ Ablation — PARTIAL RESULT UPDATE (2026-04-06)

**Script:** train_step67_safety_ablation.py (Mac Studio CPU, 50%/75ep)
**Ref + A saved to JSON** (as of 2026-04-06 ~08:00 UTC). Config B (λ=0.05) now running.

| Config | λ_safety | top1_best | vs Ref (73.68%) |
|--------|----------|-----------|-----------------|
| Ref | 0.489 (default scaled) | 73.68% | — |
| A | 0.0 (safety OFF) | **70.78%** | −2.90pp |
| B | 0.05 (minimal) | running | ? |
| C | 0.1 (light) | queued | ? |

**Interim finding:** Safety valve is NOT redundant with AH. Dropping to λ=0 costs −2.9pp.
AH suppresses direction diversity but does NOT fully substitute for Euclidean repulsion.
Watch B/C to see if there's a reduced-λ sweet spot with smaller compute cost.
