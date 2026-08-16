# VLM trajectory — part 13 (width past the retracted knee, and the noise floor)

Continues `VLM_TRAJECTORY_part12.md`. Same line: SmolVLM-256M, Imagenette 10-class, d12@512
anchor 0.711, teacher 0.7040.

## 45. vlm_step029 — d6 × r=0.125 at 50 epochs. DONE. **PARTIAL by the bar — but the bar is below the noise floor.**

Launched as `vlm_step028`, which collided with the runtime grid already holding that number; **renumbered
step029** here. The tmux session, log and results filenames keep the `step028` prefix — they are the launch
artifact and renaming mid-run breaks the writer. §46 below is the true step028.

`--student_depth 6 --ratios 0.125 --epochs 50 --n_train 9352`. **Only the width ratio changed** vs
§44 (0.25 → 0.125). Strictly one-variable. Launched because §44 retracted §43's knee and showed the
real cost of narrowing was epochs, not accuracy — so the axis was owed one more probe at a budget
that could not repeat §43's mistake.

**GATE — PASS.** Teacher d12 top1 **0.7040**, the sixth bit-identical reproduction. Naive (sliced,
untrained) **0.1000** — chance, as every naive structural cut to this tower has been.

**Param arithmetic CONFIRMED as predicted before the run.** d6's non-MLP is a fixed 14.2M; the MLP
at r=0.25 is 7.08M, at r=0.125 is 3.54M. Measured 17.74M = **0.209×**, against the 17.7M/0.208×
predicted. **Halving the MLP again bought only 0.2502× → 0.209×** — the non-MLP floor now dominates,
and width's param return is flattening. That is a finding independent of the accuracy result.

### Result — `vlm_step023_mlp_width_r0.125_e50_t9352_n500_d6__5060ti_cuda.json`

| arm | ratio | inter | final rel_mse | distilled | vs teacher | params | verdict |
|---|---|---|---|---|---|---|---|
| §44 d6 × r=0.25 | 0.25 | 768 | 0.1776 | 0.6840 | −2.00pp | 21.28M (0.2502×) | WIN |
| **§45 d6 × r=0.125** | **0.125** | **384** | **0.2041** | **0.6680** | **−3.60pp** | **17.74M (0.209×)** | **PARTIAL** |

Gain over naive **+56.80pp**, the third-largest in this line.

### The finding: the verdict bar is finer than the instrument

The script's WIN bar is teacher − 2.0pp. At n_eval=500 and p≈0.70 the binomial SE is **≈2.05pp**.
**The bar separating WIN from PARTIAL is smaller than one standard error of the measurement it is
applied to.** §45 landing PARTIAL and §44 landing WIN is therefore not evidence that the two
architectures differ — it is the bar resolving a difference the eval cannot see.

Both paired tests confirm this directly. Against the teacher:

| | student right | student wrong |
|---|---|---|
| **teacher right** | 274 | 78 |
| **teacher wrong** | 60 | 88 |

χ²(cc)=2.094, **p=0.148** (exact binomial two-sided 0.148). Not significant at α=0.05, though
materially weaker than §44's p=0.444 — the drift is in the expected direction even if it does not
clear significance.

And head-to-head against §44's student on the same 500 images, which is the sharper test because it
removes the teacher entirely:

| | r=0.125 right | r=0.125 wrong |
|---|---|---|
| **r=0.25 right** | 295 | 47 |
| **r=0.25 wrong** | 39 | 119 |

χ²(cc)=0.570, **p=0.451**. **The −1.60pp between r=0.25 and r=0.125 is noise.** Halving the MLP
intermediate a second time produced no measurable accuracy cost on this eval.

**CONFIRMED: width is not the binding axis down to 0.209× tower params at d6, at 50 epochs, to the
resolution this eval provides.** The last clause is load-bearing and is the honest limit of the
claim — see below.

### What the fit says, which the accuracy cannot

rel_mse tells a different and more sensitive story than top-1. §45 finished at **0.2041** vs §44's
**0.1776** at the identical budget. That is a real, non-noisy gap in the distillation objective:
§45's ep50 fit sits between §44's ep25 (0.2234) and its ep50 (0.1776), i.e. roughly where §44 was
around ep30.

Combined with §44's own observation that r=0.25 at ep40 matched r=0.5 at ep25, the pattern is
consistent: **each halving of the MLP intermediate shifts the fit ladder right by roughly 10–15
epochs.** HYPOTHESIS — two points on a two-point trend, and the shift is estimated by eye off the
epoch traces, not fitted.

**§45 is non-converged, exactly as §44 was.** rel_mse fell 0.2053 → 0.2041 over the last epoch,
~0.0012/ep and decaying, with no plateau. **0.6680 is a FLOOR on this architecture, not its
ceiling**, and the pre-registration's own logic applies: a PARTIAL at a non-converged budget is not
a knee. Per the step028 row as written before the run, calling this a capacity knee would require a
100ep ladder, and that ladder has not been run.

### Latency from this run is INVALID — do not quote it. Again.

The log reports `tower 26.20 -> 27.36 ms (0.96x)`, i.e. a strictly smaller tower measured as slower.
The pre-registered guard caught it on the first check: **teacher prefill 13.73 ms, student prefill
26.09 ms** — a component this cut does not touch, nearly doubled. A teammate job (13–15 GB, GPU at
100%) landed on the 5060ti during the eval phase. Nothing was killed or displaced.

This is the **second consecutive arm with no valid timing**, and the two cases differ in a way that
matters. §44 could substitute §43's uncontended numbers because §43 measured the *same architecture*
at the same params. **There is no prior uncontended measurement of r=0.125 at all**, so its speedup
is not merely re-used from elsewhere — it is **UNMEASURED**. The Pareto row for this arm has a hole
in the wall-time column, and no number should be invented to fill it.

Process defect, stated plainly: the guard worked, but a guard that only detects contamination after
12 hours of training is a detector, not a fix. Two of the last two arms have been hit.

### What this changes

**The drone operating point does not move.** §44's d6 × r=0.25 stays the recommendation: it has a
valid measured speedup (2.75× tower / 1.70× end-to-end from §43), the better fit, and the stronger
parity evidence (p=0.444 vs 0.148). §45 buys 0.2502× → 0.209× params for no measurable accuracy loss
but with an unmeasured latency column — a real candidate, not yet a substitute.

| want | pick | params | tower | end-to-end | vs teacher |
|---|---|---|---|---|---|
| cheapest to train | §42 d6 × r=0.5, 25ep | 0.333× | 2.40× | 1.61× | −1.20pp |
| **best deployed** | **§44 d6 × r=0.25, 50ep** | **0.250×** | **2.75×** | **1.70×** | **−2.00pp, p=0.44** |
| smallest, unmeasured | §45 d6 × r=0.125, 50ep | 0.209× | **UNMEASURED** | **UNMEASURED** | −3.60pp, p=0.15 |

**The eval has become the bottleneck, not the architecture.** Three arms now sit inside one SE of
each other (0.7040 / 0.6840 / 0.6680) and every pairwise test between them is null. Further width
probes at n_eval=500 cannot produce an interpretable answer — they will return "not significant"
regardless of what is true. Either the eval grows or the width axis stops here.

### Open, in priority order

1. **Widen the eval before any further width arm.** n=500 gives SE≈2.05pp; n=2000 halves it to
   ~1.02pp, which is the minimum needed to resolve the 2.0pp bar the script already uses. This is
   now the top item because it gates the interpretability of everything downstream, and it is cheap
   — eval-only, no retraining.
2. **§32 prologue ablation — DEMOTED by §46, not promoted.** The case for it was that prefill is
   13.73 ms of a ~40 ms end-to-end; §46 falsified that premise on this very page. Under the ship
   runtime prefill is **4.21 ms of 7.95 ms**, so the prologue can no longer be a large win in
   absolute terms. Superseded by **step030** (clean-compile re-timing) and then CUDA graphs/export,
   which attack the same 4.21 ms with a cheaper and already-diagnosed lever.
3. **A clean uncontended re-timing of §45** — the only way to fill the hole in the table above, and
   it costs one eval pass on a free card, no retraining. Pair it with a re-timing of §44.
4. **Learned token selection** — **DEAD as a latency lever**, killed by §46 and by step022 (a 16×
   token cut bought 1.4% of prefill). May still run as an accuracy/memory study, never on latency.
5. **A 100ep ladder at r=0.125**, only if item 1 lands first. Without a wider eval it cannot
   distinguish its own outcome from noise.
6. **d9 depth retry** — unblocked, unstarted, lowest value.

## 46. vlm_step028 — bf16 × torch.compile, no training. DONE. **WIN. The prefill wall is RUNTIME, not architecture.**

`scripts/frontier/vlm_step028_runtime.py`, full 2×2×2 grid (dtype × compile × tower), inference only,
reusing §44's r=0.25 checkpoint. Card **VERIFIED EMPTY** at launch (48 MiB / 0% util), so unlike §44 and
§45 **these timings are valid**. Pre-registered on prefill: WIN iff best accuracy-clean prefill < 5.0 ms.

**GATE — PASS.** fp32-eager teacher 0.7040, tower 25.72 ms, prefill 13.77 ms — matches uncontended
history. Retro-validates §44's quarantine: the fp32-eager student tower re-measures **9.52 ms** vs §43's
9.53 ms on the same architecture.

| cell | top1 | tower ms | prefill ms | e2e ms |
|---|---|---|---|---|
| fp32 eager teacher | 0.7040 | 25.72 | 13.77 | 39.49 |
| fp32 eager student | 0.6840 | 9.52 | 13.92 | 23.45 |
| fp32 compile teacher | 0.7040 | 24.92 | 9.45 | 34.37 |
| fp32 compile student | 0.6840 | 9.56 | 8.27 | 17.82 |
| bf16 eager teacher | 0.7040 | 8.13 | 11.00 | 19.13 |
| bf16 eager student | 0.6880 | 3.29 | 11.02 | 14.31 |
| bf16 compile teacher | 0.6980 | 8.99 | 4.21 | 13.20 |
| **bf16 compile student** | **0.6860** | **3.67** | **4.29** | **7.95** |

**VERDICT WIN — best accuracy-clean prefill 4.21 ms vs 13.77 = 3.27×, under the 5.0 ms bar.** §44's
dispatch-bound diagnosis is CONFIRMED, not falsified: a stage that is 10.5% of fp32 roofline gave up 3.3×
to *compilation alone*, which removes launch and Python dispatch and touches no arithmetic.

**The two levers hit different stages, and they are near-orthogonal.** Compile moves prefill (−31% in
fp32) and not the tower (−3%); bf16 moves the tower (3.16×) and only −20% of prefill. Neither alone broke
8.27 ms. Tower 25.72 → 3.28 ms = 7.84×, decomposing as arch 2.70× × dtype 2.90× = 7.83 — the dtype gain
shrinks 3.16× → 2.90× as the tower gets smaller and relatively more overhead-bound, which is the same
dispatch story appearing on the other stage.

**End-to-end: 39.49 → 7.95 ms = 4.97×**, of which **2.95× is PURE RUNTIME** (23.45 → 7.95 on the *same*
student) and 1.80× is architecture. §44 capped e2e at ~2.8× with prefill pinned near 14 ms; that cap is
lifted, and the runtime lever turns out to be larger than every architectural cut in this line combined.

**Numerics: the drift is real but negligible, and paired testing says so.** Only the compile cells drift
at all (teacher 0.7040 → 0.6980, student 0.6840 → 0.6860 — opposite signs). Paired against their own
fp32-eager counterparts, every cell differs on **at most 8 of 500 images**: bf16-compile teacher 3/0
(p=0.25), bf16-compile student 1/2 (p=1.0), bf16-eager student 3/5 (p=0.73), bf16-eager teacher 4/4
(p=1.0). All null, all inside the 2.0pp guard. No cell is excluded.

**Two caveats, both load-bearing.**
1. **The compile is LEAKY.** dynamo hit `config.recompile_limit (8)` on transformers'
   `output_capturing.wrapper` and fell back on part of the graph. The warning fires **once**, immediately
   before `bf16_compile_teacher`, so the fp32-compile cells are fully compiled and only the bf16-compile
   pair is degraded. **4.21 ms is therefore a FLOOR on a partially-compiled model.** Root cause is a
   HARNESS defect: dynamo's cache is keyed on a shared code object and accumulated across all four compile
   cells in one process. Each cell builds a fresh model, so carrying a compile cache between them was
   never correct. Fix = `torch._dynamo.reset()` per cell → step030.
2. **Compile makes the bf16 tower SLOWER** (8.13 → 8.99, +11%). Guard overhead on a stage with no dispatch
   cost left to recover. Compile is a prefill lever only.

**Ship config: student, bf16 + compile — 7.95 ms e2e, 21.28M tower params, 42.6 MB of bf16 weights
(8.00× fewer bytes than the fp32 teacher), top1 0.6860 = −1.80pp.** This is the drone operating point.

**Tower latency work is FINISHED.** The tower is now 3.67 ms against 4.29 ms of prefill; further narrowing
is a params/bytes argument only, never again a wall-time one. That reframes §45: r=0.125's UNMEASURED
latency column matters much less than it looked, because there is ~3.7 ms of tower left to win.
