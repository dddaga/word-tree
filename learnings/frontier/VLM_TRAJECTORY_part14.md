# VLM trajectory — part 14 (the clean compile, and the instrument that ran out of resolution)

Continues `VLM_TRAJECTORY_part13.md` (closed at 195 lines, as its own footer required). Same line:
SmolVLM-256M, Imagenette 10-class, d12@512 anchor 0.711, teacher 0.7040.

## 47. vlm_step030 — clean compile (`torch._dynamo.reset()` per cell). DONE. **WIN — the fix works and the 3.27× prefill claim survives. But the ship cell did not reproduce.**

The one experiment §46 demanded: `--tag_suffix _clean`, every other knob identical to step028, so the
only variable is the dynamo cache reset between cells. Card verified EMPTY at launch and still empty at
finish (48 MiB / 0%, no compute apps).

**GATE — PASS, and this run has the control the last two arms lacked.** fp32-eager teacher top1
**0.7040** and, more importantly, its *timings* reproduce step028's uncontended baseline to
**0.3% / 0.8%** (tower 25.65 vs 25.72, prefill 13.66 vs 13.77). §44 and §45 had no such anchor and had
their latency blocks quarantined; this one is anchored.

### Result — `vlm_step028_runtime_..._n500_clean__5060ti_cuda.json` (written 17:02)

| cell | top1 | tower ms | prefill ms | e2e ms |
|---|---|---|---|---|
| fp32 eager teacher | 0.7040 | 25.65 | 13.66 | 39.31 |
| fp32 eager student | 0.6840 | 9.51 | 14.06 | 23.57 |
| fp32 compile teacher | 0.7040 | 24.94 | 8.76 | 33.70 |
| fp32 compile student | 0.6840 | 9.52 | 8.24 | 17.76 |
| bf16 eager teacher | 0.7040 | 8.12 | 10.99 | 19.11 |
| bf16 eager student | 0.6880 | 3.29 | 11.53 | 14.81 |
| bf16 compile teacher | **0.7040** | 8.99 | **4.18** | 13.17 |
| **bf16 compile student** | 0.6840 | 4.02 | 5.39 | **9.41** |

Best accuracy-clean prefill **4.18 ms vs 13.66 = 3.27×**, under the 5.0 ms bar. §46 pre-registered
*"prefill improves or holds at 4.21 ms"* → **HOLDS. WIN reproduced.**

### What the fix bought — two CONFIRMED results

1. **The leaky compile is gone.** `grep -c recompile_limit` on the log goes **1 → 0**. §46's caveat 1
   diagnosed a harness defect — dynamo's cache keyed on a shared code object and accumulated across all
   four compile cells in one process — and `torch._dynamo.reset()` removes it exactly as predicted.
2. **§46's caveat-1 diagnosis is CONFIRMED via ACCURACY, not latency — and that is the stronger test.**
   The leak analysis named the *third* compile cell, `bf16_compile_teacher`, as the single degraded one.
   That cell and only that cell moves: top1 **0.6980 → 0.7040**, precisely the anchor it reproduces in
   all six of its other measurements. A latency argument would have been arguable; a named cell moving
   to a known value is not. **bf16 + full compile is accuracy-neutral on this model. CONFIRMED.**

### What it exposed — the ship cell got *worse*, and the instrument is why

`bf16_compile_student` e2e **7.95 → 9.41 ms (+18%)**, prefill 4.28 → 5.39, tower 3.67 → 4.02 — every
component worse, on a clean card, under a fix that helped every other cell. That combination does not
have an architectural explanation, so the run was interrogated with an internal control instead.

**The control: the language model is byte-identical between the teacher and student cells.** The width
and depth cuts touch the vision tower only and never enter prefill, so within a dtype × mode pair the
teacher/student prefill difference is *pure instrument error*. Across the four pairs it runs
**+0.40 / −0.52 / +0.54 / +1.21 ms** — a repeatability floor of roughly **±0.5 ms absolute**. That is
~4% of a 13 ms fp32-eager prefill and **~25% of a 4–5 ms compiled one.**

**HYPOTHESIS (explicitly not confirmed): sub-5 ms prefill is at or below this harness's resolution, and
the 7.95-vs-9.41 spread is instrument, not regression.** Falsifier is cheap and pre-registered: repeat
each cell N times in one process and report median + IQR → step032.

**This is §45's pathology on the other axis.** There the 2.0pp accuracy bar was finer than the eval's
own 2.05pp binomial SE; here the 5.0 ms latency bar sits about one repeatability-width from the
4.18–5.39 ms it is judging. **The instrument, not the architecture, is now the binding constraint on
both axes of this line.**

Also found, and worth recording separately: the fp32 and bf16-**eager** cells reproduce
**bit-identically** across the two runs (McNemar 0/0), while **both** bf16-compile cells moved. So
compile + bf16 is **not bit-deterministic across compile sessions** — the ship cell flips 3 images, one
each way, p=1.00, far inside the eval SE, but it is not zero and should not be assumed zero.

### A second, accidental replicate — and what it measures

At 17:03 a duplicate `--tag_suffix _clean` run was launched by a second session before the 17:02 result
was visible to it. Same tag → same output filename, so it overwrote the canonical JSON; the 17:02 file
had already been preserved as `...json.step030_1702` and has been **restored to the canonical name**,
with the duplicate kept as `..._clean_DUP1703_contended__5060ti_cuda.json`. **Quote the 17:02 numbers.**

The duplicate is contaminated — step031's eval landed on the card at 17:04, one minute into it — but
that makes it a free measurement of what co-residency costs, on cells whose clean values are known:

| cell | clean tower | dup tower | clean prefill | dup prefill | clean e2e | dup e2e |
|---|---|---|---|---|---|---|
| fp32 eager teacher | 25.65 | 31.22 (**+21.7%**) | 13.66 | 14.54 (+6.4%) | 39.31 | 45.76 (+16.4%) |
| bf16 compile student | 4.02 | 6.00 (**+49.3%**) | 5.39 | 5.18 (−3.9%) | 9.41 | 11.18 (+18.8%) |

**The damage lands on the tower, not on prefill** — +22% to +49% on the GPU-bound stage against ±6% on
the dispatch-bound one, and the smaller the tower the larger the relative hit. That is consistent with
§44/§46's picture (prefill is launch- and Python-dispatch-bound, i.e. CPU-side; the tower is the part
actually competing for SMs) and it means a co-resident job inflates precisely the number this line has
spent five arms shrinking. The duplicate independently reproduced the two CONFIRMED results above —
`recompile_limit` count 0 and `bf16_compile_teacher` at 0.7040 — so those do not rest on one run.

**Process note:** nothing was killed and no teammate or concurrent-session job was displaced. The cost
was one wasted 13-minute run and one overwritten file, both recovered.

### Ship config — restated, and unmoved

**d6 × r=0.25, bf16 + compile.** 21.28M tower params; **42.6 MB of bf16 weights = 8.00× fewer bytes
than the fp32 teacher's 340 MB** — counted, not timed, and therefore completely unaffected by
everything above; top1 0.6840–0.6860 = −1.80 to −2.00pp — **STALE, see §48: those are n=500 values and
the gap they imply is retracted. At n=2000 the config reads 0.6785 = −3.60pp, CONFIRMED, not parity.** **End-to-end 7.95–9.41 ms = 4.2–5.0×: quote
the RANGE, not 7.95 alone, until step032 lands.** step030 in fact strengthens the config, because the
compile path is now accuracy-clean outright rather than accuracy-clean-with-a-caveat.

### Open, in priority order

1. **step032 — harness hardening. Top priority; every latency claim downstream depends on it.** Three
   cheap changes: (a) **N repeats per cell**, median + IQR, so a point estimate is never quoted again;
   (b) **log CPU load and per-cell wall-clock** alongside the GPU check — an empty card is demonstrably
   not a sufficient validity gate for a dispatch-bound stage; (c) **promote the teacher/student prefill
   delta to an automatic gate** — it is free, zero by construction, and would have flagged both the
   duplicate and the ship-cell spread before any number was written down. Pre-registered: IQR ≤5% →
   the 7.95–9.41 range collapses to a point and the WIN is restored or retired on evidence; IQR >5% →
   absolute latency on this box is not publishable and the paper quotes ratios with intervals.
2. **step031 — n_eval=2000.** DONE, see §48. All three width comparisons significant; the parity
   claim was Type-II and is retracted. An accuracy arm, so none of the above touched it.
3. **§32 prologue ablation.** Demoted by §46, demoted further here: it is a fraction of a stage whose
   measurement error is ±25% at the scale that now matters.
4. **CUDA graphs / `torch.export`** — the honest successor to compile and the only remaining prefill
   lever. Blocked behind step032, for the same reason as item 3.

---

## §48 — step031: at n=2000 every width "parity" null dies. The parity claim was Type-II.

**Guard first.** Teacher re-scored on the 4× sample: **0.7145** vs 0.7040 at n=500 = +1.05pp, inside
the pre-registered ±2.0pp. PASS. Both passes were run separately and their teacher hit-vectors are
**bit-identical**, so the two students are paired against the same 2000 teacher decisions.

| arm | tower | n=500 top1 | **n=2000 top1** | gap vs teacher (n=2000) |
|---|---|---|---|---|
| teacher d12 | 85.05M | 0.7040 | **0.7145** | — |
| d6 × r=0.25 | 21.28M (0.2502×) | 0.6840 | **0.6785** | **−3.60pp** |
| d6 × r=0.125 | 17.74M (0.2086×) | 0.6680 | **0.6420** | **−7.25pp** |

**Paired McNemar, all three, all significant** (SE at n=2000 ≈ 1.04pp):

| comparison | discordant | b/c | p | at n=500 |
|---|---|---|---|---|
| teacher vs r=0.25 | 592 | 332/260 | **0.0035** | not tested — assumed parity |
| teacher vs r=0.125 | 607 | 376/231 | **<1e-5** | 0.148 NULL |
| r=0.25 vs r=0.125 | 381 | 227/154 | **0.00022** | 0.451 NULL |

**Pre-registered outcome: SIGNIFICANT.** So "a real capacity cost exists between r=0.25 and
r=0.125", and §45's CONFIRMED parity narrows — but it narrows further than the pre-registration
anticipated, because the *teacher vs r=0.25* test also fires. **There is no parity arm left.** The
width axis carries a real, monotone capacity cost at both steps, and the d6 × r=0.25 ship config sits
a measured **−3.60pp** below its teacher rather than at parity.

**What is retracted.** §45's parity finding was a **Type-II error**, not a finding. It was built from
McNemar nulls at n=500, where SE (2.05pp) was wider than the 2.0pp bar being tested — a test that
could not reject was read as evidence of no difference. This is the same failure the step031
pre-registration named in the abstract; the arm confirms it applies to its own line's headline claim.
Absence of significance was never absence of effect, and three arms sitting "inside one SE" was a
statement about the instrument.

**Point estimates moved too, by more than one SE:** r=0.125 went 0.6680 → 0.6420 (−2.60pp). The
n=500 student numbers were themselves sample-dependent, so no *value* from that eval should be quoted
either — only the direction survived.

**And the mechanism is structural, which bounds how the two columns may be compared.** `sample_images`
sets `per = max(1, n // len(wnids))`, so n=500 draws 50/class and n=2000 draws 200/class. `rng.sample`
is called with a different `k` at the very first class, the RNG state diverges from there, and **the
n=2000 set is a different balanced draw, not a superset of the n=500 one.** So every n=500 → n=2000
movement in the table above confounds resolution with sample identity and must NOT be reported as "the
gap grew" — that phrasing implies a controlled before/after that was never run. What is *not*
confounded is everything inside the n=2000 column: all three McNemar tests are paired image-by-image
on the same 2000 images against a bit-identical teacher. **Correct reading: n=2000 supersedes n=500 as
the estimate (4× data, SE 2.05 → 1.01pp). It is not a measurement of how the gap changed.**

**What survives untouched.** The bytes argument. 21.28M params / **42.6 MB bf16 = 8.00× fewer bytes**
than the fp32 teacher's 340 MB is counted, not measured, and no eval-power question touches it. The
decision not to ship r=0.125 also survives and is now *better* supported: it was declined on bytes
(7.1 MB, 16.7%) when its accuracy cost was merely unproven; that cost is now CONFIRMED at −3.65pp
against r=0.25.

**Ship config restated, honestly:** d6 × r=0.25 bf16+compile, 8.00× fewer bytes, **top-1 0.6785 =
−3.60pp below the d12 teacher (CONFIRMED, n=2000, p=0.0035)**. Not parity. Any paper text claiming
parity for this line is wrong and must be rewritten.

**Method.** Both axes of this line have now been caught with a decision bar finer than the
instrument's own resolution — §45/§48 on accuracy (2.0pp bar, 2.05pp SE) and §47 on latency (5.0 ms
bar, ±0.5 ms floor at 4–5 ms). **Pre-register the instrument's resolution alongside the bar, and
refuse to run any arm whose bar is inside it.** step032 is the latency half of that same discipline.

---

*Part 14 closed at §48 on the 200-line limit. §49 onward: `learnings/frontier/VLM_TRAJECTORY_part15.md`.*
