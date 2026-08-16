# VLM trajectory — part 15 (the noise was a tail, not a floor)

Continues `VLM_TRAJECTORY_part14.md` (closed at 176 lines, near the 200 cap as its footer required).
Same line: SmolVLM-256M, Imagenette 10-class, d12@512 anchor 0.711, teacher 0.7040.

## 49. vlm_step032 — R=5 repeats per cell. DONE. **The instrument is fine. The statistic I pre-registered was not.**

§47 inferred a **±0.5 ms symmetric repeatability floor** from four within-pair prefill deltas and used it
to explain the ship cell's 7.95 → 9.41 ms move. This arm measured the floor directly: 5 repeats of each
cell inside one process, model rebuilt every repeat, CPU load and wall-clock logged, teacher-vs-student
prefill invariance promoted to an automatic gate. Card verified empty; `load1` sat at 0.94–1.06 for all
20 passes, so nothing below is a load artifact.

### Both gates PASS, and they pass by two orders of magnitude

| gate | bar | measured |
|---|---|---|
| prefill invariance, fp32:eager | 0.5 ms | **0.02 ms** |
| prefill invariance, bf16:compile | 0.5 ms | **0.01 ms** |

The LM is byte-identical across towers, so this delta is zero by construction and the harness reproduces
that to 0.01 ms. **The control §47 leaned on is sound — but its numeric conclusion was wrong**, because
§47 read *one draw per pair* and this arm reads five.

### The per-repeat table is the result. Medians hide it.

| cell | tower ms (5 reps) | prefill ms (5 reps) |
|---|---|---|
| fp32 eager teacher | 25.71–25.73 | 13.52–13.60 |
| fp32 eager student | 9.52–9.53 | 13.58–13.63 |
| bf16 compile teacher | 9.01–9.07 | 4.18–4.23, **one at 4.67** |
| bf16 compile student | 4.02–4.04 | 4.17–4.23, **one at 5.36** |

**Every stage is deterministic to ~0.1% except one: compiled prefill, which has a rare upward
excursion.** Base mode is tighter than anything else in the run (±0.03 ms, 0.7%); one repeat in five
jumps +0.45 ms (teacher) or **+1.13 ms (student)**. Never downward. Never the tower — including the
*compiled* tower, which is as tight as the eager one. Not load: `load1` is flat across all 20 passes.

### This explains §47 exactly, and retires its hypothesis

**§47's ship-cell "regression" was one excursion draw.** step030 measured `bf16_compile_student` at
prefill 5.39 / e2e 9.41; this arm's rep1 measures **5.36 / 9.40** — agreement to 0.01 ms — while its
other four repeats sit at 8.19–8.27. step030 did not catch a slower configuration; it took a single
sample and landed in the tail.

**Retired: the ±0.5 ms symmetric floor (HYPOTHESIS, §47).** It is not a floor on all measurements. It is
a **one-sided heavy tail on compiled prefill only**, at roughly 1-in-5 incidence in this run. §47's four
within-pair deltas (+0.40 / −0.52 / +0.54 / +1.21) are exactly what that tail produces when you difference
two single draws: small when neither member excurses, ~+1 ms when one does. The mechanism is consistent
with everything else this line has measured — prefill is the launch/Python-dispatch-bound stage, so it is
the one exposed to host-side scheduling jitter, and compile makes the GPU work small enough for that
jitter to show. Still HYPOTHESIS as to *cause*; the *distribution shape* is CONFIRMED.

### The automated verdict was an artifact of my own pre-registration. Correcting it.

The script printed `worst relative IQR 1.0% vs bar 5.0% -> RESOLVABLE`. That is technically true and
practically misleading: **IQR is robust to tails by construction, so it discards precisely the events
this arm existed to find.** A 1-in-5 excursion sits at the 80th percentile and never enters the
interquartile range. I pre-registered a robust statistic to hunt a tail — a category error, and the same
species of mistake as §45's (a bar finer than its own SE) and §48's (reading a null as parity).

**Corrected reporting rule for this line: quote median with MIN–MAX, never IQR.** IQR stays in the JSON
as the dispersion of the base mode, which is a genuinely useful second number, but it is not the headline.

### Ship numbers, restated on 5 draws instead of 1

All ratios below are within-process, teacher and student measured in the same run:

| quantity | value |
|---|---|
| tower | 25.72 → **4.04 ms = 6.37×** (both deterministic, ±0.1%) |
| prefill | 13.58 → **4.22 ms = 3.22×** median; worst draw 5.36 |
| end-to-end | 39.30 → **8.26 ms = 4.76×** median; range 8.19–9.40 |
| tower params | 85.05M → 21.28M (0.2502×) |
| weights | 340 MB fp32 → **42.6 MB bf16 = 8.00×** (counted, not timed) |
| top1 | 0.7040 → **0.6860**, identical across all 5 repeats |

**The 5.0 ms prefill WIN bar holds at the median (4.22) and at four of five draws, but NOT at the worst
draw (5.36).** State it that way. The end-to-end claim is unaffected: even the tail draw, 9.40 ms, beats
the compiled teacher's 13.27 and the fp32-eager teacher's 39.30 outright. **STALE as a worst case (2026-08-15, §§54–56): 5.36 is a pass MEAN of 500 images over 10 draws; the per-frame worst case is ≥808.56 ms, measured in three processes, and no worst-case frame latency may be quoted from a pass mean anywhere in this line. Medians, ratios, params and bytes above are unaffected — read "worst draw 5.36" as "worst pass mean of 10 draws".**

Note also `bf16_compile_student` top1 **0.6860** here vs **0.6840** in step030 — 1 image out of 500,
consistent with §47's finding that compile + bf16 is not bit-deterministic *across processes* while being
perfectly stable *within* one (all 5 repeats identical).

### What this unblocks

§47 blocked CUDA graphs, `torch.export`, and the §32 prologue ablation behind step032 on the grounds that
the instrument could not resolve them. **That block is lifted.** The base mode resolves 0.03 ms, which is
ample for any of the three. The standing requirement is procedural, not instrumental: **≥5 repeats,
report median and min–max, and treat a single-draw latency number as unpublishable.**

### Open, in priority order

1. **CUDA graphs / `torch.export` on prefill.** Now the top item and the only remaining prefill lever —
   and the tail is itself an argument for it, since graph capture removes exactly the per-launch host
   dispatch that the excursion is consistent with. If it kills the tail as well as the median, that is
   two wins from one change.
2. **Re-time §44 (r=0.25) and §45 (r=0.125) under the 5-repeat protocol.** The operating-points table
   still carries single-draw numbers, and §48 has already retracted its accuracy column.
3. **Characterise the tail properly** if any paper text quotes worst-case latency: R=20 on the two
   compile cells would pin the incidence rate, which R=5 estimates only as "roughly 1 in 5".
4. **§32 prologue ablation.** Unblocked, still the lowest-value of the four.

## 50. vlm_step033 — CUDA graphs on prefill. DONE. **WIN on both cells: the median falls 33%, and the tail disappears in the cells measured here.**

> **AMENDED at §51 (step034).** The heading originally read "the tail disappears", full stop. That is
> too strong: this arm measured two cells and the tail vanished in both, but step034 later drew an
> excursion (2.67 base → **3.60 ms**) in a cudagraph cell at r=0.125. **Cudagraphs REDUCE excursion
> incidence; they do not eliminate it.** The dispatch attribution below survives — a replay does far
> less host work than N launches, so a lower rate is what that mechanism predicts — but the
> elimination claim was generalised from one clean cell, which is the third time this line has
> over-read a handful of draws. Read every "tail gone" statement below as scoped to r=0.25.

§49 left one prefill lever and one open cause. This arm pulled the lever and, in doing so, tested the
cause. `torch.compile(mode="reduce-overhead")` (inductor cudagraph trees) vs the default-inductor
`compile` baseline, R=5 per cell, teacher and student, bf16, n=500, card verified empty, `load1`
0.80–1.11 across all 20 passes.

### Result

| cell | prefill median | min–max | tail (max−med) | e2e median |
|---|---|---|---|---|
| compile teacher | 4.17 | 4.11–4.25 | 0.07 | 13.23 |
| **reduce-overhead teacher** | **2.80** | 2.73–2.82 | 0.02 | **11.71** |
| compile student | 4.27 | 4.18–**5.65** | **1.38** | 8.36 |
| **reduce-overhead student** | **2.73** | 2.73–2.80 | **0.06** | **6.74** |

Verdicts as pre-registered: teacher `WIN` (top1 Δ 0.0020, median +1.38, tail +0.05), student `WIN`
(top1 Δ **0.0000**, median **+1.54**, tail **+1.32**). Both inside the 0.02 accuracy gate, so the
KV-cache clone held and the latency is admissible.

### The student cell is the result; the teacher cell's WIN is median-only

Read the two cells differently, because their baselines differ. **The teacher baseline drew zero
excursions in five**, so its 0.07 "tail" was just base-mode dispersion and shrinking it to 0.02 is the
graph cell being tighter — there was no phenomenon there to remove. Its WIN is a median win that the
verdict rule happened to score on both axes.

**The student baseline excursed twice in five** (5.43, 5.65 against a 4.18–4.27 base) and the graph
cell returns 2.73–2.80 across all five. That is the dispatch hypothesis tested by intervention:
remove the per-launch host dispatch and the one-sided upward tail does not appear. **CONFIRMED: the
compiled-prefill excursion is caused by per-launch dispatch, not by the GPU work.** The finer
mechanism (which scheduler, which queue) stays HYPOTHESIS — the intervention is at the level of
"launches vs one replay", so that is the level the claim gets made at.

### Two corrections to §49, both from this run's own data

1. **Incidence is cell-dependent and §49's "roughly 1 in 5" was one run's worth of evidence about a
   rate.** Combined over step032+step033: teacher **1/10**, student **3/10**. §49 read a single
   teacher excursion and a single student excursion as one rate; they are not obviously one rate.
2. **The excursion is NOT locked to a rep index.** Mid-run I noted 5.39 (step030) / 5.36 (step032 rep1)
   / 5.43 (step033 rep1) and read a positional lock into three draws. step033 rep4 then excursed to
   5.65 and killed it. Logged because it is the same failure §49 and §48 committed — reading structure
   off a handful of draws — and it took four more draws to catch. The one-sided *shape* is what has
   survived every look; the *placement* has not.

### Ship numbers, restated on cudagraphs (all within-process, R=5)

| quantity | fp32 eager teacher | bf16 cudagraph student | ratio |
|---|---|---|---|
| tower | 25.72 | **4.01** | 6.41× |
| prefill | 13.58 | **2.73** | **4.97×** |
| end-to-end | 39.30 | **6.74** | **5.83×** |
| tower params | 85.05M | 21.28M | 4.00× |
| weights | 340 MB fp32 | **42.6 MB bf16** | 8.00× |
| top1 | 0.7040 | 0.6880 | −1.60pp |

**§49's 5.0 ms prefill bar now passes at every draw, not four of five** — max 2.80. The "state it with
the worst draw" caveat §49 imposed is discharged for this configuration: worst draw 2.80 prefill,
6.80 e2e. e2e improves 8.26 → 6.74 median, a further 1.52 ms on top of everything §44–§49 banked.

Accuracy note: student top1 reads **0.6880** here vs 0.6860 (step032) and 0.6840 (step030) — 1–2
images of 500, the cross-process bf16+compile nondeterminism §47 identified. Within a run all five
repeats are identical. Do not quote student top1 to four digits across runs.

### Measurement caveat that must travel with these numbers

`GraphEval` clones the KV cache out of graph-owned memory after prefill, because the constrained-choice
protocol reuses one cache across 10 teacher-forced label passes and cudagraph replay would overwrite it
(PyTorch raises; it does not corrupt silently — the error fired on the first smoke test exactly as
pre-registered). **The clone sits outside the timed window and is an artifact of this EVAL protocol,
not of deployment** — a deployed pipeline decodes inside the captured region and never hands a cache
back to Python. Do not report it as cudagraph overhead, and do not report the baseline cell as
"cudagraphs without the clone": the baseline runs plain `VLMEval`, so the comparison is one-variable.

### Open, in priority order

1. **Re-time §44 (r=0.25) and §45 (r=0.125) under the 5-repeat protocol, now with cudagraphs as the
   reference mode.** The operating-points table still carries single-draw numbers from before both
   §49's protocol and this arm's mode change.
2. **R=20 on the compile baseline cells** if any paper text quotes worst-case latency — 3/10 vs 1/10
   is still a thin estimate of a rate, and this arm showed how easily a rate is over-read.
3. **§32 prologue ablation.** Still unblocked, still lowest value.
4. Prefill is no longer the leading cost: at 2.73 ms prefill against a 4.01 ms tower, **the vision
   tower is now the larger half of e2e**. Any further latency work should target the tower.

---

§51 onward continues in `VLM_TRAJECTORY_part16.md` — this file is at its cap.
