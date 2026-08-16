# Graphiti pending — part 3

Episodes queued while the graphiti MCP is down. Replay in order once it returns.
Episodes 1–3: `graphiti_pending.md`. Episodes 4–6: `graphiti_pending_part2.md`.

---

## Episode 7 — vlm_step033 (group_id="dhiraj"), 2026-08-15

vlm_step033 asked whether CUDA graphs remove BOTH the compiled-prefill median cost and the one-sided
upward tail §49 found. `scripts/frontier/vlm_step033_cudagraph.py --modes compile reduce-overhead
--towers teacher student --repeats 5 --n_eval 500`, bf16, 5060ti_cuda, card verified empty, load1
0.80–1.11 across all 20 timed passes. VERDICT **WIN on both cells**.

Prefill median / min–max / tail(max−med) / e2e median:

| cell | median | min–max | tail | e2e |
|---|---|---|---|---|
| compile teacher | 4.17 | 4.11–4.25 | 0.07 | 13.23 |
| reduce-overhead teacher | 2.80 | 2.73–2.82 | 0.02 | 11.71 |
| compile student | 4.27 | 4.18–5.65 | 1.38 | 8.36 |
| reduce-overhead student | 2.73 | 2.73–2.80 | 0.06 | 6.74 |

Pre-registered verdicts: teacher WIN (top1 delta 0.0020, median +1.38 ms, tail +0.05 ms); student WIN
(top1 delta 0.0000, median +1.54 ms, tail +1.32 ms). Both inside the 0.02 top1 gate.

CONFIRMED by intervention: the compiled-prefill excursion is caused by PER-LAUNCH HOST DISPATCH, not
by GPU work. The student compile baseline excursed twice in five (5.43, 5.65 off a 4.18–4.27 base);
replacing per-kernel launches with one graph replay returns 2.73–2.80 across all five. §49's
dispatch-jitter HYPOTHESIS is promoted at that granularity. The finer mechanism (which scheduler,
which queue) stays HYPOTHESIS — the intervention only resolves "launches vs one replay".

Two corrections to §49 from this run's own data. (1) Excursion incidence is CELL-DEPENDENT: combined
over step032+step033 the teacher is 1/10 and the student 3/10; §49 read one draw each as a single
rate. (2) The excursion is NOT locked to a rep index — 5.39 (step030) / 5.36 (step032 rep1) / 5.43
(step033 rep1) looked positional, then step033 rep4 excursed to 5.65 and killed it. Same failure mode
as §45 and §48: structure read off a handful of draws. The one-sided SHAPE has survived every look;
the PLACEMENT has not.

Ship config restated on cudagraphs (d6 x r=0.25, bf16, within-process R=5), vs the fp32-eager teacher:
tower 25.72 -> 4.01 ms (6.41x); prefill 13.58 -> 2.73 ms (4.97x); e2e 39.30 -> 6.74 ms (5.83x); tower
params 85.05M -> 21.28M (4.00x); weights 340 MB fp32 -> 42.6 MB bf16 (8.00x EXACT, counted not timed).
top1 0.6880 vs 0.7040 at n=500; the load-bearing accuracy number remains step031's n=2000 −3.60pp.
§49's 5.0 ms prefill bar now passes at EVERY draw (max 2.80), not four of five — the "quote the worst
draw" caveat is discharged for this configuration (worst draw 2.80 prefill, 6.80 e2e).

Measurement caveat that must travel with these numbers: `GraphEval` clones the KV cache out of
graph-owned memory after prefill because the constrained-choice protocol reuses one cache across 10
teacher-forced label passes and replay would overwrite it (PyTorch raises; it does not corrupt
silently — the error fired on the first smoke test exactly as pre-registered). The clone sits OUTSIDE
the timed window and is an artifact of the EVAL protocol, not of deployment. Do not report it as
cudagraph overhead. The baseline cell runs plain `VLMEval`, so the comparison stays one-variable.

Consequence for the roadmap: at 2.73 ms prefill against a 4.01 ms tower, **the vision tower is now the
larger half of e2e**. Prefill is no longer the leading latency cost, and further latency work should
target the tower. Also: student top1 reads 0.6880 / 0.6860 / 0.6840 across step033 / step032 / step030
— cross-process bf16+compile nondeterminism, identical across all five within-process repeats. Do not
quote student top1 to four digits across runs.

Write-up: learnings/frontier/VLM_TRAJECTORY_part15.md section 50.

---

## Episode 8 — vlm_step034 + vlm_step035 + the §52 retraction (group_id="dhiraj"), 2026-08-15

vlm_step034 re-timed the operating points (r=0.5 / 0.25 / 0.125, student, bf16, reduce-overhead, R=5,
n=500) under the >=5-repeat protocol. Control PASSED: r=0.25 reproduced step033 to 0.01 ms prefill and
exact top1 0.6880. Table (tower / prefill / e2e medians, ms): r=0.5 4.30 / 2.73 / 7.04 top1 0.6940;
r=0.25 3.94 / 2.72 / 6.66 top1 0.6880; r=0.125 3.81 / 2.72 / 6.52 top1 0.6740. Epoch budgets NOT
matched across rows (r=0.5 is e25, others e50), so the top1 column is not a clean width comparison --
this is a latency re-timing only.

step034's write-up (section 51) then claimed an unexplained "tower/width coupling": the tower moved
with --ratio although --ratio supposedly rebuilt only the text model. **RETRACTED at section 52 from
source, with no experiment.** --ratio narrows the VISION TOWER: vlm_eval.py:91 sets
`self.full = list(model.model.vision_model.encoder.layers)`; vlm_eval.py:103 `set_layers` assigns back
into `vision_model.encoder.layers`; vlm_width.py:41 `build_width_student` slices `ev.full[:depth]` at
`int(round(inter * ratio))`. Section 50's own ship table already said it -- "tower params 85.05M ->
21.28M" IS the r=0.25 arm. So the tower moving with ratio is the designed width effect, the additive
decomposition e2e = tower + prefill is INTACT, no per-stage attribution needs a caveat, and section
51's companion Finding 1 ("prefill is flat in width") is the NULL EXPECTATION rather than evidence,
because --ratio never touches the text model.

vlm_step035 had been launched to explain that non-existent coupling, and its pre-registration read a
surviving gap under --modes compile as "a harness measurement bug older than section 50 -- the stage
split needs rebuilding before publication". The gap survived (tower 4.37 vs 4.03 at r=0.5 vs r=0.25,
disjoint bands, 0.34 ms uncaptured against 0.36 ms captured) because it is a genuine width difference.
The void read was annotated into the queue row and section 51 BEFORE the run reported, so the false
alarm never landed. Salvaged value: a compile-mode companion table, and the finding that CAPTURE BARELY
HELPS THE TOWER (4.03 compiled vs 3.94 captured, 0.09 ms) while taking prefill 4.22 -> 2.72 --
consistent with section 44's roofline reading that the tower is compute-bound and prefill is
dispatch-bound.

Excursion statistics revised again. Pooling compiled-student prefill draws across step032/033/035
gives roughly 8 elevated of 20, NOT section 49's "roughly 1 in 5"; at r=0.5 the elevated state is the
MAJORITY (median 5.03, draws 4.11/4.17/5.03/5.06/5.20, no identifiable base). Captured cells run 1
elevated in 15. So section 50's dispatch attribution is REINFORCED and only its "rare" and
"eliminated" language was wrong (both already amended). Consequence for method: the min-max rule from
section 49 is necessary but NOT sufficient -- once the elevated state is common it stops being a tail
and moves the centre, so raw draws must be reported for this cell. vlm_step036 (R=20, both modes) is
running to pin the incidence.

Method findings worth keeping separate. (a) Sections 48/50/51 were over-reads of small samples.
(b) Section 52 was a DIFFERENT failure: draws and statistic both fine, but the arm asked what a knob
did without reading what the knob was wired to, then bought GPU time to explain the answer. CHECK THE
SOURCE BEFORE BUYING AN EXPERIMENT -- three lines would have prevented both the claim and the run.
(c) Logged retraction inside section 53: a monotone-by-rep-index prefill pattern in r=0.5 (5/5) was
read as within-process rebuild drift with a 1/120 figure computed on the very pattern that suggested
it; r=0.25 came back non-monotone and killed it at n=2. A pattern noticed inside a 5-draw cell is a
hypothesis for the next cell, never a finding, and never carries an inference-free number.

Ship cell UNCHANGED through all of this: d6 x r=0.25, bf16 + cudagraphs -- tower 3.94, prefill 2.72,
e2e 6.66 ms, 21.28M tower params, 42.6 MB bf16 (8.00x fewer bytes, counted not timed), top1 0.6880 at
n=500 and -3.60pp at n=2000 (section 48). Nothing in sections 51-53 moved a shipped number; what moved
was what those numbers were said to MEAN.

Write-up: learnings/frontier/VLM_TRAJECTORY_part16.md sections 51-52 and part17.md section 53.

---

## Episode 9 — vlm_step036, the R=20 excursion arm (group_id="dhiraj"), 2026-08-15

Settled section 49's oldest open item. One invocation of vlm_step033_cudagraph.py at --modes compile
reduce-overhead --towers student --ratio 0.25 --repeats 20 --n_eval 500, bf16, 5060ti_cuda, card empty,
load1 0.74-1.05 across all 40 passes. Departure thresholds were COMMITTED IN WRITING BEFORE THE DECIDING
DRAWS (compile >=4.5 ms at rep9; capture >=3.0 ms at rep0) because base draws had started filling the
gap section 49's clean base/tail split assumed away, and a threshold picked after seeing draws is a
threshold tuned to a count.

Compile cell: prefill median 4.23 (4.17-6.35), tail 2.12; tower 4.07 (3.97-4.10), tail 0.04; e2e 8.29
(8.19-10.43). Departures 4.58 / 4.97 / 5.25 / 5.69 / 5.81 / 6.35 = **6 of 20**. Captured cell: prefill
median 2.75 (2.73-7.08), tail 4.33; tower 4.01, tail 0.03; e2e 6.76 (6.70-11.09). **1 of 20.** top1
0.6880 on all 40 passes -- the excursion is latency-only and never touches what the model computes.

Finding 1, CONFIRMED: incidence UPHELD. Pre-registered >=4/20 sustains section 49's "roughly 1 in 5";
6/20 = 30% (95% CI ~12-54%) so the rate was if anything understated. The excursion is a reproducible
property of the compile cell, not three unlucky early runs.

Finding 2, CONFIRMED and it SUPERSEDES a number this line has quoted since section 49: the worst case
is not 5.36 ms. It is **>=6.35 compiled and >=7.08 captured**, and ">=" is the honest operator because
20 draws bound a tail from below only. Every budgeting statement in sections 44-51 that used 5.36 must
be restated. For a drone frame budget this matters more than any median.

Finding 3, CONFIRMED: stage asymmetry is **50x** -- tower tail 0.04 ms against prefill tail 2.12 ms, same
20 passes, same process, same images. The tower is the LARGER stage by median (4.07 vs 4.23) and is
nearly noiseless. That rules out every whole-process cause (thermal, clock, host load, co-tenancy), which
would move both stages, and localises the excursion to what prefill has and the compiled tower does not.

Finding 4: incidence and tail depth now point OPPOSITE WAYS. Capture cuts incidence 6/20 -> 1/20 and
median prefill by 1.47 ms, but its single excursion (7.08) is DEEPER than any compiled one (6.35) and
worst-case e2e is WORSE captured (11.09) than compiled (10.43). Depth is 1 draw against 6 --
UNDERPOWERED, explicitly not claimed as a finding. What IS settled: capture does not cap the excursion,
only its rate. Sections 50/51 have been letting a RATE claim stand in for a TAIL claim.

Consequence for the verdict: section 50 called this exact comparison WIN at R=5. At R=20 the harness
prints PARTIAL (top1 delta 0.0000, median gain +1.47 ms, tail gain -2.21 ms). Nothing about the cell
changed; only the number of draws did. **Fourth time in this line an R=5 absence-or-rate claim has
failed** (sections 48, 50 twice, 53) -- and the first time the failure was caught by the pre-registered
arm instead of by a later accident.

Two in-run hypotheses raised and killed by the run itself, both predicted in writing before the deciding
reps: (a) tower creep, reps 0-4 rose 3.97->4.06 monotonically, then reps 5-19 sat flat at 4.05-4.10 =
early-rep warmup, KILLED; (b) escalating ramp, departures 4.97->5.25->5.69->5.81 with reps 10/11
adjacent, prediction recorded at rep11 that a ramp never returns to base -- rep12 returned 4.19 and
rep18 returned 4.21 straight after the 6.35 maximum, KILLED. The departures are independent draws that
clustered by chance. First prospective use of this discipline in the line rather than a post-hoc repair.

Ship cell restated, medians unchanged within 0.1 ms: r=0.25 captured, prefill 2.75, tower 4.01, e2e 6.76,
top1 0.6880, 42.6 MB bf16. **Worst observed e2e 11.09 ms** is new and is the honest number to budget
against until a depth arm runs. No worst-case e2e claim may be published from this run.

Next arm is tail DEPTH, not another incidence arm (R=20 yields ~1 captured excursion; needs ~R=100 or a
soak). Also open: late compile base draws (4.25/4.27/4.30) sat above early ones (4.17-4.20), deliberately
kept out of the count by the committed rule; if real it needs rep index as the variable.

Write-up: learnings/frontier/VLM_TRAJECTORY_part17.md section 54.
Artifact: results/frontier/vlm_step033_cudagraph_bf16_compile-reduceoverhead_student_r20_n500_r20__5060ti_cuda.json
