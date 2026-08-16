# VLM trajectory — part 16 (the decomposition is not additive)

Continues `VLM_TRAJECTORY_part15.md` (closed at its 200-line cap). Same line: SmolVLM-256M,
Imagenette 10-class, d12@512 anchor 0.711, teacher 0.7040.

## 51. vlm_step034 — re-timing the operating points under cudagraphs. DONE. **The table is the least interesting thing this arm produced.**

> **RETRACTED IN PART by §52 below — read it first.** Finding 2 ("the tower moves with the text
> student's width") is wrong at the premise: `--ratio` narrows the VISION tower, not the text model
> (`vlm_eval.py:91,103`, `vlm_width.py:41`). The tower moves because it was narrowed. Finding 1 is a
> true measurement whose inference is void for the same reason. Finding 3 (the r=0.125 excursion under
> capture) and the control are unaffected and stand.

§49 item 2: the operating-points table carried single-draw numbers taken before both the ≥5-repeat rule
and §50's mode change. This arm re-timed r=0.5 / 0.25 / 0.125 at R=5 under `reduce-overhead`, student
only. Three sequential invocations of the existing `vlm_step033_cudagraph.py`; no new script.

### The control passes, so the arm is reportable

Pre-registered: r=0.25 must reproduce §50's cell or the other two rows are discarded unread.

| | §50 (step033) | this run | agreement |
|---|---|---|---|
| prefill median | 2.73 | **2.72** | 0.01 ms |
| prefill min–max | 2.73–2.80 | 2.69–2.78 | overlapping |
| e2e median | 6.74 | 6.66 | 0.08 ms |
| top1 | 0.6880 | **0.6880** | exact |

### The table

| ratio | tower median (min–max) | prefill median (min–max) | e2e median | top1 |
|---|---|---|---|---|
| r=0.5 | 4.30 (4.26–4.31) | 2.73 (2.67–2.78) | 7.04 | 0.6940 |
| r=0.25 | 3.94 (3.90–3.96) | 2.72 (2.69–2.78) | 6.66 | 0.6880 |
| r=0.125 | 3.81 (3.77–3.82) | 2.72 (2.67–**3.60**) | 6.52 | 0.6740 |

Epoch budgets are NOT matched across rows (r=0.5's only 9352/n500/d6 checkpoint is e25, the other two
are e50), so **the top1 column is not a clean width comparison** and accuracy claims stay with §44/§45/§48.
This is a latency re-timing.

### Finding 1 — prefill is flat in width. CONFIRMED.

**Prefill median is 2.72–2.73 at every ratio**, across a 4× span of MLP width. Not approximately: three
medians within 0.01 ms. Under cudagraphs the width axis is **free in prefill** — the extra FLOPs
disappear into the same replay, exactly as a dispatch-bound stage predicts. Every prefill-side argument
for narrowing the MLP is now void: narrowing buys parameters and bytes, not prefill milliseconds.

### Finding 2 — the tower moves with the text student's width. CONFIRMED, mechanism HYPOTHESIS.

> **RETRACTED at §52. The premise is inverted.** `--ratio` narrows the **VISION TOWER**, not the text
> model. `vlm_eval.py:91` sets `self.full = list(model.model.vision_model.encoder.layers)`;
> `set_layers` (`vlm_eval.py:103`) assigns back into `vision_model.encoder.layers`; and
> `build_width_student` (`vlm_width.py:41`) slices `ev.full[:depth]` at
> `int(round(inter * ratio))` where `inter = ev.full[0].mlp.fc1.out_features`. §50's own table says
> it out loud: "tower params 85.05M -> 21.28M" IS the ratio arm. So the tower moving 4.30 -> 3.94 ->
> 3.81 is the tower being narrowed, which is the designed effect and not a coupling. There is no
> unexplained phenomenon here, the additive decomposition is intact, and Finding 1's flat prefill is
> the NULL EXPECTATION (`--ratio` never touches the text model) rather than evidence that width is
> dispatch-free. See §52.


The vision tower is byte-identical across these three cells; `--ratio` rebuilds only the text student.
It has no mechanistic right to move. It moves anyway, **monotonically, with disjoint min–max bands**:

    r=0.5   4.26–4.31      r=0.25  3.90–3.96      r=0.125  3.77–3.82

0.49 ms end to end, against per-cell tails of 0.00–0.02. This is not noise and it is not cross-process
drift — it is one script, one session, and the r=0.25 control reproduced §50 to 0.01 ms.

**This breaks the additive decomposition this line has used since §44.** e2e = tower + prefill has been
read as two independent stages with width touching only the second. Under cudagraphs it does not
decompose: width shows up in the *tower* number and leaves prefill flat. Total e2e still falls with
narrowing (7.04 → 6.52), so no shipped e2e claim is wrong — but every claim that attributes a saving to
a *stage* is now suspect, and §50's "the vision tower is now the larger half of e2e" is measured under
whatever this coupling is.

Leading candidate, HYPOTHESIS only: inductor captures both compiled models into a shared cudagraph
memory pool, so a wider text MLP changes the pool layout the vision replay runs against. Alternatives
not excluded: allocator fragmentation, or a timer boundary that attributes some text-side work to
`encode`.

**step035 is the decisive test and is running**: same two ratios (0.5, 0.25), `--modes compile`, no graph
capture, R=5, 10 passes. Gap vanishes → pool coupling, and cudagraph stage attributions need a caveat
wherever they appear. Gap survives → this is not cudagraphs at all but a harness measurement bug older
than §50, and the stage split needs rebuilding before any of it is published.

### Finding 3 — cudagraphs reduce the excursion rate, they do not eliminate it. §50 AMENDED.

r=0.125 rep2 drew **prefill 3.60** against a 2.67–2.72 base — +0.9 ms, one-sided upward, tower and
`load1` flat through it (3.82, 1.00). Same shape §49 described, now inside a *captured* cell.

§50's heading claimed the tail disappears. It measured two cells, both clean, and generalised. The
correct statement: **incidence drops sharply under capture but is not zero** — cudagraph cells so far
run 1 excursion in 15 (r=0.125 only; r=0.5 and r=0.25 clean at 5/5 each) against 3 in 10 for the
compiled student baseline. The dispatch attribution survives, since a replay issues far less host work
than N launches and a reduced rate is what that mechanism predicts. The elimination claim does not.

That is the **third** over-read of a handful of draws in this line (§48 read a null as parity; §50 read
a rep-index coincidence as a positional lock, then read two clean cells as elimination). The pattern is
consistent enough to name: **R=5 is enough to establish a median and nowhere near enough to establish
the absence of a rare event.** Absence claims need the R=20 arm (§49 item 3), which is now overdue.

### Ship numbers unchanged

r=0.25 remains the ship cell: prefill 2.72, tower 3.94, e2e 6.66, top1 0.6880, 42.6 MB bf16. r=0.125
buys 0.14 ms of e2e for −1.40pp top1 at an unmatched epoch budget — not a trade worth taking on this
evidence.

### Open, in priority order

> **This list is superseded by §52/§53.** Items 1 and 3 below were written under the retracted
> premise. Items 1 is void (step035 asked a question that has no referent) and item 3's block is
> lifted: the tower number means what it always meant, so the tower is actionable now.

1. ~~**step035** — resolve the tower/width coupling. Blocks any per-stage attribution.~~ VOID (§52).
2. **R=20 on the compile and cudagraph cells** — now the load-bearing open item, not an optional one:
   three separate absence-claims in this line have been wrong. → step036.
3. **The tower is the remaining latency mass** (3.94 vs 2.72) and is the leading target. ~~do not act
   on that until step035 says what the tower number actually measures.~~ Block lifted at §52.
4. §32 prologue ablation. Unblocked, still lowest value.

---

## 52. The §51 coupling was a read of the wrong knob. **No experiment needed; the source settles it.**

§51 Finding 2 reported that the vision tower moved 4.30 -> 3.94 -> 3.81 ms across `--ratio` 0.5 /
0.25 / 0.125 while "the vision tower is byte-identical across these three cells; `--ratio` rebuilds
only the text student", called that an unexplained coupling, declared the additive decomposition
broken, and launched step035 to distinguish cudagraph-pool coupling from a harness bug.

**`--ratio` narrows the vision tower.** Three lines of source, no run required:

| file:line | what it establishes |
|---|---|
| `vlm_eval.py:91` | `self.full = list(model.model.vision_model.encoder.layers)` — `full` IS the tower |
| `vlm_eval.py:103` | `set_layers` assigns into `model.vision_model.encoder.layers` |
| `vlm_width.py:41` | `build_width_student` slices `ev.full[:depth]`, MLP-sliced to width `m` |

and `vlm_step033_cudagraph.py:137,142` passes `m = int(round(inter * args.ratio))` with
`inter = ev.full[0].mlp.fc1.out_features`. The text model is never rebuilt by `--ratio`. §50's own
ship table states it directly — "tower params 85.05M -> 21.28M" is the r=0.25 arm.

What that does to §51:

1. **Finding 2 is RETRACTED, not downgraded.** The tower moves because the tower was narrowed. It is
   the intended, measured effect of the width axis and it has been the meaning of `--ratio` since
   §44. Monotonic with disjoint min-max bands is exactly what a real 4x width span should produce.
2. **The additive decomposition is intact.** e2e = tower + prefill still holds, width still lands in
   the tower only, and no per-stage attribution in this line needs a caveat. §50's "the vision tower
   is now the larger half of e2e" (3.94 vs 2.72) stands as measured.
3. **Finding 1 survives as a fact and dies as an inference.** Prefill IS flat at 2.72–2.73 across the
   three cells — but that is the null expectation, because `--ratio` does not touch the text model.
   It is a harness sanity check that passed, not evidence that width is free under dispatch. The
   prefill-is-dispatch-bound claim rests on §44's roofline and step022's token sweep, which are
   untouched; it simply gains nothing from this arm.
4. **step035's pre-registration is void.** It reads a surviving gap under `--modes compile` as "a
   harness measurement bug older than §50 — the stage split needs rebuilding before any of it is
   published." The gap will survive, because it is a genuine width difference, and its first reps
   already show r=0.5 at tower 4.33–4.39 under plain compile. Read as pre-registered that is a false
   alarm aimed at the foundation of the paper. What the run is still good for: a `compile`-mode
   companion to §51's table at two ratios, and 10 more draws toward the overdue R=20 excursion count.

**Method.** §48, §50 and §51 were each an over-read of a handful of draws. This one is different in
kind and worth separating: the draws were fine and the statistic was fine. The arm asked what a knob
did without reading what the knob was wired to, and then reached for an experiment to explain the
answer. **Before attributing a measurement to a mechanism, verify which module the flag actually
rebuilds — the cheaper check is always the source, and here it was three lines.**

---

§53 onward continues in `VLM_TRAJECTORY_part17.md`. This file stays short by design: §51 and §52
are one investigation and splitting them mid-argument was the lesser evil against the 200-line cap.
