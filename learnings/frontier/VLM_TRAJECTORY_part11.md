# VLM trajectory — part 11 (compounding the levers)

Continues `VLM_TRAJECTORY_part10.md` (full at 198 lines). Same line: SmolVLM-256M, Imagenette
10-class, d12@512 anchor 0.711, teacher 0.7040, tower = 82% of end-to-end latency.

State entering this part, all at the 25ep/9269 budget and all scored against the same 0.7040 teacher:

| arm | tower params | tower latency | vs teacher |
|---|---|---|---|
| depth only (d6) | 0.500× | 2.00× | parity |
| width only (r=0.5) | 0.667× | 1.28× | +2.80pp (parity) |
| compound (d6 × r=0.5) | **0.333×** | **2.40×** | −1.20pp (parity) |

§42 CONFIRMED the two levers compose: params landed at exactly the product, latency at 2.40× against
a predicted 2.56×, and accuracy stayed at parity — so the orthogonality argued from disjoint parameter
sets is now measured, not merely asserted.

## 42. vlm_step025 — the compound arm: d6 × r=0.5. DONE. **WIN.**

Roadmap item 1. `--student_depth 6 --ratios 0.5 --epochs 25 --n_train 9352`, every other knob
identical to §40. The teacher stays the **full d12 tower** — `cache_teacher` always runs it — so the
distillation targets are unchanged and the student is scored against the same 0.7040 teacher as §39
and §40. Teacher top1 reproduced at **0.7040 in all three runs**, so the line is one comparison line.

Result — `vlm_step023_mlp_width_r0.5_e25_t9352_n500_d6__5060ti_cuda.json`:

| arm | top1 | vs teacher | tower params | tower ms | speedup |
|---|---|---|---|---|---|
| teacher d12, full width | 0.7040 | — | 85.05M (1.000×) | 26.20 | 1.00× |
| naive (sliced, untrained) | 0.0920 | −61.20pp | 28.36M | 10.80 | chance |
| **distilled d6 × r=0.5** | **0.6920** | **−1.20pp → WIN** | **28.36M (0.3335×)** | **10.89** | **2.40×** |

Final `rel_mse 0.1915 / cosine 0.9002`. Trace still falling at the end: ep1 0.4348 → ep5 0.3030 →
ep10 0.2594 → ep15 0.2288 → ep20 0.2069 → ep25 0.1915.

**Orthogonality is now MEASURED, not argued. CONFIRMED.** §42 was pre-registered before the run
precisely because the compounding-rule licence here rested on disjoint parameter sets alone, and a
LOSS was named the *informative* outcome. It did not happen. Depth (whole layers) and width
(intermediate channels inside the survivors) compose.

Param prediction landed to the digit: each surviving layer = 7.08M − 4.72M + 2.36M = 4.72M, ×6 =
**28.36M = 0.3335×** against the predicted 0.333×.

**Speedup is slightly sublinear — state it honestly.** Measured 2.40× against the multiplicative
prediction 2.00 × 1.28 = 2.56×, i.e. ~94% of it. The levers compose on accuracy; they compose only
approximately on latency.

**End-to-end is 1.61×, not 2.40×.** Prefill was 13.70 ms (teacher) → 13.94 ms (student), essentially
unchanged — the cut touches only the tower. End-to-end ≈ 39.9 ms → 24.8 ms ≈ **1.61×**. The tower is
82% of end-to-end (§step006), so this is the arithmetic, not a surprise. Quote 2.40× as a *tower*
number only.

**−1.20pp is PARITY, not a loss.** At n_eval=500 and p≈0.70 the binomial SE is ≈2.05pp, so the gap is
inside one SE. Per-image predictions were not recorded for this arm — the `correct`-vector fix landed
*after* §42 launched — so McNemar is unavailable and no stronger claim is available either way.

**HYPOTHESIS — feature-fit is not monotone with task accuracy across architectures.** §42's 0.1915
is a *worse* feature fit than both single-lever runs (d6-alone 0.1639, width-alone 0.1570) yet it
reaches the same top-1 parity. So rel_mse ranks budgets within one architecture (§39's ladder) but
must not be used to rank architectures against each other. Untested; one comparison, three points.

**New drone operating point: 0.3335× tower params, 2.40× tower / 1.61× end-to-end, at parity.**
First arm in this line to move both drone axes at once.

Filename carries a `_d6` suffix — the TAG gains the depth field only when `student_depth != depth`,
so §39's and §40's width-only filenames stay byte-identical.

## 43. vlm_step026 — how far does width stretch once depth is halved? d6 × r=0.25. DONE. **PARTIAL.**

`--student_depth 6 --ratios 0.25 --epochs 25 --n_train 9352`, every other knob identical to §42.

**Why this and not more epochs.** §42's fit was still falling at ep25 (0.1915, vs §40's 0.1570), so
more epochs are cheap upside on an arm that already won. Pushing the width factor instead asks the
question that is actually open: **is 0.333× the knee, or does the compound keep paying?** §39's
r=0.25 result (−26.80pp) cannot answer it — that arm was starved by the same ~16× budget deficit §40
falsified for r=0.5, so it is uninterpretable and must not be cited as a floor. This run gives r=0.25
the budget at which d6, r=0.5 and d6 × r=0.5 all reached parity.

Predicted: each surviving layer becomes 7.08M − 4.72M + 1.18M = **3.54M**, so 6 layers = **21.2M =
0.25× tower params**.

Pre-registered, before the run:
- **WIN** (≥ teacher − 2.0pp) — 0.25× at teacher parity becomes the new drone operating point, and
  the compound is confirmed to keep paying past 0.333×.
- **PARTIAL** (≤10pp) — the compound has begun to pay for itself. This **locates the knee** between
  r=0.5 and r=0.25; it closes nothing, and per §42's still-falling curve the first thing it buys is
  an epoch ladder, not a verdict.
- **LOSS** (>10pp) — bounds how far the width factor stretches once depth is already halved. This is
  only interpretable because it is the budget at which every neighbouring arm reached parity, which
  is precisely what §39's r=0.25 lacked.

**Known gap, carried forward unchanged.** Per-image predictions still are not recorded, so a
near-teacher result must again be read as PARITY rather than superiority. Folding that in here would
have made this run two-variable against §42; it stays roadmap item 3.

### Result — `vlm_step023_mlp_width_r0.25_e25_t9352_n500_d6__5060ti_cuda.json`

**GATE — PASS.** Teacher d12 top1 **0.7040**, bit-identical to §39/§40/§42, so all four arms remain
one comparison line. Naive (sliced, untrained) 0.0860 — chance, as every naive structural cut here.

| arm | tower params | tower ms | distilled | vs teacher | verdict |
|---|---|---|---|---|---|
| §40 width only (d12 × r0.5) | 56.72M (0.667×) | 26.16→20.45 (1.28×) | 0.7320 | +2.80pp | WIN |
| depth only (d6, step004) | 42.5M (0.500×) | (2.00×) | 0.74 | ≈parity | WIN |
| §42 compound (d6 × r0.5) | 28.36M (0.333×) | 26.20→10.89 (2.40×) | 0.6920 | −1.20pp | WIN |
| **§43 compound (d6 × r0.25)** | **21.28M (0.250×)** | **26.21→9.53 (2.75×)** | **0.6540** | **−5.00pp** | **PARTIAL** |

Params again landed exactly on prediction: 6 × 3.54M = **21.28M = 0.250×**. Final
`rel_mse 0.2234 / cosine 0.8824`; naive→distilled **+56.80pp**.

**The knee is located, between 0.333× and 0.250×. This is the section's actual finding.** Three arms
in a row reached parity; this one did not. −5.00pp against a ≈2.05pp binomial SE at n_eval=500 is
~2.4 SE — **the first gap in this line that exceeds noise**, and so the first that must be reported
as a genuine drop rather than parity. The compound does not keep paying indefinitely.

**The trade past 0.333× is bad on the axis it was meant to buy.** Going 0.333× → 0.250× params cut
tower latency only 2.40× → 2.75×, i.e. **1.15× more speed for 5pp of accuracy**, while the param
ratio fell 1.33×. That is §39's "width buys speed poorly" reappearing at the margin, now measured
inside the compound: **the marginal width cut is a params/bytes purchase, not a latency purchase.**
For a drone that is flash-bound rather than frame-bound, 0.250× at −5pp may still be the right buy;
for a frame budget it is clearly not.

**Second finding, and it may outlast the first: tower speedup no longer survives to end-to-end.**
Prefill was unchanged (13.77 → 14.03 ms), so end-to-end went 39.98 → 23.56 ms = **1.70×**, against
§42's 2.40× tower → 1.61× end-to-end. Pushing the tower from 2.40× to 2.75× bought only 1.61× → 1.70×
overall. The tower was 82% of latency at d12; after two compounding cuts it no longer dominates, so
**further tower squeezing has diminishing end-to-end return.** The next real latency win has to come
from outside the tower — §32's prologue — or from the token count, not from more slicing.

**Budget caveat, and it is load-bearing here exactly as pre-registered.** Fit was still falling at
ep25 (0.2234, against §42's 0.1915) and ran ~0.033 behind §42 at every matched epoch from ep10 on.
The PARTIAL pre-registration said this outcome "buys an epoch ladder, not a verdict" — that stands.
**CONFIRMED by part12 §44 (was HYPOTHESIS): −5.00pp was largely budget, not capacity.** §39→§40 is the precedent
where exactly this reading was correct and worth +18.20pp. What forbids assuming it: §42's own
still-falling curve reached parity anyway, so a falling curve does not by itself predict recovery.

**Operating points now on the table, all vs the same 0.7040 teacher:**

| want | pick | cost |
|---|---|---|
| frame budget | §42 d6 × r0.5 — 0.333×, 2.40× tower / 1.61× end-to-end | parity |
| flash budget | §43 d6 × r0.25 — 0.250×, 2.75× tower | −5.00pp |

**Process defect found while writing this up, and fixed.** The `correct` per-image hit vector was
implemented locally after §42 launched, but **never copied to the 5060ti** — the box was still
running the pre-fix 189-line script, so §43 has no per-image vectors either and McNemar is *still*
unavailable. Part 10 §41 item 3 is marked DONE and is **wrong**: the fix was written, not deployed.
The script is now synced (196 lines) and `--help`-smoked on the box. The lesson is general: an edit
to a local script is not a change to the experiment until it reaches the machine that runs it.

**GATE — PASS.** Teacher d12 0.7040, the fourth identical reproduction. Params landed at **21.28M =
0.2502×** against the predicted 0.25×.

| arm | top1 | vs teacher | tower params | tower ms | tower speedup |
|---|---|---|---|---|---|
| teacher d12 | 0.7040 | — | 85.05M (1.000×) | 26.21 | 1.00× |
| naive (sliced, untrained) | 0.0860 | −61.80pp | 21.28M | 9.46 | chance |
| **distilled d6 × r=0.25** | **0.6540** | **−5.00pp → PARTIAL** | **21.28M (0.2502×)** | **9.53** | **2.75×** |

Final `rel_mse 0.2234 / cosine 0.8824`. Gain over naive +56.80pp.

**RETRACTED by part12 §44.** This section concluded "the knee is located, and it is between r=0.5 and r=0.25". At 50 epochs the same architecture reaches 0.6840 (−2.00pp, McNemar p=0.444), so the −5.00pp below was budget, not a capacity knee. Read the rest of this section with that correction applied.

| compound arm | params | tower | end-to-end | vs teacher |
|---|---|---|---|---|
| d6 × r=0.5 (§42) | 0.3335× | 2.40× | 1.61× | −1.20pp (parity) |
| d6 × r=0.25 (§43) | 0.2502× | 2.75× | 1.70× | −5.00pp |

The marginal step from r=0.5 to r=0.25 buys **0.083× of params and 0.35× of tower speed for 3.80pp**.
Compare §42's own marginal step, which bought 0.33× of params and 1.12× of speed for nothing. The
compound stops being free somewhere in between. **The r=0.5 compound remains the recommended drone
operating point**; r=0.25 is the arm to reach for only if the flash budget forces it.

**PARTIAL is the pre-registered "closes nothing" outcome, and that reading is honest here.** The fit
was still falling at ep25 (0.2234, against §42's 0.1915 and §40's 0.1570), and this arm ran ~0.03
behind §42's curve at every matched epoch — the same signature §39 showed before §40 falsified it as a
budget artifact. So −5.00pp is an upper bound on the loss at this budget, not the floor for r=0.25.
The difference from §39 is that this time the budget is d6's, r=0.5's *and* the compound's own parity
budget, so the result is interpretable as stated instead of uninterpretable.

**End-to-end, stated the same way as §42.** Prefill unchanged (13.77 → 14.03 ms), so end-to-end goes
39.98 → 23.56 ms = **1.70×**, against a tower figure of 2.75×. The extra tower speed over §42 barely
survives to end-to-end — 1.61× → 1.70× — because prefill is now the larger share of what is left.
**Squeezing the tower further has diminishing end-to-end return; the next real latency win has to come
from outside the tower** (roadmap item 5, the §32 prologue) or from the token count.

**HYPOTHESIS from §42 survives a fourth point.** rel_mse ordering across the four architectures is
0.1570 (width-only) < 0.1639 (depth-only) < 0.1915 (compound r=0.5) < 0.2234 (compound r=0.25), while
top-1 is +2.80 / parity / −1.20 / −5.00pp. Here the two orders happen to agree; §42 remains the point
where they did not. One agreeing point does not restore rel_mse as a cross-architecture ranker.

Results: `results/frontier/vlm_step023_mlp_width_r0.25_e25_t9352_n500_d6__5060ti_cuda.json`.

> **§43's knee is WITHDRAWN — see part 12 §44.** step027 reran this exact arm at 50 epochs and reached
> 0.6840 (−2.00pp, McNemar p=0.4437 vs the teacher). The −5.00pp below was a budget artifact, not a
> capacity limit. The param and timing numbers in this section stand; only the knee reading is retracted.

---

§44 onward continues in `VLM_TRAJECTORY_part12.md`.
