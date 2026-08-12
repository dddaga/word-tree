# VLM Trajectory — part 5 (from `vlm_step010`)

Continues `VLM_TRAJECTORY_part4.md` (§12–§15). Part 4 closed the token lever (§14: pruning is a bad
trade at this operating point — ~3 ms for −5.2pp) and left §11.7, quantizing the lookup tables, as
the highest-value remaining item because it is pure *memory*, which is the stubborn goal. §16
answers it and produces a −33.1%-bytes shipping configuration; **§17 then bounds that configuration**
— the byte win is unconditional, the accuracy parity is not, because depth truncation turns out to
cost fine spatial detail. Read §16 and §17 together; §16 alone overclaims.

## 16. `vlm_step010` — quantize the two lookup tables. **DONE — int8 SHIPS, int4 KILLED.**

Weight-only symmetric PTQ, no calibration set and no retraining: weights are quantized then
dequantized back to fp32, so **accuracy is measured exactly as an int kernel would produce it while
the byte saving stays analytic**. Full 2×4 grid {d12 teacher, d6 distilled} × {fp32, int8_row,
int4_row, int8_tensor} on one 3859-image eval set with per-image records, quantization applied once
per arm. mini_mps, ~1.6 h. `results/frontier/vlm_step010_quant_tables_d6_fp32-int8_row-int4_row-int8_tensor__mini_mps.json`.

**The block is 22.13%, not the 11.60% part 3 costed.** `tie_word_embeddings` is `False` on
SmolVLM-256M, so `embed_tokens` and `lm_head` are two *independent* 49280×576 tensors — 28,385,280
params each, **56,770,560 together, 22.13% of the model, a bigger block than depth truncation's
16.58%**. Both sit on the readout path (`vlm_eval.merge/forward/choose` call one or both).

**Validity anchors exact:** `d12_fp32` = 0.711 and `d6_fp32` = 0.713, reproducing step006 to three
decimals for the third independent time (step006, step009, step010).

| cell | top1 | agree | KL | vis_ms | tables_MB | model_MB |
|---|---|---|---|---|---|---|
| d12_fp32 | 0.711 | 1.000 | 0.000 | 64.9 | 227.1 | 1025.9 |
| d12_int8_row | 0.711 | 0.986 | 0.001 | 64.8 | 57.2 | 856.0 |
| d12_int4_row | 0.607 | 0.791 | 0.481 | 64.8 | 28.8 | 827.6 |
| d12_int8_tensor | 0.710 | 0.965 | 0.017 | 64.8 | 56.8 | 855.6 |
| d6_fp32 | 0.713 | 0.699 | 1.973 | 35.1 | 227.1 | 1025.9 |
| d6_int8_row | 0.708 | 0.698 | 1.987 | 35.1 | 57.2 | 856.0 |
| d6_int4_row | 0.633 | 0.618 | 2.362 | 35.1 | 28.8 | 827.6 |
| d6_int8_tensor | 0.706 | 0.700 | 1.969 | 35.1 | 56.8 | 855.6 |

Paired vs same-depth fp32, against the pre-registered rule (SHIP iff boot95 lower ≥ −2.0pp):

| arm | d12 Δ | boot95 | call | d6 Δ | boot95 | call |
|---|---|---|---|---|---|---|
| int8_row | +0.00pp | [−0.29, +0.29] | **SHIP** | −0.52pp | [−0.86, −0.18] | **SHIP** |
| int8_tensor | −0.05pp | [−0.49, +0.39] | **SHIP** | −0.67pp | [−1.11, −0.23] | **SHIP** |
| int4_row | −10.34pp | [−11.45, −9.25] | **KILLED** | −8.01pp | [−9.07, −7.00] | **KILLED** |

**int8 is free; int4 is not (CONFIRMED).** The weight-space pre-flight predicted the ordering
exactly: int8_row rel_rmse 0.0121 / row-cos 0.99993 → costs nothing measurable; int4_row 0.219 /
0.9775 → costs 8–10pp, an order of magnitude outside the ±2.0pp tolerance, at both depths, with
McNemar p < 1e-16. 4 bits is not enough for these tables *without* something the run did not try
(group-wise scales, or a calibrated/error-compensating method like GPTQ). The failure is honest and
one-variable: nothing but bit-width changed.

**Per-row scales are NOT load-bearing (CONFIRMED null).** `int8_tensor` was the one-variable
granularity control — identical bit-width, one scale for the whole tensor instead of one per
vocabulary row — and it lands within 0.15pp of `int8_row` at both depths (0.710 vs 0.711; 0.706 vs
0.708), both SHIP. It also stores 0.4 MB less and needs a simpler kernel with one less lookup. So
**the drone should use tensor-scale int8**, and the 2.7×-worse weight-space rel_rmse (0.0327 vs
0.0121) is a distinction without a task-level difference — a standing reminder that weight-space
fidelity is a proxy, not the goal metric.

**Compounding with depth (pre-registered double difference, ORTHOGONAL iff CI ⊆ ±2.0pp):**
int8_row I = −0.52pp, boot95 [−0.96, −0.08] → **ORTHOGONAL, levers compound**; int8_tensor
I = −0.62pp [−1.24, −0.03] → **ORTHOGONAL**. Read honestly: both CIs *exclude zero*, so the
truncated tower does pay marginally more for quantization than the teacher does — a real, detectable
sub-additivity of about half a point, which is well inside the registered tolerance and therefore
ORTHOGONAL **by the rule as written, not by rounding**. int4_row I = +2.33pp [+0.93, +3.71] →
INCONCLUSIVE (lower bound does not clear +2.0pp); its point estimate says quantization cost the
*teacher* more, which is moot given int4 is killed at both depths anyway.

### 16.1 The shipping configuration, and the drone number

Stacking d6 (the §13/part 3 lever) with int8 tables (this one), against the untouched fp32 teacher,
paired on the same 3859 images:

| config | top1 | Δ vs teacher | boot95 | McNemar p | bytes | Δ bytes |
|---|---|---|---|---|---|---|
| d12_fp32 (teacher) | 0.7108 | — | — | — | 1025.9 MB | — |
| d6_fp32 | 0.7129 | +0.21pp | [−1.27, +1.74] | 0.813 | 855.8 MB | −16.6% |
| **d6_int8_row** | **0.7077** | **−0.31pp** | **[−1.76, +1.19]** | **0.708** | **685.9 MB** | **−33.1%** |
| d6_int8_tensor | 0.7061 | −0.47pp | [−1.94, +1.04] | 0.564 | 685.5 MB | −33.1% |

**One third of the model's bytes deleted at −0.31pp, statistically indistinguishable from the
teacher (p = 0.71).** 42,527,232 params removed by truncation (16.58%, 170.1 MB) plus 56,770,560
params stored at 8 bits instead of 32 (169.9 MB). Bytes measured, not proxied (meditation-005).

> **SCOPE, added after `vlm_step011` (§17) — read this before quoting the number above.** Every cell
> in this table was measured on **whole-object** images. §17 shows the accuracy half of this claim
> does **not** survive when the target is small: at f=0.25 the d6 tower loses 46.07pp against the
> teacher's 21.66pp. The **byte** half is unconditional — −33.1% is arithmetic on the weights and
> holds for any input. What is scoped is "at −0.31pp": that holds **only where the object fills the
> frame**. The two levers separate cleanly under §17, and the split is the useful part: **int8 on
> the tables is input-independent** (it touches the text-side lookups, not the vision tower), while
> **depth truncation is where the scale sensitivity lives**. A drone facing small distant targets
> should treat d12 + int8 = 856.0 MB (−16.6%, Δ +0.00pp) as the unconditional configuration and d6
> as conditional on target size.

**Timing did not move and no timing claim is made:** `vis_ms` is 64.8–64.9 at d12 and 35.1 at d6
across every arm, `prefill_ms` 14.4 everywhere. Fake-quant dequantizes to fp32 before the matmul, so
this is the expected null — the byte saving is real, the *speed* of a true int8 kernel is untested.

**Caveat that survives everything:** `agree`(d6_int8_row, d12_fp32) = 0.698, essentially unchanged
from d6_fp32's 0.699. Quantization does not alter the student's relationship to the teacher — but
that relationship was already only ~0.70, and 0.499 on the §14 measurement against the *shipping*
comparison. Matched top-1 is parity, not equivalence. The drone gets a model that is as accurate as
the teacher on this task, not one that behaves like it.

## 17. `vlm_step011` — does parity survive when the object is SMALL? **DONE — NO. SUB-ADDITIVE.**

The §11.6 fine-detail control, and the most consequential run in this trajectory. One variable:
object scale. Each val image is resized to a fraction f of its own size and pasted centred on a gray
canvas of the **original** size, so the object occupies f² of the pixels while the processor's input
resolution never changes; f = 1.0 is the untouched image. Grid {d12, d6} × f {1.0, 0.5, 0.25}, one
3859-image eval set, per-image records. mini_mps, ~12 min.
`results/frontier/vlm_step011_fine_detail_d6_f100-f50-f25__mini_mps.json`.

**Validity anchors exact for the fourth independent time:** `d12_f100` = 0.711, `d6_f100` = 0.713.

| cell | top1 | agree | KL | | cell | top1 | agree | KL |
|---|---|---|---|---|---|---|---|---|
| d12_f100 | 0.711 | 1.000 | 0.000 | | d6_f100 | 0.713 | 0.699 | 1.973 |
| d12_f50 | 0.672 | 0.705 | 1.498 | | d6_f50 | **0.546** | 0.565 | 2.771 |
| d12_f25 | 0.494 | 0.554 | 2.673 | | d6_f25 | **0.252** | 0.320 | 4.449 |

**The confound was stated before the run and it is why the double difference is the registered
quantity:** f < 1 changes object scale *and* introduces OOD uniform padding. The padding is applied
**identically at both depths**, so it contaminates the main effect but **differences out of the
interaction**. Main effects are therefore descriptive only: d12 −3.89pp / −21.66pp, d6 −16.66pp /
−46.07pp at f = 0.5 / 0.25, McNemar p < 1e-5 everywhere.

**THE CALL — I(f) = [d6 shrink cost] − [d12 shrink cost], SUB-ADDITIVE iff upper < −2.0pp:**

| f | I | boot95 | verdict |
|---|---|---|---|
| 0.50 | **−12.78pp** | [−14.77, −10.75] | **SUB-ADDITIVE** |
| 0.25 | **−24.41pp** | [−26.64, −22.21] | **SUB-ADDITIVE** |

Both sit 5–11× outside the tolerance and neither CI comes near it. **CONFIRMED: depth truncation
destroys fine spatial detail, and d6's teacher parity is an artefact of whole-object framing.**

**The additive-scale objection is pre-empted.** A double difference on proportions can manufacture
an interaction if one arm is near a floor or the harder condition simply amplifies all gaps. Neither
applies: in *ratio* terms d12 retains 69.5% of its f=1.0 accuracy at f=0.25 while d6 retains 35.3%,
so the truncated tower loses about twice as much on either scale; and d6's 0.252 is still well above
the 10-way 0.10 chance floor, so nothing is saturating. `agree`(d6, d12) falls 0.699 → 0.565 → 0.320
— the towers do not merely differ in accuracy, they **diverge further the smaller the target gets**.

**Two mechanisms remain live and this run does not separate them — both HYPOTHESIS:**
(a) **CAPACITY** — the six deleted layers are the ones carrying fine spatial detail, and no retraining
recovers it. (b) **DISTRIBUTION** — d6 was distilled exclusively on whole-object Imagenette and has
never seen a small target, so scale-augmented distillation recovers most of it. They have opposite
consequences for the drone, and (b) is cheaply testable: re-distil d6 with random-scale augmentation
and re-run this exact grid. Note also that **the teacher degrades badly too** (0.711 → 0.494): small
objects are a weakness of SmolVLM-256M in general; truncation makes it roughly twice as bad.

## 18. Next steps

1. **Scale-augmented re-distillation — LAUNCHED as `vlm_step012` (mini_mps, 2026-07-29).** The
   decisive follow-up to §17 and the top item: it separates CAPACITY from DISTRIBUTION and therefore
   decides whether the d6 half of the drone recipe is recoverable at all. Same distillation harness,
   random-scale augmentation on the train images (scale ~ U(0.2, 0.9), half left untouched, same
   gray canvas §17 tests with), then re-run the §17 grid unchanged with `--ckpt` pointed at the new
   student. If I(f) shrinks inside ±2.0pp, the drone claim is restored; if it does not move, depth
   truncation is *confirmed* to cost fine detail irrecoverably and the drone ships d12 + int8 only.

   Three design points worth carrying forward, because each is a way the run could have been
   silently wrong:

   * **The control is the existing step004 d6 checkpoint, not a freshly trained plain arm** — which
     halves the compute but only holds if the train set is bit-identical. `sample_images` floors to
     `per = n // 10` per class, so step004's literal `--n_train 9469` yields **9352** images while
     the round 10000 yields 9469. step012 defaults to 9469 and was verified to return exactly 9352.
     Requesting the round number would have trained on a 117-image-different set and quietly made
     this a two-variable run. It also runs on mini_mps, the same physical box as the control.
   * **The augmentation must be deterministic per path.** Teacher targets are cached once while the
     student is re-encoded every epoch, so a per-call random transform trains the student to match
     teacher features computed from a *different view* — a corrupted objective that would still
     produce a plausible-looking loss curve. The scale is seeded from the file name, and the `tf=`
     hook added to `vlm_distill.tower_feats/cache_teacher/train_student` forwards it to the teacher
     so both see the identical image.
   * **`rel_mse` is not comparable to step004's** — the targets are features of *shrunk* images, so
     a higher loss here does not mean a worse student. The verdict is the §17 grid, never the curve.

   Pre-registered, with the second criterion being the one most likely to bite: `d6_f100` must stay
   within 2.0pp of the teacher's 0.711, because augmentation that buys small-object skill by
   spending whole-object parity has not helped — that parity *is* the original claim. And the
   asymmetry is stated before the result: **a NULL is confound-free and decisive for CAPACITY** (the
   training distribution would have matched the test manipulation exactly and still failed), while a
   POSITIVE is confounded between real scale-invariance and mere adaptation to OOD gray padding and
   needs a follow-up on genuinely small objects before the drone claim is un-scoped.
2. **int4 with group-wise scales** — the only live route to the extra 28.4 MB, and it is a genuine
   open question rather than a retry: §16 killed *per-row* int4, not all 4-bit schemes. Cheap
   (eval-only, the harness exists). Queue only if the extra 2.8% of bytes is worth it — it is the
   smallest remaining block relative to its risk.
3. **A real int8 kernel** would convert the byte win into a possible latency/energy win. Untested
   and unclaimed; needs `torch.ao` or a bitsandbytes path inside the private `vlm_libs` shadow.
4. Unchanged from part 4 §15: **do NOT** attack VLM FFNs or the connector.
