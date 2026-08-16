# The feature hint — the mechanism that closed the teacher gap (seg_step010–011)

Split out of `seg_distillation.md` at the 200-line limit. Same setup: teacher **E2** (3.2324 GMAC,
2,195,656 par, 0.2650 @60ep seed42, cached) → student **E7** (0.8757 GMAC, 851,816 par, inside the
1 GMAC/frame drone ceiling). Logit-KD background and the ±2pp noise floor are in
`seg_distillation.md` and `seg_coarse_heatmap.md`.

---

## seg_step010 — feature hint on the sliced init (T0, 10 seeds, mini_mps, 11 cells, 2979s, 2026-08-14)

α is flat (step008), so the logit channel is saturated; the next orthogonal channel is the
*features*. A FitNets hint normally needs a learned projection (student 128ch vs teacher 256ch) —
here it does **not**, because E7 is built by *slicing* E2's convs (seg_step003), so student channel
j **is** teacher channel j by construction. Hint = plain `MSE(s_enc, t_enc[:, :128])`, β=1:
**zero extra parameters, no projection to confound the result.** Teacher features are recomputed
per batch under `no_grad` (caching 2000×256×16×16 fp32 = 0.5 GB is worse than a 3.2 GMAC forward);
cost 294s/cell vs 175s (+68%).

| cell | GMAC | params | mAP (10 seeds) |
|---|---|---|---|
| teacher E2 @60ep | 3.2324 | 2,195,656 | 0.2650 (seed 42) |
| S0 student, no KD | **0.8757** | **851,816** | 0.2125 ±0.0162 |
| a2 student, logit KD | **0.8757** | **851,816** | 0.2276 ±0.0121 |
| **H1 student, KD + hint** | **0.8757** | **851,816** | **0.2644 ±0.0056** |

Paired H1−a2, seeds 42–51: **+4.89 / +3.94 / +3.78 / +1.88 / +3.71 / +3.26 / +2.67 / +5.43 /
+4.41 / +2.74**.

**CONFIRMED — the feature hint is worth +3.67pp over logit KD** (n=10 paired, sd 1.08, sem 0.34,
**t=10.77, df=9, p<1e-5, 10/10 positive**). Versus the no-KD control: **+5.18pp** (sem 0.59,
t=8.74, 10/10). The largest effect measured anywhere in the seg line, and the first to clear the
±2pp floor on its own rather than by seed accumulation — exactly the kind of mechanism the
seg_step009 cost note said to go looking for.

**CONFIRMED — the student reaches the teacher.** H1 = 0.2644 vs teacher 0.2650: the 5.2pp deficit
is **closed, not halved**, at 3.7× fewer MACs, 2.6× fewer params, and **zero extra student
parameters** — the hint is a training-time loss term only, so inference cost is byte-identical to
S0. This is the drone result the line was after: **1 GMAC-class student at teacher accuracy.**

**CONFIRMED — the hint also removes the seed noise.** H1 spread is ±0.0056 vs ±0.0121 (logit KD)
and ±0.0162 (no KD) — a 2.9× tighter sd. The ±2pp floor that dominated steps 006–009 is a property
of *under-constrained* training on this task, not of the task itself; constraining the features
collapses it. Practical consequence: future arms built on the hint need far fewer seeds.

**CONFIRMED — not a code-path confound.** The H0 reproduction cell (logit KD only, run through this
script's live-teacher path) gives seed42 = 0.2181, bit-matching the seg_step008 a2 seed42 cell from
the cached-logits path. So H1 is comparable to the existing a2/S0 cells and no control needed
re-running.

**HYPOTHESIS — why it works.** The sliced init starts the student *on* the teacher's features and
training drifts off them; the hint holds channel correspondence in place. Untested alternative: the
MSE term is acting as a generic regulariser and the channel *alignment* is irrelevant. Cheap
discriminator queued as seg_step011: hint against a channel-**shuffled** teacher (same statistics,
alignment destroyed) plus a β sweep. If shuffled ties H1, the alignment story is dead and the
result is "add a feature-MSE regulariser"; if shuffled falls back to a2, alignment is the mechanism.

Script: `scripts/seg/seg_step010_feature_hint.py`.
Results: `results/seg/seg_step010_H1_b1_seed4*__mini_mps.json`, `..._seed5[01]*`, H0 repro cell.

---
## seg_step011 — what the hint actually is (T0, 5 seeds × 3 arms, mini_mps, 15 cells, 4475s, 2026-08-14)

seg_step010 left one explanation open: is the +3.67pp from *channel alignment* (the sliced init
making student channel j = teacher channel j), or is `MSE(features)` just a generic regulariser?
Pre-registered in the queue before launch, with the decision rule written down in advance:
*"if shuffled ties H1, alignment is dead and the claim becomes 'feature-MSE regularises'; if
shuffled falls back toward a2 (0.2276), alignment is the mechanism."*

Three arms, all paired against the existing β=1 aligned cells on the SAME seeds 42–46. The shuffled
arm hints against a **fixed random permutation** of the teacher's first 128 channels — identical
target statistics, channel correspondence destroyed. The permutation is seeded independently of the
run seed, so every seed of that arm sees the same misalignment.

| arm | mAP (seeds 42–46) | paired vs β=1 aligned |
|---|---|---|
| **β=1 aligned** (step010) | 0.2649 ±0.0081 | — |
| **β=0.25 aligned** | 0.2675 ±0.0097 | +0.26pp, sd 0.99, t=+0.58, 3/5 |
| **β=4 aligned** | 0.2602 ±0.0069 | −0.47pp, sd 1.01, t=−1.05, 1/5 |
| **β=1 SHUFFLED** | 0.2596 ±0.0121 | −0.53pp, sd 1.48, t=−0.80, 2/5 |
| a2 logit KD (anchor) | 0.2285 ±0.0134 | −3.64pp, sem 0.49, t=−7.45, 0/5 |
| S0 no KD (anchor) | 0.2141 ±0.0136 | −5.08pp, sem 0.88, t=−5.80, 0/5 |

**CONFIRMED — channel alignment is NOT the mechanism.** The pre-registered "shuffled ties H1"
branch fires: −0.53pp with t=−0.80 and 2/5 positive is a null, while shuffled−a2 is **+3.11pp**
(sem 0.77, t=4.02, **5/5 positive**). Destroying the correspondence the sliced init was supposed to
provide costs, at most, a fraction of the effect and is unreadable at n=5. The seg_step010
HYPOTHESIS ("the hint holds channel correspondence in place") is **falsified**; the surviving
explanation is that matching the *distribution* of teacher features — any fixed teacher-derived
target — is what constrains the student. **The result is untouched; only its explanation changed.**

**CONFIRMED — the hint is flat in β over 16×, exactly as α was.** 0.2675 / 0.2649 / 0.2602 across
β ∈ {0.25, 1, 4}: a 0.73pp spread over a 16× weight range, with no monotone trend and every pairwise
contrast inside the noise. This is the *second* mechanism in this line with no strength knob
(seg_step008 found the same for α). Practical read: **the hint needs no tuning** — pick β=1 and move
on. It also means the effect is not a delicate loss-balance artifact, which is the usual failure mode
for auxiliary losses.

**Robustness, stated as the paper would state it:** all four hint variants (β 0.25/1/4 aligned, plus
shuffled) sit in **0.2596–0.2675**, a 0.79pp band, and every one of them beats logit KD (0.2285) by
+3.1 to +3.9pp and no-KD (0.2141) by +4.6 to +5.3pp. The effect is insensitive to both knobs it has.
A mechanism that survives a 16× weight sweep *and* deliberate target scrambling is not fragile.

**Method note — the pre-registration did its job.** The shuffled arm was written into the queue with
its decision rule *before* it ran, so the null could not be re-read as "alignment helps a little."
Worth repeating for every mechanism claim in this line.

**Bug caught pre-launch (recorded so it is not repeated):** the first shuffled implementation named
its channel permutation `perm`, colliding with `fit()`'s per-epoch batch order, and indexed 128
channels with 2000 batch indices — all 5 cells died in 17s. `--smoke_test` did **not** catch it
because that path never enters `fit()`. **Loss-path changes must be verified with a real
`--epochs 1` run, not a smoke test.**

Script: `scripts/seg/seg_step010_feature_hint.py --shuffle_hint`.
Results: `results/seg/seg_step010_H2_shuf_seed4[2-6]*`, `..._H1_b{025,4}_seed4[2-6]*__mini_mps.json`.

---

## seg_step012 — the hint without the sliced init (mini_mps, 10 cells, 2026-08-14)

Step011's shuffled control permutes channels that still *came from* the teacher's weights, so it
cannot answer whether the teacher-derived init is needed at all. Arm **E8** is E7's *exact*
architecture (width 0.5, 3 blocks — verified byte-identical at 0.8757 GMAC / 851,816 par) with a
**random init**, which removes the teacher from the student's weights entirely. Only the hint varies.

| arm (E8, random init) | mAP, seeds 42–46 |
|---|---|
| logit KD only | 0.2056 ±0.0068 |
| **+ β=1 hint** | **0.2551 ±0.0046** |

**CONFIRMED — the feature hint is architecture-general; the sliced init is not load-bearing for it.**
Paired Δ = **+4.95pp** (5.32 / 4.88 / 4.83 / 4.05 / 5.67), sd 0.61, sem 0.27, **t=18.2, 5/5
positive**. The pre-registered first branch fires. Note the honest deviation: the rule asked for
"within ~1pp of the +3.67pp sliced result" and the measured effect is **1.28pp larger**, i.e.
marginally outside the band on the *bigger* side — the opposite of the failure mode the rule was
written to catch. Read conservatively, the claim is "at least as large without the sliced init."

Two consequences worth stating in the paper:

1. **The hint is worth more than the ImageNet init it was supposed to depend on.** A *random-init*
   student with the hint (0.2551) beats a *sliced* student on logit KD alone (0.2285) by +2.66pp.
   The step003 sliced-init result (+4.26pp) was the biggest single lever in this line until now.
2. **Init and hint are additive, not redundant.** Sliced + hint (0.2644, step010) still edges
   random + hint (0.2551) by ~0.9pp — around the ±1pp resolution here, so "small but same sign".
   Best configuration remains sliced init **and** hint, which is also the free one.

Together with step011 this closes the mechanism question: the hint does not work by preserving
channel identity (shuffling is a null) and does not work by exploiting a teacher-derived init
(random init reproduces it). What survives is the plain reading — **regressing the student's encoder
onto any fixed teacher-derived feature target constrains it far more than the class logits can**,
at zero parameter and zero inference cost.

Script: `scripts/seg/seg_step010_feature_hint.py --student_arm E8`; arm E8 in `scripts/seg/seg_encoders.py`.
Results: `results/seg/seg_step010_H{0,1}_e8{kd,hint}_seed4[2-6]__mini_mps.json`.

---

## seg_step014 — iterated distillation: does a better teacher help? (mini_mps, 5 cells, 1291s, 2026-08-14)

seg_step013 found the student *beats its own teacher*: E10+hint scores 0.2795 against E2's 0.2650 at
1.8653 vs 3.2324 GMAC. That makes a free lever available — supervise the drone-budget E7 student with
the better E10 student instead of E2. It costs **zero extra inference MACs**, since the teacher only
exists at training time. Pre-registered before launch. The E10 seed-42 checkpoint was saved with the
new `--save_ckpt` flag (best epoch, not last) and reported best_mAP=0.2884, identical to its step013
cell — so the checkpoint is that exact model.

| student E7 (0.8757G) | teacher | teacher mAP | student mAP, seeds 42–46 |
|---|---|---|---|
| control (step010 β=1) | E2, 3.2324G | 0.2650 | 0.2649 |
| **R2 iterated** | **E10+hint, 1.8653G** | **0.2884** | **0.2659 ±0.0046** |

**CONFIRMED NULL — the feature hint saturates at the student's capacity; teacher quality is not the
binding constraint.** Paired Δ = **+0.10pp** (+0.41 / −0.38 / +0.99 / −0.25 / −0.27), sd 0.59,
sem 0.26, **t=+0.38, 2/5 positive**. Pre-registered branch (b) fires. The strength of the null is in
the size of the input it ignored: the R2 teacher is **+2.34pp better** than the control teacher and
that bought the student **+0.10pp**, a transfer ratio of ~4%. Iteration does not compound; there is
no round three.

**This reclassifies seg_step013's law.** The ≈1.0–1.4pp per octave of MACs is a **student-capacity
law, not a supervision law**. Supervision is already saturated at E7's width, so nothing on the
teacher side can move the drone-budget number — **MACs are the only remaining lever in this line**,
and the 1 GMAC ceiling therefore costs the confirmed 1.45pp with no way to buy it back for free.

Read together with steps 011/012, the hint's mechanism is now bounded from three sides: it does not
work by channel identity (shuffling is a null), does not need a teacher-derived init (random init
reproduces it), and does not scale with teacher quality (this null). What is left is that *any* fixed
teacher-derived feature target saturates the student's encoder, and the student's own width sets the
ceiling on what that target can deliver.

**Caveat, stated because it cuts the honest way:** the R2 teacher is E10 *seed 42*, which at 0.2884
is E10's best seed against a 0.2795 arm mean — so this compares seed-42-teacher to seed-42-teacher
(the control's E2 is also single-seed-42), and if anything **overstates** the teacher advantage,
which only strengthens the null.

Script: `scripts/seg/seg_step010_feature_hint.py --save_ckpt` (new flag), then `--teacher_arm E10`.
Results: `results/seg/seg_step010_H1_iter2_seed4[2-6]__mini_mps.json`;
checkpoint `results/seg/teacher_E10_e60_seed42__mini_mps.pt`.
