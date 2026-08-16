# Distilling the drone student (seg_step007–010)

Split out of `seg_coarse_heatmap.md` at the 200-line limit. Setup common to every step here:
teacher **E2** (3.2324 GMAC, 2,195,656 par, 0.2650 @60ep seed42, cached to disk and reused by every
student seed) → student **E7** (0.8757 GMAC, 851,816 par — inside the 1 GMAC/frame drone ceiling).
Encoder arms are defined in `seg_encoder_levers.md`; the noise floor is in `seg_coarse_heatmap.md`.

---

## seg_step007 — E2 → E7 soft-KD distillation (T0, 5 seeds, mini_mps, 10 cells, 2097s, 2026-08-14)

seg_step004 confirmed the E2−E7 gap is real and not budget-limited, which promoted distillation.
Teacher E2 trained once at 60ep (seed 42) and cached; every student seed reuses it. Loss =
BCE(hard) + α·T²·BCE_with_logits(student/T, sigmoid(teacher/T)) + λ_decor·decor, α=0.5, T=2.0.
Sigmoid-KD, not softmax: the head is multi-label (one cell can hold several event classes).
S0 = same script, same seeds, no KD term — the control shares the exact code path.

| cell | GMAC | params | mAP (5 seeds) |
|---|---|---|---|
| teacher E2 @60ep | 3.2324 | 2,195,656 | 0.2650 (seed 42) |
| S0 student, no KD | **0.8757** | **851,816** | 0.2141 ±0.0136 |
| S1 student, soft-KD | **0.8757** | **851,816** | 0.2307 ±0.0151 |

Paired S1−S0, same seed: **+2.99 / +2.98 / +2.71 / +1.09 / −1.44 → mean +1.67 ±1.91pp**.

**HYPOTHESIS (not confirmed): soft-KD is worth ~+1.7pp on the drone student.** The mean is positive
and 4/5 pairs are positive, but the spread (±1.91pp) is the full ±2pp run-to-run floor and one pair
is negative. This is *not* a readable win at 5 seeds. To settle it needs ~15 seeds, or a mechanism
change large enough to clear the floor (higher α, a feature-level term, or a stronger teacher).

**Method lesson, second confirmation of the floor.** At n=3 this looked like a clean, consistent
+2.9pp; the two remaining seeds took it to +1.67 ±1.91. Exactly the failure mode seg_step006
predicted. **Do not read a seg delta before all seeds are in.**

**CONFIRMED — the new script is not a confound (reproduction check).** The cached teacher reproduces
seg_step004 E2 seed42 exactly (0.2650), and all three overlapping S0 controls bit-match their
seg_step004 E7 twins (0.1960 / 0.2030 / 0.2246, same best_ep). seg_eval.py's factored helpers are
behaviour-identical to the inlined copies in step001/002.

**Pareto (the point of the line):** the student sits at **0.8757 GMAC / 851,816 params — inside the
1 GMAC/frame drone ceiling**, 3.7× cheaper than the teacher and 5.6× cheaper than the E0 reference,
at 0.2307 vs teacher 0.2650 (−3.4pp). KD does not close the gap; it *may* have halved it.

Script: `scripts/seg/seg_step007_distill.py`, helpers `scripts/seg/seg_eval.py`.
Results: `results/seg/seg_step007_S{0,1}_t0kd_seed4*__mini_mps.json`.

---

## seg_step008 — KD strength sweep (T0, α ∈ {2,8} × 5 seeds, mini_mps, 10 cells, 1819s, 2026-08-14)

If soft-KD is real but sub-floor, the obvious move is more of it. It is not.

| α | student mAP (5 seeds) | paired vs same-seed S0 |
|---|---|---|
| 0.5 | 0.2307 ±0.0151 | +1.67 ±1.91 |
| 2.0 | 0.2285 ±0.0134 | +1.44 ±1.74 |
| 8.0 | 0.2292 ±0.0066 | +1.51 ±1.59 |
| 0 (control) | 0.2141 ±0.0136 | — |

**CONFIRMED — the KD effect is flat in α over 16×.** No strength knob to turn: 16× more KD gradient
buys nothing. Whatever the teacher transfers, it transfers fully at α=0.5 and saturates there.

**CONFIRMED — the ±2pp floor is a SEED effect, not a run effect.** Seed 42 is good for KD at every α
(+2.99 / +2.21 / +3.96); seed 46 is bad at every α (−1.44 / +0.37 / −0.16). Because α is flat, the
three arms are replicates — and averaging replicates *within* a seed does not shrink the spread.
**Consequence for the whole seg line: re-running a seed is wasted compute; only new seeds buy
significance.** This is the mechanism behind seg_step006's floor.

**HYPOTHESIS — soft-KD ≈ +1.5pp.** Pooled 15 pairs: +1.54pp, 12/15 positive. Controls are shared
across arms so honest n=5: per-seed means +3.05 / +2.73 / +0.94 / +1.38 / −0.41 = **+1.54 ±1.41pp,
t=2.45, df=4, p≈0.07**. Extended to n=10 in seg_step009.

Results: `results/seg/seg_step007_S1_a{2,8}_seed4*__mini_mps.json`.

---

## seg_step009 — KD seed extension to n=10 (T0, seeds 47–51, mini_mps, 10 cells, 1798s, 2026-08-14)

seg_step008 CONFIRMED that only *new seeds* buy significance, so this run added five (S1 α=2.0 +
matched S0 controls) and pooled them with seeds 42–46. Pre-registered in the queue before launch:
*"predicted sem ~0.6pp → p≈0.03 if the effect holds."* Measured sem 0.58, p≈0.029 — prediction met.

| cell | GMAC | params | mAP (10 seeds) |
|---|---|---|---|
| teacher E2 @60ep | 3.2324 | 2,195,656 | 0.2650 (seed 42) |
| S0 student, no KD | **0.8757** | **851,816** | 0.2125 ±0.0162 |
| S1 student, soft-KD α=2 | **0.8757** | **851,816** | 0.2276 ±0.0121 |

Paired S1−S0, seeds 42–51: **+2.21 / +3.20 / −1.05 / +2.49 / +0.37 / +3.51 / +1.71 / −1.48 /
+3.60 / +0.55**.

**CONFIRMED — soft-KD is worth +1.51pp on the drone student** (n=10 paired, sd 1.84, sem 0.58,
t=2.60, df=9, **p≈0.029**, 8/10 positive). This upgrades the seg_step007 (+1.67 ±1.91, n=5) and
seg_step008 (+1.54 ±1.41, p≈0.07, n=5) HYPOTHESIS tags — both are now resolved by this result, and
their point estimates agree with it to within 0.2pp. First statistically significant effect in the
seg line, and the first mechanism to move mAP at all (decorrelation is a confirmed null).

**Note on the cost of the ±2pp floor:** this single +1.5pp fact cost 30 student cells (~95 min).
That is the price of every future seg claim of this size — budget accordingly, or look for
mechanisms expected to clear 2pp on their own.

**Pareto:** student **0.8757 GMAC / 851,816 params inside the 1 GMAC drone ceiling**, 3.7× cheaper
than the teacher, at 0.2276 vs teacher 0.2650. KD **halves** the gap (−3.4 → −3.7pp raw, but the
no-KD student sits at 0.2125, so KD recovers 1.5 of the 5.2pp deficit). It does not close it.

Results: `results/seg/seg_step007_S1_a2_seed4[7-9]*__mini_mps.json`, `..._seed5[01]*`, S0 twins.

---

## Feature hint (seg_step010–011) → `seg_feature_hint.md`

Moved out at the 200-line limit. Summary: a projection-free `MSE(s_enc, t_enc[:, :128])` hint,
**zero extra parameters and zero extra inference cost**, is **CONFIRMED +3.67pp over logit KD**
(n=10 paired, t=10.77, p<1e-5, 10/10 positive) and **closes the teacher gap entirely** — student
0.2644 ±0.0056 vs teacher 0.2650, at 3.7× fewer MACs and 2.6× fewer params. It also collapses the
±2pp seed floor 2.9×. seg_step011 then **falsified the channel-alignment explanation** (shuffled
target = −0.53pp, t=−0.80, a null) and CONFIRMED the hint is **flat in β over 16×** — no knob, and
robust to deliberate target scrambling. The effect stands; only its mechanism story changed.
