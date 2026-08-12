# Visual-LLM Trajectory — part 3: half the tower deleted, at parity

Continues [`VLM_TRAJECTORY_part2.md`](VLM_TRAJECTORY_part2.md), which ends at `vlm_step005`
(logit-KD KILLED) with one open HYPOTHESIS: *d6 reaches teacher parity at full budget.* Part 3 =
that run, the significance check it demanded, and the recipe as it now stands.

**Read §8 through §9 as one unit.** §8 is the n=200 run and reports a +5.5pp overtake; §9 is the
n=3859 recheck and **kills the overtake** (p=0.813). The HYPOTHESIS in part 2 §5 closed exactly as
written — at parity, not above it. §8 is preserved as-run rather than rewritten, because the size of
the correction is itself the lesson: **an n=200 eval moved a headline by 5.3pp of pure noise.**

## 8. `vlm_step004` at FULL budget — the n=200 result (SUPERSEDED by §9)

`--depth 6 --n_train 9469 --n_eval 200 --epochs 25 --batch 8 --lr 2e-4`, mini_mps, ~7.6 h. The
class-balanced sampler floors to **9352** train images (935/wnid), so this is 4.7× the T0 data at
2.5× the epochs = **11.7× the updates** of the 2000×10 row.

| arm | params | %save | top1 | same | agree | KL | vis_ms | total | speed |
|---|---|---|---|---|---|---|---|---|---|
| teacher_d12 | 256,484,928 | 0.00 | 0.685 | 1.000 | 1.000 | 0.000 | 65.7 | 80.1 | 1.00× |
| naive_d6 | 213,957,696 | 16.58 | 0.135 | 0.160 | 0.000 | 10.823 | 35.4 | 49.7 | 1.61× |
| **distil_d6** | 213,957,696 | 16.58 | **0.740** | 0.705 | 0.475 | **1.986** | 35.5 | 49.8 | 1.61× |

**As measured at n=200, the student passed the teacher by 5.5pp**, at 16.58% fewer parameters and
1.61× faster vision. Same 200 images for both arms, so the comparison is paired — pairing was never
the weakness; *n* was. §9 re-ran the identical checkpoint at n=3859 and the delta collapsed to
+0.2pp (p=0.813). **Do not cite the 0.740/0.685 pair.** Both numbers were wrong in opposite
directions: the teacher's true value is 0.711 and the student's is 0.713.

### Budget scaling, all three runs

| d6 budget | updates | rel_mse | cosine | top1 | agree | KL | teacher top1 | gap |
|---|---|---|---|---|---|---|---|---|
| 2000 × 10 ep | 1× | 0.2573 | 0.8634 | 0.560 | 0.420 | 2.512 | 0.740 (n=50) | −18pp |
| 6000 × 15 ep | 4.5× | 0.2011 | 0.8951 | 0.680 | 0.500 | 1.988 | 0.740 (n=50) | −6pp |
| 9352 × 25 ep (n=200) | 11.7× | **0.1639** | **0.9154** | 0.740 | 0.475 | 1.986 | 0.685 (n=200) | +5.5pp |
| **9352 × 25 ep (n=3859)** | **11.7×** | — | — | **0.713** | 0.499 | 1.973 | 0.711 (n=3859) | **+0.2pp (n.s.)** |

Every row's gap is measured against a **different eval set** (n=50, n=200, n=3859) and they are not
interchangeable; only the within-row pairing is valid. The last two rows are the *same checkpoint*
scored twice — the 5.3pp swing between them is the measurement, not the model. The fit is *still*
descending at ep25 (rel_mse 0.1665 → 0.1639, cosine 0.9140 → 0.9154, no flattening in 25 epochs) — three
budgets in, distillation budget has **never once been the saturating variable**.

### What the student is: a different tower, not a copy — and no label ever reached it

**The student never saw a label.** Its only supervision is the teacher's post-connector features
under relative MSE; the 10-way class labels enter *nowhere* in `train_student`, and the eval labels
come from a disjoint val split. That framing was written to argue an overtake could not be leakage.
The overtake is gone, but the observation it rested on **survives §9 and got sharper**: at n=3859
the student and teacher `agree` on only **0.499** of images — the student picks a different label
half the time — while scoring 0.713 vs 0.711. It is not a compressed copy of the teacher; it is a
differently-wrong network of equal skill.

> ~~**HYPOTHESIS — capacity-limited feature regression acts as a denoiser.**~~ **DROPPED at §9.** It
> was posited to explain a +5.5pp gain that does not exist. §9's discordance split is the direct
> refutation: b=440 student-only-right vs c=432 teacher-only-right — the student trades away almost
> exactly as many correct teacher answers as it adds. A denoiser predicts b ≫ c. Half the tower can
> be deleted **without** the remaining half being better; that is the finding.

**What this does buy — evidence about the method.** 872/3859 images (22.6%) discordant, split
440/432: two individually-71% networks disagreeing on 23% of inputs. **Feature distillation does not
converge the student onto the teacher's decision function** — a claim about the method, not this
checkpoint. (Also the textbook ensembling precondition, not pursued: two towers costs the params
back.)

**Standing caveat — this is unlabelled in-domain adaptation.** The student saw 9352 imagenette
images; the teacher saw none. The question §8 asked of the *gain* now transfers to the *parity*: it
may hold only on the distribution the distillation data came from. That is the load-bearing question
for the drone, and §11.1 / `vlm_step007` is aimed at exactly it.

## 9. `vlm_step006` — is the overtake real? **DONE — NO. Parity, not overtake.**

`scripts/frontier/vlm_step006_eval_scale.py`. **No training compute**: loads the saved student
checkpoint (`_d6_25ep9k__mini_mps.pt`, added to step004 after the 6000×15 student was lost) and
re-runs eval only, at `--n_eval 3900` → **3859** balanced val images (the `n//10` per-wnid floor),
19× the eval set the claim was made on, with **per-image records kept**.

The aggregate cannot decide this. +5.5pp at n=200 is 11 net images, and `same` 0.705 means the two
arms disagree on 59. McNemar's exact test on the discordant *accuracy* pairs spans **p ≈ 0.001 if
the split is 11/0, to p ≈ 0.19 if it is 35/24** — the same headline number is either strong or null
depending on a quantity the aggregate JSON never stored. Hence per-image records, McNemar exact,
and a paired percentile bootstrap on the delta. `naive_d6` is dropped from the arm list (CONFIRMED
at chance twice; it would cost a third of the wall-time to re-confirm).

**Pre-registered rule** (written before the run, honored below). distil > teacher is CONFIRMED only
if McNemar p < 0.05 with b > c at n=3900. If p ≥ 0.05, the honest statement is *"student reaches
teacher parity"* — itself a strong result at −16.58% params and 1.61× — and the overtake is written
off as n=200 noise. Either way the parity HYPOTHESIS from part 2 §5 closes.

### Result

| arm | params | %save | top1 | same | agree | KL | vis_ms | pre_ms | total | speed |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher_d12 | 256,484,928 | 0.00 | 0.711 | 1.000 | 1.000 | 0.000 | 64.9 | 14.3 | 79.2 | 1.00× |
| **distil_d6** | 213,957,696 | **16.58** | **0.713** | 0.699 | 0.499 | 1.973 | **35.0** | 14.5 | 49.6 | **1.60×** |

McNemar b=440, c=432, **delta +0.0021, p=0.813**, paired bootstrap 95% **[−0.0127, +0.0174]**.
`results/frontier/vlm_step006_eval_scale_d6_n3900_s42_val__mini_mps.json`.

**CONFIRMED — d6 distilled ≡ d12 teacher on top-1, at −16.58% params and 1.60× vision.** The
pre-registered null branch fired: p ≥ 0.05, so the overtake is n=200 noise and §8's headline is
withdrawn. What replaces it is *stronger evidence for a weaker claim*: the bootstrap CI is
[−1.3pp, +1.7pp], so this is not "we failed to detect a difference at low power" — it is a positive
parity result that **excludes any degradation beyond 1.3pp**. A null that only says "underpowered"
is worth little; this one bounds the effect.

Three things worth carrying forward:
1. **Discordance is enormous and balanced** (b=440, c=432 of 3859). See §8's revised subsection.
2. **KL and vis_ms reproduced across eval sets** (1.986→1.973, 35.5→35.0 ms, 1.61×→1.60×). The
   *distributional* and *latency* measurements were stable at n=200; only **top-1 was not**. Cheap
   rule for this line: latency claims survive small n, accuracy claims do not.
3. **`same` 0.699 vs `agree` 0.499.** `same` compares the constrained 10-way pick, `agree` the raw
   argmax over the full vocabulary. The constrained readout recovers ~20pp of agreement the free
   argmax loses — an independent measurement of how much of the tower's output the 10-way protocol
   is actually reading, and a reminder that any claim here is scoped to *this* readout.

## 10. Drone recipe as it now stands

| Lever | Evidence | Gain | Status |
|---|---|---|---|
| `do_image_splitting=False` | vlm_step002 CONFIRMED | **11.71×** latency | ship (retest on fine-detail task) |
| **tower truncation d6 + distillation, FULL budget** | **vlm_step004 + vlm_step006 CONFIRMED** (§8–9, n=3859) | **1.60× latency, 16.58% params, +0.2pp — parity, 95% CI [−1.3, +1.7]pp** | **ship — best lever found** |
| tower truncation d6 + distillation, 6000×15 | vlm_step004 CONFIRMED | 1.63× latency, 16.58% params, −6pp | superseded by the row above |
| tower truncation d6 + distillation, T0 budget | vlm_step004 CONFIRMED | 1.62× latency, 16.58% params, −18pp | superseded |
| tower truncation d3 + distillation, T0 budget | vlm_step004 CONFIRMED | 2.33× latency, 24.87% params, −28pp | latency-bound option; **retest at full budget** |
| top-k image tokens (32/16) | vlm_step003 CONFIRMED | 1.02–1.04× latency; 2–4× KV cache, −6/−14pp | ship where memory-bound |
| logit-KD term on the distillation loss | vlm_step005 CONFIRMED | monotone −14 to −28pp top-1 | **KILLED at every β tested** |
| vision-tower truncation, zero-shot | vlm_step003 CONFIRMED | 1.26–2.38× at **chance accuracy** | superseded |
| readout (connector + lm_head) prune | vlm_step001 CONFIRMED | ≤1.15× params | **do not spend budget here** |
| VLM FFN distillation | step989, llm_step002 CONFIRMED | — | **KILLED role** (substitution, not truncation) |

**Headline: half the vision tower deleted — 16.58% of the whole model, 1.60× faster vision — at
top-1 parity (95% CI [−1.3, +1.7]pp, n=3859), from unlabelled in-domain images alone.** No labels,
no architecture search, no new modules: the recipe is "keep the first 6 of 12 SigLIP layers, regress
the full tower's post-connector features on unlabelled target-domain images". Across three budgets
spanning 11.7× in updates the fit never saturated — distillation budget, not architecture, has been
the binding constraint on this lever every single time it was measured, which means 16.58% is a
**floor on this lever, not a ceiling**, and d3 (24.87%, 2.33×) has never been given the budget that
made d6 work (§11.3).

Corollary of §9: every accuracy claim here still resting on n ≤ 200 is provisional — §11.6's real
motivation, not just the fine-detail concern.

## 11. Next steps (gated, in order)

1. ~~Settle the overtake~~ — **DONE, §9. Null: parity, not overtake.** §8 and §10 downgraded per the
   pre-registered rule; the ordering below is unchanged, exactly as this item said it would be.

   **Successor, now the single most load-bearing open question: does the PARITY survive off-domain?**
   §9 killed the *gain*, so the question is no longer "is the student a better tower" — it is whether
   16.58% of the model can be deleted at parity on images the distillation never saw. That is the
   difference between two very different drone recipes:

   | if | drone recipe reads |
   |---|---|
   | parity holds off-domain | distil once on **any** unlabelled images, ship the d6 tower anywhere |
   | parity is domain-scoped | you **must** collect unlabelled target-domain frames before deploying |

   Both are shippable; they cost wildly different amounts of fieldwork, and we currently cannot tell
   which one we have. **Written and QUEUED as `vlm_step007`**
   (`scripts/frontier/vlm_step007_domain_split.py`, with the distillation loop extracted to
   `scripts/frontier/vlm_distill.py` so the file clears the 200-line limit). Distil on 5 wnids, eval
   on the held-out 5 — same resolution, same source, only the classes change. CIFAR-10/100 is
   available but its 32×32 resolution is a second confound, so it is the *second* choice.

   Three details make it decisive rather than suggestive: (a) the readout stays the full constrained
   10-way choice, so *only* the student's unlabelled image distribution changes and the task does
   not; (b) budget matched on **updates** (50 × ~4676 = 25 × 9352 = 233,800), so the seen-5 student
   is not handicapped by seeing half the data; (c) the existing all-10 checkpoint is re-evaluated on
   the same two halves as a **zero-cost control** — without it, a seen−held gap is indistinguishable
   from the two halves simply differing in difficulty. Read the *interaction* Δ_seen − Δ_held, not
   the raw deltas. §9 makes the control's expected value concrete: the all-10 student is now known to
   sit at +0.2pp over the whole val set, so it should read ≈0 on **both** halves; if it does not,
   the halves differ in difficulty and the seen-5 numbers must be read against that offset, not
   against zero.

   Known bias, to be stated with any result: 50 passes over half the images overfits more than 25
   over all of them, which pushes *toward* "domain-scoped". **Second caveat inherited from §9:**
   n=1000 per group gives roughly half the resolving power of the n=3859 run that just overturned a
   5.5pp claim, so only interaction effects well outside ±2pp should be called.
2. ~~Add a logit-KD term~~ — **DONE, KILLED** (part 2 §5b). Do not re-raise a β sweep: the
   dose-response covers β ∈ {0.1, 1} and both lose.
3. **Long-budget d9/d3 pair.** Depth-independence (part 2 §5 CONFIRMED #2) rested on a shared
   plateau now known to be a budget artefact, and d3 has never been run past T0. With the full
   budget shown to be worth +18pp at d6, d3's −28pp is almost certainly an underestimate — and d3
   is the 2.33× row, i.e. the one that actually matters for the drone.
4. **Compounding: depth × tokens.** Unblocked (`distil_d6` at teacher parity, §9) and cheap (the
   checkpoint exists, so this is eval-only). Compounding Rule: token top-k acts on what the tower
   emitted, depth on the emitting — plausibly orthogonal, but HYPOTHESIS until the isolation
   ablation runs.
5. **Learned token selection.** top-k beat stride, so the *criterion* is worth training: a tiny
   scorer over post-connector tokens, distilled to match the full-token output distribution.
6. **Fine-detail control task.** Everything here is 10-way whole-object imagenette. Before any
   drone claim, rerun the table where high-res tiles should matter.
7. **Quantize the embedding table** (11.60%) — a lookup, not a transform; the one param-side win
   the profile endorses. Orthogonal to everything above.
8. **Do NOT** attack VLM FFNs or the connector for param savings. vlm_step001 bounds that direction
   at 1.15×, and step989/llm_step002 bound the FFN *substitution* role at "does not transfer" — a
   verdict part 2 §5 narrows but does not overturn.
