# VLM Trajectory — part 4 (from `vlm_step007`)

Continues `VLM_TRAJECTORY_part3.md` (§8–§11). Part 3 closed with the d6 tower CONFIRMED at teacher
parity (0.713 vs 0.711, n=3859) for 16.58% of the model deleted, and named the domain question as
the top open item for the drone. §12 answers it.

## 12. `vlm_step007` — is the parity method-level or domain-scoped? **DONE — DOMAIN-SCOPED.**

Distil the d6 tower on 5 of the 10 wnids (unlabelled), keep the readout the full constrained 10-way
choice, eval on both halves. Budget matched on updates (50 × 4676 = 25 × 9352 = 233,800). The
existing all-10 checkpoint re-evaluated on the same halves is the zero-cost control.
`results/frontier/vlm_step007_domain_split_d6_50ep__mini_mps.json` (mini_mps, ~8 h).

| group | arm | n | top1 | teacher | Δ | b/c | p | boot95 |
|---|---|---|---|---|---|---|---|---|
| seen-5 | seen5_d6 | 1000 | 0.706 | 0.742 | −0.0360 | 76/112 | 0.0105 | [−0.0620, −0.0100] |
| seen-5 | all10_d6 (ctrl) | 1000 | 0.703 | 0.742 | −0.0390 | 95/134 | 0.0119 | [−0.0680, −0.0100] |
| **held-5** | **seen5_d6** | 1000 | **0.168** | 0.683 | **−0.5150** | **5/520** | 0.0000 | [−0.5470, −0.4830] |
| held-5 | all10_d6 (ctrl) | 1000 | 0.739 | 0.683 | +0.0560 | 139/83 | 0.0002 | [+0.0270, +0.0850] |

Interaction (Δ_seen − Δ_held): **seen-5 student +0.4790**, all-10 control **−0.0950**.

**CONFIRMED — the d6 tower's parity is a property of the distillation DOMAIN, not of the method.**
The three things that make this a verdict rather than a suggestion:

1. **Magnitude clears the power caveat by 24×.** §9 licensed calls only outside ±2pp at n=1000/group.
   The interaction is +47.9pp.
2. **The control rules out half-difficulty.** The all-10 student's own interaction is **−9.5pp** —
   opposite sign. Had the halves merely differed in difficulty, both arms would lean the same way.
   (The control does *not* read ≈0 on both halves as §9 predicted: it is −3.9pp on seen and +5.6pp
   on held. The halves do differ, and they differ in the direction that makes the seen-5 collapse
   *harder* to produce, not easier.)
3. **b=5, c=520.** Near-total one-sided loss. 0.168 against a 0.100 chance floor on a closed 10-way
   readout means the held classes are essentially gone from the tower, not merely degraded.

**Confound, stated as required (pre-registered in the script docstring).** Budget was matched on
updates, so the seen-5 student saw each of its images **50×** against step004's 25×. Domain
restriction and 2×-repetition overfit cannot be separated by this run. What survives the confound:
the *direction and scale* of the effect, since 2× repetition on in-domain data is not a known
mechanism for a 51.5pp off-domain collapse, and the control shares the eval but not the training.
What does **not** survive: any claim that the collapse is *purely* domain restriction — that
attribution is **HYPOTHESIS**. A matched-epoch (25 ep) rerun on 5 classes would separate them; it is
in the parking lot below, not the queue, because the operational conclusion is the same either way.

**The training loss confirms the mechanism.** seen-5 reached **rel_mse 0.1071 / cosine 0.9456**,
*below* step004's final 0.1639 on all 10 classes. It fit its own distribution strictly better while
losing the other half. Low distillation loss is therefore CONFIRMED to be a within-domain fit
statistic, not a proxy for tower quality — it cannot be used as an early-stopping or model-selection
signal across domains.

### 12.1 What this changes for the drone

The §10 recipe gains a hard prerequisite: **collect unlabelled frames from the deployment domain
before distilling.** Not a blocker — the frames are unlabelled and a drone generates them by flying,
which is the cheapest data any stage of this pipeline needs. But "distil once, ship anywhere" is
**KILLED (CONFIRMED)**, and any deployment onto a domain the tower has not seen must be treated as
untested rather than as covered by the 0.713 parity number.

Two consequences that are upside, and both are untested (**HYPOTHESIS**):
- The seen-5 arm lost only −3.6pp *in* its domain while deleting 16.58% of the model. A drone only
  ever needs its own domain, so domain-restricted distillation may be the *right* configuration
  rather than a degraded one — and a narrower domain than 5 imagenette classes is narrower still.
- The all-10 control read **+5.6pp over teacher** on the held half. A distilled tower beating its
  teacher on a subset it was not specialised to is not predicted by anything in the trajectory and
  is n=1000 noise-eligible at that CI. Do not build on it; note it.

## 13. `vlm_step008` — d3/d9 at full budget. **RUNNING** (5060ti_cuda)

`scripts/frontier/vlm_step008_depth_budget.py` (200 lines), `--depths 3 9`, 25 epochs each, 3859
eval images, d6 anchor loaded not retrained. **Budget caveat:** the balanced sampler floored
`--n_train 10000` to **9469**, not the 9352 the docstring predicted, so d3/d9 train on 1.25% more
images than the d6 anchor did. Too small to move a depth comparison, but it means the anchor is not
*exactly* budget-matched — state it if d3 lands near the ±2pp boundary. Pre-registered rule in the
docstring: SHIP iff boot95 lower bound ≥ −2.0pp, KILLED iff upper bound < −2.0pp, INCONCLUSIVE
between — plus an independent floor-vs-ceiling read on the rel_mse slope over the last 5 epochs.

### 13.1 Getting it onto the 5060ti — four blockers, all cleared without touching shared state

Recorded because every one of them will recur for any non-SGNNET job on that box.

1. **Shared venv pinned to transformers 4.44** (no Idefics3). Fixed with a private
   `pip install --target=/home/indra/sgnnet_bench/vlm_libs` (transformers 5.5.4) plus a
   `sys.path.insert` at the top of the step script, guarded by `.is_dir()` so it no-ops on macOS.
   `launch_slot.sh` passes no environment through, so the shadow *must* live inside the script.
2. **`operator torchvision::nms does not exist`** — pre-existing, not caused by the install. Proved
   by running the un-shadowed interpreter and getting the identical error: the shared venv ships
   torchvision 0.26.0 built against a different torch than its own 2.11.0+cu128. Fixed inside
   `vlm_libs` with `torchvision==0.26.0+cu128 --no-deps`. **Shared venv never modified** — it is
   teammates' runtime.
3. **imagenette absent on the box.** rsynced (13,394 JPEGs, 351 MB). Note: macOS rsync 2.6.9 rejects
   `--info=stats2` and the usage dump piped through `tail` still let an `&&` chain echo success —
   same GNU-flags-in-BSD-userland trap as `timeout`. Verify by counting files, not by exit code.
4. **CUDA OOM at `loss.backward()`**, 2.07 GiB ceiling because another user's idle Jupyter kernel
   (`ipykernel_launcher`, tiger-ml env) holds 13.36 of 15.48 GiB. **Slot-free ≠ GPU-free** —
   `slot_status.sh` derives occupancy from tmux panes and cannot see a notebook.

**The OOM diagnosis is the reusable part.** Batch 8, 4 and 2 all died at the *same* allocation point
(~1.89 GiB already held, failing on 12–20 MiB). Identical failure across a 4× batch sweep is the
signature of a **fixed-footprint** problem, not an activation problem — so gradient accumulation
could not have fixed it. Fix in `vlm_distill.park(ev, device)`: the distillation loop touches only
the vision tower and connector, so `model.text_model` + `lm_head` (~0.68 GB) and the frozen teacher
layers (~0.34 GB) are parked on CPU for the whole training phase and brought back for eval, and the
d6 anchor (~0.17 GB) is loaded *after* training instead of before. ~1.19 GB freed. Smoke then passed
end-to-end on cuda.

fp16/bf16 was rejected despite being the easier win: the run's validity check is that the loaded d6
anchor reproduces step006's 0.713 **in fp32**, and changing dtype would break that comparison and
make the whole run unreadable.

## 14. `vlm_step009` — depth × tokens compounding. **DONE — no interaction; pruning is a bad trade.**

Eval-only 2×4 grid on one paired eval set, n=3859.
`results/frontier/vlm_step009_compound_tokens_d6_k64-32-16-8__mini_mps.json` (mini_mps, ~1.6 h).

**Validity check passed exactly.** `d12_k64` = **0.711** and `d6_k64` = **0.713** — both reproduce
step006 to three decimals on an independently drawn eval pass. Everything below is readable.

| cell | top1 | agree vs d12_k64 | KL | vision_ms | prefill_ms | total_ms |
|---|---|---|---|---|---|---|
| d12_k64 | 0.711 | 1.000 | 0.000 | 65.0 | 14.3 | 79.3 |
| d12_k32 | 0.659 | 0.791 | 0.186 | 64.9 | 13.1 | 78.0 |
| d12_k16 | 0.577 | 0.625 | 0.748 | 64.9 | 11.0 | 76.0 |
| d12_k8 | 0.422 | 0.424 | 1.834 | 64.9 | 11.2 | 76.1 |
| **d6_k64** | **0.713** | 0.499 | 1.973 | **35.1** | 14.3 | **49.4** |
| d6_k32 | 0.662 | 0.490 | 1.894 | 35.1 | 11.8 | 46.9 |
| d6_k16 | 0.585 | 0.449 | 2.028 | 35.1 | 10.3 | 45.4 |
| d6_k8 | 0.421 | 0.353 | 2.561 | 35.1 | 10.5 | 45.6 |

| k | I(k) | boot95 | d6-vs-d12 at k (b/c, p) | pre-registered call |
|---|---|---|---|---|
| 32 | +0.0013 | [−0.0109, +0.0132] | 442/429, p=0.684 | **ORTHOGONAL — levers compound** |
| 16 | +0.0057 | [−0.0111, +0.0225] | 504/474, p=0.354 | INCONCLUSIVE at this n |
| 8 | −0.0034 | [−0.0223, +0.0155] | 493/498, p=0.899 | INCONCLUSIVE at this n |

**Reported as pre-registered: only k=32 formally clears.** k=16 and k=8 miss solely because their CI
half-widths run 0.2–0.3pp past the ±2.0pp tolerance; all three point estimates sit within ±0.6pp of
zero and every McNemar p is ≥ 0.35. The honest statement is *no interaction was detected at any
budget, and at k=16/8 the measurement is not tight enough to certify it* — not "they interact."
Resolving k=8/16 needs roughly 1.4× the eval set, ~40 min more on mini_mps; queued only if a pruned
config ever ships, which §14.1 argues against.

### 14.1 The real finding is the timing column, and it kills pruning for this drone

**`vision_ms` is flat to the third digit across k** (65.0/64.9/64.9/64.9 at d12; 35.1 four times at
d6). CONFIRMED: pruning happens *after* the tower, so it cannot touch vision cost by construction.
The only thing it moves is prefill, **14.3 → 11.2 ms** — about **3 ms, 3.9% of the d12 total**.

The smoke's 41.6 → 24.2 ms was warm-up noise at n=20 and **overstated pruning's payoff by ~6×**;
part 3 §11.4's "~12% of total" is superseded. Against that ~3 ms, the accuracy bill is −5.2pp at
k=32, −13.4pp at k=16, −28.9pp at k=8 — and it is the *same* bill at both depths, which is exactly
what the null interaction says.

Depth buys **29.9 ms (37.7% of total) for +0.2pp**. Tokens buy **~3 ms for −5.2pp**. Same units,
~10× the time saving, opposite sign on accuracy. **Drone recipe: ship d6 unpruned.** Token pruning
is KILLED for the 256M-parameter deployment (CONFIRMED on time-vs-accuracy, not on interaction).
Note the scope: SmolVLM-256M prefills 64 image tokens into a small text stack. On a VLM with a large
text stack or many more image tokens, prefill is a much bigger share and this arithmetic can flip —
the *method* is not killed, this **operating point** is.

**One number to carry forward: `agree` between `d6_k64` and `d12_k64` is 0.499.** Matched top-1
(0.713 vs 0.711), but the two towers disagree on half the images. The distilled tower is an
equally-accurate *different* classifier, not a functional copy — step004's agree-vs-top1 warning,
now confirmed at full n on the shipping config. Anything downstream that assumes teacher-equivalence
rather than teacher-parity is unsupported.

## 15. Next steps

Unchanged from part 3 §11 except where §12/§14 rewrite them:

1. ~~Domain question~~ — **DONE, §12. Domain-scoped.**
2. **d3/d9 full budget** — RUNNING as `vlm_step008` (§13).
3. ~~Compounding: depth × tokens~~ — **DONE, §14. No interaction; pruning is a bad trade here.**
4. **Learned token selection** (§11.5) — **DEPRIORITISED to the parking lot by §14.1.** A better
   scorer can only recover accuracy inside a 3 ms prefill envelope; the ceiling on the whole
   direction is now measured and it is ~4% of wall-time at this operating point. Revisit only for a
   VLM whose prefill share is large.
5. ~~Quantize the embedding table~~ (§11.7) — **DONE as `vlm_step010`, part 5 §16. int8 SHIPS,
   int4 KILLED.** The block was 22.13%, not the 11.60% part 3 costed (`tie_word_embeddings=False`).
   Per-row scales turned out not to be load-bearing. Stacked with d6 this reads −33.1% of the
   model's bytes at −0.31pp, McNemar p = 0.71 — **but see the next line before quoting that.**
   ~~**Fine-detail control task**~~ (§11.6) — **DONE as `vlm_step011`, part 5 §17. SUB-ADDITIVE:
   parity does NOT survive small objects** (I = −12.78pp at f=0.5, −24.41pp at f=0.25; d6 collapses
   to 0.252 top-1 where the teacher holds 0.494). **The −33.1% byte win is unconditional; the
   −0.31pp is scoped to whole-object images.** int8 on the tables is input-independent, so the
   unconditional drone config is d12 + int8 (−16.6%, Δ +0.00pp); d6 is conditional on target size
   until part 5 §18 item 1 (scale-augmented re-distillation) settles capacity vs distribution.
6. **Do NOT** attack VLM FFNs or the connector (§11.8) — unchanged.

### Parking lot
- **Matched-epoch (25 ep) 5-class rerun** to separate domain restriction from 2×-repetition overfit
  in §12. Deferred: the drone conclusion ("collect target-domain frames") is identical under both
  explanations, so the experiment buys attribution precision, not a decision.
- **Narrow-domain distillation as a feature** — distil on 2 classes, or on a single scene type, and
  ask whether in-domain accuracy *rises* above the all-10 tower. Follows from §12.1; needs the d3/d9
  curve first so it is run at the right depth.
