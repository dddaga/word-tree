# Visual-LLM Trajectory — part 2: repairing the truncated tower

Continues [`VLM_TRAJECTORY.md`](VLM_TRAJECTORY.md), which ends at `vlm_step003` and its
HYPOTHESIS: *the truncated tower needs distillation, not deletion.* Part 1 = the Amdahl profile
(§1), the redirect (§2), and the two pruning-axis sweeps — tokens (§3) and depth (§4). Part 2 =
whether the depth axis can be repaired, and what the drone recipe looks like afterwards.

## 5. `vlm_step004` — vision-tower DISTILLATION (T0)

`scripts/frontier/vlm_step004_tower_distill.py`. Teacher = the frozen 12-layer SigLIP tower plus
post_layernorm and connector. Student = a `deepcopy` of the teacher's **first d layers**, the only
trainable thing in the model. Target = the teacher's **post-connector** features (B, 64, 576).
Loss = relative MSE, `mse(s,t) / mean(t²)`, which is scale-free and therefore comparable across
depths. The teacher is cached once over the train subset in fp16 on CPU (74 KB/image), so it is
run once, not once per epoch.

**Why this is not step989/llm_step002 re-run.** Those KILLED replacing a TRANSFORM block's
*internals* with a sparse head. Here the student is the **same architecture, only shorter**, and
is asked to reproduce the stack's **OUTPUT**. Different claim → its own evidence. Arms
`teacher_d12` / `naive_d{d}` / `distil_d{d}` share the same 50 val images and the same constrained
10-way choice as step002/003, so top-1 is directly comparable to 0.740 and to step003's naive row.

### Results (mini_mps, T0, n_train=2000, 10 ep, bs=8, lr=2e-4, n=50 val, chance = 0.100)

| arm | params | %save | top1 | same | agree | KL | vis_ms | pre_ms | total | speed |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher_d12 (ref) | 256,484,928 | 0.00 | **0.740** | 1.000 | 1.000 | 0.000 | 66.1 | 14.6 | 80.7 | 1.00× |
| naive_d9 | 235,221,312 | 8.29 | 0.080 | 0.180 | 0.000 | 9.973 | 50.5 | 14.3 | 64.8 | 1.25× |
| **distil_d9** | 235,221,312 | 8.29 | 0.540 | 0.580 | 0.380 | 2.881 | 50.4 | 14.3 | 64.8 | 1.25× |
| naive_d6 | 213,957,696 | 16.58 | 0.120 | 0.140 | 0.000 | 10.669 | 35.4 | 14.3 | 49.7 | 1.63× |
| **distil_d6** | 213,957,696 | 16.58 | **0.560** | 0.660 | 0.420 | 2.512 | 35.4 | 14.3 | 49.7 | 1.62× |
| naive_d3 | 192,694,080 | 24.87 | 0.100 | 0.080 | 0.000 | 12.457 | 20.4 | 14.3 | 34.7 | 2.32× |
| **distil_d3** | 192,694,080 | 24.87 | 0.460 | 0.560 | 0.320 | 3.253 | 20.3 | 14.3 | 34.6 | **2.33×** |

Train curves (ep1 → ep10), all three monotone and **none converged**:

| arm | rel_mse | cosine |
|---|---|---|
| distil_d6 | 0.6092 → **0.2573** | 0.7598 → **0.8634** |
| distil_d9 | 0.5614 → 0.2716 | 0.7604 → 0.8551 |
| distil_d3 | 0.7630 → 0.3168 | 0.7281 → 0.8284 |

### Two CONFIRMED results

1. **The step605 / det_step002 distillation recipe recovers a truncated VLM vision tower.**
   d6: top-1 0.120 → **0.560** (+44pp over naive, chance 0.100), KL 10.669 → 2.512 (−76%),
   `agree` 0.000 → 0.420 — at an **unchanged** 16.58% param saving and 1.62× latency, since the
   student has exactly the naive arm's shape. **This is the first evidence in the whole directive
   that a TRANSFORM stack can be shrunk at all.** It does not contradict step989/llm_step002: the
   thing that transfers is *output-distillation of the same architecture*, not *substitution by a
   sparse head*. vlm_step003's "KILLED as-is" verdict on the depth axis is hereby **reopened**.
2. **Recovery is near-INDEPENDENT of depth — a plateau with a knee below d6, not a slope.** Over a
   3× range of surviving depth, distilled top-1 spans only 0.460–0.560 while the naive arms sit at
   0.080–0.120 (chance):

   | depth | %params saved | naive | **distil** | KL | agree | speed | final cosine |
   |---|---|---|---|---|---|---|---|
   | d9 | 8.29 | 0.080 | 0.540 | 2.881 | 0.380 | 1.25× | 0.8551 |
   | **d6** | 16.58 | 0.120 | **0.560** | 2.512 | 0.420 | 1.62× | 0.8634 |
   | d3 | 24.87 | 0.100 | 0.460 | 3.253 | 0.320 | **2.33×** | 0.8284 |

   d9→d6 is flat (Δ = −2pp = one image at n=50) and d9 even fit the teacher **worse** than d6
   despite three more layers; only d6→d3 bends, by 10pp. Three of twelve layers still reach 4.6×
   chance. **Consequence: take the deep cut.** d6 dominates d9 on every Pareto axis (more params
   saved, faster, equal accuracy), so the timid cut is strictly wasted. d3 is the pick where 2.33×
   latency is worth 10pp. This kills the natural intuition that recovery degrades monotonically
   with depth removed — the intuition that would have argued for d9.

   d3 is also the first arm whose *training* fit is worst while its capacity is smallest, i.e. the
   first place the loss ordering tracks depth at all — weak evidence that capacity begins to bind
   somewhere below d6, and the reason the knee is placed there rather than asserted.

### CONFIRMED #3 — the residual gap was BUDGET, not capacity

Same d6 student, same code, only `--n_train 6000 --epochs 15` (3× data, 2.25× updates):

| d6 budget | rel_mse | cosine | top1 | same | agree | KL | params | speed |
|---|---|---|---|---|---|---|---|---|
| 2000 × 10 ep | 0.2573 | 0.8634 | 0.560 | 0.660 | 0.420 | 2.512 | 16.58% saved | 1.62× |
| **6000 × 15 ep** | **0.2011** | **0.8951** | **0.680** | **0.760** | **0.500** | **1.988** | 16.58% saved | 1.63× |
| teacher_d12 | — | 1.000 | 0.740 | 1.000 | 1.000 | 0.000 | — | 1.00× |

**+12pp for budget alone** — the gap to the teacher goes 18pp → **6pp** (3 images at n=50) at
unchanged params and unchanged latency. Every metric moved the same way (KL −21%, `agree` +8pp,
`same` +10pp), so this is fidelity, not eval noise. And it is *still* not converged: cosine rose
monotonically through ep15 (0.7896 → 0.8951) with no flattening.

**Consequences.** (a) The earlier "plateau at cosine ≈ 0.86" was an artefact of a 2000-image
budget, not a property of the architecture — the d9-vs-d6 tie that motivated CONFIRMED #2 was
measured *under that artefact*, so the depth-independence claim is now weaker evidence than it
looked: it may be that all depths were budget-bound alike. It does **not** overturn "take the deep
cut" (d6 still dominates d9 at equal budget), but a long-budget d9/d3 pair is the honest retest.
(b) The 6pp residual is below what n=50 can resolve — a parity claim needs a larger eval set.

> **HYPOTHESIS — d6 reaches teacher parity at full budget.** Extrapolating a still-descending
> curve. Test = n_train 9469 (all of imagenette train) × 25–30 ep with n_eval ≥ 200. Cost ~10 h on
> mini_mps, which is why vlm_step005 (new information) goes first.

**Caveat (SURVIVES the budget fix) — feature-MSE does not preserve the teacher's function
pointwise.** `agree` sits below top-1 in every distilled arm, at both budgets: 0.420 vs 0.560 at
2000×10, and still 0.500 vs 0.680 at 6000×15. Budget lifted both by ~8–12pp but did **not** close
the ~18pp *gap between them* — so this is a property of the objective, not of undertraining. The
student reaches the right label partly by a *different route*. On a 10-way closed task that is
invisible; on open-ended VLM generation it may not be. Hence the logit-KD term — the actual step605
recipe, which distilled soft *outputs*, not intermediate features (`vlm_step005`, §5b — **KILLED**;
the caveat therefore stands as a limitation, not a solved problem).

## 5b. `vlm_step005` — logit-KD term: KILLED

`scripts/frontier/vlm_step005_logit_kd.py`. Same student, same teacher, same budget as the 2000×10
d6 row, plus a second loss: KL between student and teacher next-token logits at the answer position,
weight β, on top of the feature relative-MSE at weight α. Logit targets live at **one** token
position, so the forward is bs=1 and `--accum 8` restores step004's effective batch.

| d6 arm, 2000×10 | β | rel_mse | cosine | top1 | same | agree | KL |
|---|---|---|---|---|---|---|---|
| §5 feature-MSE only (α=1) | 0 | **0.2573** | **0.8634** | **0.560** | **0.660** | **0.420** | 2.512 |
| α=1, β=0.1 | 0.1 | 0.4029 | 0.7737 | 0.420 | 0.440 | 0.280 | 2.614 |
| α=1, β=1 | 1 | 0.5218 | 0.7006 | 0.280 | 0.360 | 0.380 | 2.347 |
| α=0, β=1 (pure KD) | 1 | 29.11 | 0.4985 | 0.300 | 0.460 | 0.220 | **2.161** |
| naive_d6 | — | ~11–15 | — | 0.120 | 0.140 | 0.000 | 10.669 |

**CONFIRMED — logit-KD is a monotone liability, and does not fix the caveat it was built for.**
top-1 falls monotonically in β (0.560 → 0.420 → 0.280) and `agree` never beats the β=0 baseline at
any weight. The dose-response is what makes this a kill rather than a tuning failure: β=0.1 — KD as
a 10% regulariser on a healthy feature fit — still costs 14pp top-1 and still degrades the feature
fit 57% (rel_mse 0.2573 → 0.4029). The two gradients genuinely conflict; no mixing weight escapes.
Pre-registered rule (top-1 ≥ 0.560 **and** `agree` > 0.420) failed on both clauses by every arm.

**KD buys exactly one thing: eval-KL**, the metric nearest its own objective — pure KD is *best* on
KL (2.161) while *worst* on `agree` (0.220). Soft-distribution match ≠ argmax identity.

> **Side-finding worth a paper sentence (from the α=0 arm).** Pure KD ends at rel_mse **29.1**,
> cosine 0.4985, down only 11% over 10 epochs — the features drift essentially unconstrained and
> keep almost no geometric relation to the teacher's. Yet the frozen text model still reads
> **0.300 top-1 (3× chance)** out of them. **The readout tolerates near-arbitrary feature geometry
> provided the output distribution is steered** — it decodes what is presented rather than matching
> a learned code. Same "readout is permissive" theme as step605 / det_step002, from the opposite
> direction.


---

Continues in [`VLM_TRAJECTORY_part3.md`](VLM_TRAJECTORY_part3.md): the full-budget parity run
(§8, where the student **overtakes** the teacher), the significance check that decides whether
that is real (§9), the drone recipe (§10) and next steps (§11).
