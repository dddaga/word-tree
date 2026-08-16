# VLM trajectory — part 10 (tower WIDTH)

Continues `VLM_TRAJECTORY_part9.md` (full at 199 lines). Same line: SmolVLM-256M, Imagenette
10-class, d12@512 anchor 0.711, tower = 82% of end-to-end latency.

## 39. vlm_step023 — MLP intermediate width, sliced and distilled. DONE.

**Why this axis.** Depth is done (d6 = 2.0× at parity). Resolution is CLOSED (§38: the training-free
oracle caps any res-256 student at 0.4160). Readout/token pruning is dead (step009). Width is the one
structural axis never touched, and the only one that shrinks **params** — resolution shrank neither
params nor bytes, which the drone goal cares about.

Of the two width axes, the tower's **MLP intermediate** (3072) was chosen over residual width (768)
because slicing it needs **no adapters**: fc1 loses rows, fc2 loses the matching columns, the residual
stream stays 768 end to end, so the run is strictly one-variable. It is also 2·768·3072 = 4.72M of
each layer's 7.08M = **67% of the tower**. Init = magnitude-ranked slice, keeping the top-m channels
by `||fc1_row_j|| · ||fc2_col_j||` (a unit matters only if it both fires and is read) — the width
analogue of depth truncation, warm-started from teacher weights, so "too narrow" is not confounded
with "started from nothing".

**Setup.** d12 stock, 15 epochs, 1000 train images, lr 1e-4, n_eval 500, seed 42. Only the sliced
encoder layers move; patch embeddings, post_layernorm, connector and the text stack stay frozen.
Loss = relative MSE on post-connector features, identical to step004/007/008/021.

**GATE (pre-registered) — PASS.** Teacher d12 top1 0.7040 vs anchor 0.711 (0.7pp, bar was ±2.0pp).
Tower params 85.05M, mlp intermediate 3072.

| ratio | inter | naive | distilled | gap vs teacher | gain over naive | params | tower ms | verdict |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 1536 | 0.1160 | 0.5500 | −15.40pp | **+43.40pp** | 56.72M (0.667×) | 26.22→20.41 (1.28×) | LOSS |
| 0.25 |  768 | 0.0860 | 0.4360 | −26.80pp | +35.00pp | 42.56M (0.500×) | 26.22→17.76 (1.48×) | LOSS |

Both naive controls read at chance, as every naive structural cut to this tower has (naive d6 = 0.120,
naive res-256 = 0.160) — the distilled number is the only informative one.

**Both arms LOSS by the pre-registered absolute bar. The bar is not the finding.** Two things say so:

1. **Neither loss curve converged.** r=0.5: 0.3421 → 0.2769 → 0.2357 over ep 5/10/15, still falling
   at the last epoch. r=0.25: 0.3696 → 0.3275 → 0.2852, likewise. No plateau in either.
2. **The budget control, which is decisive.** Unlike §38 there is no training-free oracle for a
   sliced MLP, so the "at this budget" caveat is load-bearing — and step004's own d6 ladder shows
   exactly how load-bearing:

| d6 budget | final rel_mse | d6 top1 | vs its teacher |
|---|---|---|---|
| 10ep / 2k | 0.2573 | 0.56 | **−18pp** |
| 15ep / 6k | 0.2011 | 0.68 | −6pp |
| 25ep / 9.3k | 0.1639 | 0.74 | **parity** |

step023 r=0.5 reached rel_mse 0.2357 on **15ep / 1k** — a smaller budget than d6's 10ep/2k row — and
landed −15.40pp, i.e. *better than depth's −18pp at comparable fit*. **The depth win was bought with
~16× the training budget step023 was given.**

> **Correction, recorded deliberately.** On first reading the r=0.25 arm I was about to file
> "depth strictly dominates width" as CONFIRMED, on the grounds that r=0.25 and d6 sit at the *same*
> 0.500× param budget while d6 got parity and r=0.25 got −26.8pp. That comparison is matched on
> params and **unmatched on training budget**, and the d6 ladder above shows the training budget is
> worth ~18pp on its own. The claim would have been an artifact. It is not made.

**VERDICT: LOSS at this budget; the axis is NOT closed.** Tagged HYPOTHESIS, not CONFIRMED, and
explicitly *not* the mirror of §38 — resolution died against a ceiling that no amount of training
could lift, whereas width died against an epoch count.

**What is CONFIRMED here**, independent of budget: the naive→distilled recovery of **+43.40pp** at
r=0.5 is the largest any arm in this line has produced, so a magnitude-ranked MLP slice is
recoverable-in-principle, not destroyed. Also confirmed and unflattering: **width buys speed poorly.**
0.667× params → 1.28×, 0.500× params → 1.48×, while depth got 2.0× at 0.5×. The tower is not
MLP-bound in wall-time even though the MLP is 67% of its params. For the drone that split matters —
width is the params/bytes lever, depth is the latency lever, and they are not the same lever.

## 40. vlm_step024 — width at d6's winning budget. DONE. **WIN.**

The one experiment §39 demands: r=0.5, **25 epochs / 9269 train images**, every other knob identical.
That is exactly the budget at which d6 reached parity, so the comparison becomes matched-budget and
the axis question gets a real answer instead of a budget artifact.

Pre-registered, before the run:
- **WIN** iff distilled ≥ teacher − 2.0pp. Width then joins depth as a live lever, and the two can be
  compounded (they touch different params, so the compounding rule's orthogonality test is met).
- **PARTIAL** (within 10pp) reads as "recoverable but depth is the better spend at equal budget".
- **LOSS** at *this* budget is the one that closes the axis, because it is d6's own winning budget —
  and only then is the closure CONFIRMED rather than budget-limited.

**GATE — PASS.** Teacher d12 top1 **0.7040**, identical to §39's teacher (same seed, same 500-image
eval sample), so §39 and §40 are directly comparable and the only variable is the training budget.

| run | budget | final rel_mse | naive | distilled | vs teacher | params | tower ms | verdict |
|---|---|---|---|---|---|---|---|---|
| §39 r=0.5 | 15ep / 1k | 0.2357 | 0.1160 | 0.5500 | −15.40pp | 56.72M (0.667×) | 26.22→20.41 (1.28×) | LOSS |
| **§40 r=0.5** | **25ep / 9.3k** | **0.1570** | 0.1160 | **0.7320** | **+2.80pp** | 56.72M (0.667×) | 26.16→20.45 (1.28×) | **WIN** |

**§39's LOSS was a budget artifact. CONFIRMED, by direct replacement of the single variable.** Same
slice, same init, same seed, same eval — only epochs and train images changed, and the arm moved
−15.40pp → +2.80pp. The deliberate correction recorded in §39 was correct to withhold "depth strictly
dominates width": that claim is now positively falsified, not merely unproven.

**Read the +2.80pp as PARITY, not as "the slice beats the full tower."** At n_eval=500 and p≈0.70 the
binomial SE is ≈2.05pp, so +2.80pp is ~1.4 SE — well inside noise. The honest claim is that a tower
with **half its MLP intermediate removed matches the full teacher**, and the naive→distilled recovery
is **+61.60pp**, the largest in this line by a wide margin (§39's own +43.40pp was the prior record).

**What this changes.** Width is a live lever, and by the compounding rule it is *orthogonal* to depth:
depth removes whole layers, width removes intermediate channels inside the layers that remain —
different params, different signal paths. Compounding d6 × r=0.5 is therefore permitted without an
isolation ablation, and is the obvious next arm: 0.5 × 0.667 ≈ **0.33× tower params** if it holds.

**The Pareto split from §39 survives intact and is now sharper.** Width bought parity for 0.667×
params but only 1.28× wall-time; depth bought parity for 0.5× params at 2.0×. **Width is the
params/bytes lever, depth is the latency lever.** For a drone that must fit weights in flash *and*
hit a frame budget, the compound arm is the one that serves both.

Results: `results/frontier/vlm_step023_mlp_width_r0.5_e25_t9352_n500__5060ti_cuda.json` (step024 runs
the step023 script, so the filename carries the step023 stem — the TAG `e25_t9352` is what
distinguishes it, which is exactly why the TAG was added to the checkpoint path before launch).

## 41. Roadmap after §40

1. **vlm_step025 — compound d6 × r=0.5 at the 25ep/9.3k budget.** Both factors reached parity alone
   at this budget; they are orthogonal by the compounding rule, so no isolation ablation is owed.
   Target ≈0.33× tower params and ≈2.5× tower latency. This is now the top arm: it is the only one
   that moves *both* drone axes at once, and both of its ingredients are already CONFIRMED.
2. **r=0.25 rematch at the matched budget.** §39's r=0.25 was starved by the same ~16× deficit that
   §40 just falsified for r=0.5, so its −26.80pp is uninterpretable and must not be cited as a floor.
   0.500× params at 1.48× is worth one run — and it is the cheaper alternative to arm 1 if the
   compound fails.
3. **Record per-image predictions in every future arm. DONE — closed by part12 §44.** Written locally
   after §42 but never synced to the 5060ti, so §41's "done" was premature and §43 still ran the
   pre-fix script with no vectors (see part11's process-defect note). Script synced 2026-08-14;
   **step027/§44 is the first arm that actually ran with it**, and its McNemar (p=0.444, 74
   teacher-only vs 64 student-only) is the first paired test in this line. Below as originally filed:
   `evaluate()` now returns a `correct`
   hit vector in fixed `sample_images` order, so teacher and student are paired image-by-image and
   McNemar becomes available. §40 could not test "student beats teacher" because only aggregate top-1
   was kept, and that cannot be reconstructed post-hoc. Landed *after* §42 launched, so §42 is the
   last arm without it.
4. **Learned token selection** — still the only route to the ~2.9× that average pooling cannot buy,
   and explicitly *not* bound by §38's ceiling (which bounds average-pooled budgets only).
5. **§32 prologue ablation** — time the quantization prologue separately from `tri_run`, or hoist it
   into the preceding layer. Lives in the 18% of latency that is not the tower.
6. **d9 depth retry** — unblocked, unstarted, lowest value.

---

§42 onward continues in `VLM_TRAJECTORY_part11.md`.
