# VLM trajectory — part 9: the tower was the target all along

Continues part 8 (§24–§32). Part 8 ended by admitting that §24–§32 optimised the wrong thing.
This part starts from the corrected Amdahl reading and works the vision tower.

## 33. The Amdahl re-read — where the milliseconds actually are

Nothing new was measured for this; the numbers were already in step006's stored output and went
unread for three parts of the trajectory.

| depth | vision_ms | prefill_ms | total | tower share |
|---|---|---|---|---|
| d12 teacher | 64.85 | 14.33 | 79.18 | **82%** |
| d6 distilled | 35.05 | 14.54 | 49.59 | **71%** |

**CONFIRMED — the vision tower is the bottleneck, and depth is the only tower lever the trajectory
had ever pulled.** The entire int8 `lm_head` line (§24–§32) targeted a ~0.3 ms sliver of a ~50 ms
pipeline and moved zero milliseconds end-to-end (§32 measured 0.94× vs fp16 — a loss). The byte
result from that line stands (3.97× vs fp32 on the head); the latency motivation does not.

This is a process failure worth naming, not just a wrong turn: the profile that falsifies §24–§32's
premise was sitting in a results JSON the whole time. **Read the profile before choosing the
optimisation, every time.**

## 34. `vlm_step020` — the resolution ladder. **DONE. Naive resolution is REJECTED, and my own "two-for-one" framing is FALSIFIED.**

SigLIP here is patch-16 on a 512 grid (1024 patches) and the connector pools 4×4, so decoder tokens
= patches / 16. The hope was that cutting the pixel grid cuts *both* `vision_ms` and `prefill_ms`,
which depth never did. d12 stock, no checkpoint anywhere, so resolution is the only variable.
n = 500, 5060ti, fp32.

| res | tokens | top1 | Δpp | vision_ms | prefill_ms | tower× |
|---|---|---|---|---|---|---|
| 512 | 64 | 0.704 | — | 26.18 | 13.87 | 1.00 |
| 384 | 36 | 0.456 | −24.8 | 14.55 | 13.67 | 1.80 |
| 256 | 16 | 0.160 | −54.4 | 8.92 | 13.63 | 2.93 |
| 192 | 9 | 0.094 | −61.0 | 7.00 | 13.30 | 3.74 |
| 128 | 4 | 0.106 | −59.8 | 4.61 | 13.59 | 5.68 |

Both pre-registered gates PASS: validity (res-512 d12 = 0.704 vs step006's 0.711, Δ −0.0070) and
the per-image token-count assert (connector tokens == (res/16)²/16, and exactly that many image
slots surviving the merge) fired on no image in any cell.

**CONFIRMED — cutting image tokens buys NOTHING in prefill.** `prefill_ms` is flat at 13.3–13.9 ms
while image tokens go 64 → 4. Prefill is dominated by the text prompt, not by image tokens. The
"two-for-one" claim I wrote into the step020 queue row before running it is dead: **resolution is a
tower-only lever, exactly like depth.** This also retro-explains §9's token-pruning results being
worth less wall-time than their token counts suggested.

**CONFIRMED — naive resolution reduction is catastrophic.** 192 and 128 sit at 0.094 and 0.106
against a 0.100 chance floor, i.e. the model is destroyed, not degraded. Even 384 — a mere 1.8× —
costs 24.8pp.

**But the naive arm is uninformative here, and the trajectory's own history says so.** Naive depth
did precisely this: naive d6 = 0.120 and naive d3 = 0.100, both at chance, and distillation lifted
them to 0.560 and 0.460 (§4, §8). A stock tower handed interpolated position embeddings it never
trained on is off-distribution by construction. So step020 is a rejection filter on *naive*
resolution and a pointer to §35 — it is not a verdict on the lever.

Implementation note that cost a design: the processor's `size={'longest_edge': N}` knob is **inert**
— every N in {512, 384, 256, 192, 128} returns a 512×512 tensor. Resolution has to be set model-side.
Downsampling the processor's own normalised output keeps mean/std preprocessing identical across
arms, so it is also the cleaner one-variable change. Harness extracted to `scripts/frontier/vlm_res.py`.

## 35. `vlm_step021` — the distilled low-res tower. **DONE. 256 PARTIAL, 384 IRRECOVERABLE.**

| res | tokens | naive | distilled | gain | verdict | final rel_mse / cos |
|---|---|---|---|---|---|---|
| 256 | 16 | 0.1600 | 0.2980 | +13.80pp | **PARTIAL** | 0.0545 / 0.9738 |
| 384 | 36 | 0.4560 | 0.4980 | +4.20pp | **IRRECOVERABLE** | 0.0611 / 0.9706 |

Validity gate PASS (teacher 0.7040 vs 0.711). The in-run naive control at 256 reproduced step020's
independent draw **exactly** (0.1600), so the two experiments' eval paths agree.

Distillation recovers more where the naive arm is more broken (+13.8pp at 256 against +4.2pp at
384), but neither cell reaches the teacher. So resolution is **not** the free lever depth was: d6
kept parity at 512 (0.720 vs 0.711), whereas the best distilled low-res tower here gives up 20.6pp
for 1.87× — a far worse trade than depth's 2.0× at parity. **The loss was still falling
monotonically in both cells at the 15-epoch budget, so these are verdicts at this budget, not
ceilings.**

**A confound I introduced and only saw afterwards.** The students fit their targets unusually well
— cosine 0.9738 is tighter than any depth distillation in this line ever reached — and still scored
0.298. A student matching its target that well is not failing to optimise; the target is the limit.
And the target was the 512 teacher's 64 tokens *pooled to 16*, i.e. a blurred teacher. So this
experiment cannot separate "256 px destroys information" from "16 tokens cannot carry the answer".
§38 settles that training-free.

**A bug my own pre-registered assert caught.** `pool_to` used an integer-stride `avg_pool2d` and
asserted `g % k == 0` — true for 8×8 → 4×4 (res 256), false for 8×8 → 6×6 (res 384), which crashed
the second cell. Fixed to `adaptive_avg_pool2d` and verified **bit-identical** on the divisible case
(`torch.allclose` PASS for 8→4), so res 256 stands unchanged and only 384 was re-run. Without that
assert the 384 cell would have silently produced a quotable, meaningless number.

### 35a. Original pre-registration (kept verbatim)

Teacher = stock d12 @512 (64 tokens, 8×8 grid). Student = full-depth **d12** trainable copy fed at
256 (16 tokens, 4×4) and at 384 (36 tokens, 6×6), so depth is held fixed and resolution is the only
structural variable.

The one non-obvious piece: the feature MSE every prior distillation in this line used is
**shape-incompatible** across resolutions. Resolved by average-pooling the teacher's grid to the
student's (8×8 → 2×2 pool → 4×4) — each student token reproduces the mean of the four teacher
tokens covering the same region, which is what the connector's own 4×4 pooling already does one
level up. Loss stays relative MSE so numbers remain comparable to §4/§7/§8.

Verdict rule, fixed before the run: **RECOVERABLE** iff distilled ≥ naive + 20pp at 256 (the size of
the depth-distillation recovery, 0.120 → 0.560 — the precedent's own bar, not a fitted one);
**PARTIAL** iff >5pp but <20pp; **IRRECOVERABLE** iff ≤5pp. Naive controls are re-measured inside
this run rather than taken from step020's separate draw. An IRRECOVERABLE result is a real and
publishable negative: it would say the 512 grid carries information this task needs, and that no
amount of tower training gets it back.

T0 scout budget (1000 train images, 15 epochs), so a null is "at this budget" — the loss curve is
reported so a still-falling curve cannot be read as convergence.

## 36. `vlm_step011` re-run on the step012 checkpoint — **DONE. CAPACITY CONFIRMED (at step004's epoch budget).**

The decisive follow-up from part 5 §18 item 1: does scale-augmented distillation close d6's
small-object gap? n = 3859, mini_mps, 6 cells.

| cell | top1 | agree | vis_ms |
|---|---|---|---|
| d12_f100 | 0.711 | 1.000 | 65.7 |
| d12_f50 | 0.672 | 0.705 | 64.9 |
| d12_f25 | 0.494 | 0.554 | 64.8 |
| d6_f100 | 0.720 | 0.685 | 35.4 |
| d6_f50 | 0.666 | 0.670 | 35.9 |
| d6_f25 | 0.423 | 0.504 | 36.6 |

Double difference I(f) = [d6 shrink cost] − [d12 shrink cost]:
**I(0.25) = −0.0801, boot95 [−0.1003, −0.0593]** — upper bound far below the pre-registered −2.0pp
bar → **CAPACITY**. I(0.5) = −0.0143, boot95 [−0.0334, +0.0052] straddles zero, so the break sits
between f = 0.5 and f = 0.25 and gets reported at that scale rather than rounded into "d6 fails on
small objects".

Both validity anchors pass: d12_f100 = 0.711 exactly, and the independent second criterion — d6_f100
within 2.0pp of the teacher — passes at 0.720. So augmentation did not buy small-object skill by
spending whole-object parity; it simply did not buy small-object skill.

**This is the confound-free direction, as pre-registered.** The training distribution matched the
test manipulation exactly (same gray canvas, same centring, continuous U(0.2, 0.9) against discrete
test scales) and the gap survived — no confound rescues that. The stated caveat travels with the
claim: step012's loss was still falling monotonically at the pinned budget (rel_mse 0.3795 → 0.1514,
last five epochs still dropping), so this is CAPACITY *at step004's epoch budget*, the budget the
control had, and not a proven ceiling.

**Consequence for the drone:** d6's −16.6% params is **scoped to whole-object framing**. Small-object
scenes need d12, or a recovery mechanism other than more scale-augmented feature distillation.

## 38. `vlm_step022` — token budget or pixel grid? **DONE. MIXED — and it caps the whole low-res branch.**

Training-free oracle: tower at **full 512 px** (pixels perfect by construction), its 64 output tokens
average-pooled to k, fed to the frozen decoder. No student at budget k can beat this. n=500, d12 stock.

| k | oracle | step020@k (naive low-res) | pixel_cost_pp |
|---|---|---|---|
| 64 | 0.7040 | 0.7040 | +0.00 |
| 36 | 0.5720 | 0.4560 | **+11.60** |
| 16 | 0.4160 | 0.1600 | **+25.60** |
| 9 | 0.2580 | 0.0940 | +16.40 |
| 4 | 0.1780 | 0.1060 | +7.20 |

Validity gate PASS (k=64 identity pool = 0.7040, Δ 0.7pp from 0.711). Curve is monotone in k.

**CONFIRMED — the two costs are separable and they decompose exactly.** At k=16 the total naive gap
is 54.40pp (0.7040 → 0.1600), and it splits into **28.80pp of token-budget cost** (0.7040 → 0.4160,
perfect pixels) plus **25.60pp of pixel-grid cost** (0.4160 → 0.1600, token count held fixed). Both
walls are real and comparable in size, so MIXED is the honest call — neither hypothesis wins.

**The consequence is sharper than the call, and it is a ceiling, not a budget-limited null.** No
res-256 student can ever exceed 0.4160, because that is the oracle handed the real teacher's
features. So the best possible low-res arm is **−28.8pp for 2.93× tower**; at res 384 it is
**−13.2pp for 1.80×**. Compare depth: d6 held *parity* (0.720 vs 0.711) at 2.0×. **Depth strictly
dominates resolution as a tower lever** — better speedup at zero accuracy cost.

It also re-reads §35 correctly: step021's res-256 student at 0.2980 recovered 13.80pp of the 25.60pp
pixel cost (54%) and has 11.80pp of headroom left to its oracle. The still-falling loss was real
headroom — but buying it maxes out at 0.4160, so it is not worth the epochs.

LIMIT, pre-registered: this bounds **average-pooled** token budgets, not all k-token encodings. A
learned or content-selective 16-token encoder is not covered by this ceiling.

**Process lesson, recorded against myself:** this should have run BEFORE step021. When a
distillation underperforms, first bound what its target could possibly deliver — a training-free
ceiling costs minutes and would have told me which hypothesis step021 was even testing.

## 37. Live roadmap

1. **Resolution is CLOSED as a lever** (§38). Do not spend the d6 × res-256 2×2 the MIXED branch
   pre-registered: its ceiling is now known to be below d6-alone at 512, so the compounding test has
   nothing left to win.
2. **Tower WIDTH is next** — the one structural axis never touched (depth done, resolution capped,
   readout dead). It is also the axis that shrinks params, which resolution never did.
3. **Learned token selection**, if anything revisits tokens — §38's ceiling explicitly does not bind
   it, and it is the only route left to the 2.9× that average pooling cannot buy.
4. **The §32 prologue ablation** — still strictly prior to any int4 work, still in the 18% of
   latency that is not the tower.
5. **d9 retry** — unblocked, unstarted, least valuable.
