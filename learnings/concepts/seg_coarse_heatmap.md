# Coarse-grid event heat-map — drone reframe (seg_step0xx)

**Reframe (user, 2026-08-14):** object detection as *coarse semantic segmentation*. Input 128×128,
output stride 8 → **16×16 grid**, C=4 event classes × K=2 channels. Localisation accuracy is
deliberately sacrificed; a drone operator needs "something happened THERE, and it looks like THIS
kind of event". K channels per class give explainability (confidence = max over K; argmax channel
names which sub-pattern fired).

**Budget:** ≤ 1 GMAC/frame. **Supervision:** cached `fasterrcnn_mobilenet_v3_large_fpn` boxes
rasterised onto the grid — no pixel masks, no annotation, no download (imagenette2-320 on disk).

---

## seg_step001 — baseline + MAC reference (T0, seed 42, mini_mps, 2026-08-14)

Deliberately expensive encoder (VGG16 b1–b4, frozen below block4) so step002's encoder levers have
an honest reference. Teacher cache: 2000 train / 500 val, pos_frac 0.057 / 0.047.

| arm | encoder | M_FPM | decor λ | params | GMAC/frame | best mAP | person | vehicle | animal | object |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | VGG b1–4 | yes | 0 | 7,226,312 | 4.898 | 0.2084 | 0.212 | 0.043 | 0.106 | 0.341 |
| B1 | VGG b1–4 | yes | 0.05 | 7,226,312 | 4.898 | **0.2200** | 0.231 | 0.084 | 0.124 | 0.391 |
| B2 | VGG b1–4 | **no** | 0 | 5,903,880 | 4.559 | 0.2022 | **0.245** | **0.088** | **0.149** | 0.281 |

**CONFIRMED (MAC arithmetic, measured by forward hook, not estimated): the encoder is the entire
problem.** Every arm is ~4.6–4.9 GMAC, i.e. **~5× over the 1 GMAC drone ceiling**, and the head is
free (16K params, ~4 MMAC). M_FPM costs 0.34 G = **7% of encoder**, matching the analytic estimate.
VGG b1–4 cannot meet the drone budget at any stride → step002's encoder levers carry the whole win;
there is nothing to gain by tuning the head.

**CONFIRMED NULL (7 paired seeds): anti-Hebbian decorrelation does not move mAP.** The original
single-seed read (+1.16pp, all four classes up) did not survive replication. Paired B1−B0, same
slot / same seed: mps +1.16 / −1.41 / +1.58 / −0.24 / +0.49 (s42–s46), cpu +3.21 / +0.78 (s43/s44).
**Mean +0.80pp over 7 pairs, 5/7 positive; mps-only mean +0.32pp over 5 pairs** — inside the ±2pp
run-to-run noise floor measured below. Consistent with every prior test (A4b ≈ A4, step003 S2
−0.30pp): the mechanism is inert on accuracy. It costs ~nothing, so it may still be kept **if and
only if the K-channel explainability it was built for is measured and earns it** — that is now the
only open claim for decorrelation, and it needs its own metric, not mAP.

**seg_step020 measured it (5 paired seeds, pre-registered metric and rule).** Over teacher-positive
val cells, on the K=2 pre-max logits: specialisation S = 1−|r(ch0,ch1)|, balance B = 2·min(p,1−p).
**CONFIRMED: decorrelation raises specialisation, +0.121, t=6.57, 5/5 seeds positive** — 2.4× the
pre-registered +0.05 threshold. **The literal pre-registered verdict is nonetheless branch (b)**,
because the rule also required dB ≥ 0 and B came in at **−0.014 with t=−0.38** — a null, but on the
wrong side of a sign test. **The rule as written stands; it was not rewritten after seeing the
data.** Flagged: that clause meant "decor must not kill a channel", and **neither arm shows gate
death** (both ≈0.7 B, *above* the 0.61 random-init floor), so an equivalence test on B is the
disambiguating experiment. **CONFIRMED independent of the verdict — a finding about the K-channel
design, not the penalty: training COLLAPSES specialisation.** Random init is already 0.948
specialised (two random projections of the same features are near-uncorrelated by default);
supervision drives both channels onto the same evidence, down to **0.475–0.596** without decor.
Decor holds **0.623–0.706**. So the penalty *slows a collapse*, it does not *create* structure —
and the headroom for any such mechanism is bounded above by an untrained network.

**HYPOTHESIS (single seed): M_FPM is not earning its 7%.** B2 drops the pyramid for −1.32M params
and −0.34 GMAC, loses only −0.62pp mAP overall, and **beats B0 on 3 of 4 classes** (person, vehicle,
animal). Its entire deficit is the `object` catch-all (0.281 vs 0.341) — the class most likely to
need global context, since it is a grab-bag. On a Pareto basis B2 already dominates B0 outside that
one class. If this survives 3 seeds, the pyramid should be dropped before any other cut.

**Caveats:** single seed; mAP is cell-level AP against a *teacher*, not ground truth, so it measures
teacher agreement; imagenette is COCO-class-narrow (`vehicle` AP is low because vehicles are rare
there — garbage truck is essentially the only source).

**Infrastructure note (CONFIRMED):** `fasterrcnn_mobilenet_v3_large_fpn` **hangs on MPS** — 26 min
elapsed at 0% CPU for a single batch of 32, no error. The teacher therefore runs on its own device
(`--teacher_device`, default `cpu`, ~10s per batch of 32 at 128px) while training stays on MPS.

Scripts: `scripts/seg/seg_common.py`, `scripts/seg/seg_step001_coarse_heatmap.py`.
Results: `results/seg/seg_step001_*.json`.

---

## seg_step005 / seg_step006 — method: noise floor + slot (mini_mps + mini_cpu, 2026-08-14)

**CONFIRMED — the run-to-run noise floor on this task is ~±2pp, not ±0.5pp.** B0 on mini_mps,
5 seeds: 0.2084 / 0.2616 / 0.2302 / 0.2206 / 0.2424 = **0.2326 ±0.0203**. Every 3-seed error bar
earlier in this line therefore *understates* the noise. Implication: the sliced-init result
(+4.26pp, E6 vs E3) is ~2σ of this floor and survives; **no lever delta under ~2pp in this line is
readable at n=3**, and single-seed deltas are worthless. Budget 5 seeds minimum from here.

**Slot (mps vs cpu) is NOT a systematic variable — earlier CONFIRMED tag was wrong and is
retracted.** It rested on one seed-matched pair (B0 s42: 0.2084 mps vs 0.2304 cpu, +2.20pp). Adding
the other two pairs gives **+2.20 / −1.90 / +1.19 (mean +0.50pp, mixed signs)** — indistinguishable
from the ±2pp floor above. The apparent perfect separation of seg_step001's cells (mps
0.2084/0.2200/0.2022 vs cpu 0.2426/0.2421/0.2747) was coincidence over n=3 unpaired cells.
Cross-slot comparison is still bad practice (it adds a nuisance factor for free), but it does not
void prior results by itself — the ±2pp floor does. Method lesson: **do not stamp CONFIRMED on n=1.**

---

## Encoder levers (seg_step002–004) → `seg_encoder_levers.md`

Moved out at the 200-line limit. Summary: width 0.5× is free, dropping block4 is cheap,
depthwise-separable and stride-2 stem both lose; **sliced pretrained init is worth +4.26pp free**;
60-epoch budget does NOT close the E2−E7 gap (it widens 4.29 → 5.18pp). seg_step013 re-ran the width
lever *with* the hint: width is **no longer free** (0.25× costs −1.69pp), the curve is a knee-free
**≈1pp per octave of MACs**, and **the 1 GMAC drone ceiling is CONFIRMED binding at ~1.45pp** —
w0.75 (1.87G) scores 0.2795 and **beats the 3.23G teacher**.

---

## Distillation (seg_step007–009) → `seg_distillation.md`; feature hint (010–011) → `seg_feature_hint.md`

Moved out at the 200-line limit. Summary: logit soft-KD is **CONFIRMED +1.51pp** (n=10 paired,
p≈0.029) but **flat in α over 16×** — it recovers only 1.5 of the 5.2pp teacher deficit. Adding a
**feature hint** on the sliced init (`MSE(s_enc, t_enc[:, :128])`, no projection, zero extra params)
is **CONFIRMED +3.67pp on top of that** (n=10, t=10.77, p<1e-5, 10/10 positive) and **closes the gap
entirely**: student 0.2644 ±0.0056 vs teacher 0.2650, at 3.7× fewer MACs and 2.6× fewer params. It
also collapses the ±2pp seed floor 2.9× — that floor was under-constrained training, not the task.
seg_step011 **falsified** the channel-alignment story (shuffled target is a null, −0.53pp) and
confirmed the hint is **flat in β over 16×**: no knob, robust to scrambling, mechanism = matching the
teacher's feature *distribution*, not its channel identities. seg_step012 then removed the teacher
from the student's weights entirely (random-init E8) and the hint is **CONFIRMED +4.95pp** there
(t=18.2, 5/5) — **architecture-general, sliced init not load-bearing**; random-init + hint even beats
sliced + logit-KD by +2.66pp. seg_step014 closed the mechanism from the third side: distilling from a
**+2.34pp better teacher** (the E10 student itself) bought **+0.10pp** (t=+0.38, 2/5) — **CONFIRMED
NULL, the hint saturates at student capacity**. Consequence: step013's ≈1pp/octave is a
**student-capacity law, not a supervision law**, and **MACs are the only remaining lever** here.

---

## Wall-time Pareto (seg_step015) → `seg_latency_pareto.md`

First **goal-metric** measurement in this line — everything before it is mAP + MACs, a proxy.
**CONFIRMED: 63% of E7's batch-1 latency is fixed overhead, not compute** (fit over E7/E10/E2), so
E2/E7 is only **1.99×** faster against a 3.69× MAC ratio. **CONFIRMED: E9 is Pareto-dominated by E7**
— 3.32× fewer MACs but **1.18× slower** at batch 1, plus −1.69pp from step013. **E7 is the wall-time
floor of the ladder at batch 1.** seg_step016 then tested this file's own "dispatch count is the
next lever" prediction and **FALSIFIED it** (pre-registered branch c): at iso-MAC, 7→3 convs made
the model **6.5% slower**, so the 0.51 ms fixed cost is **per-model, not per-launch**. Pooled,
**b1 latency is architecture-independent below ~1 GMAC** — 0.26–0.88 G all land in 0.81–0.96 ms.
**Further MAC reduction below 1 G buys no wall-time**; both proxies (MACs, dispatch count) are
exhausted for latency, leaving accuracy-per-MAC at ~0.9 G and the unmeasured memory/Joules (MPS has
no per-model peak counter; Joules need the blocked 5060ti). seg_step017/018 then closed the accuracy
side of the same three arms: at iso-MAC and iso-latency, E11 (4 convs) is **−0.31pp** and E12
(3 convs) **+0.38pp, t=+1.79 at n=10** — both null. **CONFIRMED: accuracy-per-MAC at ~0.9 G is flat
to architecture**; depth is not load-bearing and width does not substitute for it. **E7 stays the
operating point on params (0.85M vs 1.09M).** seg_step019 measured memory on CUDA (per-process, so
valid with a co-tenant): **E12/E7 = 1.00×**, whole ladder in **4–27 MB** at b1 — **CONFIRMED memory
is flat to allocation too, and is not binding at batch 1**. **seg_step021 re-ran the latency on a
FREE card and OVERTURNS the flatness claim** → `seg_latency_slot_dependence.md`. Clean b1 is
0.376–0.758 ms, not the contaminated 2.553–3.012 ms; **E9/E7 = 0.878×, so E9 is 12% FASTER than E7**
(p90 0.386 < p10 0.426, outside noise), failing the pre-registered ≥0.95× rule. **"b1 flat below
1 GMAC" was a measurement-resolution artifact of the MPS slot, not a property of the ladder**, and
step015's "E9 is Pareto-dominated by E7" does not hold on CUDA — with E9's −1.69pp it is an
accuracy/latency **trade**, now live for a drone at batch 1. Standing: memory (bit-identical under
contention, so per-process as claimed), b32 compute-bound (71.1% of the MAC ratio), and accuracy-
per-MAC flat. So **params and accuracy are flat to architecture; latency is not, and it is
slot-dependent**. **Joules alone remain unmeasured.**

## Next — seg_step005

1. **Distillation (now justified):** E2 @60ep (0.2597, pretrained) → E7 student, soft-KD on the
   class logits — the step605 K=1 move. The gap is confirmed real and budget cannot close it.
2. ~~Slot control~~ DONE (seg_step005/006) — see the noise-floor block above. Decorrelation is a
   confirmed null on mAP; the ±2pp floor is the real finding.
3. **Slice ratio sweep:** width 0.25/0.75 with sliced init, to find where the free lunch ends.
   Must be ≥5 seeds — the expected effect is near the floor.

