# Coarse heat-map — encoder levers (seg_step002–004)

Split out of `seg_coarse_heatmap.md` (200-line limit). That file holds the baseline, the MAC
reference and the noise-floor / method findings; this one holds the encoder-lever experiments.
All cells here are mini_mps.

---
## seg_step002 — encoder FLOP levers (T0, 3 seeds 42/43/44, mini_mps, 2026-08-14)

Six arms, one orthogonal lever each, isolated per the Compounding Rule (all share the encoder
signal path). Every arm ends at stride 8 so the head is untouched and any delta is the encoder's.
All arms carry M_FPM + decor λ=0.05. 18 cells, no failures.

| arm | lever | GMAC | params | mAP (3 seeds) | Δ vs its control |
|---|---|---|---|---|---|
| E0 | VGG b1–4 **pretrained** | 4.898 | 7,226,312 | 0.2378 ±0.0155 | — (pretrained ref) |
| E1 | same shape, **random init** | 4.898 | 8,961,800 | 0.1439 ±0.0049 | — (scratch ref) |
| E2 | DEPTH: drop block4 | 3.232 | 2,195,656 | 0.2349 ±0.0066 | −0.29pp vs E0 |
| E3 | WIDTH 0.5× | 1.331 | 2,630,248 | 0.1467 ±0.0085 | +0.28pp vs E1 |
| E4 | FACTORISATION: separable | **0.896** | 2,200,584 | 0.1343 ±0.0142 | −0.96pp vs E1 |
| E5 | RESOLUTION: stride-2 stem | 2.611 | 8,961,800 | 0.1477 ±0.0106 | +0.38pp vs E1 |

Pretrained weights only exist for unmodified VGG shapes, so E3/E4/E5 are random-init and are read
against **E1 only**. Reading them against E0 would confound the lever with loss of the ImageNet init.

**CONFIRMED — ImageNet init is worth +9.4pp** (E0 0.2378 vs E1 0.1439, ~6σ of the scratch spread).
This is by far the largest effect measured on this task — 10–30× any architecture lever. **The drone
budget is not FLOP-limited, it is initialisation-limited.** Any encoder that hits 1 GMAC must inherit
pretrained weights or be distilled; scratch-training it on 2000 images throws away more than every
architectural cut combined.

**CONFIRMED — block4 is dead weight at a 16×16 grid.** E2 costs −0.29pp (well inside E0's ±1.55pp)
for **−34% MACs, −70% params, −22% wall-time**. Consistent with seg_step001's B2 (M_FPM near-free):
the deep, wide, low-resolution end of VGG contributes almost nothing once the output is coarse.

**CONFIRMED — VGG width is ~4× over-provisioned here.** E3 halves every channel for **3.7× fewer
MACs and 3.4× fewer params at +0.28pp** — the cleanest free lunch in the sweep.

**CONFIRMED — all three cheap levers are ~free; 1 GMAC is reachable on architecture alone.** E4
alone is already **0.896 GMAC, under the drone ceiling**, at −0.96pp — inside its own ±1.42pp spread,
so not a demonstrated cost. E5 confirms the deep blocks can be pushed to a coarser grid for free.

**Noise floor (load-bearing):** E0's pretrained/frozen-trunk arm is the noisy one at ±1.55pp; the
all-trainable scratch arms are tight (±0.49 to ±1.42pp). Any future pretrained-arm lever needs
**> ~3pp** to clear noise at 3 seeds.

**Caveat — scratch arms are epoch-limited, not capacity-limited.** E1/E4/E5 peak at ep16–20, i.e. at
or near the budget end, while pretrained E0 peaks at ep11–14. Every scratch number here is a lower
bound; the 20-epoch budget understates the cheap architectures specifically.

**KILLED — the seg_step001 decorrelation win was seed noise.** B0 seed43 = 0.2426 exceeds B1 seed42
= 0.2200, and E0's 3-seed spread (±1.55pp) is larger than the claimed +1.16pp delta. The step001
HYPOTHESIS above is superseded: anti-Hebbian decorrelation remains null, now on this task too.
(3-seed B0/B1 confirmation still finishing on mini_cpu; the direction is already settled.)

Scripts: `scripts/seg/seg_encoders.py`, `scripts/seg/seg_step002_encoder_flops.py`.
Results: `results/seg/seg_step002_*.json`.

---

## seg_step003 — sliced pretrained init (T0, 3 seeds, mini_mps, 2026-08-14)

seg_step002 left one question: the cheap architectures are free, but they are *scratch*, and the
init is worth +9.4pp. Distillation is one way to hand them that knowledge. **Slicing is cheaper** —
take the first `width` fraction of every pretrained VGG conv's output filters (and the matching
input channels of the next conv). Filter order in VGG is arbitrary, so "first k" is unbiased, not a
selection heuristic. No teacher pass, no extra loss term, no extra training: it is only an init.

| arm | encoder | GMAC | params | mAP (3 seeds) | vs same-shape scratch |
|---|---|---|---|---|---|
| E3 | width 0.5×, random init | 1.331 | 2,630,248 | 0.1467 ±0.0085 | — |
| **E6** | width 0.5×, **sliced init** | 1.331 | 2,630,248 | **0.1893 ±0.0103** | **+4.26pp** |
| **E7** | E6 + drop block4 | **0.876** | **851,816** | **0.1920 ±0.0085** | (no scratch twin) |

**CONFIRMED — a narrow net can inherit the ImageNet init by slicing: +4.26pp (~4σ) at byte-identical
params, MACs and wall-time.** E6 is shape-identical to E3; the *only* variable is the init. This
recovers ~45% of the 9.4pp gap for free, and it is the first mechanism in this line that buys
accuracy without buying compute.

**CONFIRMED — the two free levers compound cleanly on top of it.** E7 stacks width 0.5× + drop
block4 + sliced init and *matches* E6 (+0.27pp, well inside noise) at **1.5× fewer MACs and 3.1×
fewer params**. Compounding was the risk here (both levers touch the encoder signal path, per the
Compounding Rule) and it did not materialise.

**Standing Pareto vs the E0 reference (VGG b1–4 pretrained, 4.898 G / 7.23 M / 0.2378):**

| | GMAC | params | mAP | ceiling |
|---|---|---|---|---|
| E0 reference | 4.898 | 7,226,312 | 0.2378 ±0.0155 | 4.9× over |
| **E7 candidate** | **0.876 (5.6× less)** | **851,816 (8.5× less)** | 0.1920 ±0.0085 (−4.6pp) | **under** |

The drone budget is met on MACs and params. The open item is the remaining **−4.6pp**, and it is
probably not architectural: E7 peaks at ep16–19, i.e. still improving at the budget end, while the
full-width pretrained arms peak at ep10–14. **Longer training is the cheapest next test, before
distillation.**

Results: `results/seg/seg_step002_E6_*.json`, `..._E7_*.json` (same script, `--arm E6/E7`).

---

## seg_step004 — 60-epoch budget rerun (T0, 3 seeds, mini_mps, 6 cells, 1494s, 2026-08-14)

Every seg_step002/003 arm peaked at or near the 20-epoch budget end, so every cheap number was a
lower bound and the E7 gap was an upper bound of unknown tightness. Rerun E7 (candidate) and E2
(best pretrained arm) at 60 epochs, nothing else changed.

| arm | 20 ep | 60 ep | Δ | best_ep @60 |
|---|---|---|---|---|
| E2 (pretrained, 3.232 G) | 0.2349 ±0.0066 | **0.2597 ±0.0037** | +2.48pp | 20 / 21 / 23 |
| E7 (sliced, 0.876 G) | 0.1920 ±0.0085 | **0.2079 ±0.0122** | +1.59pp | 23 / 24 / 29 |
| **gap E2 − E7** | **4.29pp** | **5.18pp** | **widened** | |

**CONFIRMED — the epoch-limit caveat is resolved, and it did not favour the cheap arm.** Both arms
now peak at ep20–29, i.e. well inside the 60-epoch budget, so neither is boundary-limited any more.
Longer training helps both, but helps the *pretrained* arm more: the gap widens from 4.29pp to
5.18pp. **E7's deficit is real capacity/init loss, not a training-budget artifact.**

This closes the cheapest explanation for the gap and promotes distillation from "only if a real gap
survives" to the next experiment. It also upgrades the teacher: E2 @60ep = 0.2597 is now the
strongest pretrained arm measured, above the E0 reference (0.2378 @20ep).

*Method: the ±2pp noise floor and the retracted slot claim live in `seg_coarse_heatmap.md` (seg_step005/006).*

Results: `results/seg/seg_step002_{E2,E7}_t0e60_seed*__mini_mps.json`.

---

## seg_step013 — slice-ratio sweep WITH the feature hint (mini_mps, 10 cells, 2026-08-14)

seg_step002 found width 0.5× essentially free *without* distillation. Once the hint closed the
teacher gap (`seg_feature_hint.md`), the question changed: with the student now constrained by the
teacher's features, is width still free — and is the 1 GMAC drone ceiling costing anything? Three
sliced 3-block students, β=1 hint, paired on seeds 42–46.

| arm | width | GMAC | params | mAP | Δ vs E7 (paired) |
|---|---|---|---|---|---|
| E9 | 0.25 | 0.2636 | 374,680 | 0.2480 ±0.0096 | −1.69pp, sd 0.69, t=−5.49, 0/5 |
| E7 | 0.50 | 0.8757 | 851,816 | 0.2644 ±0.0056 | — (incumbent) |
| E10 | 0.75 | 1.8653 | 1,545,528 | **0.2795 ±0.0053** | **+1.45pp, sd 1.02, t=+3.19, 5/5** |

**CONFIRMED — the 1 GMAC ceiling is binding and costs ~1.45pp.** This was pre-registered as the
branch that would hurt us (a big E10 win means the drone budget constrains the whole approach), and
it fired: E10 beats E7 by +1.45pp on 5/5 seeds. Nothing in this line should henceforth be described
as "the ceiling is free."

**Width is NOT free under the hint** — and neither pre-registered E9 branch fires cleanly. −1.69pp
falls between the "≤1pp ⇒ cut the budget to 0.26G" and ">2pp ⇒ E7 is at the knee" thresholds, so it
is reported as the intermediate it is. Note the contrast with seg_step002, where 0.5× width was free
*without* distillation: a student pinned to the teacher's features has less slack to give up.

**CONFIRMED — ≈1.0–1.4pp per octave of MACs, with no knee.** 0.95pp/octave from E9→E7 and
1.38pp/octave from E7→E10 over a 7× MAC range. The curve through these three points is a shallow
straight line, which means **E7 is not a sweet spot — it is a budget choice**. Any MAC target in
this range can be priced directly: halve the compute, pay about one point of mAP.

**The student beats its teacher.** E10 (0.2795) exceeds the E2 teacher (0.2650) by +1.45pp at
1.8653 vs 3.2324 GMAC — 1.7× less compute than the model supervising it. Consistent with the hint
acting as a regulariser (seg_step011) rather than as pure imitation: a student that only copied the
teacher could not pass it.

Arms `E9`/`E10` in `scripts/seg/seg_encoders.py`; driver `scripts/seg/seg_step010_feature_hint.py`.
Results: `results/seg/seg_step010_H1_w{025,075}_seed4[2-6]__mini_mps.json`.

**Pyramid head levers moved out.** The M_FPM branch ablation and the global-context test
(seg_step022/023) are in `learnings/concepts/seg_mfpm_pyramid.md` — `pool` is dead weight,
`d1` is a Pareto win, and a true global vector is worth nothing at a 16x16 grid.
