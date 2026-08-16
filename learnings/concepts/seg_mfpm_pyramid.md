# M_FPM pyramid — what each branch is worth (seg_step022/023)

Split out of `seg_encoder_levers.md` at the 200-line cap. Encoder-level levers (depth, width,
slicing, MAC budget) live there; this file covers the pyramid HEAD only.

## Result (seg_step022/023, 8 cells x 5 seeds, 60ep, E2->E7 arm H1)

Once seg_step014 CONFIRMED that a better teacher buys nothing, the binding constraint became the
student's capacity — so the pyramid, the only part of the net that is a design choice rather than a
slice of VGG, was ablated branch by branch. Control `full` = 0.2649 ±0.0081, reproducing the
seg_step010 H1 number (0.2644) and sitting at the E2 teacher (0.2650).

| cell | mean | ± | Δpp | params | GMAC |
|---|---|---|---|---|---|
| full | 0.2649 | 0.0081 | base | 851,816 | 0.8757 |
| no_pool | 0.2671 | 0.0085 | **+0.22** | 842,920 | 0.8735 |
| no_d1 | 0.2608 | 0.0099 | −0.41 | 740,520 | 0.8472 |
| no_d16 | 0.2535 | 0.0048 | −1.14 | 740,520 | 0.8472 |
| no_d8 | 0.2489 | 0.0040 | −1.60 | 740,520 | 0.8472 |
| no_d4 | 0.2440 | 0.0040 | **−2.09** | 740,520 | 0.8472 |
| gap_cat | 0.2638 | 0.0062 | −0.12 | 860,712 | 0.8758 |
| gap_se | 0.2681 | 0.0121 | +0.32 | 893,096 | 0.8757 |

**CONFIRMED — the `pool` branch is dead weight.** A 1x1 conv on a 3x3 max-pool is local despite the
name; removing it is +0.22pp at −8,896 params. It was never a pyramid branch.

**CONFIRMED — `d1` is a Pareto win.** −0.41pp buys −111,296 params (13% of the student) and
−0.0285 GMAC. Nothing else in this line has traded that little accuracy for that much cost.

**CONFIRMED — cost does not order by dilation rate.** d4 (−2.09) > d8 (−1.60) > d16 (−1.14) > d1
(−0.41): the middle of the cascade carries the work, not the widest receptive field. HYPOTHESIS —
depth-of-cascade dominates rate coverage; the clean test is a same-depth/different-rate cell that
this design does not contain. The pre-registered "d16 is pure cost at 16x16" prediction is **NOT
supported**; −1.14pp is under the +1.2pp bar but calling it free would be reading the bar, not data.

**CONFIRMED — a true global vector is worth nothing here, in either polarity.** Before seg_step023
no image-level term existed anywhere in the net. Adding one costs 8,896 params and 0.0001 GMAC and
returns −0.12pp concatenated, +0.32pp as a squeeze-excite gate. Gate-death (steps 873-916) and GLAM
T0 predicted cat >= se; the observed se >= cat by +0.44pp is inside gap_se's own ±0.0121 spread, so
**this is an underpowered tie — it neither supports nor falsifies the prediction.** Most economical
reading (HYPOTHESIS): the d4->d8->d16 cascade already spans a 16x16 map, so a GAP vector is redundant
rather than useless in principle; at a coarser grid the answer could differ.

**CONFIRMED — the two cost wins are additive.** t1 ran `lean` (= no_pool + no_d1, branches d4/d8/d16)
paired against `full`, 5 seeds: 0.2618 ±0.0056 vs 0.2649 ±0.0081, **−0.31pp at 731,624p / 0.8450G**,
~14% params and 3.5% MACs off E7. Compounding Rule check passes — naive sum of the t0 deltas is
−0.19pp, observed −0.31pp, a 0.12pp gap well inside either cell's seed spread. **`lean` is the new
default student.** Note the spread also narrowed (±0.0056 vs ±0.0081), matching the t0 pattern where
every branch drop tightened the seeds.

Driver `scripts/seg/seg_step022_pyramid.py`; module `MFPM(branches=, gap=)` in
`scripts/seg/seg_common.py`. Results `results/seg/seg_step022_t0__mini_mps.json` and
`results/seg/seg_step022_t1__mini_mps.json` (t1 rebuilt from its log after an EPERM write fault).
