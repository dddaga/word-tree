# VLM Trajectory — part 6 (from `vlm_step008`'s salvage)

Continues `VLM_TRAJECTORY_part5.md` (§16–§18). Part 5 closed the memory lever (§16: int8 on the two
lookup tables ships, −33.1% bytes) and bounded it (§17: depth-6 parity does **not** survive when the
object is small — sub-additive). This part records §19, the depth×budget sweep that asks whether
d6's 16.58% saving can be pushed to 24.87%, and §20, two harness bugs the sweep exposed that were
both fixed at the category level because both would have silently recurred. §21 then **reopens a
result part 5 had closed**: §16's kill of int4 was a kill of one *scale layout*, and int4 at group
granularity ships.

## 19. `vlm_step008` — is there a shallower tower than d6? **DONE — d3 KILLED at 25 epochs.**

The question is narrow and worth asking exactly once: §13 established d6 as the deepest truncation
that keeps parity (0.713 vs the teacher's 0.711), but d6 was never shown to be the *shallowest*.
d3 doubles the saving (24.87% of params vs 16.58%) and is 1.84× faster on the vision tower vs d6's
1.44×, so if a longer distillation budget could close its gap the drone configuration changes
materially. The sweep trains d3 and d9 at 25 epochs — d9 as the "budget helps" control — and
evaluates them against the *existing* step004 d6 checkpoint on the same 3859 images, seed 42.

**The d6 anchor is the validity gate, and it passed to the digit:** delta +0.0021, b=440, c=432,
p=0.81263, boot95 [−1.27pp, +1.74pp] → SHIP, reproducing step006's [−1.3pp, +1.7pp] for the fourth
independent time. Without that match nothing in the table is readable.

| arm | params | % saved | top-1 | vision ms | speed |
|---|---|---|---|---|---|
| teacher_d12 | 256,484,928 | 0.00 | 0.711 | 25.8 | 1.00× |
| distil_d6 | 213,957,696 | 16.58 | **0.713** | 13.6 | 1.44× |
| distil_d3 | 192,694,080 | 24.87 | 0.631 | 7.5 | 1.84× |

**d3 is KILLED: delta −0.0796, b=426, c=733, p=0.00000, boot95 [−9.67pp, −6.19pp].** The entire
interval sits far outside the ±2pp parity band — this is not a marginal call, and d6 stands as the
shipping depth. The extra 8.29% of params costs a quarter of the accuracy the whole VLM line exists
to preserve; there is no Pareto reading in which that trades well. Note n = 1159 discordant pairs,
**above the ~1020 overflow threshold of §20.2** — this exact number is why that fix was load-bearing
rather than defensive.

The agreement columns say the same thing from a second direction: d3 matches the teacher's
prediction on 0.609 of images and its argmax agrees on 0.422 with KL 2.721, against d6's
0.699 / 0.499 / 1.973. d3 is not a slightly noisier teacher; it is a different function.

**But the KILL has a scope limit that must travel with it: d3's rel_mse was still falling at epoch
25** (0.2370 → 0.2236 over the last five epochs, ~0.0035/ep, never flattening). Under the
pre-registered floor/ceiling rule (`slope < −1e-3` ⇒ floor) this is a FLOOR reading, so what is
killed is **d3-at-25-epochs, not d3**. That distinction is not a hedge — it is the exact signature
d6 itself showed before its budget grew (0.2573 → 0.2011 → 0.1639), which is what motivated running
this sweep at all. Reopening d3 requires a longer budget, and is worth it only if 8.29% more params
becomes load-bearing for the drone target.

**d9 was never trained.** It hit a CUDA OOM (`tried to allocate 24.00 MiB … 5.94 MiB free`) caused
by another user's 13.36 GiB process on the shared card — an external cause, not a design fault. It
is not retried while that kernel holds ~13.7 of 16.3 GiB, and it is also the least interesting arm:
d9 saves less than d6 and d6 already has parity, so d9 can only confirm a monotone trend that d3's
result already implies the shape of.

## 20. Two harness bugs the sweep exposed. **Both fixed in the library, not in the step script.**

Neither is about VLMs, and both would have recurred on any future sweep, which is why they are
recorded here rather than left in the queue entry.

### 20.1 A late failure discarded every arm that had already finished

`vlm_step008.main()` trained all depths first and ran ONE shared `evaluate()` at the very end. When
d9 OOM'd, the run lost d3's *completed* 5.5-hour student **and** the eval — the external cause was
unfixable, but the blast radius was ours. The fix is `vlm_distill.train_or_resume(ev, ckpt, …)`:
reuse the checkpoint if it exists, else train and save it. A depth sweep is now restartable at arm
granularity, and reusing a saved arm costs nothing in validity because **the checkpoint IS the
object the run would have produced**. Recovered d3 without repeating its distillation.

Two deliberate details:

* **`hist` is None for a resumed arm**, so a caller reading the loss curve reports *unknown* rather
  than inventing one from weights that contain no curve. `verdict(cmp, hist)` already returns
  `(call, None)` when `hist is None`. The d3 floor reading above was therefore taken by hand from
  the crashed run's log — which is where the curve actually lives.
* **Resume-by-existence creates its own hazard**, closed in the same edit: a 1-epoch
  `--smoke_test` checkpoint would silently resume as a real 25-epoch arm, and a stale
  `vlm_step008_d3_1ep__local.pt` was already sitting on the remote. `SLOT` now carries a `_smoke`
  suffix, tagging every artifact of a smoke run (JSON and all checkpoints) at zero net line cost.

### 20.2 `mcnemar_exact` overflowed after a successful 38-minute eval

`OverflowError: int too large to convert to float`. The tail sum `sum(comb(n,i) …)` is a Python int
of order 2^n; multiplying it by `0.5**n` converts it to float first, which **overflows once n passes
~1020 discordant pairs**. This is a threshold, not a fluke — d6 survived only because its n was 872,
meaning every prior VLM result sat just under a cliff that any larger or noisier arm crosses. Fixed
as one exact int ratio, `2 * sum(...) / 2**n`: int/int division is correctly rounded without ever
materialising either side as a float, so it is exact **and** overflow-proof. Verified it reproduces
d6's p to all printed digits (0.81263 from b=440, c=432) and now returns finite p at n=1129
(8.6e-58) and n=20000 (2.0e-45), both previously unreachable.

The eval is deterministic (seed 42, fixed checkpoints), so the crashed run's table stands and the
rerun reproduced it exactly — the numbers in §19 are the same ones the OOM'd run had already
computed.

## 21. `vlm_step013` — int4 at GROUP granularity. **DONE — `int4_g64` SHIPS at both depths.**

§16 killed int4 at *row* granularity (−10.34pp at d12) and the trajectory recorded that as "int4 is
dead". It was not: what died was one *scale layout*. A row scale spans all 576 columns, so a single
outlier column forces a coarse step across every weight in the row; a group scale confines that
damage to its own block. Same script (`vlm_step010_quant_tables.py`, byte-identical), same images,
same pre-registered rule — the only change is a granularity axis in `vlm_quant.fake_quant`.

| cell | top-1 | agree | KL | tables MB | d vs same-depth fp32 | boot95 | verdict |
|---|---|---|---|---|---|---|---|
| d12_fp32 | 0.711 | 1.000 | 0.000 | 227.1 | — | — | ref |
| d12_int4_row | 0.607 | 0.790 | 0.488 | 28.8 | −10.39pp | [−11.51, −9.30] | KILLED |
| d12_int4_g192 | 0.660 | 0.868 | 0.228 | 29.6 | −5.08pp | [−5.99, −4.17] | KILLED |
| d12_int4_g64 | 0.700 | 0.897 | 0.233 | 31.9 | −1.09pp | [−1.84, −0.31] | **SHIP** |
| d6_fp32 | 0.713 | 0.699 | 1.973 | 227.1 | — | — | ref |
| d6_int4_row | 0.627 | 0.617 | 2.365 | 28.8 | −8.60pp | [−9.64, −7.59] | KILLED |
| d6_int4_g192 | 0.689 | 0.687 | 2.072 | 29.6 | −2.44pp | [−3.29, −1.58] | INCONCLUSIVE |
| d6_int4_g64 | **0.720** | 0.696 | 2.278 | 31.9 | **+0.67pp** | [+0.00, +1.35] | **SHIP** |

**The effect is monotone in group size across three points at both depths** — row (576 cols/scale)
→ g192 → g64 recovers 10.39pp → 5.08pp → 1.09pp at d12 and 8.60 → 2.44 → −0.67 at d6. That is the
outlier-confinement account making a quantitative prediction and the grid matching it, not a single
arm squeaking past a threshold. The weight-space POC that motivated the run predicted the same
ordering before any GPU time was spent (rel_mse 0.0304 → 0.0192 → 0.0123).

**Validity gates.** `fp32` reproduced step006's 0.711 / 0.713 for the fifth independent time.
`int4_row` — the control against the granularity refactor — reproduced step010 exactly at d12
(0.607, agree 0.790 vs 0.791) and drifted by **6 images at d6** (0.627 vs 0.633). That drift is
cross-device (step010 ran mps, this ran cuda) on quantized weights of the distilled tower, and is an
order of magnitude below the 10.39pp effect being measured, but it is recorded rather than rounded
away: it means int4_row cells are reproducible to ~0.6pp across devices, not to the digit.

**`g192` at d6 is the rule declining to answer**, not a soft kill: [−3.29, −1.58] straddles the
−2.0pp band, so it is INCONCLUSIVE while the *same arm* at d12 is a clean KILL. Reporting it as
"nearly ships" would be exactly the over-read the pre-registered rule exists to prevent.

**The byte case is real but smaller than the accuracy story suggests.** Group scales are counted,
not assumed: g64 is 4.5 effective bits (31.9 MB/table) against int4_row's 4.06 (28.8 MB) and
int8_row's 8.06 (57.2 MB). So against the §16 shipping config, group-int4's marginal gain is
**25.3 MB of table bytes** — exact, from the run's own accounting — which on the compound d6 config
is roughly 669 MB vs int8's ~686 MB, both against a 1026 MB fp32 teacher. Worth having, not
transformative.

**Interaction: all three arms INCONCLUSIVE, all three positive.** I(g64) = +1.76pp [+0.78, +2.80].
The sign says quantization costs the *truncated* tower less than the teacher — the opposite of a
compounding risk — but every interval crosses the +2pp edge, so nothing is called. Consistent with
step010's int4_row I = +2.33pp; this run's +1.79pp for the same arm sits inside it.

**What ships, and the caveat that decides it.** `d6 + int4_g64` is the new memory champion on
bytes-on-disk and holds 0.720 top-1 against the teacher's 0.711. But **no int4 kernel exists in this
stack** — the eval is quantize→dequantize, so today the win is storage only, while int8 has a real
`torch.ao` path. Until §22 item 3 lands, `d6 + int8_row` remains the config with a runtime story and
`d6 + int4_g64` is the one with the smaller footprint. Do not collapse those two claims.

## 22. `vlm_step014` — does the byte win convert to a latency win? **PROBE, ~2 min. No — and the
reason is the kernel, not the theory.**

§21 left every quantization claim storage-only. Before building an int8 deployment path, one
microbenchmark at the real `lm_head` shape ([T, 576] @ [576, 49280]) on the 5060ti (torch 2.11,
sm_120). `torch._int_mm` is present; no third-party dependency needed.

| probe | fp16 | int8 | ratio |
|---|---|---|---|
| `clone` — pure memory traffic | 0.293 ms / 387 GB/s | 0.147 ms / 386 GB/s | **2.00×** |
| `lm_head` GEMM, T=64 / 128 / 576 | 0.147 / 0.174 / 0.682 ms | 0.150 / 0.175 / 0.711 ms | 0.98 / 1.00 / 0.96× |
| dequantize-then-matmul, T=64 | 0.147 ms | 0.431 ms | **0.34×** |

Three facts, in the order that makes the conclusion unavoidable. **(1) CONFIRMED — the fp16 GEMM is
memory-bound**: 56.8 MB of weights in 0.147 ms = 386 GB/s, ~86% of the card's ~448 GB/s peak, so
halving the bytes *should* halve the time. **(2) CONFIRMED — the hardware delivers exactly that**:
`clone` moves int8 at the same 386 GB/s as fp16, so half the bytes take half the time. The 2× is
physically available. **(3) CONFIRMED — `torch._int_mm` does not capture any of it**, at any of
three token counts spanning 9×.

**HYPOTHESIS for the cause** (inference from the three measurements, not a direct observation):
`_int_mm`'s sm_120 path is unoptimized for this skinny-K, huge-N shape and leaves the available 2×
on the floor. What is *not* in doubt is the actionable half: the ceiling is real and the stock
kernel is what forfeits it, so the route is a Triton int8 GEMM or `torchao`, not abandoning int8.

**The third row is the one that changes a deployment decision.** Dequantize-then-matmul — precisely
what a fake-quant config does if shipped naively — is **2.9× SLOWER** than just keeping fp16. So
storing int8/int4 weights and expanding them at runtime is not latency-neutral, it is a regression.
The §21 byte savings are bytes *at rest* only, and any drone claim must say so until a real kernel
lands. A quantized model with no quantized kernel costs energy rather than saving it.

Cost of learning this: ~2 minutes of GPU time against what would have been days of integration work
predicated on a 2× that the stock kernel never had.

**SUPERSEDED IN PART, by `vlm_step015` (part 7 §24).** The prescription above — "a Triton int8 GEMM
… not abandoning int8" — was taken, and it worked: a hand-written Triton int8 kernel is **1.98×**
the matched fp16 kernel at T=576 and 1.17–1.55× cuBLAS fp16 across T, bit-exact against `_int_mm`.
So `_int_mm` was the *whole* problem, and the storage-only caveat is **lifted for the `lm_head`
GEMM**. What survives from this section unchanged: the memory-bound analysis, the 2× ceiling it
predicted, and the dequantize-then-matmul regression (0.34×) — that path is still the wrong one.

*Continues in `VLM_TRAJECTORY_part7.md` (§24 onward), which also carries the live next-steps list.*
