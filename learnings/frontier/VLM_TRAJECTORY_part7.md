# VLM Trajectory — part 7 (the energy lever)

Continues `VLM_TRAJECTORY_part6.md` (§19–§22). Parts 5–6 closed the *memory* lever completely: int8
on the two lookup tables ships (§16), depth-6 is the shipping truncation (§19), and int4 at group
granularity ships on top of both (§21). Every one of those wins was **bytes at rest** — §22 measured
`torch._int_mm` at 0.96–1.00× and concluded the energy lever was untouched. This part opens it.

## 24. `vlm_step015` — a hand-written int8 GEMM. **DONE — the 2× is real and capturable. `_int_mm` was the whole problem.** *(All §24/§28 ratios are BATCHED-PREFILL numbers at T=64–576; part 8 §30 shows the deployed shape is T=1 and the number there is 3.05×. Correct as measured, wrong shape to quote for the drone.)*

§22 left a sharp, falsifiable prescription: the memory system offers 2.00×, the fp16 `lm_head` GEMM
is memory-bound at ~86% of peak, and the stock int8 kernel captures none of it — *"so the route is a
Triton int8 GEMM or `torchao`, not abandoning int8."* This takes that route. Triton ships inside
torch, so nothing was installed into the teammates' shared venv.

**The control is the load-bearing part of the design.** Timing a Triton int8 kernel against cuBLAS
fp16 confounds two variables at once — int8-vs-fp16 *and* Triton-vs-cuBLAS — and a mediocre Triton
kernel would then masquerade as "int8 doesn't help on this card", which is exactly the wrong
conclusion §22 was at risk of. So `_gemm` is **one kernel used for both dtypes**: same tiling, same
loop, same autotune grid, differing only in the accumulator (`tl.int32` → IMMA vs `tl.float32` →
HMMA). The `8/16 tri` column is then a one-variable measurement, and `8/cuBLAS` separately prices
what a deployment would actually gain.

| T | cuBLAS fp16 | `torch._int_mm` | Triton fp16 | Triton int8 | 8/16 matched | 8/cuBLAS | correctness |
|---|---|---|---|---|---|---|---|
| 64 | 0.1519 | 0.1560 | 0.1732 | **0.1012** | 1.71× | 1.50× | EXACT |
| 128 | 0.1770 | 0.1815 | 0.2229 | **0.1519** | 1.47× | 1.17× | EXACT |
| 576 | 0.6911 | 0.7187 | 0.8820 | **0.4458** | **1.98×** | 1.55× | EXACT |

Shape `[T,576] @ [576,49280]`, RTX 5060 Ti sm_120, torch 2.11.0+cu128, triton 3.6.0, median of 50.

**CONFIRMED — int8 beats fp16 at every token count, on a matched kernel, with bit-exact output.**
The correctness gate is not a tolerance argument: both sides are exact integer arithmetic, so the
Triton result is compared to `torch._int_mm` with `torch.equal`, and it matched at all three T. A
fast kernel that computed something else would be worthless, and this rules that out rather than
assuming it.

**T=576 recovers 1.98× of the 2.00× §22 proved was physically available.** That is the cleanest cell
and it closes the loop: the ceiling §22 derived from bandwidth was not theoretical, and a competent
kernel reaches it. `_int_mm` was a kernel-quality failure on sm_120, exactly as §22 hypothesized —
that hypothesis is now **CONFIRMED**, by the direct method of writing a kernel that doesn't fail.

**The measurement understates itself, which is the right direction to be wrong in.** The Triton int8
kernel writes an **int32** output — 4 bytes/element against the fp16 baseline's 2 — so at T=576 it
moves 142 MB (28.4 read + 113.5 written) against fp16's 113 MB, i.e. it wins 1.55× while moving
*more total bytes than its competitor*. A real deployment fuses the dequant scale into the epilogue
and writes fp16, removing 56.7 MB of traffic that this benchmark pays for. The number to quote is
therefore a **floor**, not a best case.

**HYPOTHESIS — the mechanism changes with T, and both regimes favour int8.** At T=64 the GEMM is
bandwidth-bound (§22: 386 GB/s ≈ 86% of peak) and int8 wins by halving weight reads. At T=576 the
fp16 arm is doing 32.7 GFLOP in 0.691 ms ≈ 47 TFLOP/s, which is throughput territory, and int8
tensor cores run at 2× the fp16 rate. Not separately ablated; the two regimes are inferred from the
arithmetic, not measured apart.

**The weak cell is reported, not smoothed: T=128 gives only 1.17× vs cuBLAS** — well below its
neighbours at 1.50× and 1.55×. HYPOTHESIS: an autotune gap, since M=128 with `BM=128` leaves
`grid_m = 1` and the config list has no tiling that suits it. Untested. It does not change the
verdict but it does mean the speedup is **not uniform in T**, and a claim quoting 1.98× without
naming the T it came from would be misleading.

### 24.1 The gap this opens, which matters more than the speedup

**The accuracy experiments and the kernel experiment are not measuring the same quantization
scheme, and the trajectory must not silently join them.**

§16 and §21 measured accuracy under **weight-only** quantization — `fake_quant` quantizes the
weights and dequantizes them back to fp32, activations untouched. That is W8A16 (or W4A16 for
`int4_g64`). But `torch._int_mm` and the Triton kernel both require **two int8 operands**: the
runtime win needs **W8A8**, with activations dynamically quantized per forward.

So the state of the evidence is:

| claim | scheme | status |
|---|---|---|
| −33.1% table bytes at parity (§16) | W8A16 | CONFIRMED on accuracy, no runtime measured |
| `int4_g64` ships, +0.67pp at d6 (§21) | W4A16 | CONFIRMED on accuracy, **no int4 kernel at all** |
| 1.17–1.98× on `lm_head` at T=64–576 (§24) | W8A8 | CONFIRMED on speed, **accuracy unmeasured** |

**No single configuration currently has both halves.** Quoting §21's accuracy alongside §24's speed
would be assembling a result that no experiment produced. The honest one-line summary today is:
*int8 weight-only ships on bytes at measured parity; an int8×int8 kernel delivers 1.17–1.98× on the
lm_head GEMM at T=64–576; whether W8A8 holds §16's parity is the next experiment, not a known.*

Two further scope limits on the speed number. It is a **microbenchmark of one GEMM**, not end-to-end
latency — of the two tables §16 quantizes, only `lm_head` is a GEMM; the input embedding is a gather
and no kernel applies. And the benchmark feeds pre-quantized activations, so the per-forward cost of
quantizing them (a `[T,576]` pass — small, but not zero) is excluded.

## 26. `vlm_step016` — W8A8 accuracy. **DONE — SHIPS at both depths. §24.1's gap is closed: the config with the kernel also holds the parity.**

§24.1 said the accuracy line and the speed line described different models, and named the fix. This
is the fix, run: activations dynamically quantized to int8 per token on `lm_head`, weights int8
per row, everything else — harness, 3859 images, seed, `call_ship` rule — untouched from §16.

| cell | top1 | agree | KL | tables MB | paired d vs same-depth fp32 | verdict |
|---|---|---|---|---|---|---|
| `d12_fp32` | 0.711 | 1.000 | 0.000 | 227.1 | — | reference |
| `d12_int8_row` | 0.711 | 0.986 | 0.001 | 57.2 | +0.0000 [−0.0029, +0.0029] | SHIP |
| `d12_w8a8_row` | 0.710 | 0.983 | 0.003 | 57.2 | −0.0010 [−0.0041, +0.0021] | **SHIP** |
| `d6_fp32` | 0.713 | 0.699 | 1.973 | 227.1 | — | reference |
| `d6_int8_row` | 0.708 | 0.698 | 1.987 | 57.2 | −0.0052 [−0.0086, −0.0018] | SHIP |
| `d6_w8a8_row` | 0.708 | 0.698 | 1.987 | 57.2 | −0.0052 [−0.0088, −0.0016] | **SHIP** |

**CONFIRMED — quantizing activations costs essentially nothing on top of quantizing weights.** At
d6 the W8A8 row is *identical to the weight-only row in every metric to three decimals* (0.708 /
0.698 / 1.987) and its paired point estimate is the same −0.0052; the CI widens by 0.0004 and that
is the whole difference. At d12 it costs 0.1pp (0.711 → 0.710, agree 0.986 → 0.983, KL 0.001 →
0.003) — non-zero, and well inside the ±2.0pp tolerance. Interaction with depth is
**ORTHOGONAL** for both arms (`w8a8_row` I = −0.0041 [−0.0091, +0.0005]), so quantization and
truncation still compound.

**Three validity gates passed, and the third is the strong one.** (1) `d12_fp32` = 0.711 and
`d6_fp32` = 0.713 for the sixth time. (2) `int8_row` reproduces step010's cells exactly. (3) Its
*paired lines* are character-identical to step010's — `+0.0000 [−0.0029, +0.0029] b=16 c=16
p=1.00000` at d12, `−0.0052 [−0.0086, −0.0018] b=12 c=32 p=0.00366` at d6, and the same interaction
`−0.0052 [−0.0096, −0.0008]` — despite step010 running on **mps** and this on **cuda**. The
activation-axis refactor of `vlm_quant` introduced zero drift, and that is measured, not assumed.

**HYPOTHESIS — the truncated tower masks the activation noise, which is why d6 pays less than d12.**
d6's output is already far from the teacher (agree 0.699, KL 1.973 against d12's 1.000 / 0.000), so
the extra per-token rounding lands inside an error the truncation already dominates, while at d12
there is nothing to hide behind. Consistent with the numbers but not ablated apart; a d9 cell would
test it, and d9 stays unretried while the foreign kernel holds the card.

**What this does and does not license.** It closes §24.1: the scheme with the 1.17–1.98× kernel is
the same scheme that holds parity, so the two claims may now be quoted about one model. It is still
**two measurements, not one end-to-end run** — this grid uses `fake_quant` (quantize→dequantize in
fp32), so `vis_ms` is unchanged at 13.6 ms by construction and *no speedup is measured here*. And it
covers `lm_head` only; the §16 byte win also quantizes `embed_tokens`, which is a gather that no
kernel touches, so its weight-only status is correct rather than a gap.

**A real trade now exists where §21 read as a clean win.** `int4_g64` beat int8 on accuracy
(+0.67pp at d6) and on bytes — but there is still **no int4 kernel**, while W8A8 has both halves.
So the shipping choice is: `int4_g64` for the smallest static footprint with fp16-speed inference,
or `w8a8_row` for 1.17–1.98× at T=64–576 (3.05× at the deployed T=1, part 8 §30) at −0.52pp. The drone decision depends on
whether it is storage-bound or energy-bound, and the trajectory should stop implying one config
dominates.

## 28. `vlm_step017` — fused dequant epilogue. **DONE — the floor lifts to 1.45–2.14× at T=64–576, and T=64 breaks §22's 2.00× ceiling.** *(Batched-prefill again. Part 8 §30 measures the same kernel at the deployed T=1 and finds fusion worth only 1.01× there — so the fusion recommendation below does NOT carry to the drone.)*

§24 called its own number a floor and named the reason: the kernel wrote **int32**, 4 B/element
against fp16's 2, so at T=576 it won 1.55× while moving *more total bytes than its competitor*.
This fuses the per-token and per-column scales into the epilogue and writes fp16. One kernel with a
`FUSE` constexpr serves both arms — not merely an equivalent main loop but literally the same code,
so `fus/i32` is a one-variable measurement of the epilogue and nothing else.

| T | cuBLAS fp16 | Triton→int32 | unfused total | **fused** | fus/i32 | fus/unfused | **fus/cuBLAS** |
|---|---|---|---|---|---|---|---|
| 64 | 0.1527 | 0.1016 | 0.1987 | **0.0714** | 1.42× | 2.78× | **2.14×** |
| 128 | 0.1773 | 0.1504 | 0.5553 | **0.1226** | 1.23× | 4.53× | **1.45×** |
| 576 | 0.6898 | 0.4471 | 2.6331 | **0.3975** | 1.12× | 6.62× | **1.74×** |

**CONFIRMED — fusion wins at every T, and the drift gate passed for free.** `tri_i32` reproduced
§24's column to within 0.4–1.0% (0.1016/0.1504/0.4471 vs 0.1012/0.1519/0.4458) and cuBLAS to within
0.2%, across two sessions on the same card — so the improvement is the epilogue, not a warmer GPU.
Correctness is **EXACT at all three T**: the fused fp16 output is bit-identical to
`torch._int_mm(...).float() * s_a * s_w` cast to fp16, so nothing was bought with precision.

**HYPOTHESIS — T=64's 2.14× exceeds §22's 2.00× bandwidth ceiling because of L2 residency, not
just byte-halving.** This card's L2 is **33.55 MB** (measured, `L2_cache_size`): the int8 weight
table is 28.4 MB and **fits**, the fp16 table is 56.7 MB and **does not**. Halving the bytes changes
*where they live*, not only how many move, and §22's ceiling was derived from DRAM bandwidth alone
so it cannot bound a cache-resident arm. Not ablated — a direct test would shrink N until the fp16
table also fits and see the ratio fall back toward 2.00×.

**The unfused column is the deployment warning, and it is bigger than the fusion win.** Scaling in a
second pass costs 2.6331 ms at T=576 — **5.9× the GEMM it decorates**, and 3.8× slower than just
keeping fp16. That is the same shape of failure §22 found in dequantize-then-matmul, now in the
epilogue: an int8 model with a naive scale pass is a latency *regression*. Scope limit stated
plainly: this baseline is **eager PyTorch**, which materializes full-size fp32 temporaries, so
`fus/unfused` is an upper bound on fusion's value — a compiled epilogue would land in between. The
defensible numbers are `fus/i32` (1.12–1.42×, one-variable) and `fus/cuBLAS` (1.45–2.14×) — both at
T=64–576, which part 8 §30 shows is a serving shape, not this pipeline's.

**What this still is not: an end-to-end latency number.** It remains a microbenchmark of one GEMM
at the `lm_head` shape, with activations fed pre-quantized. §26 gives the accuracy of this exact
scheme (`w8a8_row`, per-token activations, per-row weights — the same scales this kernel consumes),
so the two now describe one model, but joining them into a single measured forward pass is still
open.

## 29. Next steps

1. **`vlm_step012` — scale-augmented distillation.** RUNNING on mini_mps (ep 17/25 at last check).
   The direct test of §17's CAPACITY vs DISTRIBUTION split, NULL/POSITIVE asymmetry pre-registered.
   The last thing standing between the trajectory and an unscoped drone claim.
2. ~~W8A8 accuracy~~ — **DONE, §26. SHIPS at both depths; §24.1's gap is closed.**
3. ~~A real int8 kernel~~ — **DONE, §24. 1.17–1.98× at T=64–576, 3.05× at T=1 (part 8 §30), bit-exact.**
4. ~~Fused dequant epilogue~~ — **DONE, §28. 1.45–2.14× vs cuBLAS, bit-exact.** What remains of
   this item is the **end-to-end** W8A8 latency number: §26 (accuracy) and §28 (speed) now describe
   one scheme, but no single run measures both. That means wiring the kernel into the real
   `lm_head` forward — a `torch.autograd.Function` or module swap in the eval harness — and
   re-reading `vis_ms`, which every quantization grid so far has left unchanged by construction.
5. **An int4 kernel is now a live question, not a nicety.** §26 leaves a genuine trade —
   `int4_g64` wins bytes and accuracy but runs at fp16 speed; `w8a8_row` wins speed at −0.52pp.
   Either build the kernel or state the trade in the paper; do not quote §21 and §24 together.
6. Unchanged from part 4 §15: **do NOT** attack VLM FFNs or the connector.
7. **Depth is settled at d6** by §19 — do not re-sweep depth without a longer-budget hypothesis for
   d3 specifically. d9 stays unretried while the foreign kernel holds ~13.7 of 16.3 GiB; it would
   also test §26's masking HYPOTHESIS.
