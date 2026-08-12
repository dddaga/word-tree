# VLM trajectory — part 8

Continues `VLM_TRAJECTORY_part7.md`, which closed at §29 (Next steps) and is at its 200-line limit.
Section numbering continues unbroken. Part 7 §29 remains the live roadmap; this part starts by
correcting a premise underneath items 3, 4 and 5 of it.

## 30. `vlm_step018` — what shape does `lm_head` ACTUALLY run at? **DONE — T=1, not T=576. The published speedup was measured on a shape this pipeline never executes, and the true number is BIGGER: 3.05× at T=1, peaking 3.45×.**

**How this was found — by reading the harness, not by running anything.** §24, §28 and every kernel
number in part 7 were benchmarked at `T ∈ {64, 128, 576}`, chosen as plausible prefill lengths. But
`vlm_eval.VLMEval` calls `lm_head` in exactly two places, and neither is a prefill-length GEMM:

- `forward()`: `self.m.lm_head(o.last_hidden_state[:, -1])` — the last position only. **T = 1.**
- `choose()`: `self.m.lm_head(o.last_hidden_state)` over one label's token run — **T ≈ 1–3**, ten
  times per image, and this call is **outside the timed region entirely**.

The 64 image tokens and the prompt go through the *text tower* at full length; they never reach
`lm_head`, because a classification readout needs one distribution, not T of them. So the entire
part-7 kernel line measured a shape the deployment does not run. This is not a small mismatch of
degree — T=1 is a **GEMV**, a different regime from a T=576 GEMM, and the two need not agree in
either magnitude or ranking.

**Result** (same script, same card, `--T 1 2 3 8 16 64`; ms, median of 50):

```
    T  cuBLAS16   tri_i32   unfused     fused  fus/i32  fus/unf  fus/cuB
    1    0.1470    0.0487    0.0602    0.0482    1.01x    1.25x    3.05x
    2    0.1621    0.0482    0.0633    0.0490    0.98x    1.29x    3.31x
    3    0.1629    0.0480    0.0674    0.0477    1.00x    1.41x    3.41x
    8    0.1664    0.0481    0.0685    0.0482    1.00x    1.42x    3.45x
   16    0.1558    0.0494    0.1065    0.0491    1.01x    2.17x    3.17x
   64    0.1522    0.1006    0.1851    0.0715    1.41x    2.59x    2.13x
```

**CONFIRMED — at the deployed shape the int8 readout is 3.05× the fp16 one, and 3.41–3.45× at
T=3–8.** Correctness gate **EXACT at every T**. The error ran in the safe direction: part 7 quotes
1.45–2.14× and understates the deployed win by ~1.4×, rather than claiming a win that is not there.

**CONFIRMED — fusion buys NOTHING at the deployed shape.** `fus/i32` is 0.98–1.01× for every T ≤ 16
and only becomes real at T=64 (1.41×). The mechanism is arithmetic, not mysterious: the epilogue
saves *output* bytes, and at T=1 the output is 49,280 elements (197 KB int32 vs 98 KB fp16) against
a **28.4 MB weight read** that both arms pay identically. Fusion's benefit scales with T; the weight
read does not scale with T at all. **Deployment consequence: the drone can ship the simpler int32
kernel and a separate scale, and lose nothing** — §28's fused epilogue is the right choice for a
batched-prefill serving path and an unnecessary complication for this one. That is the opposite of
what part 7 item 4 implies, and it is the cheaper option that wins.

**The fp16 baseline is flat in T (0.147–0.166 ms from T=1 to T=64) — CONFIRMED weight-bandwidth
bound.** A 576×49280 fp16 table is 56.7 MB; reading it costs the same whether one row or sixty-four
rows are multiplied against it. This is why the readout is a *memory* problem and why quantization,
not sparsity or a smarter GEMM, is the lever that moves it.

**HYPOTHESIS strengthened, still not ablated: L2 residency, not just byte-halving.** §28 measured
`L2_cache_size` = 33.55 MB; the int8 table (28.4 MB) fits and the fp16 table (56.7 MB) does not.
A pure byte-halving argument caps the win at 2.00× (§22's ceiling), and **3.45× exceeds it by 1.7×**
— at *five* separate T values, not one, so it is not a lucky cell. The named ablation is unchanged:
shrink N until fp16 also fits in L2 and check the ratio falls back toward 2.00×.

**Two validity gates passed, one of them free.** (1) The **T=64 overlap cell reproduces §28 within
1%** — cuBLAS 0.1522 vs 0.1527, `tri_i32` 0.1006 vs 0.1016, fused 0.0715 vs 0.0714, `fus/cuB` 2.13×
vs 2.14× — a third session on the same card. (2) That same cell re-passed the correctness gate
**after the reference implementation was changed**, so the change did not quietly weaken it.

**The gate had to be rebuilt to reach T=1, and the fix is a category fix.** `torch._int_mm` raises
`self.size(0) needs to be greater than 16, but got 1` — the reference itself could not evaluate the
deployed shape. The replacement is `(x8.float() @ w8.float()) * s_a * s_w`, which is **bit-exact,
not merely close**: with int8 operands and K=576, every partial sum is an integer bounded by
127·127·576 = 9.29e6 < 2²⁴, hence exactly representable in fp32 **regardless of accumulation order**.
The bound is `assert`ed in the script rather than assumed, so a larger K fails loudly instead of
silently degrading to an approximate gate, and `allow_tf32` is pinned off so exactness does not
depend on a global flag. **One code path for every M** — deliberately not a branch keyed to the
M ≤ 16 case that happened to break, which would have left the same trap for the next shape.

**What this does NOT license.** Still a microbenchmark of one GEMM in isolation:

1. **The end-to-end number is still unmeasured, and is now known to be Amdahl-limited.** `prefill_ms`
   times the whole text tower *plus* one T=1 `lm_head`. Saving 0.099 ms on a call inside a
   multi-millisecond prefill is a small fraction of it. Quoting 3.05× as an end-to-end speedup would
   be exactly the composition error §24.1 warned about, one level up.
2. **The `choose()` calls — ten per image — are untimed by construction.** They are `lm_head` work
   the model genuinely does and the harness has never counted. An honest per-image readout cost must
   include them; this makes the *absolute* saving larger than `prefill_ms` alone can show, so the
   omission is not conservative in the direction one would assume.
3. **The eval harness runs the model in fp32**, so its real `lm_head` baseline is an fp32 GEMM, not
   the fp16 one benchmarked here. fp16 is the right *deployment* baseline and the honest comparison;
   an fp32 comparison would flatter int8 further and should not be quoted.
4. Activations are fed pre-quantized, so the per-forward `[T,576]` quantization cost is excluded —
   at T=1 that is 576 elements, negligible, but it is excluded rather than measured.

## 32. `vlm_step019` — the same kernel, inside the real model. **DONE — the 3.05× is 0.94×. The int8 readout is SLOWER end-to-end than the fp16 baseline it beats 3.05× in isolation, and the composition error §30 warned about was real and pointing the other way.**

§30 ended by naming its own limit: the 3.05× is a microbenchmark of one GEMM, quoting it end-to-end
would be §24.1's composition error one level up, and the honest move is to measure rather than
assert. This measures it. `lm_head` is swapped for a module carrying the **identical** quantization
scheme, the kernel is `import`ed verbatim from `vlm_step017` so there is one definition and no
re-typed variant to drift, and all **11 calls per image** are timed — the `forward()` call plus the
ten `choose()` calls the harness has never counted. Full eval set, n = 3859, depth 12 stock.

```
  arm           top1  agr_fp32  head_ms  fwd_ms  chz_ms  calls  pre_ms  head_MB
  fp32         0.711     1.000   3.2471  0.3036  2.9435   11.0   14.23    113.5
  fp16         0.711     1.000   1.9471  0.1902  1.7569   11.0   14.08     56.8
  w8a8_fake    0.710     0.989   3.8684  0.3786  3.4898   11.0   14.28     28.6
  w8a8_tri     0.710     0.989   2.0694  0.2347  1.8346   11.0   14.15     28.6
  w8a8_trif    0.710     0.989   2.0179  0.2299  1.7880   11.0   14.14     28.6
```

**Both pre-registered gates PASS.** Kernel fidelity `agree(w8a8_tri, w8a8_fake)` = **1.0000** against
a 0.99 bar — the Triton kernel and the pure-PyTorch reference of the same scheme make **identical
predictions on all 3859 images**, so `tri`-vs-`fake` is a clean one-variable measurement of the
kernel. Validity `d12 fp32` = **0.711 vs step006's 0.711 (Δ −0.0002)** — exact for the fifth
independent time. `calls = 11.0` exactly confirms the 1-forward + 10-choose pattern §30 read out of
the source.

**CONFIRMED — against the honest fp16 deployment baseline the int8 readout LOSES: 0.94× on head
time, 0.81× on the `forward()` call.** Fusion does not rescue it (0.96× / 0.83×). Against fp32 it
"wins" 1.57–1.61×, and that is exactly the flattering comparison §30 note 3 said not to quote: the
harness happens to run fp32, but a drone would ship fp16, and fp16 is 1.67× faster than fp32 here
for free. **A 3.05× kernel became a 0.94× module.** The direction matters — an unmeasured
composition claim would have overstated the win by 3.2×.

**HYPOTHESIS for where it goes — per-token activation quantization and launch overhead, not the
GEMM.** The load-bearing evidence is `w8a8_fake`: **3.87 ms against fp32's 3.25 ms**, i.e. *slower
than the baseline with no Triton kernel involved at all*. That 0.62 ms gap is pure quantization
cost — an `abs().amax()`, a divide, a `round`, a `clamp`, a cast, at T=1 where each is a kernel
launch moving 576 elements. §30 note 4 named this exclusion explicitly ("activations are fed
pre-quantized"), and it turns out to be the whole story. Not yet ablated: the named test is to time
the quantization prologue separately from `tri_run`, or to hoist it into the preceding layer.

**CONFIRMED — §30's deployment recommendation survives, for a new reason.** Fusion is worth 1.03×
at the module level (2.0179 vs 2.0694 ms), against 1.01× at the kernel level. So the drone should
still ship the simpler unfused int32 kernel — but not because fusion is marginal, rather because
**fusion is not where the time is**.

**CONFIRMED — the byte win is untouched and is what actually ships.** `head_MB` 113.5 → 28.6 is
**3.97× vs fp32 and 1.99× vs fp16**, arithmetic on the weights and independent of every timing
number above. Accuracy holds: 0.711 → 0.710 with `agree` 0.989 vs fp32. This is the same shape as
§16/§21 — the memory-footprint half of the goal is real and measured; the latency half is not.

**Amdahl, as §30 predicted.** `prefill_ms` is 14.08–14.28 ms across every arm while *total* head
time over all 11 calls is 1.9–3.9 ms, and the harness's `prefill_ms` contains only the `forward()`
share of that (0.19–0.38 ms). Even a free `lm_head` would move `prefill_ms` by ~2%. **The readout is
not the bottleneck of this pipeline**, which is the finding that should govern where effort goes
next.

**Scope limits.** (1) One card (RTX 5060 Ti), one framework, eager PyTorch — a compiled or
CUDA-graph-captured module would amortize the launch overhead this arm pays, and that is precisely
the HYPOTHESIS above, so the result is a statement about *this* deployment path, not about int8. (2)
Depth 12 stock, because vision depth cannot touch `lm_head` latency; nothing here depends on a
distillation checkpoint. (3) `top1` here is 10-way whole-object Imagenette and inherits §17's
scoping unchanged.

## 31. Next steps

Supersedes part 7 §29 where they conflict.

1. **`vlm_step012` — scale-augmented distillation. TRAINING DONE; the VERDICT is still open and is
   now the single blocking item.** 25 ep, rel_mse 0.3795 → 0.1514, cosine 0.8159 → 0.9219, falling
   monotonically through the last five epochs — so a CAPACITY verdict would be *CAPACITY at step004's
   epoch budget*, not a proven ceiling, and must be written that way. The `vlm_step011` re-run that
   reads the verdict off the six-cell grid died mid-cell-1 when the box restarted and needs relaunch.
2. ~~End-to-end W8A8 latency~~ — **DONE, §32, and it answered NO.** 0.94× vs fp16, not 3.05×.
   Both gates passed at n=3859 (fidelity 1.0000, validity Δ −0.0002). The int32-kernel
   recommendation survives; the latency claim does not. Bytes (1.99× vs fp16) are the surviving half.
3. ~~Re-quote the kernel line at T=1~~ — **DONE.** Part 7 §24 and §28 headings now carry an explicit
   *batched-prefill at T=64–576* qualifier pointing here, and the five in-body quotes of 1.17–1.98× /
   1.45–2.14× (§25 table, §25 summary, §26, §28, §29 item 3) are shape-labelled. Nothing was deleted:
   those numbers are correct as measured, they are simply a serving shape this pipeline never runs.
4. **An int4 kernel — attractive for BYTES, no longer for latency.** §30's weight-bandwidth argument
   still says halving weight bytes keeps paying, and §26's `int4_g64` wins accuracy (+0.67pp at d6)
   *and* bytes. But §32 rewrites the latency half of this item: if quantization prologue and launch
   overhead dominate at T=1, int4 will not show up end-to-end either, and int4 needs a *wider*
   prologue than int8. **Build it as a memory-footprint result, and do not pre-commit to a speedup.**
   Cheaper and strictly prior: the §32 prologue ablation, which decides whether item 4 is worth it.
5. Unchanged from part 4 §15: **do NOT** attack VLM FFNs or the connector. Depth stays settled at d6
   (part 7 §19). The d9 retry is **now unblocked** — the foreign process that held ~13.7 of 16.3 GiB
   is gone and the 5060ti is idle (48 MiB used) — but it stays unstarted pending the item-1 verdict.
