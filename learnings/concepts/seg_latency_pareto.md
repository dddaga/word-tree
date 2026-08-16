# Latency Pareto — measuring the goal metric instead of the MAC proxy (seg_step015)

New file rather than an append: `seg_feature_hint.md` is at 197/200 lines and `seg_encoder_levers.md`
at 162, and this is a distinct axis anyway — every seg result in steps 001–014 is reported in mAP and
MACs, and MACs are a **proxy**. The charter requires the full Pareto (accuracy + params + FLOPs +
wall-time + memory + Joules). bench_step608 already CONFIRMED this codebase can be dispatch-bound
rather than FLOP-bound, so "E7 is 0.8757 GMAC, inside the 1 GMAC drone ceiling" never established
that E7 is fast.

**Method:** random weights (architecture alone sets latency), 200 timed forwards after 30 warmup,
median reported with p10/p90, explicit `mps.synchronize()` per repeat so the timer captures the work
and not the dispatch. Batch 1 is the deployment case — a drone processes one frame at a time.
Decision rule pre-registered in the queue *before the script was written*.

## seg_step015 — the ladder in wall-time (mini_mps, 2026-08-14)

| arm | GMAC | params | b1 ms/frame | b32 ms/frame | mAP (step013/010) |
|---|---|---|---|---|---|
| E9 | 0.2636 | 374,680 | **0.958** | 0.172 | 0.2480 |
| **E7** | **0.8757** | **851,816** | **0.810** | 0.365 | **0.2644** |
| E10 | 1.8653 | 1,545,528 | 1.209 | 0.662 | 0.2795 |
| E2 (teacher) | 3.2324 | 2,195,656 | 1.617 | 0.994 | 0.2650 |

p10/p90 are within ±5% of the median on every cell, so these are not noise reads.

**Pre-registered branch (b) fires — partially compute-bound; the MAC ceiling is directionally right
but overstated.** Measured latency(E2)/latency(E7) = **1.99× at batch 1** and **2.72× at batch 32**,
against a MAC ratio of 3.6912× — i.e. **54% and 74% of the MAC ratio** respectively. MACs are a
loose upper bound on the achievable speedup, not a prediction of it.

**CONFIRMED — most of batch-1 latency is fixed overhead, not compute.** Fitting
`latency = fixed + slope·MACs` across E7/E10/E2 gives fixed = **0.51 ms** at batch 1, which is
**63% of E7's entire 0.810 ms**; at batch 32 the fixed share falls to 36% (0.131 of 0.365 ms/frame).
This is the seg-line instance of the step608 dispatch-bound finding, and it is worse in the drone's
own regime, because batch 1 is exactly where fixed cost cannot amortise.

**CONFIRMED — E9 is Pareto-dominated by E7, and only wall-time could show it.** E9 has **3.32× fewer
MACs than E7 and is 1.18× SLOWER at batch 1** (0.958 vs 0.810 ms). It sits **0.36 ms above** the
fitted line — it does not merely fail to gain, it pays a *penalty*. Mechanism (HYPOTHESIS): E9 has
the same 7 convs and therefore the same dispatch count as E7, but at widths 16/32/64 each kernel is
too small to occupy the GPU, so it pays full per-launch cost for a fraction of the work. Since
seg_step013 already measured E9 at **−1.69pp** accuracy, E9 loses on **both** axes: this is the
first outright Pareto kill in the seg line, and against the MAC proxy alone E9 looked like the
cheapest arm on the ladder.

**This bounds seg_step013's law rather than overturning it.** The "≈1.0–1.4pp per octave of MACs"
result stands as stated — it was a statement about MACs. What changes is the exchange rate: at batch
1, an octave of MACs is **not** an octave of time, and below E7 it buys negative time. So the
accuracy/latency curve has a floor that the accuracy/MAC curve did not: **E7 is the wall-time floor
of this ladder at batch 1**, and the 1 GMAC ceiling's true cost is ~2× time, not ~3.7×.

**Consequence for the drone objective.** The line has been optimising MACs, and MACs recovered only
54% of the available speedup at the deployment batch size. The remaining lever is no longer width —
it is **dispatch count**: fewer, fatter kernels (fusion, or fewer conv layers at greater width) is
predicted to beat further narrowing, which is the opposite of what the MAC objective recommends.
That is the next experiment in this line, and it is a prediction this file is on record for.
**[RETRACTED by seg_step016 — see below. The prediction was tested and is false: 7→3 convs at
iso-MAC made the model 6.5% slower. The paragraph is left standing as written so the retraction has
something to point at.]**

**NOT measured — stated so it is not mistaken for a result:**
- **Memory.** MPS `driver_allocated_memory()` is a driver-wide total, not a per-model peak: every
  arm reported an identical 53 MB at b1 and 1119 MB at b32, which is plainly the allocator pool and
  not the model. The memory column of the Pareto table remains **unmeasured** on this slot; it needs
  CUDA's `max_memory_allocated`.
- **Joules.** Needs `nvidia-smi` power telemetry (see `bench_step997_energy_joules.py`), so it is
  5060ti-only and was blocked by a teammate job at the time of this run.
- **Slot generality.** One slot (mini_mps), one resolution (128). The dispatch-bound conclusion is
  expected to be *stronger* on faster silicon (compute shrinks, launch cost does not), but that is a
  HYPOTHESIS until the 5060ti frees up.

**Method note — the pre-registration named the wrong pair.** The rule keyed on E2/E7, which fired the
middling branch (b), while the load-bearing result came from E9, an arm the rule never mentioned.
Pre-registering the *rule* was still right (it stopped 1.99× being written up as a win), but the
lesson is to key the rule on the **cheapest** arm, since that is where a fixed-overhead floor shows
up first. Recorded so the next latency experiment in any line does not repeat it.

---

## seg_step016 — the dispatch-count prediction above is FALSIFIED (mini_mps, 2026-08-14)

The prediction this file went on record for is **wrong**, and it is retracted here rather than
quietly dropped. Three arms at iso-MAC (within 8%) inside the same 3-block stride-8 stack, varying
only **conv count**, all re-timed in one session so the E7 baseline is same-run:

| arm | width × reps | convs | GMAC | b1 ms/frame | vs E7 | b32 ms/frame | mem |
|---|---|---|---|---|---|---|---|
| **E7** | 0.5 × [2,2,3] | **7** | 0.8757 | **0.855** | 1.000× | **0.365** | 45 MB |
| E11 | 0.75 × [1,1,2] | 4 | 0.8461 | 0.871 | 1.018× | 0.364 | 45 MB |
| E12 | 1.0 × [1,1,1] | **3** | 0.8164 | 0.910 | **1.065×** | 0.383 | 53 MB |

**Pre-registered branch (c) fires — prediction FALSIFIED.** The rule required E12 ≤ 0.65 ms to
confirm and called ≥ 0.79 ms a falsification; E12 came in at **0.910 ms, 6.5% SLOWER than E7** while
also carrying 7% *fewer* MACs. Cutting the conv count from 7 to 3 — a 2.3× reduction in kernel
launches, the exact lever this file predicted would win — **bought nothing and cost a little.**

**CONFIRMED: the 0.51 ms fixed term is per-MODEL, not per-launch.** If it were per-conv it would be
~0.073 ms/conv and E12 should have saved ~0.29 ms. It saved none, so the overhead lives in the
Python/dispatch entry and MPS graph submission for the *forward as a whole*, and **no reordering,
fusing, or thinning of convs can reach it.** The step015 fit stands as a fit; its per-launch
*interpretation* was an unlabelled HYPOTHESIS presented too confidently, and that is the error.

**The larger CONFIRMED result — b1 latency is essentially architecture-independent below ~1 GMAC.**
Pooling both runs, every arm from 0.26 G to 0.88 G lands in **0.81–0.96 ms** at batch 1 (E7 0.810/
0.855, E11 0.871, E12 0.910, E9 0.958) — a 3.3× MAC range and a 2.3× conv-count range compressed
into a **1.18× latency range**. Only E10 (1.87 G, 1.209 ms) and E2 (3.23 G, 1.617 ms) climb out of
it. There is a **~0.85 ms floor** on this slot at batch 1, and E7 sits on it.

**Consequence for the drone objective, and it is the useful one:** *below ~1 GMAC, further MAC
reduction buys no wall-time at all on this slot.* Both proxies the seg line has been optimising —
MACs (step015) and now dispatch count (step016) — are exhausted for latency. E7 is confirmed as the
operating point from two independent directions: narrower is slower (E9), and fatter-and-shallower
is slower (E11/E12). The remaining levers are therefore **accuracy per MAC at ~0.9 G**, and
**memory/Joules**, which are still unmeasured — not further shrinking.

Also worth recording: E12's peak memory reads 53 MB against 45 MB for E7/E11 at b1. On MPS that
counter is a driver-wide pool and was declared unmeasured above, so this is **not** a memory result;
it is only consistent with E12's wider block-3 activations. It needs CUDA to become a claim.

**Method note (second time in this line).** The rule was keyed on the extreme arm this time, which
was the step015 lesson, and it worked — the falsification was unambiguous and cost one cheap
latency run instead of five training seeds. Pre-registration earned its keep by making a prediction
*this file authored* refutable in ~2 minutes. No accuracy runs were launched for E11/E12, per the
rule.

Scripts: `scripts/seg/seg_step015_latency.py` (arms via `seg_encoders.py` `reps` kwarg).
Results: `results/seg/seg_step015_lat__mini_mps.json`, `results/seg/seg_step015_dispatch__mini_mps.json`.

---

## seg_step017 / seg_step018 — the accuracy side of the same three arms (mini_mps, 2026-08-15)

step016 showed the three iso-MAC arms are also iso-*latency*, which made them free sample points on
the one axis left: **does allocation of a fixed ~0.9 GMAC budget change accuracy?** Same H1 hint
config as seg_step010, paired per seed against the existing E7 baseline.

| arm | convs × width | params | mAP | paired Δ vs E7 | sd | t | pos |
|---|---|---|---|---|---|---|---|
| **E7** | 7 × 0.5 | **0.85M** | 0.2644 | — | — | — | — |
| E11 | 4 × 0.75 | 1.11M | 0.2618 | −0.31pp | 1.20 | −0.57 | 2/5 |
| E12 | 3 × 1.0 | 1.09M | 0.2682 | **+0.38pp** | 0.67 | **+1.79** | 7/10 |

**CONFIRMED NULL — accuracy per MAC at ~0.9 G is flat to architecture.** Both arms land inside the
±0.56pp post-hint noise floor. VGG's 7-conv depth is **not** load-bearing, and width does **not**
substitute for it either: the *budget* sets accuracy, not how it is spent. E7 wins the tiebreak on
params alone — same accuracy, same b1 latency, **−22% params** against E12.

**Method note, and it is the point of the pair.** E12 read **+0.55pp, t=+1.59 at n=5** — over the
pre-registered +0.5pp threshold but under significance. Rather than call branch (a) on the mean, the
call was **deferred** and a rule fixed for n=10 (`mean ≥ +0.5pp AND t ≥ 2.26`) *before* seeds 47–51
ran, with the burden deliberately on the favourable branch. The extension **halved the effect**
(+0.55 → +0.38pp) while barely moving t. That is regression to the mean, not an under-powered true
effect — the n=5 read was a lucky draw. Recorded because the cheap, tempting move (stop at n=5, bank
the win) would have put a false "width beats depth" claim into the paper.

**Where this leaves the seg ladder.** Latency is retired below 1 GMAC (step016) and allocation is
retired at ~0.9 G (here). Every axis measured so far is at a ceiling, so the remaining headroom is
**memory and Joules** — both unmeasured at the time of writing. seg_step019 closed memory.

---

## seg_step019 — memory, and the ladder on CUDA (5060ti_cuda, 2026-08-15)

The 5060ti has no cached vgg16, so a bench-only `--no_pretrained` flag rebuilds the sliced/pre arms
as `shaped` at **identical shapes** (GMAC verified bit-equal: 0.8757 / 0.8461 / 0.8164) — weight
*values* cannot change latency or peak memory. Never valid for a training script, and labelled so.

| arm | GMAC | b1 mem | b1 ms | b32 mem | b32 ms/frame |
|---|---|---|---|---|---|
| E9 | 0.264 | **4 MB** | 2.553 | 142 MB | 0.233 |
| **E7** | 0.876 | 12 MB | 2.670 | 278 MB | 0.511 |
| E11 | 0.846 | 10 MB | 2.662 | 187 MB | 0.492 |
| E12 | 0.816 | 12 MB | 2.644 | 246 MB | 0.509 |
| E10 | 1.865 | 19 MB | 2.893 | 415 MB | 0.916 |
| E2 | 3.232 | 27 MB | 3.012 | 553 MB | 1.307 |

**CONFIRMED — memory is flat to allocation, and it is not a binding constraint at batch 1.** CUDA
`max_memory_allocated` is per-process, so it is valid despite a co-tenant. Pre-registered rule was
E12 ≥ 1.10× E7 → memory breaks the tie; measured **1.00×** → branch (b). The whole ladder fits in
**4–27 MB** at b1, i.e. 2.25× spread across a 12.3× MAC range; E7's 0.85M fp32 weights (3.4 MB) are
a *third* of its own footprint. This also **falsifies the MPS 53-vs-45 MB reading** that hinted E12
was heavier — that was the driver pool, exactly as it was declared not to be a result.

**The consequence is the finding: below 1 GMAC the seg ladder has no architectural headroom left.**
Latency (step016), accuracy (step017/018) and now memory are all flat to how the budget is spent.
Only the budget itself moves anything, and step013 priced that at ≈1pp per octave.

**Cross-slot latency — branch (a) fires, but tagged HYPOTHESIS not CONFIRMED.** E9/E7 = 0.956× at
b1 (rule: ≥0.95× → the floor generalises), and the whole b1 column spans 1.18× over 12.3× of MACs —
the same compression MPS showed. The cpu shape-check had predicted the opposite (E12 1.32× *faster*
than E7 there) and that did not reproduce: E12 is 1.01× E7 on CUDA. **Why not CONFIRMED: a teammate
process was resident, contention inflates the fixed term, and that bias points at the branch that
fired.** One re-run on a free GPU settles it. b32 is the clean half and agrees with MPS: E2/E7 =
2.56× against 3.69× MACs (69%, vs 74% on MPS) — **batched, MACs do buy throughput on both slots; it
is batch 1 that is overhead-bound.** Joules remain blocked: power telemetry is device-wide.

> **RETRACTED by seg_step021 (clean card).** The re-run above was made and it **overturns branch
> (a)**, and with it the "flat below 1 GMAC" generalisation and the E9-is-dominated claim. The
> contaminated read was wrong in exactly the predicted direction. See
> **`seg_latency_slot_dependence.md`** — split out at the 200-line limit. The memory result and the
> b32 result are unaffected and stand as written.
