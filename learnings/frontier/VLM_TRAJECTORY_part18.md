# VLM trajectory — part 18

§§1–48 are in parts 1–14, §§49–50 in part15, §§51–52 in part16, §§53–54 in part17.

---

## 55. vlm_step037 — the SHAPE arm. DONE. **Unanimous SPIKE. §49's per-launch dispatch-jitter mechanism is FALSIFIED as stated: the excursion is ONE image stalling 246–802 ms.**

`scripts/frontier/vlm_step037_tail_shape.py --modes compile --repeats 16 --n_eval 500`, bf16, r=0.25,
d6 student, 5060ti_cuda. Departure gate carried over **verbatim** from step036 (compile ≥ 4.5 ms), so
this arm could not retune a threshold to a count. **5/16 departures** — consistent with step036's 6/20.

### The pre-registered classification, applied

| pass | mean | excess | max image | top-image share | median-shift share | shape |
|---|---|---|---|---|---|---|
| rep0 | 4.53 | 164 ms | **246.1 ms** | 1.48 | −0.40 | SPIKE |
| rep3 | 4.97 | 382 ms | **399.9 ms** | 1.04 | −0.02 | SPIKE |
| rep5 | 5.24 | 518 ms | **512.2 ms** | 0.98 | 0.04 | SPIKE |
| rep10 | 5.68 | 738 ms | **736.3 ms** | 0.99 | 0.02 | SPIKE |
| rep11 | 5.80 | 800 ms | **802.3 ms** | 1.00 | 0.01 | SPIKE |

Gate was 50%. **Every departure came in at 98–148%.** This is not a marginal call that needed a
threshold argument — one image carries the entire excess in all five, and the median-shift share is
≈0 or negative. rep0's 1.48 is not an anomaly: the single image (246 ms) exceeds the pass excess
(164 ms) because the *other* 499 images ran slightly faster than base, which is what the −0.40
median-shift share says.

### Finding 1 — the mechanism is a STALL, not jitter. §49's attribution is FALSIFIED as stated. CONFIRMED.

§§49–54 have said "per-launch host dispatch jitter" and step033 promoted it by intervention. That
reading is now dead in the form it was written. Per-launch jitter distributes over launches: 500
images each +1.6 ms would produce the identical pass mean. **It does not happen.** The base image
median is 4.157 ms in every departure pass — unmoved — and one image goes to a quarter- to
four-fifths of a *second*.

What survives from step033 is narrower and still true: replacing per-kernel launches with one graph
replay reduced the *incidence* 6/20 → 1/20 (§54). Capture removes most opportunities for the stall.
It does not follow that the stall *is* dispatch jitter, and §54 already showed it does not cap the
depth (the single captured excursion, 7.08 ms, is the deepest in the line).

**Correction to make everywhere:** the excursion is a rare multi-hundred-millisecond stall on a single
image, and the mechanism is now UNKNOWN. Every "dispatch jitter" phrase in §§49–54 is downgraded to
HYPOTHESIS-KILLED; nothing replaces it yet.

### Finding 2 — the stall is positionally random **in the image axis**. CONFIRMED. (In the rep axis it is not — see Finding 4.)

Spike indices: **67, 175, 139, 248, 356** of 500. Not the first image (rules out warm-up and
first-call recompilation), not a fixed index, not a fixed input. Non-departure passes top out at
5.11–5.49 ms — ordinary spread with no intermediate regime. The distribution is **bimodal by pass and
unimodal within a pass**: a pass either contains one ~0.25–0.8 s event or it does not.

### Finding 3 — magnitude grows with rep index. HYPOTHESIS **within this cell** — but see Finding 4, which promotes it.

Departure magnitudes in rep order: 246 → 400 → 512 → 736 → 802 ms. Strictly increasing across all
five. **The §53 rule applies and I applied it to myself here:** a pattern noticed inside the cell that
suggested it, five draws, and any ordering p-value computed on it would be computed on the very
pattern that prompted the look. On this cell alone it is a hypothesis for the *next* cell.

It is the most informative hypothesis available, because it discriminates: an accumulating cost
(allocator fragmentation, cache growth) escalates with rep; an external interrupt (driver, host page
fault, co-tenant) does not. Reps 12–15 ran clean, which a naive escalation reading does not predict —
but step036 also ran clean at reps 12–16 and then departed at rep17 with 6.35, continuing the ramp.
The escalation is in the *departures*, not in every pass.

### Finding 4 — the whole pattern REPLICATES across independent processes, to 0.05 ms. CONFIRMED. **This is the decisive datum of the arm.**

step036 (2026-08-15, R=20) and step037 (separate process, separate script, R=16) ran the same compile
cell. Paired per-rep prefill means:

    036  4.58 4.20 4.17 4.97 4.24 5.25 4.19 4.20 4.19 4.27 5.69 5.81 4.19 4.25 4.30 4.21 | 4.21 6.35 4.21 4.19
    037  4.53 4.21 4.16 4.97 4.24 5.24 4.16 4.18 4.19 4.25 5.68 5.80 4.17 4.24 4.28 4.20 | (R=16)
    |Δ|  .047 .009 .009 .007 .002 .015 .024 .026 .003 .025 .014 .003 .018 .011 .029 .009

**Max |Δ| over all 16 paired reps = 0.047 ms.** Departures fire at the same rep indices (0, 3, 5, 10,
11) with the same magnitudes; base passes match too. Two predictions were written before their
deciding draws and both held — at rep5, "if accumulation is real a later departure exceeds 512 ms"
(rep10 drew 736); at rep10, "rep11 departs at ≈5.81" from step036's value (rep11 drew 5.80).

Three consequences, in increasing order of how much they cost this line:

1. **Finding 3 is promoted.** Escalation-with-rep is no longer a within-cell pattern — it reproduced
   in a second process, which is exactly the next-cell test §53 demands. **CONFIRMED** for pass-level
   magnitude.
2. **The stall is deterministic in the REP axis and random in the IMAGE axis, simultaneously.** Total
   stall per pass is reproducible to 0.05 ms across processes, while the image it lands on moves (67,
   175, 139, 248, 356). Any mechanism must produce a fixed *budget* per rep spent at an unpredictable
   *moment*. That is a strong constraint and it fits item 1 of the hunt below better than anything
   else proposed: reclaim cost is set by allocation history (deterministic in rep), and fires whenever
   the next allocation happens to cross the threshold (arbitrary in image).
3. **Every incidence statistic since §49 was fitted to a reproducible event.** "Roughly 1 in 5",
   6/20, 5/16 — these are not draws from a rate. They are counts of how many fixed positions fall
   inside the chosen R, and per-image they are also a function of `n_eval`: 1 in 1700 at n_eval=500,
   1 in 17 at n_eval=100. **The right unit is stall magnitude (≥802 ms), not incidence.**

### Two of my own kills were wrong. Both reversed here.

1. **§50's "positional lock" was REAL.** §51 dismissed it as reading a rep-index coincidence out of
   five draws, and that dismissal was carried forward for four sections. The *reason* for doubting it
   was sound and the conclusion was false, and no arm was run to check because the correction felt
   like the disciplined move. **Over-correction is also an error mode**; a cheap confirmatory arm is
   the answer, not a stronger prior.
2. **§54's "escalating ramp" kill was MIS-AIMED.** It was killed because rep12 and rep18 returned to
   base. That refutes a *global* ramp — every pass slower than the last — but the pattern was about
   growth in departure magnitude, which is monotone 6/6 in step036 and 5/5 in step037. **Kill a claim
   by testing the claim, not an adjacent one.**

An in-run hypothesis was also raised and killed by the run itself, per the §53 rule: at rep0 a 246 ms
spike in the *first* pass could have been warm-up residue surviving the 20-image warm-up, and rep0 was
named in advance as the weakest position to conclude from. **Rep3's 399.87 ms spike killed it.**

### Why this matters for the drone more than any median in this line

The ship cell's median e2e is 6.66–6.76 ms. **A single frame in roughly one pass of three ate an extra
0.25–0.8 s.** At 500 frames per pass that is one frame in ~1500, but it is a *hard* miss, ~100× the
frame budget, not a soft overrun. No amount of median improvement addresses it, and §54's "worst
observed e2e 11.09 ms" — computed from a per-pass *mean* — understates the per-frame worst case by
two orders of magnitude. **The honest per-frame worst case is ≥802 ms, and this is the number a
real-time claim has to survive.** Nothing in this line may quote a worst-case frame latency from a
pass mean again.

This does not move the shipped efficiency numbers: params 21.28M, 42.6 MB bf16 (8.00× fewer bytes,
counted), top1 0.6880 at n=500 / −3.60pp at n=2000. Those are unaffected. It moves the *latency
guarantee*, which was never separately measured.

### Open — the next arm is a mechanism hunt, not a latency arm

Ranked by cheapness of decisive evidence:

1. **CUDA caching-allocator reclaim.** `torch.cuda.memory_stats()` exposes `num_alloc_retries` and
   `num_ooms`; a retry is exactly the blocking `cudaFree`-and-retry path, and its cost scales with how
   much has to be released. Costs one counter read per image — no new timing methodology — and a
   counter that increments on precisely the stalled image would be decisive in a single pass. It also
   predicts Finding 3's escalation. **Run this first.**
2. Host-side page fault / allocation (process RSS growth over reps).
3. Driver or co-tenant preemption — would not correlate with rep index.
4. `torch.compile` guard failure and silent recompilation mid-pass — testable with
   `torch._dynamo` recompile counters, but disfavoured: recompilation would fire early, not at
   image 356 of rep 11.

**The rebuild is a HARNESS property, and this may not be a shipping defect at all.** Every rep does
`torch._dynamo.reset()`, `del ev`, `torch.cuda.empty_cache()` and rebuilds — a deployed drone builds
once. If the mechanism is reclaim across rebuilds, the stall may not exist in deployment, making it an
EVAL-protocol artifact like the §44 `GraphEval` clone rather than a latency guarantee the paper must
survive. **step038 must distinguish the two**, and the cheapest form is a no-rebuild soak: one build,
R×n_eval images straight through. That is the only version of the question the paper actually needs,
and it costs one run. Do it alongside the `memory_stats()` instrumentation, not after it — if the soak
is clean, item 1's counter tells you *why* it was clean.

Also carried forward from §54 and still open: **tail depth under capture** (n=1; needs ~R=100 or a
soak). Note the two hunts have merged — if the stall is allocator reclaim, the captured cell's deeper
excursion (7.08) is the same event seen through a graph pool, and one mechanism arm answers both.

---

§56 onward continues in `VLM_TRAJECTORY_part19.md` — this file closed at 156 lines.
