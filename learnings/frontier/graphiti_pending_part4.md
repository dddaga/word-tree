# Graphiti pending — part 4

Episodes queued while the graphiti MCP is down. Replay in order once it returns.
Episodes 1–3: `graphiti_pending.md`. Episodes 4–6: `graphiti_pending_part2.md`.
Episodes 7–9: `graphiti_pending_part3.md`.

---

## Episode 10 — vlm_step037, the SHAPE arm (group_id="dhiraj"), 2026-08-15

The most consequential VLM result since the ship cell was fixed. New script
`scripts/frontier/vlm_step037_tail_shape.py --modes compile --repeats 16 --n_eval 500`, bf16, r=0.25,
d6 student, 5060ti_cuda. Departure gate carried over VERBATIM from step036 (>=4.5 ms) so the arm could
not retune a threshold to a count. 5/16 departures. top1 0.6880 on all 16 passes.

**Verdict: unanimous SPIKE, 5/5, not close.** Pre-registered gate was a 50% top-image share; the five
departures came in at 1.48 / 1.04 / 0.98 / 0.99 / 1.00, with median-shift share ~0 or negative. One
image carries the ENTIRE pass excess: stalls of 246 / 400 / 512 / 736 / 802 ms while the base image
median stays at 4.157 ms, unmoved, and the other 499 images are unchanged. (rep0's 1.48 > 1 because
the rest of that pass ran slightly FASTER than base.)

**Consequence 1 — section 49's "per-launch host dispatch jitter" is FALSIFIED.** Jitter distributes
over launches; 500 images each +1.6 ms would give the identical pass mean, and that is not what
happens. What survives from step033 is only that graph capture reduces INCIDENCE (6/20 -> 1/20,
section 54); it does not follow that the stall IS dispatch jitter, and section 54 already showed
capture does not cap depth. Section 54's 50x stage asymmetry still rules out process-global causes, so
the LOCALISATION survives -- only the named mechanism dies. Every "dispatch jitter" phrase in sections
49-54 is HYPOTHESIS-KILLED with nothing yet replacing it.

**Consequence 2, the decisive datum — the whole pattern REPLICATES ACROSS INDEPENDENT PROCESSES to
0.05 ms.** step036 (R=20) and step037 (separate process, separate script, R=16) paired per-rep: max
|delta| over all 16 paired reps = 0.047 ms. Departures fire at the SAME rep indices (0, 3, 5, 10, 11)
with the SAME magnitudes; base passes match too (rep12 4.19/4.17, rep13 4.25/4.24, rep14 4.30/4.28).
Two predictions were written before their deciding draws and both held: at rep5 "a later departure
exceeds 512 ms" (rep10 drew 736), at rep10 "rep11 departs at ~5.81" (rep11 drew 5.80). **The stall is
DETERMINISTIC in the rep axis and RANDOM in the image axis simultaneously** -- spike indices 67 / 175 /
139 / 248 / 356 of 500, never image 0 (rules out warm-up and first-call recompilation). Any mechanism
must produce a fixed BUDGET per rep spent at an unpredictable MOMENT.

**Consequence 3 — every incidence statistic since section 49 was fitted to a reproducible event.**
"Roughly 1 in 5", 6/20, 5/16 are not draws from a rate; they count how many fixed positions fall
inside the chosen R. Per-image they are also a function of n_eval: ~1 in 1700 at n_eval=500, ~1 in 17
at n_eval=100. The right unit is STALL MAGNITUDE (>=802 ms), not incidence and not mean prefill.

**TWO OF THIS LINE'S OWN KILLS REVERSED.** (a) Section 50's "positional lock" was REAL -- dismissed at
section 51 as a 5-draw coincidence and carried as dismissed for four sections. The reason for doubting
was sound and the conclusion was false, and no arm was run to check because the correction FELT like
the disciplined move. **Over-correction is also an error mode; the answer is a cheap confirmatory arm,
not a stronger prior.** (b) Section 54's "escalating ramp" kill was MIS-AIMED -- it refuted a GLOBAL
ramp (rep12/rep18 returned to base) but the claim was growth in DEPARTURE MAGNITUDE, monotone 6/6 in
step036 and 5/5 in step037. **Kill a claim by testing the claim, not an adjacent one.** In-run
hypothesis raised and killed by the run: rep0's 246 ms spike as warm-up residue, killed by rep3's
399.87 ms spike (not a first pass).

**DRONE READING REFRAMED, and this is the paper-relevant part.** Not a fat latency tail -- a ~250-800
ms FREEZE at a reproducible point. At 30 fps an 800 ms stall drops ~24 consecutive frames. Section
54's "worst observed e2e 11.09 ms" came from a per-pass MEAN and understates the per-FRAME worst case
by two orders of magnitude. **Honest per-frame worst case >=802 ms. No worst-case frame latency may
ever again be quoted from a pass mean.** Deterministic is far more likely fixable than stochastic, so
this is a lead, not just a liability. Median ship numbers untouched: prefill 2.75, tower 4.01, e2e
6.76, top1 0.6880, 21.28M params, 42.6 MB bf16 (8.00x fewer bytes, counted). What moved is the LATENCY
GUARANTEE, which had never been separately measured.

**Next arm is a mechanism hunt, not a latency arm. Leading candidate, HYPOTHESIS: caching-allocator
reclaim.** Each rep rebuilds the model (`torch._dynamo.reset()`, `del ev`, `torch.cuda.empty_cache()`),
so the allocation sequence and the fragmentation the allocator must walk evolve with rep index -- the
one candidate predicting determinism, rep-index dependence, monotone growth, AND the fixed-budget /
arbitrary-moment split. Decisive and cheap: `torch.cuda.memory_stats()` `num_alloc_retries` read PER
IMAGE (a retry is exactly the blocking cudaFree-and-retry path). One counter read, no new timing
methodology, settled in a single pass. Disfavoured: host page fault / RSS growth; driver or co-tenant
preemption (would not be deterministic); torch.compile guard failure with silent recompilation (would
fire early, not at image 356 of rep 11). Section 54's open tail-depth arm MERGES into this hunt -- if
the stall is allocator reclaim, the captured 7.08 ms excursion is the same event through a graph pool.

Write-up: learnings/frontier/VLM_TRAJECTORY_part18.md section 55.
Artifact: results/frontier/vlm_step037_tail_shape_bf16_compile_r16_n500_shape__5060ti_cuda.json

---

## Episode 11 — vlm_step038 + vlm_step039: THE PROBE SUPPRESSED THE EFFECT (group_id="dhiraj"), 2026-08-15

step038 (`--arms rebuild soak --mode compile --repeats 16 --n_eval 500`) ran step037's exact protocol
plus allocator instrumentation and got **0/16 departures in BOTH arms** -- against 6/20 (step036) and
5/16 (step037). Base cell unmoved (prefill 4.20-4.28, tower 4.05-4.09, top1 0.6880 every pass, wall
55-60 s, load1 ~1.0), so the box was in the same regime; the departures simply did not happen.
`reserved` pinned flat (628 MB rebuild, 592 MB soak), 0 stalls >=50 ms, **0 alloc retries anywhere**.

**NEITHER pre-registered branch was reachable.** Both the "eval-protocol artifact" read and the
"deployment defect" read required the `rebuild` CONTROL to be dirty. A control that fails is not a
result about the treatment -- reading `soak` CLEAN as "no defect" would have been reading a broken
instrument. This is the trap the arm was one sentence away from falling into.

**step039 -- rerun the UNMODIFIED step037 script as a control -- resolved it immediately.** rep0
departed at prefill 4.55 / img max 250.59 ms against step037's 4.53 / 246.1, and rep3 at 4.99 / 403.19
against 4.97 / 399.9. A THIRD process, matching to 0.02 ms. **Section 55's Finding 4 (determinism
across processes) is UPHELD, not degraded, and Finding 3's promotion stands.** The non-replication was
the PROBE, not the phenomenon.

**Method finding, and the expensive one: an added observation can be an intervention.** step038's only
deltas from step037 were `torch.cuda.memory_reserved()` per image (500x per pass) and
`torch.cuda.memory_stats()` twice per pass. Both are documented as cheap reads. One of them removes a
246-802 ms stall. Had step039 not been run, the honest-looking conclusion "the stall is an eval-protocol
artifact; no latency guarantee is at risk" would have been published off a silently broken control.
**Rule: when a mechanism arm instruments the thing it measures, run the UNINSTRUMENTED control in the
same session.** This is the cousin of the section 52 rule (check the source before buying an
experiment) at the measurement layer instead of the config layer.

**The suppression is itself the strongest mechanism evidence this line has.** A per-image allocator
query eliminating the stall points AT the caching allocator -- a poll that incrementally drains work
which otherwise batches up and fires as one multi-hundred-ms event. Note this survives the zero
`num_alloc_retries`: retries count the blocking cudaFree-and-retry path only, so the reading is
"allocator-adjacent deferred work", not "reclaim" specifically. **If it holds, it is also a FIX** -- a
periodic no-op query costing microseconds that removes an 800 ms freeze is exactly the kind of thing
the drone target needs, and it would convert this whole detour into a deliverable.

**Next arm (step040), pre-registered:** isolate WHICH call suppresses -- step037 verbatim plus ONLY
`torch.cuda.memory_reserved()` per image, versus step037 verbatim plus ONLY the per-pass
`memory_stats()`. Departures survive in the per-pass arm and vanish in the per-image arm -> the
per-image allocator poll is the suppressor and the mechanism is allocator-adjacent deferred work.
Both clean -> suppression is not specific to either call and the hunt widens to any per-image host-side
CUDA API call. Both dirty -> the step038 result is unexplained by these two calls and something else in
that script is responsible. Gate carried over verbatim (compile >=4.5 ms), R>=16 (departures fire at
reps 0 and 3, so a short arm still has power).

**Loose end, logged and NOT claimed:** excess-vs-wall for step036's six compile departures reads
rep0 +185 ms excess / wall +555, rep3 +383 / -410, rep5 +523 / -186, rep10 +744 / +54, rep11 +799 /
+1020, rep17 +1071 / +1069. Base wall spread is ~2 s, which swamps everything below ~1 s, so this
CANNOT falsify the stalls -- but reps 3 and 5 show several hundred ms of "freeze" that never appeared
in elapsed time. Whether a per-image stall shows up in pass wall time is worth one cheap check before
any per-frame worst case goes in the paper.

Artifacts: `results/frontier/vlm_step038_stall_mechanism_bf16_rebuild-soak_r16_n500__5060ti_cuda.json`,
`logs/vlm_step038_stall_mechanism__5060ti_cuda.log`.

---

Episodes 12 onward continue in `graphiti_pending_part5.md` — this file closed at 133 lines.
