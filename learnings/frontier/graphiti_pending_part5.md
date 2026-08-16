# Graphiti pending — part 5

Episodes queued while the graphiti MCP is down. Replay in order once it returns.
Episodes 1–3: `graphiti_pending.md`. Episodes 4–6: `graphiti_pending_part2.md`.
Episodes 7–9: `graphiti_pending_part3.md`. Episodes 10–11: `graphiti_pending_part4.md`.

---

## Episode 12 — section 58: the pass is 93% HOST work, and the drone latency claim is model-only (group_id="dhiraj"), 2026-08-15

Re-analysis of existing artifacts, no GPU time: step037 and step039 JSONs, 16 reps each.

**Accounting per 500-image pass:** prefill 2.08-2.91 s (3.6-5.0%), vision tower ~2.0 s (3.5%),
**unaccounted ~54 s (93%)**, wall 57.7-59.8 s. That 93% is host-side `Image.open` / PIL JPEG decode /
HF processor -- ~108 ms per image, single-threaded.

**Finding 1 (CONFIRMED) -- the excess-vs-wall loose end is an INSTRUMENT LIMIT, closed, and it was
never evidence against the stalls.** Base wall spread is 1.71 s (step037) / 2.10 s (step039) on a pass
that is 93% host work; step039 rep3 carried a 403.19 ms stall and ran 2.87 s FASTER than base median.
Host noise is ~5x the signal, so pass wall can neither corroborate nor falsify a sub-second GPU stall
in either direction. The `sync`-bracketed per-image timer (vlm_eval.py:151-155) is the only instrument
with the resolution and it stands; section 55 Findings 1-4 unchanged. Consistency check passes:
prefill totals track the stalls exactly (2.10 s clean -> 2.91 s at the 808 ms departure).

**Finding 2 (CONFIRMED, arithmetic) -- `e2e 6.76 ms` is MODEL-ONLY and has been quoted as a drone
number without that qualifier in every latency sentence of sections 44-56.** It excludes acquisition,
decode, resize, normalize and upload, which in this harness cost ~108 ms/image = **16x the model
time**. The EFFICIENCY claim is untouched -- 21.28M tower params, 42.6 MB bf16, 8.00x fewer bytes are
counted, not timed, and independent of preprocessing. Only the latency framing moves. HYPOTHESIS
(untested): deployed preprocessing is far cheaper -- a camera yields raw frames so JPEG decode
disappears, and resize/normalize can run on GPU. Until measured the honest statement is "6.76 ms
model-only; end-to-end frame latency not yet measured." Queued as step042 (three arms: instrumented
decode/processor/H2D; raw-frame path; GPU resize+normalize), ~10 min, no training.

**Method note.** Sections 55-57 spent five arms on 0.6% of the pass while 93% went unlooked-at for
eight sections. The stall work was right to do -- a per-frame freeze is a hard miss, not a throughput
cost -- but nobody had ever added up where a pass's time goes. **Total-accounting is cheap, needs no
GPU, and is worth doing ONCE per measurement harness before optimizing inside it.**

Write-up: learnings/frontier/VLM_TRAJECTORY_part19.md section 58.

---

## Episode 13 — section 59: CORRECTION, the stall is deterministic in BOTH axes (group_id="dhiraj"), 2026-08-15

Cross-process spike IMAGE indices (same seed 42, identical image order, indices comparable):
rep0 img67 / img67 / img67 (246.1, 250.6, 250.6 ms); rep3 img175 / img175 / img175 (399.9, 403.2,
402.4); rep5 img139 vs img247 (512.2, 513.9); rep10 img248 / img248 (736.3, 738.6); rep11 img356 /
img356 (802.3, 808.6). Processes = step037, step039, step040's in-flight rebuild control.

**4 of 5 spike images reproduce EXACTLY across independent processes. Section 55 Finding 2
("positionally random in the image axis") is WRONG as stated** -- it was inferred from indices
differing across REPS WITHIN one run, which they do, and the across-RUN comparison that actually tests
it was never made even though the artifacts were already on disk.

**Cost to the mechanism argument:** section 55 Finding 4 consequence 2 and the section 56 allocator
ranking both leaned on "a fixed BUDGET per rep spent at an unpredictable MOMENT", and reclaim was
favoured because it explained a deterministic cost fired at an arbitrary threshold crossing. The
moment is NOT unpredictable. New constraint is stronger and simpler: the stall is a deterministic
function of execution history -- same rep, same image, same magnitude to ~1%, in any process.
Allocator-adjacent deferred work still FITS (allocation history is deterministic too) so nothing is
killed, but the "arbitrary moment" half of the argument that SELECTED it is withdrawn and it no
longer discriminates against fully-deterministic candidates.

**rep5 is the informative exception** (img139 vs img247 at 512.2 vs 513.9 ms): one flipped position at
fixed cost is what a threshold between two adjacent candidate sites looks like. HYPOTHESIS; step040's
rep5 is a free third draw and is reported either way.

**Method, second instance in two sections:** section 58 and section 59 both overturned claims by
RE-READING ARTIFACTS ALREADY ON DISK, no new arm. Both were inferences from collected data, stated
without running the comparison that tests them. **Before buying an arm, check whether the results
directory already answers it** -- section 52's "check the source before buying an experiment", pointed
at results/ instead of at the code.

Write-up: learnings/frontier/VLM_TRAJECTORY_part19.md section 59.

