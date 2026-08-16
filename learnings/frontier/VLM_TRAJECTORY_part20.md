# VLM trajectory — part 20

§§1–48 in parts 1–14, §§49–50 part15, §§51–52 part16, §§53–54 part17, §55 part18, §§56/58/59 part19.
§57 was reserved for vlm_step040 and lands here because part19 closed at 197 lines.

---

## 57. vlm_step040 — the clean soak. **The stall is an EVAL-PROTOCOL ARTIFACT of per-rep rebuilding. §55's drone paragraph is RETRACTED: the ≥808 ms per-frame worst case does NOT bind a build-once deployment.**

`scripts/frontier/vlm_step040_clean_soak.py --arms rebuild soak --mode compile --repeats 16
--n_eval 500`, bf16, r=0.25 d6 student, 5060ti_cuda. step038's two-arm design with **every allocator
probe removed** — `timed_pass` is step037's verbatim, so nothing here touches CUDA that step037 did
not also touch. Gate carried over verbatim (compile ≥4.5 ms), `STALL_MS` 50.0, and the control's
acceptance band [3,7] plus the known indices [0,3,5,10,11] hardcoded **before any draw**.

### The control reproduced. That is what makes the rest readable.

| rep | step037 | step039 | **step040 rebuild** | step040 stall image |
|---|---|---|---|---|
| 0 | 4.53 | 4.55 | **4.55** | img67, 250.58 ms |
| 3 | 4.97 | 4.99 | **4.99** | img175, 402.40 ms |
| 5 | 5.24 | 5.24 | **5.22** | img139, 499.82 ms |
| 10 | 5.68 | 5.71 | **5.70** | img248, 736.62 ms |
| 11 | 5.80 | 5.83 | **5.83** | img320, 805.60 ms |

**5/16 at reps [0, 3, 5, 10, 11] — the identical index set drawn by step036, step037 and step039.
Four independent processes, one hit-list.** This retroactively settles step038: its 0/16 was its own
probe's doing, not an environment change, and step039's attribution is now confirmed from a second
direction.

### The soak arm: 0/16. Build once, 16 passes, 8000 images.

| | rebuild arm | soak arm |
|---|---|---|
| departures (≥4.5 ms) | **5/16** | **0/16** |
| pass prefill mean | 4.03–5.83 ms | **4.19–4.26 ms** |
| worst single image, whole arm | **805.60 ms** | **5.94 ms** |
| stalls ≥50 ms | 5 | **0** |
| top1 | 0.6880 every pass | 0.6880 every pass |

The soak arm's worst image over 8000 consecutive frames is **5.94 ms** — inside the ordinary spread,
two orders of magnitude below the rebuild arm's worst. The pre-registered branch **rebuild DIRTY +
soak CLEAN** fires, and unlike step038 — whose identically-shaped 0/16 was VOID because its own
control was clean — this read is valid, because the control landed exactly on its band.

### What this retracts

1. **§55's "Why this matters for the drone more than any median in this line" is RETRACTED.** It said
   a real-time claim has to survive a ≥802 ms frame. It does not. That frame is produced by
   `_dynamo.reset()` / `del ev` / `empty_cache()` / rebuild between passes — a **harness** property,
   like the §44 `GraphEval` clone. A deployed drone builds once and did not hit it in 16 consecutive
   passes.
2. **The ≥808.56 ms per-frame worst case stands as a fact about the EVAL protocol and only that.**
   The supersession notes written into §49 (part15) and §54 (part17) remain correct as written — they
   forbid quoting a worst-case frame from a pass mean, which is still forbidden — but both must now
   also carry the scope: ≥808.56 ms is the rebuild-protocol worst case, not the deployment one.
3. **Nothing about the shipped efficiency numbers moves.** 21.28M params, 42.6 MB bf16 (8.00× fewer
   bytes, counted), top1 0.6880 at n=500. Those were never at issue.

### Power — stated, not assumed

**A clean soak at R=16 bounds a RATE, never a mechanism (§51/§53).** What was measured is: over 8000
consecutive frames in a build-once process, zero excursions ≥50 ms. That is an upper bound on the
per-frame stall rate of roughly 1 in 8000 at this confidence, and nothing more. It is **not** proof
that a build-once process never stalls, and a drone flying for hours executes far more than 8000
frames. Two things follow:

- The deployment latency claim the paper may now make is **"no excursion in 8000 consecutive frames
  under the deployed build-once protocol"** — a bound with its n attached, not "the stall is gone".
- A longer soak is the only thing that tightens it. Cost scales linearly and it needs no new code:
  the same script at `--repeats 200` is ~3.3 hours of card time for a ~1-in-100k bound.

### The sharpest new datum, and the one that constrains mechanism

**rep11's stall landed on img320 where step037's and step039's rep11 both hit img356 — while the pass
mean matched to 0.00 ms (5.83 vs 5.83).** The budget is fixed; the position moved. §55's "fixed budget
per rep spent at an unpredictable moment" is demonstrated *within* a single run here rather than
inferred across two. See §60 — this is the datum that keeps the image axis from being fully
determined, and any mechanism must produce both halves.

### Open

The mechanism is still UNKNOWN and is now **demoted in priority**: it explains behaviour that does not
ship. step041 (the probe-isolation ladder, renumbered from step040) is still the arm that would name
it, but it should be reassessed against card time rather than run on momentum. The one hypothesis it
would settle cheaply — which of `memory_reserved()` per image vs `memory_stats()` per pass does the
suppressing — is now a question about the harness, not the product.

---

## 60. The image axis, settled at three processes: **3 of 5 positions fixed, 2 of 5 mobile at fixed cost.** §59's "not random" survives; its "deterministic" does not.

§59 read the cross-process indices with step040's rebuild arm still in flight and counted 4 of 5
fixed. The completed arm gives the third draw on every rep:

| rep | step037 | step039 | step040 | verdict |
|---|---|---|---|---|
| 0 | img67 246.1 | img67 250.6 | img67 250.6 | **fixed 3/3** |
| 3 | img175 399.9 | img175 403.2 | img175 402.4 | **fixed 3/3** |
| 5 | img139 512.2 | img247 513.9 | img139 499.8 | **mobile** (139 / 247 / 139) |
| 10 | img248 736.3 | img248 738.6 | img248 736.6 | **fixed 3/3** |
| 11 | img356 802.3 | img356 808.6 | **img320** 805.6 | **mobile** (356 / 356 / 320) |

**Both moves are at unchanged cost.** rep5: 512.2 / 513.9 / 499.8 ms across two distinct sites.
rep11: 802.3 / 808.6 / 805.6 ms across two distinct sites, with the pass mean identical to 0.00 ms.
Magnitude is a deterministic function of rep index; position is not a function of anything yet
identified.

§59's own hypothesis for rep5 — *"a threshold sitting between two adjacent candidate sites"* — was
written before step040's rep5 landed and predicted nothing specific about which way it would fall, so
its return to img139 neither confirms nor refutes it. But rep11 is a **second independent instance of
the same shape**, drawn after that hypothesis was written, which is what §53 asks for: the pattern
noticed in one cell showed up in the next. Two mobile positions out of five, both at fixed cost, is a
stronger and more specific constraint than either "random" (§55) or "deterministic" (§59).

**Corrected claim, replacing both:** the stall's **cost** is deterministic in the rep axis across
processes (~1%, four processes); its **position** is fixed at 3 of 5 sites and flips between a small
number of nearby sites at the other 2. Neither §55's "positionally random" nor §59's "deterministic in
both axes" is the right description, and both are superseded here.

### Method — the same lesson twice more

§59 closed on *"before buying an arm, check whether the artifacts on disk already answer it"*. §60 is
the other half: **a claim written from a cell still in flight gets amended by that cell, not
defended.** §59's 4-of-5 was stated with three of fifteen comparisons unavailable, and it flagged
that. Flagging it is what made this a one-line amendment instead of a retraction.

---

§61 onward continues in this file until it nears 200 lines.
