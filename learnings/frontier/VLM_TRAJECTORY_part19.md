# VLM trajectory — part 19

Continues `VLM_TRAJECTORY_part18.md` (§55). §§1–48 in parts 1–14, §§49–50 part15, §§51–52 part16,
§§53–54 part17. Read §55 first: §56 is the arm that reverses its apparent kill.

---

## 56. vlm_step038 + vlm_step039 — the probe suppressed the effect. **Allocator reclaim goes from best-fitting guess to the only hypothesis with an intervention behind it.**

step038 ran two arms (`rebuild` = §55's protocol plus allocator instrumentation, `soak` = build once) and
**both drew 0/16 departures.** Read alone that looks like a kill. It is not, and the reason is the
methodological point of this section.

### The null was a measurement artifact, and step039 is what showed it

step038's `rebuild` arm was the CONTROL: it used step037's protocol precisely so the counters would have
a stall to explain. It reproduced nothing. **A control that fails to reproduce voids its own
experiment's reads** — both pre-registered branches ("EVAL-PROTOCOL ARTIFACT" and "deployment defect")
required rebuild to be dirty. Neither is reachable. The script nonetheless *printed*
`soak CLEAN ... EVAL-PROTOCOL ARTIFACT`, because that line hardcoded the assumption that the control
would be dirty. **That printed conclusion is wrong and must never be quoted.** Pre-registration protects
against tuning a threshold to a count; it does not protect against a branch whose premise fails.

Zero alloc retries across 8000 images was likewise read as "allocator hypothesis KILLED" by the
pre-registration. It says nothing of the sort: **a null drawn from a cell containing no signal is
uninformative, not negative.** Retries absent from a pass with no stall cannot bound whether retries
accompany a stall. Logged as the void-premise rule of §53, applied a second time.

### step039 — the unmodified script, byte-identical, as the discriminating arm

`vlm_step037_tail_shape.py` rerun with no edits (md5 `8bb93b021f9ba4d675519f4d8c82132c`, verified equal
on both machines). Two outcomes only, written before the launch: departs ≥1/16 → step038's added
instrumentation suppressed the effect; 0/16 → the environment changed and §55's Finding 4 degrades to
"deterministic within a window", withdrawing Finding 3's promotion.

**It departed 5/5, at the same indices, with the same magnitudes.**

| rep | step036 | step037 | step039 | max image (037 → 039) |
|---|---|---|---|---|
| 0 | 4.58 | 4.53 | **4.55** | 246.1 → 250.6 |
| 3 | 4.97 | 4.97 | **4.99** | 399.9 → 403.2 |
| 5 | 5.25 | 5.24 | **5.24** | 512.2 → 513.9 |
| 10 | 5.69 | 5.68 | **5.71** | 736.3 → 738.6 |
| 11 | 5.81 | 5.80 | **5.83** | 802.3 → 808.6 |

Max |Δ| vs step037 = 0.03 ms, inside the 0.047 ms of §55's pairing. Reps 1, 2, 4, 6–9 stayed at base,
so this is the whole pattern reproducing, not five cherry-picked hits. **Two predictions were written
down before their deciding draws and both held** — rep5 "≈5.24" drew 5.24, rep11 "≈5.80" drew 5.83.
That is now the third and fourth prospective prediction this line has landed.

### Finding 1 — §55's Finding 4 is UPHELD and strengthened. Finding 3's promotion stands. CONFIRMED.

Determinism across independent processes is now 3-agree / 1-disagree, and the disagreeing process is
the one carrying the probe. P(0 departures in 16 | historic 5/16) = 0.0025 under an iid model and 0
under the determinism §55 argues for — so step038's double-clean was never absorbable as an unlucky
draw. It was a real change, and step039 attributes it to the only thing that differed.

### Finding 2 — reading the allocator 500× per pass makes the stall disappear. **This is an INTERVENTION, and it is the first one this mechanism hunt has had.**

step038's `rebuild` arm differs from step037 in exactly two ways: a `torch.cuda.memory_reserved()` call
after every image, and a `torch.cuda.memory_stats()` call per pass. Nothing else — same protocol, same
gate, same seed, same images, same build path. That probe touches the caching allocator 500 times a
pass and takes its lock each time, and under it a reproducible 250–800 ms stall does not occur.

Every prior statement about allocator reclaim in §55 was HYPOTHESIS from a fit to two axes
(deterministic in rep, random in image). This is different in kind: perturbing the allocator changed
the outcome. **Allocator reclaim is promoted to the LEADING mechanism on interventional evidence** —
still not CONFIRMED, because a lock-taking call also perturbs timing generally and the arm was not
designed as an intervention. The next arm must vary the probe deliberately, and the cheap decisive
form is a probe-frequency ladder: instrument every image / every 50th / every pass / none, in one run.
If incidence rises monotonically as the probe thins, the suppression is the probe's, and the mechanism
is allocator timing. Items 2–4 of §55's hunt (page faults, driver preemption, guard failure) predict no
such gradient.

Corollary worth stating plainly: **a probe that suppresses the effect is evidence about the effect.**
step038 was scored as a failed measurement for one turn before that flipped.

### Finding 3 — the per-frame worst case rises again, to ≥808.56 ms.

Fourth successive supersession: §49's 5.36 → §54's ≥6.35 compiled / ≥7.08 captured (both pass MEANS,
superseded in kind by §55) → §55's ≥802.3 → **≥808.56 ms**, now measured in three separate processes.
"≥" stays the honest operator. At 30 fps that is ~24 consecutive dropped frames at a reproducible point
in the run. Ship medians are untouched: prefill 2.75, tower 4.01, e2e 6.76, top1 0.6880, 42.6 MB bf16.

### What did NOT get answered, and must not be claimed

- **The soak question is still open.** step038's `soak` arm was clean, but so was its control, so the
  arm cannot distinguish "building once fixes it" from "the probe fixed it". **The paper's actual
  question — does a deployed drone that builds once hit this stall? — remains UNANSWERED.** It needs a
  soak arm with NO per-image instrumentation. This is the more important of the two follow-ups: the
  probe ladder explains the mechanism, the clean soak decides whether the drone has a defect at all.
- Tail depth under capture (§54) is still n=1 and still untouched.

---

§57 was reserved for vlm_step040 and is written in `VLM_TRAJECTORY_part20.md` — this file closed at
197 lines before step040 landed. §§58–59 below were written in the gap; §60 (part20) amends §59.

---

## 58. Excess-vs-wall, closed — and the pass is 93% HOST work. **The line's `e2e 6.76 ms` is MODEL-ONLY and has been quoted as a drone number without that qualifier.**

Re-analysis only, no GPU time: `results/frontier/vlm_step037_tail_shape_bf16_compile_r16_n500_{shape,rerun}__5060ti_cuda.json`, 16 reps each (step037 and step039).

### Accounting, per 500-image pass

| component | time | share |
|---|---|---|
| prefill (sum of per-image `ev.forward`) | 2.08–2.91 s | 3.6–5.0% |
| vision tower (`vision_ms` x n) | 1.98–2.05 s | 3.5% |
| **unaccounted (host: `Image.open`/decode/processor)** | **~54 s** | **~93%** |
| pass wall | 57.7–59.8 s | 100% |

~108 ms per image of host-side work — single-threaded PIL JPEG decode from disk plus the HF processor.

### Finding 1 — the excess-vs-wall discrepancy is an INSTRUMENT LIMIT, not a contradiction. CONFIRMED.

§55/§56 logged (and correctly declined to claim) that step036's departures showed hundreds of ms of
excess that never reached elapsed time — rep3 −410 ms, rep5 −186 ms. Resolved: base wall spread is
1.71 s (step037) and 2.10 s (step039), on a pass whose wall is 93% host work. The clincher is
step039 rep3, which carried a **403.19 ms stall and ran 2.87 s FASTER than the base wall median**.
Host-path noise is ~5x the signal. **Pass wall time can neither corroborate nor falsify a sub-second
GPU stall, in either direction.** The GPU-side per-image timer (`vlm_eval.py:151–155`, real wall,
`sync`-bracketed) is the only instrument in this harness with the resolution, and it stands. Nothing
in §55's Findings 1–4 changes. The loose end is CLOSED — it was never evidence against the stalls.

Internal consistency check passes: prefill totals track the stalls exactly (2.10 s clean -> 2.91 s at
the 808 ms departure), so the per-image timer is self-consistent.

### Finding 2 — the drone latency claim is under-qualified. CONFIRMED (arithmetic, not inference).

`prefill 2.75 + tower 4.01 = e2e 6.76 ms` measures **the model and nothing else**. It excludes image
acquisition, decode, resize, normalization and tensor upload. In this harness those cost ~108 ms per
image — **16x the model time**. Every drone-facing latency sentence in §§44–56 quotes 6.76 ms with no
such qualifier.

This does **not** move the efficiency claim, which is the counted one: 21.28M tower params, 42.6 MB
bf16, **8.00x fewer bytes**. Bytes are unaffected by preprocessing. It moves the latency framing only.

**HYPOTHESIS (untested): deployed preprocessing is far cheaper than this harness's.** A drone camera
yields raw frames, so JPEG decode disappears; resize/normalize can run on GPU. Plausible, and the
harness number is certainly an over-estimate of deployment. But it is unmeasured, and the honest
statement until it is measured is: **"6.76 ms model-only; end-to-end frame latency is not yet
measured."** Cheap to fix — one arm timing decode/preprocess separately, with a GPU-resize variant.

### Method note

The stall hunt (§§55–57) spent five arms on 0.6% of the pass while 93% went unlooked-at for eight
sections. Both matter and the stall work was right to do — a per-frame freeze is a hard miss, not a
throughput cost — but **no one had ever added up where a pass's time goes.** Total-accounting is
cheap, needs no GPU, and is worth doing once per measurement harness before optimizing inside it.

---

## 59. **Correction to §55 Finding 2: the stall is NOT random in the image axis.** It is deterministic in BOTH axes, 4 spike images of 5 reproducing exactly across three independent processes.

Cross-process spike **image indices** (same `--seed 42`, so `sample_images` yields the identical order
and the indices are directly comparable):

| rep | step037 | step039 | step040 rebuild (in flight) |
|---|---|---|---|
| 0 | img**67** 246.1 ms | img**67** 250.6 ms | img**67** 250.6 ms |
| 3 | img**175** 399.9 ms | img**175** 403.2 ms | img**175** 402.4 ms |
| 5 | img139 512.2 ms | **img247** 513.9 ms | — |
| 10 | img**248** 736.3 ms | img**248** 738.6 ms | — |
| 11 | img**356** 802.3 ms | img**356** 808.6 ms | — |

**4 of 5 reproduce exactly.** §55 Finding 2 read "positionally random in the image axis" — inferred
from the indices differing **across reps within one run** (67, 175, 139, 248, 356), which they do.
That inference does not license randomness: the across-**run** comparison, which is the one that tests
it, was never made. It was available in the artifacts the whole time.

### What this costs the mechanism argument

§55 Finding 4 consequence 2 and the §56 allocator ranking both leaned on: *"a fixed BUDGET per rep
spent at an unpredictable MOMENT"*, and reclaim was favoured precisely because it explained a
deterministic cost fired at an arbitrary threshold crossing. **The moment is not unpredictable.** The
constraint is now strictly stronger and simpler: the stall is a deterministic function of execution
history — same rep, same image, same magnitude to ~1%, in any process. Allocator-adjacent deferred
work still fits (allocation history is itself deterministic), so nothing is killed; but the
"arbitrary moment" half of the argument that *selected* it is withdrawn, and it no longer
discriminates against candidates that are deterministic end-to-end.

**AMENDED by §60 (part20) once step040 completed its rebuild arm: the count is 3 of 5 fixed, not 4 of 5 — rep11 drew img320 in step040 against img356 twice, at the same cost (805.6 vs 802.3 / 808.6 ms), and rep5 returned to img139. Reps 0/3/10 are fixed across all three processes. The correction below still stands in kind — the axis is not random — but it is not fully determined either; read §60 for the settled version.**

**rep5 is the informative exception** — img139 vs img247, at near-identical magnitude (512.2 vs
513.9 ms). A single flipped position at fixed cost is what a threshold sitting between two adjacent
candidate sites looks like. HYPOTHESIS; step040's rep5 is a free third draw on it and is reported
whichever way it lands.

### Method

This is the second time in two sections that re-reading existing artifacts overturned a claim no new
arm was needed for (§58 was the first). Both claims were *inferences from data already collected*,
stated without running the comparison that would test them. **Before buying an arm, check whether the
artifacts on disk already answer it.** §52's rule was "check the source before buying an experiment";
this is the same rule pointed at the results directory.

