# VLM trajectory — part 17

Continues `VLM_TRAJECTORY_part16.md` (§§51–52). SmolVLM-256M, Imagenette 10-class, d12@512 anchor
0.711, teacher 0.7040. Read §52 first: it retracts §51's Finding 2 and voids this arm's premise.

## 53. vlm_step035 — ran on a void premise. DONE. **Salvaged as a `compile`-mode companion table and 10 excursion draws.**

Launched to distinguish "cudagraph shared-pool coupling" from "a harness bug older than §50" as the
cause of §51's tower/width coupling. §52 then showed there is no coupling to explain: `--ratio` narrows
the vision tower, so the tower moving with ratio is the width axis working as designed. **The
pre-registered read is void and is NOT applied** — under it, the surviving gap would have been called a
foundational harness bug, which would have been a false alarm aimed at the paper's stage split.

What the run is still worth: the `compile`-mode half of §51's table at two ratios, and 10 more draws
toward the overdue R=20 excursion count.

| ratio | mode | tower median (min–max) | prefill median (min–max) | e2e | top1 |
|---|---|---|---|---|---|
| r=0.5 | reduce-overhead (§51) | 4.30 (4.26–4.31) | 2.73 (2.67–2.78) | 7.04 | 0.6940 |
| r=0.5 | compile | 4.37 (4.33–4.39) | 5.03 (4.11–5.20) | 9.42 | 0.6940 |
| r=0.25 | reduce-overhead (§51) | 3.94 (3.90–3.96) | 2.72 (2.69–2.78) | 6.66 | 0.6880 |
| r=0.25 | compile | 4.03 (3.98–4.06) | 4.22 (4.16–4.96) | 8.27 | 0.6880 |

Reading it the right way round: **the tower gap between ratios is the width effect** — 0.36 ms captured,
0.34 ms uncaptured, i.e. narrowing 0.5→0.25 buys ~0.35 ms of tower in either mode. **Capture buys a
further 0.07–0.09 ms of tower and ~2.3 ms of prefill.** top1 is identical to §51 in both cells, so the
two modes compute the same thing. Nothing here needs a mechanism; §50's ship table already said the
width axis is a tower axis ("tower params 85.05M → 21.28M" at r=0.25).

### Excursion count, the one genuinely new datum

Compile-mode student prefill draws, in rep order:

    r=0.5    4.11  4.17  5.03  5.06  5.20     median 5.03, spread 1.09
    r=0.25   4.53  4.19  4.16  4.96  4.22     median 4.22, spread 0.80

r=0.25 keeps the §49 shape: a 4.16–4.22 base with two upward departures (4.53, 4.96), 2/5. r=0.5 does
NOT have that shape — it is two draws near 4.1 and three near 5.1, with no base to depart from. Reported
as spread, not classified as base+tail, because assigning a "base" to a bimodal five-draw sample is the
same over-read this line keeps making. Prefill medians here (5.03 / 4.22) also differ by ratio despite
`--ratio` not touching the text model; at these spreads that is within instrument noise, but it is a
loose end the R=20 arm should close rather than something to explain now.

### Retraction, logged — the "rebuild drift" reading. KILLED at n=2.

r=0.5's prefill ran monotone increasing by rep index, 5/5. I called that a within-process drift across
rebuilds, computed 1/120 from the rank order, and wrote that if it replicated every median back to §44
needed re-reading. It did not replicate: r=0.25 ran 4.53, 4.19, 4.16, 4.96, 4.22, non-monotone with its
largest draw at rep3.

The probability was computed on the very pattern that suggested the hypothesis, which makes it not a
probability. Standing correction: **a pattern noticed inside a 5-draw cell is a hypothesis for the next
cell, never a finding, and never carries an inference-free number.**

### Method note, distinct from the over-reading pattern

§48/§50/§51 were over-reads of small samples. §52 was a different failure and worth keeping separate:
the draws and the statistic were both fine, and the arm still went wrong because it asked what a knob
did without reading what the knob was wired to, then reached for GPU time to explain the answer.
**Check the source before buying an experiment.** Here it was three lines and would have prevented both
§51's Finding 2 and this entire run.

### Standing state after §§51–53

- Ship cell unchanged: r=0.25, prefill 2.72, tower 3.94, e2e 6.66, top1 0.6880, 42.6 MB bf16.
- Additive decomposition intact; no per-stage caveat needed. The tower is the larger half of e2e and
  remains the leading latency target.
- **R=20 excursion arm is the top open item** — four separate small-sample claims in this line have now
  been wrong, and the compile-mode incidence is still unpinned.
- §32 prologue ablation: unblocked, lowest value.

---

## 54. vlm_step036 — the R=20 excursion arm. DONE. **Compile 6/20, captured 1/20. The rate was right; the tail was not, and the R=5 WIN became a PARTIAL.**

§49 item 3, overdue since §51 made it load-bearing. One invocation of `vlm_step033_cudagraph.py` at
`--modes compile reduce-overhead --towers student --ratio 0.25 --repeats 20 --n_eval 500`. Card verified
empty (48 MiB, 0%) at launch; `load1` 0.74–1.05 across all passes, so nothing below is a load artifact.

**Classification rule, committed at rep9 before reps 10–19 were seen: departure = prefill ≥ 4.5 ms.**
Stated in advance on purpose. Base draws were filling the 4.24–4.30 gap that §49's clean base/tail split
assumed away, and a threshold chosen after seeing the draws is a threshold tuned to a count.

### Compile cell, all 20 draws in rep order (n=500, bf16, student, r=0.25)

    4.58* 4.20  4.17  4.97* 4.24  5.25* 4.19  4.20  4.19  4.27
    5.69* 5.81* 4.19  4.25  4.30  4.21  4.21  6.35* 4.21  4.19      (* = departure)

| stage | median | min | max | tail |
|---|---|---|---|---|
| prefill | 4.23 | 4.17 | **6.35** | 2.12 |
| tower | 4.07 | 3.97 | 4.10 | 0.04 |
| e2e | 8.29 | 8.19 | 10.43 | 2.13 |

top1 0.6880 on every one of the 20 passes. The excursion is a latency phenomenon only; it never touches
what the model computes.

### Finding 1 — incidence UPHELD at 6/20. CONFIRMED.

Pre-registered: ≥4/20 upholds §49's "roughly 1 in 5". **6/20 = 30%**, so the rate stands and was if
anything understated. The 95% interval is ~12–54% — wide, and quoting the point estimate without it
would repeat this line's habit. What is now settled is the thing R=5 could not settle: the excursion is
a real, reproducible property of the compile cell, not an artifact of three unlucky early runs.

### Finding 2 — the tail is far heavier than measured. CONFIRMED. **This supersedes the worst case §49 has been quoting.**

Departures ran **4.58 / 4.97 / 5.25 / 5.69 / 5.81 / 6.35**, spanning 2.12 ms above a 4.17–4.30 base with
no visible upper bound. §49 measured a max of 5.36 from 10 draws and that number has been the assumed
worst case ever since. It is **at least 1.0 ms low** — 6.35 is +18% on it and +50% on base prefill.

For the drone target this matters more than the median. A frame budget is set by worst-case latency, not
by a median, so every budgeting statement in this line that used 5.36 needs restating at ≥6.35, and
"≥" is the honest operator: 20 draws bound the tail from below only.

### Finding 3 — the stage asymmetry is stark, and it localises the cause. CONFIRMED.

Tower tail 0.04 ms against prefill tail 2.12 ms — **50×**, in the same 20 passes, same process, same
images. The tower is the *larger* stage by median (4.07 vs 4.23) and is nearly noiseless; the excursion
sits entirely in prefill. Rules out anything acting on the process as a whole (thermal, clock, host load,
co-tenancy) — those would move both stages. Consistent with per-launch host dispatch, which is what
prefill has that the compiled tower does not.

### Two in-run hypotheses, both raised and both killed by the run itself

Logged because the §53 rule says a mid-cell pattern is a hypothesis for the next draws, and this is the
first cell long enough to actually test its own patterns:

1. **Tower creep.** Reps 0–4 rose 3.97 → 4.06 monotonically; I flagged it as possible rebuild drift.
   Reps 5–19 sat flat at 4.05–4.10. Early-rep warmup, not drift. **Killed.**
2. **Escalating ramp.** Departures 4.97 → 5.25 → 5.69 → 5.81 escalated, with reps 10 and 11 adjacent —
   suggesting a thermal or clock ramp. Prediction recorded at rep11: a ramp keeps climbing and never
   returns to base; independent draws return. **Rep12 returned 4.19, and rep18 returned 4.21 straight
   after the 6.35 maximum. Killed.** The departures are independent draws that clustered by chance.

Both predictions were written down before the deciding reps arrived. That is the intended discipline and
it is the first time in this line it has been followed prospectively rather than repaired afterwards.

### Reduce-overhead cell, all 20 draws in rep order — **WIN at R=5 became PARTIAL at R=20.**

**Rule committed at rep0 before any draw was seen: departure = prefill ≥ 3.0 ms** (base sits at 2.7x,
so 3.0 is the same ~+10% offset the compile rule used, fixed in advance for the same reason).

    2.80  2.74  2.74  2.82  2.79  2.76  2.77  2.74  2.76  2.74
    2.73  2.75  2.75  2.77  2.75  7.08* 2.79  2.83  2.74  2.74      (* = departure)

| stage | median | min | max | tail |
|---|---|---|---|---|
| prefill | 2.75 | 2.73 | **7.08** | 4.33 |
| tower | 4.01 | 3.93 | 4.03 | 0.03 |
| e2e | 6.76 | 6.70 | **11.09** | 4.33 |

top1 0.6880 on all 20 passes — identical to the compile cell, so capture is still computing the same
thing. `reduce-overhead_student vs compile: top1 delta 0.0000  median gain +1.47 ms  tail gain −2.21 ms
-> PARTIAL`.

**§50 called this same comparison WIN on R=5. At R=20 it is PARTIAL.** Nothing about the cell changed;
only the number of draws did. That is the fourth time in this line an R=5 absence claim has failed, and
the first time the failure was caught by the pre-registered arm rather than by a later accident.

### Finding 4 — incidence and tail-depth now point OPPOSITE ways. CONFIRMED (incidence), UNDERPOWERED (depth).

- **Incidence: capture wins, decisively.** 1/20 against compile's 6/20. §51's 1/15 estimate holds.
- **Depth: capture's single excursion is the worst draw in the entire run.** 7.08 exceeds compile's
  largest (6.35), and worst-case **e2e is worse captured (11.09) than compiled (10.43)**. Median e2e is
  1.53 ms better captured, so this is purely a tail statement.

The depth comparison is **1 draw against 6 and cannot be claimed as a finding** — a single observation
bounds nothing. What IS settled is that capture does not cap the excursion: the mechanism that made it
rarer did not make it shallower, and at least one captured excursion runs deeper than any compiled one.

**This matters for the drone more than the median does.** A frame budget is set by worst case. §50/§51
have been letting a *rate* claim stand in for a *tail* claim — "cudagraphs reduce the excursion" is true
of frequency and unsupported for depth. On worst-case e2e, capture is currently behind.

### Open

- **Tail depth under capture is the load-bearing unknown**, and n=1 does not touch it. Needs an arm
  sized for tail depth, not incidence — R=20 gives ~1 captured excursion, so ~R=100 or a targeted
  long-run soak. Ship claim stays on median; no worst-case e2e claim may be published from this run.
- **§49's 5.36 ms worst case is superseded twice over** — ≥6.35 compiled, ≥7.08 captured. Every
  budgeting line in §§44–51 that quotes 5.36 needs restating, with "≥".
  **Superseded again (2026-08-15, §§55–56): ≥6.35 and ≥7.08 are themselves pass MEANS. §55 showed the
  excursion is ONE image, and §§55–56 measured it at ≥808.56 ms in three processes. Restate every
  worst-case frame claim in §§44–54 at ≥808.56 ms; the 11.09 ms "worst observed e2e" below is a mean and
  is NOT a per-frame number.** Medians and the ship table are untouched.
- Base draws late in the compile cell (4.25/4.27/4.30) sat above early ones (4.17–4.20). Not acted on —
  the committed rule kept it out of the count. If real it needs an arm with rep index as the variable.

### Ship cell, restated

r=0.25 captured: prefill 2.75, tower 4.01, e2e 6.76 median, top1 0.6880, 42.6 MB bf16. Unchanged on
medians (within 0.1 ms of §51). **Worst observed e2e 11.09 ms** — new, and the honest number to budget
against until the depth arm runs.

---

§55 onward continues in `VLM_TRAJECTORY_part18.md` — this file closed at 193 lines.
