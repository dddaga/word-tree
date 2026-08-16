# VLM trajectory — part 12 (the epoch ladder, and the knee that wasn't)

Continues `VLM_TRAJECTORY_part11.md` (closed at 196 lines). Same line: SmolVLM-256M, Imagenette
10-class, d12@512 anchor 0.711, teacher 0.7040.

## 44. vlm_step027 — d6 × r=0.25 at 50 epochs. DONE. **WIN — and it retracts §43's knee.**

`--student_depth 6 --ratios 0.25 --epochs 50 --n_train 9352`. **Only the epoch count changed** vs
§43 (25 → 50). Strictly one-variable, in the §39→§40 pattern.

**GATE — PASS.** Teacher d12 top1 **0.7040**, the fifth bit-identical reproduction, so every arm from
§39 on remains one comparison line. Naive (sliced, untrained) 0.0860 — chance, identical to §43's
naive, and tower params 21.28M (0.2502×) identical too. The two runs differ in epochs and nothing else.

### Result — `vlm_step023_mlp_width_r0.25_e50_t9352_n500_d6__5060ti_cuda.json`

| arm | epochs | final rel_mse | distilled | vs teacher | verdict |
|---|---|---|---|---|---|
| §43 d6 × r=0.25 | 25 | 0.2234 | 0.6540 | −5.00pp | PARTIAL |
| **§44 d6 × r=0.25** | **50** | **0.1776** | **0.6840** | **−2.00pp** | **WIN** |

Gain over naive **+59.80pp** — the second-largest in this line after §40's +61.60pp.

**The epoch ladder reproduced §43 before extending it.** ep1 0.4333, ep5 0.3237, ep10 0.2900,
ep15 0.2619, ep20 0.2405, ep25 0.2239 vs §43's 0.4333 / 0.3237 / 0.2900 / 0.2620 / 0.2403 / 0.2234 —
agreement to ~1e-4, the size of nondeterministic reduction-order drift on this card. The run is a
genuine *extension* of §43's curve, not an independent draw, so the +3.00pp is attributable to
epochs 26–50 and nothing else.

### The finding: §43's −5.00pp was largely BUDGET. CONFIRMED.

Doubling epochs moved the same architecture from −5.00pp to −2.00pp. **The knee I located in §43 —
"between 0.333× and 0.250×" — is retracted.** It was measured at a budget that had not converged for
this arm, and §43's own budget caveat pre-registered exactly this retraction as the WIN outcome.
This is the second time in this line the same mistake was available and the second time the
pre-registration caught it (§39→§40 was the first, worth +18.20pp; this one is worth +3.00pp).

What *is* confirmed: **narrower students need more epochs to reach the same fit, not more capacity.**
r=0.25 hit rel_mse 0.1915 / cosine 0.9002 at **ep40** — numerically identical to what r=0.5 (§42)
reached at ep25. Same fit, 1.6× the epochs. The width cut costs optimization time, and at 50 epochs
it overshoots §42's fit outright (0.1776 vs 0.1915).

### McNemar — the first rigorous parity claim in this line

§44 is the first arm to actually *run* with the per-image `correct` vector (roadmap item 3, written
after §42, synced to the box only on 2026-08-14 — see part11's process-defect note). Paired against
the teacher over the same 500 images in fixed order:

| | student right | student wrong |
|---|---|---|
| **teacher right** | 278 | 74 |
| **teacher wrong** | 64 | 84 |

Discordant n=138, χ²(cc)=0.587, **p=0.444** (exact binomial two-sided p=0.444). **Teacher and student
are statistically indistinguishable.** Every prior parity claim in this line (§40's +2.80pp, §42's
−1.20pp) rested on a binomial-SE argument over aggregate top-1; this is the first one that is tested.
Roadmap item 3 is now genuinely closed.

Note the 64 images the student gets right and the teacher gets wrong — the disagreement is two-sided,
not a strict subset relation. A 21.28M tower is not doing a degraded version of the teacher's job;
it is doing a *different* job of equal measured quality on this task.

### Latency from this run is INVALID — do not quote it

The log reports `tower 26.17 -> 29.01 ms (0.90x)`, i.e. the student slower than the teacher. That is
impossible from a strictly-smaller tower, and the tell is prefill: **13.72 → 25.00 ms**, nearly
doubled, in a component this cut does not touch at all. A teammate job (12.9 GB, GPU at 98%) landed
on the 5060ti partway through and contended with the eval. Timings taken under contention are
meaningless.

**Use §43's timings** — same architecture, same params, bit-identical construction, measured on an
uncontended card: tower **26.21 → 9.53 ms = 2.75×**, end-to-end **39.98 → 23.56 ms = 1.70×**.
Accuracy is unaffected by contention; only wall-time is.

Process note: nothing was killed or displaced — the teammate's job was left alone and ours finished
alongside it. The cost was a slowdown from ~10 to ~15 min/epoch and one unusable timing column.

### What this changes

**New recommended drone operating point: d6 × r=0.25 — 0.250× tower params, 2.75× tower / 1.70×
end-to-end, at statistical parity (McNemar p=0.44), for 50 epochs of distillation.** It replaces
§42's 0.333× recommendation: same parity status, strictly fewer params, strictly faster tower. The
only thing §42 still wins is training cost.

| want | pick | params | tower | end-to-end | vs teacher |
|---|---|---|---|---|---|
| cheapest to train | §42 d6 × r=0.5, 25ep | 0.333× | 2.40× | 1.61× | −1.20pp |
| **best deployed** | **§44 d6 × r=0.25, 50ep** | **0.250×** | **2.75×** | **1.70×** | **−2.00pp, p=0.44** |

**0.6840 is a FLOOR on this architecture, not its ceiling.** rel_mse was *still falling* at ep50
(0.1786 → 0.1776, slope ~0.001/ep and decaying), so the arm had not converged and the same
budget caveat that made §43 retractable applies to §44 in the same direction. The difference is that §44 now sits at *better* fit than the §42 arm it is compared
against, so the comparison is no longer budget-confounded in the direction that matters.

**The cross-architecture rel_mse HYPOTHESIS gains a second disagreeing point.** §44 has a strictly
better feature fit than §42 (0.1776 < 0.1915) and a slightly *worse* top-1 (−2.00 vs −1.20pp, both
inside noise). §42 was the first point where fit-order and accuracy-order disagreed; this is the
second. **rel_mse ranks budgets within one architecture and must not be used to rank architectures.**
Still HYPOTHESIS — the accuracy differences involved are inside the noise band.

### Open, in priority order

1. **§32 prologue ablation.** Promoted twice now. Tower squeezing has hit diminishing end-to-end
   return (2.40× → 2.75× tower bought only 1.61× → 1.70× overall); prefill is 25% of what remains
   and is untouched by every lever tried so far. This is where the next real latency win lives.
2. **Learned token selection** — the only route to the ~2.9× average pooling cannot buy, and not
   bound by §38's oracle ceiling.
3. **Push width further, now that the knee is retracted.** r=0.125 at 50ep is the obvious probe:
   §43's knee argument is gone, and the *actual* cost of narrowing turned out to be epochs, not
   accuracy. Budget it at 50ep minimum — anything less repeats §43's mistake by construction.
4. **A clean re-timing of §44 on an uncontended card**, if the deployed number is ever quoted from
   this run rather than §43's.
5. **d9 depth retry** — unblocked, unstarted, lowest value.

---

§45 onward continues in `VLM_TRAJECTORY_part13.md`.
