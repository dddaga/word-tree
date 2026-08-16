# Batch-1 latency is slot-dependent — seg_step021 retracts "flat below 1 GMAC"

Split out of `seg_latency_pareto.md` at the 200-line limit. This file exists because a single clean
measurement overturned a claim that three prior steps had been building on.

## seg_step021 — the re-run seg_step019 asked for (5060ti_cuda, free card, 2026-08-15)

seg_step019 measured the CUDA ladder with a teammate process resident (13006 MiB, 100% util), fired
branch (a), and **tagged it HYPOTHESIS rather than CONFIRMED** on the grounds that contention
inflates the fixed term and that bias points at the branch that fired. The card freed; this is that
re-run, same command, same `--no_pretrained` arms, `--tag step021_clean`. Co-tenant at run time:
one idle process, 804 MiB, **0% util**.

**The contamination was ~6×, and it was entirely in the fixed term.**

| arm | GMAC | b1 contaminated (step019) | **b1 clean (step021)** | b32 clean ms/frame | mem b1 |
|---|---|---|---|---|---|
| E9 | 0.264 | 2.553 | **0.376** | 0.090 | 4 MB |
| E7 | 0.876 | 2.670 | **0.428** | 0.223 | 12 MB |
| E11 | 0.846 | 2.662 | **0.420** | 0.202 | 10 MB |
| E12 | 0.816 | 2.644 | **0.406** | 0.222 | 12 MB |
| E10 | 1.865 | 2.893 | **0.645** | 0.412 | 19 MB |
| E2 | 3.232 | 3.012 | **0.758** | 0.585 | 27 MB |

p10/p90 are within ~1% of the mean on every clean cell, so the ordering is resolvable, not noise.

### What is retracted

**1. Branch (a) fails on clean data — CONFIRMED, and it reverses.** The pre-registered rule was
E9/E7 ≥ 0.95× → the flat floor generalises to CUDA. Contaminated: 0.956×, just inside. Clean:
**0.376/0.428 = 0.878×**, well outside. E9's p90 (0.386) does not overlap E7's p10 (0.426).

**2. "b1 latency is architecture-independent below ~1 GMAC" does not hold on CUDA.** That claim
came from MPS (step016), where the sub-1G arms all landed in 0.81–0.96 ms. On a clean CUDA card the
same arms separate cleanly. **The flatness was a measurement-resolution property of the MPS slot,
not a property of the architecture ladder.** Both slot readings are individually correct; the
generalisation across them was not.

**3. "E9 is Pareto-dominated by E7" is slot-specific and does not hold here.** On MPS E9 was 1.18×
*slower* at b1 despite 3.32× fewer MACs. On clean CUDA E9 is **1.14× faster**. E9 still costs
−1.69pp (step013), so on CUDA it is an accuracy-for-latency **trade**, not a domination — and for a
drone at batch 1 that trade is now live rather than closed.

### What survives

- **Memory is unchanged, byte for byte** (4/12/10/12/19/27 MB). This is the positive control for
  the whole exercise: CUDA `max_memory_allocated` is per-process, so it was always valid under
  contention, exactly as claimed at the time. Only the wall-clock needed the clean card.
- **b32 still tracks MACs**: E2/E7 = 2.6236× against a 3.6912× MAC ratio (71.1%, vs 69% under
  contention and 74% on MPS). Batched throughput is compute-bound on both slots.
- **b1 is still mostly fixed overhead.** Linear fit over the six clean points gives ≈0.129 ms/GMAC
  and a **≈0.342 ms intercept** — 80% of E7's 0.428 ms. The ladder separates *despite* that,
  because the clean measurement resolves 12% differences that MPS jitter buried.

### Method lesson

The HYPOTHESIS tag did its job. The rule fired in my favour, the confound biased it in exactly that
direction, and refusing the read cost one re-run and saved three downstream claims. **A branch that
fires under a known bias pointing the same way is not evidence.** Note also that the contaminated
b32 column was *fine* (69% vs 71% clean) — contention corrupts the fixed term, so batch-1 numbers
are the fragile ones and any future b1 timing must record card state alongside the result.

### Open

**Joules remain the only unmeasured axis in this line.** Power telemetry is device-wide, so it
needs a card with no co-tenant at all — the 804 MiB idle process here is fine for wall-clock but
not for energy. The queue item is otherwise unblocked.
