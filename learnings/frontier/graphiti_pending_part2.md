# Graphiti pending — part 2

*Episodes awaiting replay into the graph (MCP down). Continues `graphiti_pending.md`. All `group_id="dhiraj"`. Delete an episode once it is in the graph.*

## Episode 4 — vlm_step030 (group_id="dhiraj"), 2026-08-15

**Name:** vlm_step030 clean-compile re-timing — WIN reproduced, leak diagnosis confirmed, latency instrument found to be the binding constraint

**Body:** Re-ran the step028 runtime grid with `torch._dynamo.reset()` per cell, one variable, on a card
verified empty before and after (0% util, 48 MiB, no compute apps) after teammate PID 204307 exited.
VERDICT WIN as pre-registered: best accuracy-clean prefill 4.18 ms vs the 5.0 ms bar and vs step028's
4.21 — the rule said "improves or holds", it holds.

VALIDITY CONTROL, which step027 and step029 both lacked: the fp32-eager teacher cell reproduced its own
uncontended baseline to 0.3% / 0.8% (tower 25.65 vs 25.72, prefill 13.66 vs 13.77).

CONFIRMED — the leaked-dynamo-cache diagnosis, via ACCURACY not latency. It predicted the THIRD compile
cell (bf16_compile_teacher) was the degraded one. That cell and only that cell moved: top1 0.6980 →
0.7040, exactly the anchor reproduced in all six other teacher measurements. recompile_limit warnings
went 1 → 0. The arm was designed around the latency column and was settled by the accuracy column.

THE REAL FINDING — the ship cell did NOT reproduce: bf16_compile_student e2e 7.95 → 9.41 ms (+18%),
prefill 4.28 → 5.39, tower 3.67 → 4.02, all worse on a clean card under a fix that helped every other
cell. Diagnosed with an internal control rather than a rerun: the LM is byte-identical between the
teacher and student cells of a row, so within-pair prefill differences are pure instrument error. They
run +0.40 / −0.52 / +0.54 / +1.21 ms across the four dtype×mode pairs → a repeatability floor of about
±0.5 ms absolute, ~4% at 13 ms but ~25% at 4–5 ms. HYPOTHESIS (not confirmed): sub-5 ms prefill is at or
below this harness's resolution and the 7.95-vs-9.41 spread is instrument, not regression. Falsifier is
cheap: repeat each cell N times in one process, report median + IQR (step032).

STRUCTURAL LESSON — this is section 45's pathology repeated on the orthogonal axis. There the 2.0pp
accuracy bar was finer than the eval's own 2.05pp SE; here the 5.0 ms latency bar sits about one
repeatability-width from the 4.18–5.39 ms it is judging. Both axes have now reached the point where the
INSTRUMENT, not the architecture, is the binding constraint, and both fixes are cheap measurement work
rather than training. A line can be limited by its ruler while every individual arm still looks clean.

SHIP CONFIG restated honestly: d6 × r=0.25 bf16+compile, 21.28M params, 42.6 MB bf16 = 8.00× fewer bytes
than the fp32 teacher's 340 MB — EXACT, counted not timed, unaffected by any of the above, and this is
the actual drone claim. top1 0.6840–0.6860. e2e 7.95–9.41 ms = 4.2–5.0×; quote the RANGE, never 7.95
alone, which is the better of two draws. Also: fp32 and bf16-eager cells reproduce BIT-IDENTICALLY
across runs (McNemar 0/0) while both bf16_compile cells moved, so compile+bf16 is not bit-deterministic
across compile sessions (ship cell 3 flips each way, p=1.00, far below the eval SE).

PROCESS FINDING (step031, launched immediately after): the pre-registered command for the eval-widening
arm was BROKEN. It specified `vlm_step023_mlp_width.py --epochs 0`, but that script saves student
checkpoints and never loads one — no `--ckpt` argument exists — so it would have scored naive
sliced-untrained students at ~0.10 = chance and reported it as a width result at n=2000. Caught by
reading the script before launching instead of trusting the queued command; rerouted through
vlm_step028_runtime.py, which already loads a checkpoint. LESSON: pre-registration fixes the HYPOTHESIS,
it does not verify the CODE PATH — a pre-registered command is not a checked command.

Write-up: learnings/frontier/VLM_TRAJECTORY_part14.md section 47 (part13 hit 195 lines against the
200-line cap, so part14 was opened as its own footer required).

---

## Episode 5 — vlm_step031 (group_id="dhiraj")

**name:** vlm_step031 eval widened to n=2000 retracts the width line's parity claim

**body:**
vlm_step031, 2026-08-15, 5060ti_cuda. Eval-only re-scoring of the two existing width checkpoints
(d6 x r=0.25 from step027, d6 x r=0.125 from step029) against the d12 teacher on 2000 Imagenette val
images (200/class, seed 42), fp32 eager. No training. Run via vlm_step028_runtime.py, not the
pre-registered vlm_step023_mlp_width.py --epochs 0, which was a broken code path: that script saves
student checkpoints but has no load path and no --ckpt argument, so --epochs 0 would have scored naive
untrained sliced students at chance (~0.10) and reported it as a width result. Caught by reading the
script before launching. Process lesson: a pre-registered command is not a checked command --
pre-registration fixes the hypothesis, it does not verify the code path.

Control: the teacher was independently re-scored in both passes and returned a bit-identical per-image
hit vector, top1 0.7145 both times. Guard passes (0.7145 vs the 0.7040 anchor = 1.05pp, inside 2.0pp).

Results at n=2000, SE ~1.01pp: teacher 0.7145 (85.05M tower); r=0.25 0.6785 (21.28M, 0.2502x);
r=0.125 0.6420 (17.74M, 0.2086x). All three paired McNemar tests SIGNIFICANT --
teacher vs r=0.25 p=0.0035 (592 discordant, 332/260); teacher vs r=0.125 p=4.3e-09 (607, 376/231);
r=0.25 vs r=0.125 p=0.00022 (381, 227/154). Every one of these was NULL at n=500 (p=0.4437, 0.1476,
0.4505).

Verdict and consequence. Section 45's CONFIRMED parity claim is RETRACTED: it was inferred from
underpowered nulls produced by a test whose SE (2.05pp at n=500) was wider than the 2.0pp bar it was
testing. Absence of significance was read as absence of effect. The width axis carries a real, monotone
capacity cost at both steps. The ship config's accuracy cost is -3.60pp, not the -1.80/-2.00pp this line
had quoted since section 44; every downstream statement needs that number.

Caveat that bounds the cross-n reading: sample_images uses per = n // len(wnids), so n=500 and n=2000 are
DIFFERENT balanced draws, not nested samples. The cross-n movement confounds resolution with sample
identity. What is not confounded is the n=2000 column itself, which is paired image-by-image against a
bit-identical teacher. n=2000 supersedes n=500 as the estimate; it is not a controlled before/after.

What survives untouched: the bytes argument (21.28M params = 42.6 MB bf16 = 8.00x fewer bytes than the
fp32 teacher's 340 MB) is counted, not estimated, and is the actual drone claim. The step030 latency work
is unaffected. The decision to ship r=0.25 over r=0.125 is retroactively vindicated -- declined on bytes
when its accuracy cost was unproven, that cost is now CONFIRMED at -3.65pp.

Structural finding, second axis in two arms: step030 found the latency instrument's repeatability floor
(~+/-0.5 ms) sits inside the 5.0 ms verdict bar; step031 finds the accuracy instrument's SE sat inside
the 2.0pp verdict bar. THE INSTRUMENT, NOT THE ARCHITECTURE, WAS THE BINDING CONSTRAINT ON BOTH AXES,
and both fixes are cheap measurement work rather than training.

Open: the 100ep ladder at r=0.125 is now warranted (rel_mse was still falling at ep50, so -3.65pp may be
a budget artifact rather than a capacity floor); vlm_step032 repeat-timing harness still blocks any
further latency claim; every n=500 accuracy verdict in this line, including the d6 depth cut, needs
re-examination for the same type-II failure.

Write-up: learnings/frontier/VLM_TRAJECTORY_part14.md section 48.

---

## Episode 6 — vlm_step032 (group_id="dhiraj"), 2026-08-15

**name:** vlm_step032 repeat-timing shows the latency noise is a one-sided tail on compiled prefill, not a floor

**body:**
vlm_step032, 2026-08-15, 5060ti_cuda, card verified empty, load1 flat 0.94-1.06 across all 20 timed
passes. New script scripts/frontier/vlm_step032_repeat.py, R=5 repeats of every cell inside one process
with the model REBUILT (and dynamo reset) every repeat, so it measures across compile sessions rather
than measuring one compile session five times. Cells fp32:eager and bf16:compile, teacher and d6 x r=0.25
student, n_eval=500.

Motivation: section 47 saw the ship cell's e2e move 7.95 -> 9.41 ms (+18%) between two runs of an
identical config on a clean card, and INFERRED a ~+/-0.5 ms symmetric repeatability floor from an
internal control. This arm measured it instead of inferring it.

Result. Both automatic prefill-invariance gates PASS by two orders of magnitude: fp32:eager
|teacher-student| prefill = 0.02 ms, bf16:compile = 0.01 ms, against a 0.5 ms bar. That delta is zero by
construction (the LM is byte-identical across towers) and the harness reproduces zero to 0.01 ms. Every
cell/stage is deterministic to ~0.1% EXCEPT ONE: compiled prefill has a rare ONE-SIDED UPWARD EXCURSION.
Base mode is +/-0.03 ms; one repeat in five jumps +0.45 ms (bf16_compile_teacher rep0, 4.67 vs 4.18-4.23)
or +1.13 ms (bf16_compile_student rep1, 5.36 vs 4.17-4.23). Never downward, never the tower -- including
the COMPILED tower, which is as tight as the eager one.

This explains section 47 exactly and RETIRES its hypothesis. step030 measured the ship cell at prefill
5.39 / e2e 9.41; this arm's rep1 measures 5.36 / 9.40, agreement to 0.01 ms, while its other four repeats
sit at 8.19-8.27. step030 did not catch a slower configuration -- it took ONE sample and landed in the
tail. The +/-0.5 ms SYMMETRIC FLOOR is RETIRED: not a floor on all measurements, but a one-sided heavy
tail on COMPILED PREFILL ONLY at ~1-in-5 incidence. Distribution shape CONFIRMED; the cause (host-side
dispatch jitter on the launch-bound stage, exposed once compile shrinks the GPU work) stays HYPOTHESIS.

SELF-CORRECTION, and it is the methodological finding. The script printed "worst relative IQR 1.0% vs
bar 5.0% -> RESOLVABLE", which is true and misleading: IQR is robust to tails BY CONSTRUCTION, so it
discards precisely the events this arm existed to find -- a 1-in-5 excursion sits at the 80th percentile
and never enters the interquartile range. A robust statistic was pre-registered to hunt a tail. Same
species of error as section 45 (decision bar finer than its own SE) and section 48 (null read as parity).
CORRECTED REPORTING RULE for this line: quote median with MIN-MAX, never IQR.

Ship numbers restated on 5 draws instead of 1, all ratios within-process: tower 25.72 -> 4.04 ms = 6.37x;
prefill 13.58 -> 4.22 ms = 3.22x median, worst draw 5.36; e2e 39.30 -> 8.26 ms = 4.76x median, range
8.19-9.40; params 85.05M -> 21.28M; weights 340 MB fp32 -> 42.6 MB bf16 = 8.00x; top1 0.6860 identical
across all 5 repeats. The 5.0 ms prefill WIN bar holds at the median and at 4 of 5 draws but NOT at the
worst draw -- state it that way. The e2e claim is unaffected: even the tail draw (9.40) beats the
compiled teacher's 13.27 and the fp32-eager teacher's 39.30 outright. Ship cell top1 0.6860 here vs
0.6840 in step030 = 1 image in 500, so compile+bf16 is non-bit-deterministic ACROSS processes while
perfectly stable WITHIN one (5 rebuilds, identical top1).

UNBLOCKS: section 47 blocked CUDA graphs, torch.export and the section 32 prologue ablation on the
grounds that the instrument could not resolve them. That block is LIFTED -- base mode resolves 0.03 ms.
The standing requirement is now procedural, not instrumental: >=5 repeats, report median and min-max,
treat any single-draw latency number as unpublishable.

Write-up: learnings/frontier/VLM_TRAJECTORY_part15.md section 49.

*Episodes 7 onward: graphiti_pending_part3.md (200-line limit).*
