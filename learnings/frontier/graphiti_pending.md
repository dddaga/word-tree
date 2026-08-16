# Pending graphiti episodes (replay when the MCP server is back)

The graphiti MCP server dropped mid-session on 2026-08-14 (`MCP error -32000: Connection closed`)
while writing the step027 episode. Each block below is a ready-to-send `mcp__graphiti__add_memory`
call with `group_id="dhiraj"`. Delete a block once it has actually landed in the graph.

---

## vlm_step027 — d6 x r=0.25 epoch ladder: WIN, §43 knee withdrawn as budget artifact

`source_description`: SGNNET/neuro_graph VLM drone-pruning line (Branch C, SmolVLM-256M)

vlm_step027 (2026-08-14, 5060ti_cuda) DONE — VERDICT: WIN.

CONFIG. Identical to vlm_step026 except `--epochs 25 -> 50`. SmolVLM-256M-Instruct, Imagenette
10-class, compound student d6 x MLP ratio r=0.25, 9352 train images, n_eval 500, seed 42, lr 1e-4,
relative-MSE distillation on post-connector features. Runs the `vlm_step023_mlp_width.py` script
(steps 023/024/025/026/027 all share it), so the results file is
`vlm_step023_mlp_width_r0.25_e50_t9352_n500_d6__5060ti_cuda.json`.

GATE PASS. Teacher d12 top1 0.7040 — the FIFTH identical reproduction (steps 023/024/025/026/027),
so the whole line shares one comparison baseline.

RESULT. distilled top1 0.6840 = -2.00pp vs teacher, landing EXACTLY on the pre-registered WIN bar
(>= teacher - 2.0pp). +59.80pp over the naive sliced-untrained control (0.0860). Tower params
21.28M = 0.2502x of the d12 teacher's 85.05M. Final rel_mse 0.1776, cosine 0.9077 (vs step026's
0.2234 / 0.8824 at 25ep). Loss curve STILL FALLING at ep50 (0.1786 -> 0.1776), so 0.6840 is a
FLOOR, not a ceiling.

BIT-FOR-BIT REPRODUCTION. Epochs 1-25 reproduced step026 exactly (ep13 0.2726 vs 0.2725, ep18
0.2481 vs 0.2480, ep25 0.2239 vs 0.2234; ~1e-4 drift = nondeterministic GPU reduction order). Only
ep26-50 carried new information — a clean one-variable extension.

McNEMAR — FIRST PAIRED TEST IN THIS LINE. p = 0.4437 (exact, two-sided). Cells: 278 both-right, 84
both-wrong, 74 teacher-only, 64 student-only, 138 discordant. The -2.00pp gap is NOT distinguishable
from zero. Roadmap item 3 (part10 §41) is CLOSED — step027 is the first arm that actually RAN with
the per-image `correct` hit vector. CAVEAT: the 138 discordant images show the two towers are
equally ACCURATE but are visibly NOT the same function. Aggregate parity is not behavioural
equivalence.

§43's KNEE IS WITHDRAWN. CONFIRMED. step026 read d6 x r=0.25 at -5.00pp as a capacity knee; step027
shows it was a BUDGET artifact. Same failure mode as §39 -> §40 (which falsified the same reading
for width alone), recurring one level deeper. LESSON (CONFIRMED): a COMPOUND cut needs MORE budget
than either of its factors needed alone; reading a starved compound as a capacity limit
under-reports the architecture. Both part11 §43 and part12 §45 now carry the withdrawal.

DRONE OPERATING POINT MOVES. From §42's d6 x r=0.5 (0.3335x tower params) to d6 x r=0.25 (0.2502x)
— a further 25% off the weights that must fit in flash, for zero measured accuracy cost.

LATENCY FROM THIS RUN IS INVALID AND IS NOT QUOTED. A teammate GPU job (PID 151328, 12940 MiB)
landed on the shared card mid-run. Student reads prefill 25.0 ms vs teacher's 13.72 ms on an
IDENTICAL text stack, and tower 26.17 -> 29.01 ms (0.90x — student "slower" than its own teacher).
Both physically impossible => pure contention, not architecture. Use §43's uncontended 9.53 ms /
2.75x for this same d6 x r=0.25 tower — valid because wall-time depends on the architecture, not the
trained weights. Accuracy is unaffected (deterministic eval).

WRITE-UP: `learnings/frontier/VLM_TRAJECTORY_part12.md` §45. Queue row flipped RUNNING -> DONE.

NEXT: vlm_step028 (runtime arm — bf16 x torch.compile 2x2x2 grid, inference-only, script staged and
smoke-tested at `scripts/frontier/vlm_step028_runtime.py`). It is a LATENCY arm so it must not launch
under contention; blocked as of 2026-08-14 on the teammate job still holding the card at 100% util.

---

## vlm_step028 — runtime grid: WIN, prefill 4.21 ms, dispatch-bound diagnosis CONFIRMED

`source_description`: SGNNET/neuro_graph VLM drone-pruning line (Branch C, SmolVLM-256M)

vlm_step028 (2026-08-14, 5060ti_cuda) DONE — VERDICT: WIN.

CONFIG. Inference-only, NO training. 2x2x2 grid: dtype {fp32, bf16} x mode {eager, compile} x tower
{teacher d12, student d6 x r=0.25 reusing step027's checkpoint}. n_eval 500, seed 42, same eval path
as the whole line. Launched only on a VERIFIED-EMPTY card (48 MiB / 0% util) because step027's
latency block had to be discarded under teammate contention. Script
`scripts/frontier/vlm_step028_runtime.py`, results
`results/frontier/vlm_step028_runtime_fp32-bf16_eager-compile_n500__5060ti_cuda.json`.

GATE PASS. fp32-eager teacher top1 0.7040 (SIXTH reproduction) with tower 25.72 / prefill 13.77 ms,
back on the uncontended history of 26.2 / 14.0.

VERDICT WIN. Best accuracy-clean prefill 4.21 ms vs 13.77 = 3.27x, under the pre-registered 5.0 ms
bar. Section 44's roofline reading (prefill at 10.5% of roofline => ~90% dispatch) is CONFIRMED, not
falsified. The ~2.8x end-to-end cap is lifted.

GRID. fp32_eager teacher 25.72/13.77/39.49 @0.7040, student 9.52/13.92/23.45 @0.6840;
fp32_compile teacher 24.92/9.45/34.37 @0.7040, student 9.56/8.27/17.82 @0.6840;
bf16_eager teacher 8.13/11.00/19.13 @0.7040, student 3.28/11.02/14.31 @0.6880;
bf16_compile teacher 8.99/4.21/13.20 @0.6980, student 3.67/4.28/7.95 @0.6860. (tower/prefill/e2e ms)

THE TWO LEVERS HIT DIFFERENT STAGES. CONFIRMED. Compile moves the overhead-bound prefill (-31% in
fp32) and not the compute-bound tower (-3%); bf16 does the mirror image (tower 3.16x, prefill -20%).
Neither alone broke 8.27 ms; the combination reached 4.21.

COMPOUNDING NEAR-ORTHOGONAL. Tower 25.72 -> 3.28 ms = 7.84x vs architecture 2.70x x dtype 2.90x =
7.83. Mild sub-multiplicativity: the dtype gain shrinks 3.16x (big tower) -> 2.90x (small tower)
because a smaller tower is relatively more overhead-bound.

END-TO-END 39.49 -> 7.95 ms = 4.97x, of which 2.95x is PURE RUNTIME (23.45 -> 7.95 on the same
student) and 1.80x is architecture. The cheapest lever was the biggest one on latency — a dtype
switch plus a compile flag beat the entire distillation program on wall-time. It does NOT demote the
architecture work: runtime buys ZERO params, and params are what must fit in drone flash.

TWO MATERIAL CAVEATS. (1) COMPILE, not bf16, perturbs numerics: both bf16-eager cells reproduce their
fp32 counterparts exactly (0.7040 / 0.6880) while both compile cells drift (teacher -0.60pp, student
+0.20pp) — inside the 2.0pp guard so the verdict stands, but real and inconsistent in sign.
(2) The compile is LEAKY: dynamo hit `config.recompile_limit (8)` on transformers'
`output_capturing.wrapper` and fell back on part of the graph, so 4.21 ms is a FLOOR on a
partially-compiled model. Also negative: compile makes the bf16 tower SLOWER (8.13 -> 8.99, +11%).

WATCH-ITEM RESOLVED AS VARIANCE. The fp32-compile pair showed student prefill 1.18 ms BELOW teacher
prefill (8.27 vs 9.45) though prefill is tower-independent; it did not repeat in bf16-compile (4.21
vs 4.28) and both eager pairs matched. Nothing claimed from it.

RETRO-VALIDATES step027's QUARANTINE. fp32-eager student tower re-measured 9.52 ms vs section 43's
uncontended 9.53 ms — the substitution made when discarding step027's contaminated timings was right
to 0.01 ms, now by direct measurement rather than argument.

SHIP CONFIG FOR THE DRONE: student (d6 x r=0.25) in bf16 with compile — e2e 7.95 ms, 21.28M tower
params, 42.6 MB of bf16 tower weights (8.00x fewer bytes than the fp32 teacher's 340 MB), top1 0.6860
= -1.80pp vs the fp32 teacher.

TOWER LATENCY WORK IS FINISHED. At 3.67 ms tower vs 4.28 ms prefill the tower is no longer the
majority of end-to-end; further narrowing is a params/bytes argument only.

NEXT (supersedes step027's list): (1) fix the leaky compile and re-time prefill — cheapest remaining
latency win; (2) CUDA graphs / export for the residual; (3) r=0.125 at 50ep as the params/bytes probe;
(4) section 32 prologue ablation DEMOTED (prefill is now 4.21 ms, not 25% of a 24 ms residual);
(5) learned token selection DEAD as a latency lever — step022 showed a 16x token cut buys 1.4% of
prefill, and prefill is now 4.21 ms.

WRITE-UP: `learnings/frontier/VLM_TRAJECTORY_part13.md` section 45 (part12 hit the 200-line cap and
was split at section 45). Queue row flipped RUNNING -> DONE.

---

## vlm_step029 — d6 x r=0.125 at 50ep: PARTIAL by the bar, NULL by the paired test; width axis closed on bytes

`source_description`: SGNNET/neuro_graph VLM drone-pruning line (Branch C, SmolVLM-256M)

vlm_step029 (2026-08-15, 5060ti_cuda) DONE — VERDICT: PARTIAL.

CONFIG. One variable vs step027: MLP ratio 0.25 -> 0.125. SmolVLM-256M-Instruct, Imagenette 10-class,
d6 student, 9269 train images, n_eval 500, seed 42, lr 1e-4, 50 epochs, relative-MSE distillation on
post-connector features. Script `vlm_step023_mlp_width.py`; results
`vlm_step023_mlp_width_r0.125_e50_t9352_n500_d6__5060ti_cuda.json`.

GATE PASS. Teacher d12 top1 0.7040 — SEVENTH reproduction — and its per-image hit vector is
BIT-IDENTICAL to step027's, so both students are paired against the same teacher decisions.

RESULT. distilled top1 0.6680 = -3.60pp vs teacher, inside the pre-registered PARTIAL band
(WIN >= 0.6840, PARTIAL >= 0.6040). Naive sliced-untrained control 0.1000 = chance EXACTLY, so all
+56.80pp is distillation. Tower 17.74M = 0.2086x (predicted 0.208x before the run). Final rel_mse
0.2041 / cosine 0.8931 vs step027's 0.1776 / 0.9077 at identical budget.

TWO PAIRED TESTS, BOTH NULL. teacher vs r=0.125: 138 discordant, 78/60, p=0.1476. r=0.25 vs r=0.125
(the sharper test, teacher removed): 86 discordant, 47/39, p=0.4505. At n_eval=500 with p~0.70 the
binomial SE is ~2.05pp, which is WIDER than the 2.0pp bar separating WIN from PARTIAL. So the verdict
bar is finer than the instrument: 'r=0.125 is genuinely worse than r=0.25' is HYPOTHESIS, not
CONFIRMED. The eval, not the architecture, is now the bottleneck.

NON-CONVERGED. rel_mse still falling at ep50 (0.2053 -> 0.2041). Same signature that turned step026's
'capacity knee' into a budget artifact in step027. 0.6680 is a FLOOR, not a ceiling.

DECISION — SHIP CONFIG DOES NOT MOVE. Unlike step027, this arm does not move the drone operating
point. r=0.125 buys 42.6 -> 35.5 MB of bf16 tower weights (7.1 MB, 16.7%) for an unresolved accuracy
cost on a probably-starved run. Ship stays d6 x r=0.25. Recorded as a deliberate DECLINE.

WIDTH AXIS CLOSED ON THE BYTES ARGUMENT. CONFIRMED. Halving the MLP a second time moved the tower only
0.2502x -> 0.2086x because d6's non-MLP mass (attention, embeddings, layernorms) is a fixed ~14.2M —
already 80% of the r=0.125 tower. Further width work cannot pay for its compute regardless of how the
accuracy question resolves; a bytes win from here must attack attention/embeddings or quantize below
bf16.

LATENCY QUARANTINED — SECOND CONSECUTIVE ARM. Run reports tower 26.20 -> 27.36 ms (0.96x): a 0.21x-param
student measured SLOWER than its own teacher, physically impossible; plus student prefill 26.09 vs
teacher 13.73 ms on an identical text stack. Teammate job PID 204307 (`train_tiger.py`, 13006 MiB)
landed ~ep 20; cadence slipped from ~10.3 min/epoch with nothing else changed. Accuracy unaffected
(deterministic eval). NO latency number from this run is quoted. Unlike step027 there is no prior
uncontended measurement of r=0.125 to substitute, so its wall-time is UNMEASURED, not merely reused.
Teammate process was NOT killed.

WRITE-UP CONFLICT. A concurrent session wrote this arm into
`learnings/frontier/VLM_TRAJECTORY_part13.md` as SECTION 45 under the label `step028` at 09:27 IST —
colliding with the runtime grid, which already held section 45 and the `step028` label. Its analysis is
correct (both McNemar tables independently reproduced from the JSON). RESOLVED at 09:31 by that same
session, which renumbered its section to `vlm_step029` and added section 46 for the runtime grid, so both
arms are written up and nothing was lost. One residual contradiction was left behind and then fixed: the
open-items list still argued prefill is 13.73 ms of a ~40 ms end-to-end, a premise section 46 falsified on
the same page (4.21 ms of 7.95 ms); the section 32 prologue ablation is now recorded as DEMOTED and
superseded by step030. LESSON: two sessions writing one trajectory file collided on a section number and
briefly produced a document that contradicted itself — the collision was detected only because the arm's
result was cross-checked against the JSON rather than trusted from the prose.

---

---

*Episodes 4 onward: `graphiti_pending_part2.md` (200-line limit).*
