# Experiment Queue

## Strategic Direction (2026-04-09)
**All experiments from run10 onwards drop the nn.Linear FFN head.** The GNN must classify directly (output_nodes=10, act_strength as logits). `diagnose.ipynb` showed intermediate/input node weights barely changed during training — the FFN was doing most of the classification work. Since the project goal is to prove the *GNN* can replace VGG16's FC layers, keeping an FFN head defeats the purpose. Target: match run6's 85.81% without the FFN.

## Active Investigation
**run16 RUNNING** — N=4146, T=4.0 fixed, 100% data. Tests whether run11 (N=4146) + run12 (T=4.0) improvements compound. Best FFN-free so far: run13 (85.40%).

---

## Queue (active + pending only)

| Run | Status | Hypothesis | Key delta | Expected result |
|-----|--------|------------|-----------|-----------------|
| run16 | RUNNING | Combine run11 (N=4146) + run12 (T=4.0) — test if improvements compound | N=4146, T=4.0, 100% data, flat, no FFN, 40ep | Expected best FFN-free result; compare to run12 (82.19%) and run11 (61.32%) |

---

## Completed Runs — FFN-free (runs 10–15)

**Note:** All FFN-free runs use output_nodes=10 with act_strength values fed directly into CrossEntropyLoss. No nn.Linear head.

| Run | Date | Status | Val Best | Epochs | N | T | Data% | Key delta from run10 | Verdict |
|-----|------|--------|----------|--------|---|---|-------|----------------------|---------|
| run10 | 2026-04-09 | DONE (stopped ep23) | 40.33% @ep23 | 23/40 | 15454 | 1.0 | 100% | Baseline FFN-free (no FFN, output_nodes=10) | GNN learns without FFN but very slowly; gradient starvation confirmed |
| run11 | 2026-04-09 | DONE | 61.32% @ep37 | 40/40 | 4146 | 1.0 | 100% | total_nodes=4146 (3136+1000+10) | +20.99pp vs run10 (HYPOTHESIS: fewer params, denser connectivity, simpler landscape — confounded) |
| run12 | 2026-04-09 | DONE | 82.19% @ep39 | 40/40 | 15454 | 4.0 | 50% | routing_temperature=4.0 | +41.86pp vs run10; T=4.0 CONFIRMED key lever (single-variable ablation) |
| run13 | 2026-04-10 | DONE | **85.40% @ep40** | 40/40 | 15454 | 7.0 | 50% | routing_temperature=7.0 | +3.21pp vs run12; T=7.0>T=4.0>T=1.0 monotonic; 0.41pp gap from FFN target (85.81%) |
| run14 | 2026-04-10 | DONE | 45.48% @ep40 | 40/40 | 4146 | 1.0 | 50% | topology=layered (delta from run11: layered + 50% data) | -15.84pp vs run11 (HYPOTHESIS: layered worse than flat, but data fraction confounds — not clean ablation) |
| run15 | 2026-04-11 | DONE | 76.97% @ep27 | 40/40 | 15454 | 7→1 | 50% | routing_temperature annealed 7.0→1.0 per-step (delta from run13) | -8.43pp vs run13 (CONFIRMED: annealing HARMFUL — low-T end causes starvation, val collapses to 52.66% by ep40) |

### FFN-free Learnings

- **routing_temperature is the primary lever (CONFIRMED):** T=1.0→40.33%, T=4.0→82.19%, T=7.0→85.40%. Clean single-variable ablations. Gradient starvation at T=1.0 (0.4% nodes carry 50% gradient) is fixed by higher T.
- **GNN nearly matches FFN target at T=7.0:** run13 reached 85.40% without FFN, without full data — only 0.41pp below run6 (85.81% with FFN). Proof-of-concept is nearly complete.
- **Convergence not reached at T=7.0:** run13 best at final epoch (ep40), still improving. More epochs or full data likely closes remaining gap.
- **Fewer nodes helpful (HYPOTHESIS from run11):** 1K intermediates → +20.99pp vs run10. Confounded: fewer params (66K vs 247K), denser connectivity (5% vs 1.3%), simpler landscape. Not a clean ablation.
- **Layered topology worse than flat (HYPOTHESIS from run14):** layered→45.48% vs flat→61.32% (run11), -15.84pp gap. CONFOUNDED: run14 used 50% data vs run11's 100%. Clean ablation (run14 repeated with 100% data) needed to confirm.
- **Temperature annealing 7→1 is HARMFUL (CONFIRMED from run15):** val_best=76.97% @ep27 vs run13 T=7.0 fixed (85.40%), -8.43pp. Single-variable ablation. Low-T end-state causes gradient starvation to return; val collapsed from 76.97% to 52.66% by ep40 as T→1. Do NOT anneal temperature downward.

---

## Completed Runs — With FFN (runs 1–9)

**Note:** Runs 1–9 all used `nn.Linear(output_nodes=256, 10)` as a classification head after the GNN. These results reflect the combined GNN+FFN system, not the GNN alone.

| Run | Date | Status | Topology | Iters | Key delta | Val best | Epochs | Verdict |
|-----|------|--------|----------|-------|-----------|----------|--------|---------|
| run1 | 2026-03-25 | DONE | flat | 5 | baseline (no layernorm, no dropout, no radiation) | 76.05% | 20/20 | Baseline |
| run2 | 2026-03 | DONE | flat | 5 | total_nodes=30000, cardinality=1000 | <15% | 4 (stopped) | Unstable, inconclusive |
| run3 | 2026-03-25 | DONE | flat | 5 | +layernorm=true | 81.38% | 20/20 | +5.33pp from run1; layernorm CONFIRMED helpful |
| run4 | 2026-03-26 | DONE | flat | 5 | +radiation_targets=32, scattering_prob=0.8, +lr_decay | 16.89% | 3 (stopped) | Collapsed; radiation too aggressive |
| run5 | 2026-03-27 | DONE | flat | 5 | radiation_targets=16, scattering_prob=0.25, 40ep | 71.46% | 40/40 | Radiation still hurts vs run3 (HYPOTHESIS) |
| run6 | 2026-03-28 | DONE | flat | 5 | radiation=0 (off), +dropout=0.2 | **85.81%** | 40/40 | **BEST with FFN — FFN-free target** |
| run7 | 2026-04-02 | DONE | layered | 7 | +layered, temporal_decay=0.8 | 23.08% | 16 (stopped) | Poor; 3 vars changed, confounded |
| run8 | 2026-04-02 | DONE | layered | 15 | +beam_width=500, CPU | 13.32% | 23 (stopped) | Poor; highly confounded |
| run9 | 2026-04-02 | DONE | layered | 7 | +layered only (closest to run6 ablation) | 25.45% | 38/40 | Poor; 60pp gap vs run6 |

### FFN Learnings

- **LayerNorm (CONFIRMED):** +5.33pp in clean run1→run3 ablation. Always on from run3+.
- **Radiation hurts (HYPOTHESIS):** run4/5 worse than run3/run6. No clean single-variable ablation isolating radiation.
- **Dropout (HYPOTHESIS):** added in run6 alongside radiation removal — unconfirmed individually.
- **Layered topology poor with FFN (HYPOTHESIS):** run7/8/9 all <26%. run9 is closest to clean ablation (layered only vs run6) but still confounded by iters change.

---

## Consolidated Results: Accuracy, FLOPs & Parameters

FLOPs per forward pass: `(I-1) * B * N * (8*C*V + 7*C + 24*V)`. Parameters: `2*N*V + [2V if LN] + [n_out*10+10 if FFN]`. See `concepts/flops.md` for derivation.

Variables: `I`=iterations, `B`=batch size, `N`=total_nodes, `C`=cardinality (edges/node), `V`=vector_dim, `n_out`=output_nodes.

| Run | Val Best | Epochs | Params | FLOPs/fwd | FLOPs/step | N | C | I | FFN | T | Data% |
|-----|----------|--------|--------|-----------|------------|-------|-----|---|-----|-----|-------|
| run1 | 76.05% | 20/20 | 249,842 | 3.56B | 10.7B | 15454 | 200 | 5 | Yes | 1.0 | 100% |
| run2 | <15% | 4 (stopped) | 482,570 | 6.91B | 20.7B | 30000 | 1000 | 5 | Yes | 1.0 | 100% |
| run3 | 81.38% | 20/20 | 249,858 | 3.56B | 10.7B | 15454 | 200 | 5 | Yes | 1.0 | 100% |
| run4 | 16.89% | 3 (stopped) | 249,858 | 3.56B | 10.7B | 15454 | 200 | 5 | Yes | 1.0 | 100% |
| run5 | 71.46% | 40/40 | 249,858 | 3.56B | 10.7B | 15454 | 200 | 5 | Yes | 1.0 | 100% |
| **run6** | **85.81%** | 40/40 | 249,858 | 3.56B | 10.7B | 15454 | 200 | 5 | Yes | 1.0 | 100% |
| run7 | 23.08% | 16 (stopped) | 249,858 | 5.34B | 16.0B | 15454 | 200 | 7 | Yes | 1.0 | 100% |
| run8 | 13.32% | 23 (stopped) | 249,858 | 17.8B | 53.4B | 15454 | 200 | 15 | Yes | 1.0 | 100% |
| run9 | 25.45% | 38/40 | 249,858 | 5.34B | 16.0B | 15454 | 200 | 7 | Yes | 1.0 | 100% |
| run10 | 40.33% | 23 (stopped) | 247,280 | 3.56B | 10.7B | 15454 | 200 | 5 | No | 1.0 | 100% |
| run11 | **61.32%** | 40/40 | 66,352 | 0.96B | 2.88B | 4146 | 200 | 5 | No | 1.0 | 100% |
| run12 | **82.19%** | 40/40 | 247,280 | 3.56B | 10.7B | 15454 | 200 | 5 | No | 4.0 | 50% |
| run13 | **85.40%** | 40/40 | 247,280 | 3.56B | 10.7B | 15454 | 200 | 5 | No | 7.0 | 50% |
| run14 | 45.48% @ep40 | 40/40 | 66,352 | 0.96B | 2.88B | 4146 | 200 | 5 | No | 1.0 | 50% |
| run15 | 76.97% @ep27 | 40/40 | 247,280 | 3.56B | 10.7B | 15454 | 200 | 5 | No | 7→1 | 50% |
| run16 | RUNNING | /40 | 66,352 | 0.96B | 2.88B | 4146 | 200 | 5 | No | 4.0 | 100% |

Notes:
- FLOPs/fwd = forward pass only. FLOPs/step = fwd + backward (3x with grad checkpointing).
- run2 had C=1000 (5x edges), explaining its high FLOPs despite same architecture.
- run11 has 3.7x fewer FLOPs than run10 (fewer nodes: 4146 vs 15454).
- run12/13 have same per-step FLOPs as run10 but half the steps/epoch (50% data).
