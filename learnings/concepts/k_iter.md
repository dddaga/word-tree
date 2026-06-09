# K_iter (Routing Iterations)

## What It Controls

Number of message-passing steps through graph topology. At step k, every neuron h gathers normalized signals from K_hh structural neighbors:

```
Z[k+1][h] = normalize(sum_{j in neighbors(h)} score[h,j] * Z[k][j])
```

At K_iter=k, neuron integrates signal from k-hop neighborhood. On small-world graph with diameter ~log(N), K_iter >= log(N) reaches all neurons. Receptive field grows exponentially with K_iter, bounded by graph diameter (~9 hops at N=512).

## Confirmed Optimal Values

| Scale | Optimal K_iter | Step | Evidence |
|-------|---------------|------|----------|
| N=1024, D=64, AH=1.0 (buggy arch) | 16 | step68 | 74.14% (+0.61pp vs K_iter=8 Ref 73.53%) |
| N=4096, D=64, AH=1.0 (patched arch) | 12 | step71 | 96.66% (+0.79pp vs K_iter=8 Ref 95.87%) |

K_iter=8 was default throughout Phase 5 until step68 discovered gain from higher values.

## Non-Monotone Behavior

### N=1024 (step68, buggy arch, 50%/75ep)

| K_iter | Accuracy | vs Ref (K_iter=8) |
|--------|----------|-------------------|
| 8 | 73.63% | -- |
| 10 | 73.58% | -0.05pp |
| 12 | 72.94% | -0.69pp |
| 16 | 74.14% | +0.51pp |
| 24 | 70.06% | -3.57pp |

Shape: dip at 12, peak at 16, cliff at 24. Optimum narrow.

### N=4096 (step71, patched arch, 50%/75ep)

| K_iter | Accuracy | vs Ref (K_iter=8) |
|--------|----------|-------------------|
| 4 | 92.82% | -3.05pp |
| 6 | 95.11% | -0.76pp |
| 8 | 95.87% | -- |
| 12 | 96.66% | +0.79pp |
| 16 | 96.31% | +0.44pp |

Shape: monotone up to 12, peak at 12, slight decline at 16. Low end (4-6) monotone.

### Why the optimal shifts with N

At N=1024, K_iter=16 wins. At N=4096, K_iter=12 wins. Optimal K_iter is N-dependent.

Hypothesis from step68: at K_iter=24, message-passing converges too deep, over-smoothing phase representations. K_iter=16 (N=1024) and K_iter=12 (N=4096) each hit sweet spot — sufficient neighborhood context without phase collapse. Larger N provides more diverse neighbors per hop, fewer hops needed for full integration.

Note: step68 (K_iter=16 winner) calibrated on buggy arch. step71 (K_iter=12 winner) on patched arch. Not directly comparable due to architecture change. K_iter optimal may have shifted due to `alpha_reflect` fix restoring temporal persistence across routing steps.

## K_iter=3 vs K_iter=8 Gap

Gap ~15pp at D=64, confirmed three independent times:

| Measurement | K_iter=3 | K_iter=8 | Gap | Source |
|-------------|----------|----------|-----|--------|
| step34 RefK3 vs Ref | 42.01% | 56.79% | 14.78pp | LEARNINGS_phase5_p7 |
| step34 MoD effective | ~18-20% (degrades to K_iter=2-3) | 56.79% | ~37pp | LEARNINGS_phase5_p7 |
| Signed coupling K_iter=3 reference | 40.00% (step42) | 56.79% (step34 Ref) | ~17pp | LEARNINGS_phase5_p7/p9 |

All 8 routing iterations contribute meaningfully. No early-exit point where neurons "confident." MoD adaptive depth (step34) confirmed: effectively degraded most neurons to K_iter=2-3, collapsed to 18-20%.

## Interaction with Other Mechanisms

### With AntiHebbian (AH)

- **Without AH, higher K_iter hurts.** step48: K_iter=12 without AH = 55.08% vs K_iter=8 = 58.24% (-3pp). Over-smoothing without inhibitory pressure to maintain diversity.
- **With AH, higher K_iter helps (up to point).** step68: K_iter=16 with AH=1.0 = 74.14% (+0.51pp vs K_iter=8). AH prevents over-smoothing by suppressing structurally similar neighbors each step, preserving directional diversity across iterations.
- AH suppression applied before gather step; `F.normalize()` at end of each routing step restores magnitude to unit vectors. Net per-step signal conserved. This conservation allows deep K_iter without signal collapse.

### With alpha_reflect (reflection accumulator)

Reflection adds leaky self-inhibition memory across K_iter steps: what threshold suppressed leaks back next step with decay=0.5. Residual connection through time — each routing step remembers what it suppressed. Without it, each routing step starts from scratch. `alpha_reflect` fix (+3.54pp, step69) may have shifted optimal K_iter by improving information persistence across iterations.

### With gates (gate-death theorem)

Any multiplicative gate g in [0,1] applied per routing step compounds across K_iter:

```
After K_iter steps: signal proportional to product(g_k)
K_iter=8, g=0.7: 0.7^8 = 0.06x original signal
K_iter=8, g=0.5: 0.5^8 = 0.004x original signal
```

Gradient through product vanishes accordingly. Gate-death theorem (confirmed across steps 58, 59, 60, 61, 63, 65, 66, 51). Higher K_iter amplifies gate-death exponentially. Any per-step multiplicative gate with g < 1 dies in training for K_iter >= 4.

### With MoD / early-exit

step34 KILLED. MoD adaptive depth (exit_thresh=1.5) collapsed to 18-20% at e110, loss flat. MoD prematurely freezes neurons, effectively degrading to K_iter=2-3 for most. All iterations necessary; early exit destroys iterative refinement.

### With Oja's rule

step41 KILLED. Oja's rule compresses each routing step toward principal component. With K_iter=8, compounding compression causes all neurons to collapse toward shared direction by iteration 4-5. Directional diversity driving K_iter=3-to-8 gain destroyed.

### With group topology

K_iter provides inter-group mixing when groups randomly assigned (step82 design). After K_iter routing steps over K_random cross-group wires, near-complete mixing: every neuron has indirect access to every group's specialized representations. K_iter is integration mechanism for group specialization.

## Open Questions

1. **K_iter optimal on patched arch at N=1024**: step68 (K_iter=16 winner) was on buggy arch. `alpha_reflect` fix may shift optimal. Not yet tested.
2. **K_iter interaction with turing**: step69 showed turing=0.3 beneficial on patched arch (+1.68pp at N=1024). Joint K_iter x turing calibration not done.
3. **K_iter at N=10000**: step56 showed N=10000 regresses vs N=4096. One hypothesis: K_iter=8 insufficient at N=10000 — higher K_iter might recover. Not tested.
4. **K_iter with softmax redistribution routing**: step73 (running) tests conservative routing with summed weights=1. If redistribution works, does optimal K_iter differ from static AH routing?

## See Also

[[gate_death]], [[n_scaling]], [[antihebbian]]