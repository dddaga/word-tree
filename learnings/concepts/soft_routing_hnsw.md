# Soft Routing with HNSW Inference (HYPOTHESIS)

**Status:** HYPOTHESIS — not implemented, not validated. Design from 2026-04-16 research session.

## Motivation

Current routing uses static `conn_hh` (Watts-Strogatz at init, frozen). Routing topology independent of learned W_pos — gap **confirmed** by step852 (hot-rebuilding conn_hh mid-training destroys learning: −1.91pp best, −79pp worst). Static topology correct; question: should routing **weights** respond to current activation state while edges stay fixed — or drop entirely for distance-based weighting.

**Dual-mode proposal:**

- **Training:** dense soft routing — differentiable, all N nodes contribute, weighted by distance between activation Z and all W_pos.
- **Inference:** sparse hard routing — HNSW lookup retrieves top-K nearest W_pos, only those contribute.

**Key principle:** dense gradient signal during training (smooth optimization), sparse efficient compute at inference (O(log N) HNSW queries).

## Relationship to step852

step852 KILLED hot-rebuilding `conn_hh` — edges-as-identity carry learned state (W_pos, AH coupling) that breaks when edges change. This design **avoids trap**: doesn't rebuild `conn_hh`; replaces it entirely with distance-based weights. No discrete edges to rebuild — every forward pass computes continuous distance field, softmaxes over it. HNSW step at inference = **lossless-under-low-temperature approximation**, not learned structural change.

## Training-Time Math

For each active (beam) node m with state Z_m ∈ R^D:

```
dist[m, n] = ||Z_m - W_pos[n]||²        for n ∈ [0, N)
w[m, n]    = softmax_n(-β · dist[m, n])  # differentiable, peaks on nearest
signal[n]  = Σ_m w[m, n] · Z_m           # dense aggregation
```

- All N nodes receive weighted signal from beam broadcasters.
- Gradient flows to W_pos through distance term (nodes learn to position near/far from activations they want).
- β (inverse temperature) controls softness.

## Inference-Time Math (Hard Switch)

```
HNSW index: built once on W_pos (static post-training)
For each beam node m:
  top_k = HNSW.query(Z_m, K)                        # O(log N)
  w[m, n] = softmax(-β · dist[m, n]) for n ∈ top_k  # truncated softmax
  signal[n] += Σ_m w[m, n] · Z_m                    # sparse, K nodes contribute
```

**Truncation error bound:**

```
error ≤ Σ_{n ∉ top_k} exp(-β · dist[m, n]) / Z_partition
```

At β=10 with distance gap 0.1 between top-K boundary and tail, tail contributes <37% of top-K mass per step. Sharp enough β → HNSW approximation near-lossless.

## β Scheduling (Critical)

- **Start soft:** β = 0.5 (wide gradient distribution, all N contribute ~equally).
- **Anneal sharp:** β → 10 linearly over epochs (distribution concentrated on K nodes).
- **End of training:** softmax mass on top-K captures ≥99% routing weight.
- **HNSW truncation after training:** ~lossless.

Analogous to temperature annealing in Gumbel-softmax / soft-to-hard clustering. Risk symmetric: too aggressive kills training; too slow leaves routing un-sharp, HNSW truncation lossy at eval.

## HNSW on W_pos (not Z)

- **Index from learned W_pos [N, D_pos]** → static at inference.
- **Query with activation Z_m** → returns nearest W_pos nodes.
- **Rebuild cadence during training:** every 10 epochs (step852: slow cadences less harm; fast rebuilds catastrophic).
- **For N=2048:** brute-force k-NN via `torch.cdist` ~0.1ms on CUDA — HNSW only needed at N≥16384.

## Composition

- **Replaces `conn_hh` entirely:** routing becomes distance-based, not edge-list-based. Cleanest way to avoid step852 failure — no discrete edges to update inconsistently.
- **With [[sparse_bfs_routing]]:** top-M active nodes perform HNSW queries. Beam gating reduces HNSW calls per step from N to M.
- **With cascading beam:** narrower beam in deeper iterations; fewer queries as frontier stabilises.
- **With [[delta_w]] projection:** potentially orthogonal — proj modulates signal *magnitude* along relational axis; soft routing determines *which* neighbours to route to. Compose only after isolating each mechanism.
- **With [[activation_retention]]:** retained Z biases distance field smoothly across time steps.

## Risks

- **Training FLOPs:** O(B · M · N · D) per iter — potentially slower than static gather at large N unless beam-gated.
- **β schedule:** too aggressive → training fails; too slow → HNSW truncation lossy at eval.
- **HNSW query mismatch:** if Z drifts during training, stale index on old W_pos returns wrong neighbours. Rebuild cadence matters.
- **W_pos cluster collapse:** if all W_pos cluster near origin (AH not holding diversity), HNSW returns ~random neighbours, routing degenerates. AH diversity pressure becomes load-bearing.
- **Non-differentiability of top-K at inference:** gradients only flow through K selected nodes during training (if training-time truncation used). Fully soft training avoids this.

## Paper Claim (If Successful)

> Dense differentiable routing during training, O(log N) sparse routing at inference, with <X% accuracy loss from soft-hard transition.

Reframes SGNNET efficiency story: instead of fixed edges, *topology itself learned and queried* — genuinely novel mechanism for O(log N) inference.

## Proposed Experiment — step856_soft_routing_hnsw

| Config | Training routing | Inference routing | Tests |
|--------|------------------|-------------------|-------|
| Ref | static conn_hh | same | baseline |
| A_soft_β1 | soft, β=1 fixed | soft β=1 | is dense soft better? |
| B_soft_anneal | soft, β: 0.5→10 anneal | soft at final β | annealing value |
| C_soft→topK | soft annealed | hard top-K exact | transition cost |
| D_soft→HNSW | soft annealed | HNSW top-K | full pipeline |
| E_hard_only | hard top-K | hard top-K | pure sparse baseline |

**Measurements:**

- Train acc, infer acc (C/D may differ from training acc).
- **B − E** = softness-during-training benefit.
- **C − B** = exact truncation cost (should be small if β annealed high).
- **D − C** = HNSW approximation cost (index recall).
- **D vs Ref FLOPs** = paper's efficiency claim.

**Implementation cost:** ~1 week. Needs:
- Soft-routing module (dense distance softmax).
- β scheduler.
- HNSW integration at eval (hnswlib or torch-implementation).
- Dual forward path (training vs inference).

## Open Questions

1. β annealing need per-layer or per-K_iter-step, or global schedule sufficient?
2. Soft routing implicitly do what [[delta_w]] projection does? Both weight neighbours by similarity from W_pos.
3. At what N does HNSW become necessary? N=2048 brute-force `cdist` cheap; paper may not motivate HNSW unless validated at N≥16384.
4. W_pos get sufficient gradient through distance term, or still need explicit AH for diversity?
5. **Training** FLOP increase (dense distance field) outweigh **inference** FLOP decrease? Training cost matters for paper but less than inference cost.

## See Also

- [[sparse_bfs_routing]] — complementary; soft routing provides weights, BFS provides broadcaster gating
- [[delta_w]] — another W_pos-based modulation; may compose or be redundant
- [[antihebbian]] — AH enforces W_pos diversity; load-bearing for meaningful distance fields
- [[activation_retention]] — Z_{t-1} biases distance field temporally
- [[architecture_dead_ends]] — step852 dynamic-edge-rebuild failure informs why this design avoids edge updates