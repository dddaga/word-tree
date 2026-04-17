# Soft Routing with HNSW Inference (HYPOTHESIS)

**Status:** HYPOTHESIS — not implemented, not validated. Design captured from 2026-04-16 research session.

## Motivation

Current routing uses static `conn_hh` (Watts-Strogatz at init, frozen). Routing topology is independent of learned W_pos — a gap **confirmed** by step852 (hot-rebuilding conn_hh mid-training destroys learning: −1.91pp at best, −79pp at worst). Static topology is correct; the question is whether routing **weights** should respond to the current activation state while edges stay fixed in a different sense — or be dropped entirely in favour of distance-based weighting.

**Dual-mode proposal:**

- **Training:** dense soft routing — differentiable, all N nodes contribute, weighted by distance between current activation Z and all W_pos.
- **Inference:** sparse hard routing — HNSW lookup retrieves top-K nearest W_pos, only those contribute.

**Key principle:** dense gradient signal during training (smooth optimization), sparse efficient compute at inference (O(log N) HNSW queries).

## Relationship to step852

step852 KILLED hot-rebuilding `conn_hh` because edges-as-identity carry learned state (W_pos, AH coupling) that breaks when edges change. This design **avoids the trap**: it doesn't rebuild `conn_hh`; it replaces `conn_hh` entirely with distance-based weights. There are no discrete edges to rebuild — every forward pass computes a continuous distance field and softmaxes over it. The HNSW step at inference is a **lossless-under-low-temperature approximation**, not a learned structural change.

## Training-Time Math

For each active (beam) node m with state Z_m ∈ R^D:

```
dist[m, n] = ||Z_m - W_pos[n]||²        for n ∈ [0, N)
w[m, n]    = softmax_n(-β · dist[m, n])  # differentiable, peaks on nearest
signal[n]  = Σ_m w[m, n] · Z_m           # dense aggregation
```

- All N nodes receive weighted signal from beam broadcasters.
- Gradient flows to W_pos through the distance term (nodes learn to position themselves near/far from activations they want to receive).
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

At β=10 with distance gap 0.1 between top-K boundary and tail, the tail contributes <37% of top-K mass per step. Sharp enough β → HNSW approximation is near-lossless.

## β Scheduling (Critical)

- **Start soft:** β = 0.5 (wide gradient distribution, all N contribute approximately equally).
- **Anneal sharp:** β → 10 linearly over epochs (final distribution concentrated on K nodes).
- **By end of training:** softmax mass on top-K captures ≥99% of routing weight.
- **HNSW truncation after training:** approximately lossless.

Schedule is analogous to temperature annealing in Gumbel-softmax or soft-to-hard clustering. The risk is symmetric to those: too aggressive kills training; too slow leaves routing un-sharp and HNSW truncation becomes lossy at eval.

## HNSW on W_pos (not Z)

- **Index built from learned W_pos [N, D_pos]** → static at inference.
- **Query with activation Z_m** → returns nearest W_pos nodes.
- **Rebuild cadence during training:** every 10 epochs (step852 suggests slow cadences cause less harm; fast rebuilds catastrophic).
- **For N=2048:** brute-force k-NN via `torch.cdist` is ~0.1ms on CUDA — HNSW only needed at N≥16384.

## Composition

- **Replaces `conn_hh` entirely:** routing becomes distance-based, not edge-list-based. This is the cleanest way to avoid the step852 failure mode — there are no discrete edges to update inconsistently.
- **With [[sparse_bfs_routing]]:** the top-M active nodes perform the HNSW queries. Beam gating reduces HNSW calls per step from N to M.
- **With cascading beam:** narrower beam in deeper iterations; fewer queries as frontier stabilises.
- **With [[delta_w]] projection:** potentially orthogonal — proj modulates signal *magnitude* along relational axis; soft routing determines *which* neighbours to route to. Compose only after isolating each mechanism first.
- **With [[activation_retention]]:** retained Z biases the distance field smoothly across time steps.

## Risks

- **Training FLOPs:** O(B · M · N · D) per iter — potentially slower than static gather at large N unless beam-gated.
- **β schedule:** too aggressive → training fails; too slow → HNSW truncation lossy at eval.
- **HNSW query mismatch:** if Z drifts during training, a stale index built on old W_pos returns wrong neighbours. Rebuild cadence matters.
- **W_pos cluster collapse:** if all W_pos cluster near origin (AH not holding diversity), HNSW returns essentially random neighbours and routing degenerates. AH's diversity pressure becomes load-bearing for this design.
- **Non-differentiability of top-K selection at inference:** gradients only flow through the K selected nodes during training (if any training-time truncation is used). Fully soft training avoids this.

## Paper Claim (If Successful)

> Dense differentiable routing during training, O(log N) sparse routing at inference, with <X% accuracy loss from the soft-hard transition.

This would reframe the SGNNET efficiency story: instead of fixed edges, *topology itself is learned and queried* — a genuinely novel mechanism for O(log N) inference.

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
- HNSW integration at eval time (hnswlib or torch-implementation).
- Dual forward path (training vs inference).

## Open Questions

1. Does β annealing need to be per-layer or per-K_iter-step, or is a global schedule sufficient?
2. Does soft routing implicitly do what [[delta_w]] projection does? Both weight neighbours by a similarity score derived from W_pos.
3. At what N does HNSW become necessary? N=2048 brute-force `cdist` is cheap; paper may not motivate HNSW unless validated at N≥16384.
4. Does W_pos get sufficient gradient through the distance term, or do we still need explicit AH to enforce diversity?
5. Does the **training** FLOP increase (dense distance field) outweigh the **inference** FLOP decrease? Training cost matters for paper but less than inference cost.

## See Also

- [[sparse_bfs_routing]] — complementary; soft routing provides the weights, BFS provides the broadcaster gating
- [[delta_w]] — another W_pos-based modulation; may compose or be redundant
- [[antihebbian]] — AH enforces W_pos diversity; load-bearing for meaningful distance fields
- [[activation_retention]] — Z_{t-1} biases the distance field temporally
- [[architecture_dead_ends]] — step852 dynamic-edge-rebuild failure informs why this design avoids edge updates
