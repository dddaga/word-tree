# Sparse BFS Routing (HYPOTHESIS)

**Status:** HYPOTHESIS — not implemented, not validated. Design captured from 2026-04-16 research session.

## Motivation

Current SGNNET routing in `SGNNET_SmallWorld._route()` is dense — every iteration touches ALL N=2048 hidden nodes (`Z = Z[:, conn_hh, :].sum(dim=2)` at `model_smallworld.py:273`). Cost: O(B·N·K_hh·D) per iter × K_iter iters. At N=2048, K_hh=2, D=16, K_iter=5: **327K routing MACs**. The `beam_size` parameter in phase inhibition only gates the inhibition path, not the main routing.

The observation: many hidden nodes are near-quiescent per input. Broadcasting from every node wastes compute on nodes that carry no meaningful activation. Beam-gated BFS restricts broadcasters to the top-M most active per iteration, tracking the reachable frontier.

## Proposed Design

**Beam-gated BFS with frontier tracking.** Only the top-M active nodes broadcast messages per iteration; the frontier (reachable set) is bounded by M · K_hh^(K_iter-1). At M=16, K_hh=2, K_iter=5: frontier ≤ 176 nodes ≪ N=2048.

### Key math

```
Frontier bound:   |F_t| ≤ |F_{t-1}| + M · K_hh       (tree upper bound; in practice lower due to edge overlap)
topk cost:        O(|F_t| · log M)                   (scales with frontier, not N)
Routing cost:     O(M · K_hh · D)  per iter          (vs O(N · K_hh · D) dense)
```

**At M=16 vs N=2048: 128× cheaper per iteration.**

### Cascading beam schedule

Beam narrows over iterations, e.g., `beam = [16, 16, 8, 4, 4]` for K_iter=5. Early iters discover; later iters refine.

```
Total routing FLOPs = Σ beam · K_hh · D
                    = 48 · 2 · 16
                    = 1.5K MACs
```

**vs 327K dense = 218× reduction.**

## Implementation Sketch

```python
def _route_bfs(self, Z_seed, conn_hh, beam_schedule):
    # Initial frontier: all seeded nodes (or top-M of seed norms)
    active_idx = torch.arange(N).expand(B, -1)         # or topk(Z_seed.norm(-1), M_init)
    Z = Z_seed                                         # dense [B, N, D] — preserves readout
    for t, M in enumerate(beam_schedule):
        # Select broadcasters from current frontier
        Z_active = Z[batch, active_idx]                # [B, A, D]
        beam_idx = topk_by_norm(Z_active, M)           # [B, M]
        # Gather broadcasters' neighbors
        new_idx = conn_hh[beam_idx.flatten()]          # [B, M, K_hh]
        # Route: update only newly-active indices
        Z[batch, new_idx] = route_messages(Z[beam_idx], ...)
        active_idx = unique(cat([active_idx, new_idx.flatten()]))
    return Z                                           # dense for readout
```

## Readout Compatibility

Z stays dense [B, N, D] throughout — quiet (non-frontier) nodes keep their seed values. Readout `Z @ W_out.T` unchanged. No readout refactor required for the baseline variant. Config D variant below tests a sparse readout as an aggressive efficiency extension.

## Static Topology Preserved

**Critical constraint (step852 evidence):** hot-rebuilding `conn_hh` mid-training KILLS learning (−1.91pp at best, −79pp at worst). Sparse BFS **does not rebuild edges**. It uses the same static Watts-Strogatz `conn_hh` as Ref — only the **selection of active broadcasters** changes per step.

This is the key distinction from [[soft_routing_hnsw]]: sparse BFS keeps edges static and gates broadcasters; soft routing replaces edges entirely with distance-based weights.

## Composition

- **With W_pos-nearest edges:** combines if edges are static post-init (step852 forbids hot rebuild). Init-time W_pos-nearest works.
- **With cascading beam schedule:** natural fit; beam size is the degree of freedom.
- **With [[delta_w]] projection:** orthogonal — ΔW proj modulates signal magnitude along the relational axis; sparse BFS selects which senders contribute. Different gradient paths.
- **With [[activation_retention]]:** the retained state Z_{t-1} naturally provides the initial frontier at step t, no cold start needed.
- **With [[soft_routing_hnsw]]:** sparse BFS can use HNSW-retrieved neighbors instead of `conn_hh`; beam nodes do the HNSW queries.

## Risks

- **Quiet-node capacity loss:** non-frontier nodes never update beyond seed → may lose representational capacity. Test via Config C (quiet nodes zeroed).
- **Batch variability:** different inputs produce different frontiers → deduplication cost across batch, irregular memory access.
- **CUDAGraph compatibility:** dynamic sizes break torch.compile graphs; need pre-registered shape budgets (pad frontier to max size, mask).
- **Gradient flow:** top-k is non-differentiable in the selection; gradients flow through selected values only (straight-through on selection). May harm learning of which nodes should be active.
- **M-schedule sensitivity:** untested — may need N-dependent tuning like K_iter.

## Expected Paper Value

- **Inference FLOPs:** ~60× routing reduction at fixed M=16; ~218× with cascading schedule.
- **Accuracy:** HYPOTHESIS — frontier nodes carry sufficient signal. Must test.
- **Wall-time:** dependent on Triton gather/scatter implementation. Dense routing is already compiled on CUDA (step500); sparse must beat a compiled dense baseline to be worth the complexity.

## Proposed Experiment — step855_sparse_bfs

| Config | Beam | Readout | Tests |
|--------|------|---------|-------|
| Ref | dense (no beam gate) | `Z[all] @ W_out` | baseline |
| A_fixed_M16 | `[16]*5` | same | does beam-gate survive? |
| B_cascade | `[16,16,8,4,4]` | same | progressive narrowing |
| C_quiet_zero | `[16,16,8,4,4]`, quiet nodes zeroed | same | do quiet seeds matter? |
| D_readout_active | `[16,16,8,4,4]` | `Z[active] @ W_out[active]` | sparse readout too |

**Scale:** N=2048, T0 (20ep, 50% data). ~1 week implementation including Triton GPU kernels for gather+scatter.

**Success criteria:**
- A_fixed_M16 within −0.5pp of Ref → beam gating viable
- B_cascade ≥ A_fixed_M16 → cascading narrowing works
- C_quiet_zero ≥ B_cascade → quiet seed values don't matter (aggressive sparsification possible)
- D_readout_active within −1.0pp of Ref → sparse readout viable; unlocks further FLOP reduction

## Open Questions

1. Should beam selection use norm, logit of final readout, or a learned scorer?
2. Does the frontier grow uniformly across a batch, or do outlier inputs dominate frontier size?
3. Does static vs learned beam schedule matter? (Gate-death risk if learned via multiplicative gate — see [[gate_death]].)
4. Interaction with [[delta_w]]: does ΔW proj selectivity already do what beam gating attempts?

## See Also

- [[soft_routing_hnsw]] — complementary design, replaces edges with distance weights
- [[activation_retention]] — provides warm frontier across sequence steps
- [[streaming_input]] — frontier grows as segments seed new nodes
- [[gate_death]] — why learned beam scores risk multiplicative collapse
- [[delta_w]] — orthogonal signal-path mechanism that could compose
