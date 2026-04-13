# Concept: Beam Search (beam_width)

## What it is

Per-iteration filtering of *source* nodes during message passing. Only the top-K most active nodes (by `activation_strength`, per sample) are allowed to emit messages in a given iteration. All other nodes keep their previous activation unchanged.

Config: `model.beam_width: K` (int). Default `0` = disabled (all active nodes can emit).

Implemented in `native/layer.py:_apply_beam_and_replicate` (lines 303–331).

## Why it might help

1. **FLOPs reduction.** Aggregation cost scales with the number of edges used per iteration. If only K sources emit, edges drop from `~N*C` to `~K*C` (roughly — some sources may share destinations). For `N=4146, K=512` → ~8x fewer edges per iteration.
2. **Focus compute on informative nodes.** At low routing temperatures, a small subset of nodes dominates gradient flow anyway (see `gradient_starvation.md`). Hard-pruning the long tail may be a "free lunch" since those nodes barely contribute.
3. **Acts as implicit regularization.** Different top-K sets selected each iteration → message-passing path changes → similar effect to edge dropout, but structured.

## Why it might hurt

1. **Input nodes are exempt but intermediate/output nodes are not.** An output node could be pruned from sources in late iterations if its act_strength drops — but outputs aren't typically sources anyway, so this is minor.
2. **Winner-take-all dynamics.** Early in training, act_strength rankings are noisy. Hard top-K could amplify random early advantages ("rich-get-richer"), starving late-bloomer nodes of any gradient updates.
3. **Incompatible with the all_destinations fast path.** When beam is on, `effective_all_active` is forced to `False` in `layer.py:216`, so each destination must be checked individually for incoming edges. Adds a small constant overhead per iteration.
4. **topk cost.** `candidate_strengths.topk(K, dim=1)` costs `O(B * N * log K)` per iteration — usually negligible but worth noting.

## Implementation details

### Who gets filtered
- **Input nodes**: always exempt (they're the only source of fresh signal from the data). A separate `_beam_exempt` mask forces inputs into the allowed set regardless of act_strength.
- **Intermediate/output nodes**: eligible for top-K ranking.

### How filtering works (per iteration)
```python
candidate_mask = active_mask & ~exempt_mask              # eligible sources
if num_candidates <= beam_width:                         # pass-through
    return replicate(iter_edge)
as_2d = act_strength.view(B, N)                          # per-sample strengths
candidate_strengths = where(candidate_mask, as_2d, -inf)
_, topk_idx = candidate_strengths.topk(K, dim=1)         # per-sample top-K
allowed = exempt | topk_idx                              # per-sample allowed-source mask
for b in range(B):
    keep = allowed[b][edge_src]                          # edges whose src is allowed
    edges_b = iter_edge[:, keep] + offset[b]             # per-sample filtered edges
```

### Per-sample filtering
Each sample in the batch selects its own top-K independently. This is the right behavior semantically (different images activate different nodes) but means the edge tensor is no longer shared across samples — each sample has its own filtered set. This is why the fused `_apply_beam_and_replicate` exists: to avoid constructing the full `B*E` mega-graph upfront, then filtering.

### When beam activates
Beam only prunes sources after `iterations - 1` propagation steps have begun. Inputs are always exempt, so iteration 1 (input injection) is never filtered.

## FLOPs impact

From `concepts/flops.md`, per-iteration FLOPs are dominated by `8 * E * V`, where `E = B*N*C` in the no-beam case.

With beam width `K`:
- **Effective edges**: roughly `B * K * C` (each of K allowed sources contributes C outgoing edges on average).
- **FLOPs reduction**: `K/N` fraction of aggregation cost.
- **New total per iteration**: `B * (8*K*C*V + 7*K*C + 25*N*V + 1)`
  (Note: `25*N*V` term — LN, act_strength recompute, temporal decay — is per-destination and still runs for all N destinations, so that cost stays.)

### Worked example (run17 config, beam sweep)

Run17: `N=4146, C=200, V=8, I=5, B=4` → `(I-1)*B*N*(8CV + 7C + 25V) ≈ 0.96B FLOPs/fwd`.

| beam_width | Effective edges (B=4) | `8*K*C*V` term | `25*N*V` term | FLOPs/fwd | Savings |
|-----------:|----------------------:|---------------:|--------------:|----------:|--------:|
| 0 (all)   | 4*4146*200 = 3.32M   | ~2.12M per-sample | 829K per-sample | ~0.96B | 1x |
| 2048      | 4*2048*200 = 1.64M   | ~1.05M         | 829K           | ~0.57B   | 1.7x |
| 1024      | 4*1024*200 = 820K    | ~524K          | 829K           | ~0.37B   | 2.6x |
| 512       | 4*512*200 = 410K     | ~262K          | 829K           | ~0.27B   | 3.6x |
| 256       | 4*256*200 = 205K     | ~131K          | 829K           | ~0.22B   | 4.4x |
| 128       | 4*128*200 = 102K     | ~65K           | 829K           | ~0.19B   | 5.1x |

**Diminishing returns:** The `25*N*V` floor (LN + act_strength recompute, runs for all N destinations) caps beam's savings at ~5x. To go lower, cardinality reduction is more effective.

## Interaction with other levers

- **Cardinality**: Independent. `8*K*C*V` scales linearly with both K and C. Reducing one doesn't obviate the other; they multiply.
- **Iterations**: Independent. Each of `(I-1)` iterations applies the same beam filter.
- **Routing temperature**: **Likely non-trivial interaction.** At low T, act_strength is concentrated on few nodes → top-K selection is stable, beam=small should work. At high T (run17: T=4.0), act_strength is more uniform → top-K is less discriminative, small beam may drop important contributors. Hypothesis: optimal beam scales with T.
- **Edge dropout**: Orthogonal. Beam filters by source identity; dropout filters by edge identity. Both can be on.
- **LayerNorm placement (post-update)**: Beam filters happen *before* the update, so LN doesn't directly interact. But LN affects act_strength magnitude — so whichever LN regime is active sets the ranking used for topk.

## Prior results

Beam was tested in runs 7-9 of the FFN era — never cleanly.

| Run | beam_width | Topology | Other deltas | Val best | Verdict |
|-----|-----------:|----------|--------------|----------|---------|
| run7 | 0         | layered  | I=7, temporal_decay=0.8 | 23.08% (ep9) | Stopped; confounded (3 vars) |
| run8 | 500       | layered  | I=15, CPU               | 13.32%       | Stopped; highly confounded  |
| run9 | 0         | layered  | I=7                      | 25.45% @ep29 | Nearest-ctrl vs run6; poor |

**Status: NEVER cleanly ablated.** The only run with `beam_width > 0` was run8, which was also layered + I=15 + CPU — unsalvageable as a beam-only ablation. **The beam sweep planned for run23+ is the first clean single-variable test under the best-known config (run17: N=4146, T=4.0, flat, no FFN, new LN).**

## Open questions

1. **Optimal beam width at T=4.0?** With softer routing, more nodes carry meaningful signal. Maybe beam needs to be larger than it would be at T=1.0.
2. **Does beam stabilize or destabilize training?** Winner-take-all risk at early epochs.
3. **Can we warm-start beam?** e.g. beam=0 for first 5 epochs, then K=256 for rest. Would avoid early rich-get-richer.
4. **Should output nodes ever be sources?** In flat topology they can be — beam might implicitly remove them since their act_strength is low initially, which might actually be desirable.

## Cross-references

- `native/layer.py:169-208` — beam exempt mask + filtering call site
- `native/layer.py:303-331` — `_apply_beam_and_replicate` implementation
- `concepts/flops.md` — full FLOPs derivation (beam changes Phase 3)
- `concepts/gradient_starvation.md` — context for why top-K may be "free"
- `concepts/architecture.md` — core layer architecture
- `EXPERIMENT_QUEUE.md` — run23-26 beam sweep plan
