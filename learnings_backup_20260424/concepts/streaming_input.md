# Streaming / Cascading Input (HYPOTHESIS)

**Status:** HYPOTHESIS — not implemented, not validated. Depends on [[activation_retention]]. Design captured from 2026-04-16 research session.

## Motivation

Current SGNNET seeds all N hidden nodes at once from the full input X (all 25088 VGG features → all 2048 hidden). This is fine for fixed-size inputs but forecloses:

1. **Variable-size inputs** (different sequence lengths, variable-resolution images).
2. **Partial / early-exit inference** (stop when confident).
3. **Sequential modalities** where input arrives incrementally (audio frames, video, streaming text).

## Proposed Mechanism

**Cascading/streaming seed with concurrent routing.**

```
X split into sub-segments: [X_1, X_2, ..., X_k]

Z_0 = seed(X_1)                       # initial partial seed
for step in range(1, k):
    Z = route(Z, 1 iter)              # routing propagates prior seed
    Z = Z + gate · seed(X_step)       # add new segment's seed to evolving Z
# final K_iter_final steps on complete Z
for _ in range(K_iter_final):
    Z = route(Z)
```

Each segment's arrival triggers one routing iter, interleaved with fresh seeding. After all segments are ingested, a final burst of K_iter_final settles the representation before readout.

## Sub-Segment Choices

- **Spatial partition of VGG features:** rows / columns / quadrants of the 7×7×512 feature map.
- **Sequential bits of an input:** bit-plane progressive (low bits → high bits).
- **Frequency bands** (low → high) for audio.
- **Resolution cascades** (low-res → high-res) for images.
- **Natural temporal structure** (time steps for time-series, frames for video, sentences for long text).

## Routing During Ingestion

- Each new segment's seeded nodes join the **active frontier** (connects naturally to [[sparse_bfs_routing]]).
- Existing activations influence which new nodes are "important" via `gate` — may be identity (all new seeds admitted) or learned / activation-dependent.
- **Inter-neurons** (nodes without direct seed from current segment but connected to currently-active nodes) act as routing intermediaries, bridging time-separated seeds. (HYPOTHESIS — to test whether intermediate nodes carry information or are dead weight.)

## Progressive Inference / Early Exit

- After each segment ingestion, a readout head produces a **partial logit**.
- If max-logit confidence > threshold, return — no need to ingest full input.
- **Inference cost is proportional to segments processed, not total input.**
- Confidence calibration is non-trivial (entropy, logit margin, or learned confidence head).

**Contrast with prior work:** step703 (per-sample adaptive K_iter at inference) was KILLED — all thresholds catastrophic (−80pp), logit cosine_sim was not a reliable convergence proxy. Streaming early-exit differs in that it exits **input ingestion**, not **routing iterations**. The two axes are orthogonal; streaming early-exit may still be viable even though K_iter early-exit was not.

## Role of Inter-Neurons (HYPOTHESIS)

Nodes that don't receive direct seed from current segment but are connected to currently-active nodes act as routing intermediaries. Their function is to **bridge time-separated seeds**. If ablated (frozen while off-frontier), streaming should degrade more than non-streaming.

**Test:** compare a streaming variant that allows intermediate nodes to update vs one that only updates directly-seeded nodes.

## Composition

- **With [[activation_retention]]:** retention IS what makes streaming work — Z must persist across segments. Without retention, Z_0 is overwritten and streaming degenerates to "route on the last segment only".
- **With [[sparse_bfs_routing]]:** the active frontier naturally grows as new segments seed new nodes; frontier tracking is a native fit.
- **With [[soft_routing_hnsw]]:** soft weights handle "should this new seed influence this already-active node" differentiably via distance field update.
- **With [[delta_w]] projection:** ΔW proj modulates intra-segment routing; streaming provides inter-segment structure. Orthogonal.

## Risks

- **Segment ordering:** different orders may produce different outputs (non-commutative routing). Could be a **feature** (temporal/causal data) or **bug** (spatial partitions where order is arbitrary). Must test order-permutation invariance for spatial cases.
- **Training data:** needs sequence-structured data; random segmentation of Imagenette may not help and may hurt (step306-like failure mode — adding sequence to non-sequential data).
- **Inference timing:** gives flexibility but complicates wall-time measurement — per-sample compute varies with early-exit behaviour.
- **Readout calibration:** partial logits at early segments are under-informed; may need temperature scaling or confidence head.
- **Training-inference mismatch:** if trained with all segments, but deployed with early-exit, the final-segment-only readout was never trained.

## When SGNNET Shines (HYPOTHESIS — to test)

- **Time series** (natural segmentation: time steps).
- **Audio** (natural segmentation: frames; ESC-50 reload possible).
- **Long documents** (natural segmentation: sentences / paragraphs).
- **Progressive images** (natural segmentation: multi-resolution or multi-scale).

## When SGNNET May Struggle

- **Fixed-size classification (current paper):** no segmentation advantage over current full-input-at-once seed.
- **Symbolic/discrete tasks** where order matters in specific ways the network doesn't learn.
- **Any task where step306 regime applies** (single-image-like, no temporal structure).

## Paper Direction

This is a **follow-up paper extension**, not Paper 1. Together with [[activation_retention]], it forms a natural sequel:

> "Streaming SGNNET: O(1) state + progressive input ingestion for variable-length sequences with early exit."

## Proposed Experiments (FUTURE — needs [[activation_retention]] first)

| Step | Task | Config |
|------|------|--------|
| step880 | Sequential MNIST (per-pixel stream) | vs full-image baseline |
| step881 | Time-series UCI with variable-length sequences | streaming + retention |
| step882 | Audio progressive classification (ESC-50 partial clips) | 1–5 frame early-exit |

**Scale for all:** T0 first (20ep, 50% data, short sequences). Advance only if stateless SGNNET baseline is matched on the same task.

**Success criteria (step880):**
- Full-pixel streaming recovers stateless baseline (sanity — no bug in streaming path).
- Early-exit at 50% pixels loses <2pp → streaming inference viable.
- Permuted-pixel order degrades significantly → confirms inter-neuron bridging is load-bearing (tests the HYPOTHESIS).

## Open Questions

1. Should `gate` be identity, learned scalar, or learned per-node? Learned risks gate-death ([[gate_death]]).
2. How many routing iters per segment? One may be too few for propagation; too many wastes compute.
3. Is order-invariance desirable (images) or order-sensitivity (time-series)? Different tasks need different designs.
4. Does streaming training need curriculum (short sequences → long)?
5. Can the "final burst" K_iter_final be removed, or is it load-bearing for readout quality?
6. Progressive logit calibration: does a single readout at the end suffice, or do per-segment readouts need independent calibration?

## See Also

- [[activation_retention]] — **prerequisite** — streaming requires state persistence
- [[sparse_bfs_routing]] — frontier grows with segments; natural composition
- [[soft_routing_hnsw]] — smooth distance-based integration of new seeds
- [[delta_w]] — orthogonal intra-segment modulation
- [[architecture_dead_ends]] — step703 (K_iter early exit) killed — informs why we exit input, not routing
- [[gate_death]] — gate on new-segment admission must not be pure multiplicative
