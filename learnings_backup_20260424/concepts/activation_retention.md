# Activation Retention (HYPOTHESIS)

**Status:** HYPOTHESIS — not implemented for sequences, not validated. Design captured from 2026-04-16 research session.

**Prior attempt KILLED (different context):** step306 tested static/decay/norm_cons/reinject retention variants on single-image Imagenette; all hurt (−0.10 to −18.67pp). That setting has **no inherent sequence structure**, so retention has nothing to retain. The design below applies retention to **genuinely sequential inputs** (audio, video, time-series, sequential text), a different regime.

## Motivation

Current SGNNET resets Z every forward pass — stateless. For single-image classification this is correct (Imagenette). For sequences (time series, video, text), SGNNET loses continuity between samples. Adding a running context Z_{t-1} with decay turns SGNNET into a recurrent architecture with minimal inference cost.

## Proposed Mechanism

```
Z_t = α · Z_{t-1} + (1 − α) · route(seed(X_t) + β · Z_{t-1})
```

where:

- α ∈ [0, 1] = retention coefficient (0 = current SGNNET, 1 = full memory, no new input effect)
- β ∈ [0, 1] = feedback coefficient (how much prior state biases new input routing)
- Z_{t-1} persists across forward calls (carries forward through sequence)

**Interpretation:** each hidden node's activation is a running average of its trajectory. The "direction of context" at node n = integrated activation history at n. Conditional routing can depend on this direction.

## Biologically-Inspired Reflection

Activation at a node doesn't fully transfer to neighbors — some portion is retained and decays:

```
Z_t[n] = decay_n · Z_{t-1}[n] + incoming_messages_t[n]
```

where `decay_n` can be per-node learnable (different neurons retain different history lengths — fast vs slow channels). This is structurally analogous to leaky integrate-and-fire, or to the forget-gate in an LSTM generalised to per-neuron time constants.

## Context-Conditioned Routing

The stored state Z_{t-1}[n] is an aggregation of the historical trajectory at n. Routing of incoming activations can be conditioned on this:

```
gate[m, n] = f(direction_similarity(Z_{t-1}[n], incoming_t[m]))
```

Nodes route incoming signals based on which neighbours match their "learned purpose" as defined by historical activation. This is a **context-aware** variant of [[delta_w]] projection, where the "direction" is the running state rather than W_pos relational axis.

**Gate-death risk:** if `gate` is a multiplicative factor applied across K_iter × T steps, it compounds. See [[gate_death]] — multiplicative gates g^K_iter → 0 is a known SGNNET failure pattern. Any gate here must use additive/redistributive form, not pure multiplication.

## Inference Cost

- **Essentially unchanged:** just carry forward Z [B, N, D] as state.
- **FLOPs per step:** ~same as current forward pass.
- **Memory:** 2× current briefly (Z_{t-1} and new computation coexist for one step).

This is the design's key efficiency property: O(1) state amortised across a sequence, vs Transformer-style O(T²) or O(T) KV-cache growth.

## Training Cost

- **BPTT-like** through sequence length T.
- **Gradient accumulation** through K_iter × T steps.
- **Risk of gradient explosion/vanishing** for long sequences — standard RNN problem. May need gradient clipping, or clamping on α (prevent α → 1 which would make gradient history infinite).

## When It Applies

- **Time series** (weather, stock, sensor streams).
- **Video** (frame-to-frame correlation).
- **Sequential text** (running meaning accumulation).
- **Audio** (future paper vehicle — ESC-50 data already present, though step406 showed raw cross-modal fails without retention).

## When NOT Relevant

- **Current paper (Imagenette single-image):** out of scope. step306 already confirmed retention hurts on static inputs.
- **Tasks without inherent sequence structure:** nothing to retain; retention becomes noise.

## Composition

- **With [[sparse_bfs_routing]]:** running frontier persists across sequence (new frontier = old ∪ newly-seeded). Retention solves the cold-start problem for BFS at each time step.
- **With [[soft_routing_hnsw]]:** soft weights naturally integrate with decayed Z — the distance field at time t reflects accumulated context.
- **With [[streaming_input]]:** activation retention IS the mechanism streaming relies on. Segments arrive over time; retention preserves the partial state between segment arrivals.
- **With [[delta_w]] projection:** ΔW direction can be augmented with the retained-state direction for per-step gating.
- **With AH ([[antihebbian]]):** AH is position-only (static W_pos gating); retention is activation-level. Orthogonal signal paths. Safe to compose on paper; must still isolate first.

## Risks

- **Training stability:** recurrent gradients may explode for long sequences. Known RNN failure mode.
- **α tuning:** too high → stale representation, new input can't overwrite; too low → no benefit over stateless.
- **Forget-vs-remember tradeoff:** learnable per-node α is more flexible but adds parameters and training difficulty.
- **Applying to wrong task:** step306 is evidence that retention on static inputs hurts. The mechanism must match the data structure.
- **Data requirement:** sequential benchmarks (ESC-50 frames, UCI time-series, K400 video clips) need pipelines that don't currently exist in this repo.

## Paper Direction

This is a **follow-up paper vehicle**, not Paper 1. Paper 1 focuses on single-image efficiency (SGNNET as FC replacement for VGG16 on Imagenette). A sequence paper would be:

> "SGNNET for sequence modelling — O(1) inference state with learned retention."

That framing matches the project's long-term goal (improve memory footprint + energy efficiency of DL in general — see `project_goal_efficiency_of_dl.md`) and naturally follows the single-image efficiency story.

## Proposed Experiments (FUTURE — after Paper 1)

| Step | Task | Config |
|------|------|--------|
| step870 | Audio ESC-50 with α ∈ {0, 0.1, 0.3, 0.5, 0.7} | frame-by-frame seed + retention |
| step871 | Time-series UCI datasets | baseline vs SGNNET+retention |
| step872 | Video frame classification (subset of K400) | T=4 frame context |

**Scale for all:** start at T0 (short sequences, 20ep, 50% data). Do not advance to longer sequences until stateless baseline is matched.

**Success criteria (step870):**
- α=0 recovers stateless baseline (sanity check — no bug in retention path).
- Some α > 0 beats α=0 on ESC-50 → retention mechanism works on sequential audio.
- Per-node learned α > fixed α → parameterised retention pays for itself.

## Open Questions

1. Is per-node α (learnable vector) better than scalar α? Adds N parameters.
2. Does the feedback coefficient β (prior state biasing new input routing) matter, or is simple EMA (β=0) sufficient?
3. How does retention interact with K_iter? Does K_iter=5 still hold, or does recurrence effectively provide extra iterations?
4. Can retention replace K_iter entirely (K_iter=1 with retention ≈ K_iter=5 stateless)? If so, huge FLOP win per time step.
5. Does it gate-death? Multiplicative decay α^T → 0 for long T is structurally similar to gate-death. Mitigations: residual path, learnable per-node α with prior ≈ 1.

## See Also

- [[sparse_bfs_routing]] — retained state provides warm frontier
- [[soft_routing_hnsw]] — retained Z biases distance field
- [[streaming_input]] — retention is the substrate streaming relies on
- [[delta_w]] — similar mechanism (direction-based gating) but on static W_pos
- [[gate_death]] — why multiplicative decay must be constrained
- [[architecture_dead_ends]] — step306 static-input retention killed; informs scope
