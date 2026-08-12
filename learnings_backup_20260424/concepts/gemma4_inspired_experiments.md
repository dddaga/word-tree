# Gemma 4 → SGNNET Inspired Experiments

Source: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4 (read 2026-04-19)

## Background

Gemma 4 introduces several architectural ideas that map onto SGNNET's signal routing and
positional encoding mechanisms. Four experiments proposed below, ordered by novelty and
estimated impact.

---

## Experiment 1 — Local/Global K_iter Alternation (step933)

**Gemma 4 analogue:** Local attention (K=1 window) interleaved with global attention (K=all)
every N layers. Local = cheap, focused; global = integration, long-range.

**SGNNET mapping:** Alternate K_iter rounds between:
- "Local" round: aggregate only from conn_hh (existing K_hh neighbors, short-range)
- "Global" round: aggregate from a random K_global subset across all N nodes (long-range)

Pattern: `[local, local, global, local, local, global, ...]` — 1 global every G steps.

**Hypothesis:** Local rounds refine, global rounds inject non-local context. May improve
multi-class (CIFAR-100) where far-flung class features co-activate. CONFIRMED safe by
additive aggregation — no gate-death risk.

**Key parameter:** global_every ∈ {2, 3, 5}; K_global ∈ {4, 8}.

**Status:** PROPOSED — novel, no prior equivalent found.

---

## Experiment 2 — p-RoPE Dim Pruning in ΔW (step934)

**Gemma 4 analogue:** p-RoPE applies RoPE to only 25% of dimensions (p=0.25), leaving
75% non-rotated. Reduces positional coupling cost; helps model focus on content dimensions.

**SGNNET mapping:** In ΔW-proj, `dw = normalize(W_h[i] − W_h[conn])`. Apply position-only
transformation to p fraction of D dims; leave (1−p)×D dims as pure ΔW content.

Concrete: split W_h features into `W_pos[:, :p*D]` (rotated by Fourier angle) and
`W_content[:, p*D:]` (raw ΔW). Routing coefficient uses both but with separate weighting.

**Hypothesis:** Separating positional vs content channels in ΔW may reduce interference,
especially at larger D. Low-risk additive change.

**Key parameter:** p ∈ {0.25, 0.50}; D=16 baseline.

**Status:** PROPOSED — novel. Related: step903 explored topology rebuild but not channel split.

---

## Experiment 3 — MoE Additive Topology Routing (step935)

**Gemma 4 analogue:** MoE with 128 experts, 8 active + 1 shared expert. Routing is additive
(shared expert always active). Experts are independent FFNs.

**SGNNET mapping:** Multiple K_hh topology "experts" (e.g., spatial near, random, high-degree
hub, low-degree peripheral). Router selects top-k experts per node; aggregate their signals
additively. Shared "expert" = always-active local neighbor (K_r=1).

This is **additive** — safe from gate-death. Essentially a weighted multi-topology readout.

**Hypothesis:** Different node types (hub vs leaf vs random) benefit from different neighbor
strategies. MoE topology lets each node self-select its routing strategy.

**Key parameter:** n_experts ∈ {2, 4}; active_k ∈ {1, 2}; topology types: spatial/random/hub.

**Status:** PROPOSED — novel. Hub topology explored in step875 but not as MoE mixture.

---

## Experiment 4 — Per-Iter W_pos (step936)

**Gemma 4 analogue:** Per-Layer Embeddings (PLE) — learned per-layer scale/shift applied
to activations. Each layer sees a unique "positional personality."

**SGNNET mapping:** Instead of one shared W_pos: D×D matrix, use separate `W_pos^{(k)}`
for each of K_iter iterations. Each routing round uses its own projection of the Fourier
embedding, allowing the network to "look" at different angular slices per iteration.

```python
self.W_pos_list = nn.ParameterList([
    nn.Parameter(torch.randn(D, D) * 0.01) for _ in range(K_iter)
])
# In forward loop at iteration k:
pos_k = x @ self.W_pos_list[k]  # [N, D]
dw = normalize(pos_k[i] - pos_k[conn_hh[i]])
```

**Prior work check (2026-04-19):**
- **step903**: Rebuilds K_random slot topology per epoch — NOT the same as per-iter W_pos.
- **step852**: Full conn_hh rebuild every N epochs → −1.91pp (KILLED). Different mechanism.
- **step700**: Multi-hop gather — ALL KILLED (−3.5 to −13pp). Different mechanism.
- **W_pos_list / per_iter_wpos**: No script found with this pattern. **NOVEL.**

Result: Experiment 4 has NOT been run. Closest prior is step903 (epoch topology, not per-iter).

**Parameter cost:** K_iter × D² extra params. At K_iter=5, D=16: 5×256=1,280 params (negligible).

**Status:** PROPOSED — confirmed novel. Medium priority (PLE in Gemma was marginal).

---

## Priority Table

| Step | Name | Novelty | Risk | Est. Impact | Priority |
|------|------|---------|------|-------------|----------|
| 933 | Local/Global alternation | High | Low (additive) | Medium-High | **1st** |
| 935 | MoE topology routing | High | Low (additive) | Medium-High | **2nd** |
| 934 | p-RoPE dim pruning | Medium | Low | Medium | **3rd** |
| 936 | Per-iter W_pos | Medium | Low | Medium | **4th** |

All four use additive aggregation — gate-death theorem does not apply.

## Suggested Sequencing

1. Run step933 T0 on mini:cpu (small, quick iteration). Use CIFAR-100 as testbed (harder).
2. Run step935 T0 on mini:mps alongside.
3. If either shows ≥+0.5pp vs Ref → promote to T1; compound winner with step936 T0.
4. step934 last — channel splitting requires architectural surgery on ΔW core.
