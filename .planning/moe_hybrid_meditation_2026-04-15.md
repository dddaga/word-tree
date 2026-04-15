# MoE×SGNNET Hybrid — Design Meditation
# 2026-04-15

## Context

MLP_37 (929K params, 1.86M FLOPs) hits 97.71% on Imagenette — matching SGNNET's best accuracy
at matched FLOPs. The FLOPs-Pareto claim is falsified at 10 classes. What survives: SGNNET wins
on (params × FLOPs) Pareto (25× fewer params at matched FLOPs/accuracy) and on wall-clock vs
VGG_FC (5.6×). The open question: **what does the routing mechanism actually buy us?**

SGNNET baseline (step235 ΔW proj): N=2048, D=16, K_hh=2, K_iter=5, n_groups=256 (= N//8).
Per-neuron ΔW projection selects neighbors; W_pos encodes position on S^{D-1}.
~34,976 params, 0.98M routing MACs, 97.30% accuracy.

---

## Part 1: MoE Literature Survey

### 1.1 Switch Transformer / GShard / Mixtral / DeepSeek-MoE

**Switch Transformer** (Fedus 2021): Top-1 routing, aux loss for balance. Even top-1 beats dense at matched params via specialization. Collapse risk at few (2-4) experts.

**GShard** (Lepikhin 2021): Hard-coded k=2 for stability. Expert parallelism > depth scaling at same compute.

**Mixtral 8×7B** (Mistral 2023): k=2 of 8 experts. k=2 beats k=1 substantially; k=3 marginal; sweet spot k/E ≈ 0.25.

**DeepSeek-MoE** (2024): Many small experts > few large at matched total params (64 fine vs 8 coarse, +1-2pp). Mechanism: smaller experts reduce redundancy. **Prediction for SGNNET:** Fine-grained should win — but DeepSeek experts are dense FFNs, not single weight vectors. Analogy may not hold.

### 1.2 Soft-MoE (Puigcerver 2023, arXiv:2308.00951)

Fully differentiable — each expert receives a weighted combination of all tokens. No token dropping, no routing collapse, no auxiliary loss. 128 experts: matches or beats hard routing.

**Key for SGNNET:** Group-level ΔW (step612) is structurally Soft-MoE where each group shares one routing vector. Not an approximation — a hypothesis that the routing signal is group-level, not neuron-level.

### 1.3 Expert Choice vs Token Choice

**Token Choice** (standard): each token picks top-k experts. Expert load unbalanced by default.
**Expert Choice** (Zhou et al., 2022): each expert picks top-k tokens. Perfect load balance.
Throughput win of 2× over token choice at matched accuracy.

SGNNET analog: "Expert Choice" = each group selects which neurons it services, rather than each
neuron selecting its group. This is the inversion step612 implicitly tests — shared group ΔW is
experts broadcasting to their neurons, not neurons querying their expert.

### 1.4 Granularity Trade-offs

DeepSeek-MoE's result: given a fixed param budget, many small experts > few large experts.
The gain diminishes beyond ~64 experts. Key mechanism: smaller experts reduce "expert redundancy"
(two experts learning the same sub-function).

SGNNET at n_groups=256 is already at fine granularity — each "expert group" is 8 neurons.
The question is whether 8-neuron groups are too fine to maintain a coherent routing signal, or
fine enough to specialize.

**Literature prediction:** If per-neuron routing ≈ MLP at matched FLOPs, this is consistent with
one of two explanations:
- (a) The routing signal is too noisy at single-neuron granularity (too fine)
- (b) The routing signal collapses to a near-MLP solution regardless of granularity

These have different implications: (a) predicts group routing would WIN; (b) predicts group routing
also collapses. Step612 directly tests which is true.

---

## Part 2: Experiment Designs

### Baseline reference for all three experiments
- **Ref A:** step235 ΔW proj, N=2048, D=16, K_hh=2, K_iter=5. 34,976 params. 0.98M routing MACs. 97.30%.
- **Ref B:** MLP_37, h=37, N_in=25088. 929K params. 1.86M FLOPs. 97.71%.
- All experiments must run against both references.

---

### step611 — Hierarchical Routing (MoE×SGNNET Hybrid)

**Architecture:**
Two-level routing over the existing n_groups=256 topology.

Level 1 (group gate): A learned gate W_gate ∈ R^{D × g_active} computes per-sample group scores.
```
s = softmax(Z_mean @ W_gate)    # Z_mean = mean pooled node repr [B, D]
top-k groups selected per sample (k_g = 32, so 12.5% of groups active)
```
Level 2 (within active groups): For the k_g=32 active groups (64 neurons each = 2048 neurons),
run ΔW projection routing as in step235 for K_iter=5 steps. Inactive groups: zero contribution
(neurons still maintain state but receive no messages from outside their group).

The group gate uses Z_mean (mean-pooled over all neurons), not a per-neuron query. This is closer
to sample-level routing than Soft-MoE's per-token routing — one gate decision per sample, not
per neuron.

**Parameter count vs baseline:**
- W_gate: D × g = 16 × 256 = 4,096 new params
- All other SGNNET params: 34,976
- Total: ~39,072 params (11.7% increase)
- Effective routing FLOPs: only 32/256 = 12.5% of groups active → routing MACs for active neurons
  = 0.98M × (32/256 × 256/256) = 0.98M (same — all N=2048 neurons still in active groups since
  k_g=32 of 256 groups × 8 neurons = 2048 neurons = ALL neurons. Need k_g<32 to actually prune.)

CORRECTION: to get actual sparsity, k_g=8 selects 512 neurons, reducing routing cost by 4×.
Config: k_g ∈ {8, 32} to probe routing-vs-accuracy tradeoff.

**FLOPs per sample:**
- Gate forward: D × g = 16 × 256 = 4,096 MACs (negligible)
- k_g=8 config: 0.98M × (8/256) = 0.031M routing MACs + 4K gate MACs ≈ **0.035M total**
  (35× fewer routing MACs than baseline, IF accuracy holds)
- k_g=32 config: 0.98M × (32/256) = 0.122M routing MACs ≈ **0.126M total**

**Ablation controls required:**
- k_g=256 (all groups active, gate present but no pruning) = step235 + gate overhead
- k_g=8, no gate (random group selection) = null control for routing benefit
- k_g=8, gate = primary experiment

**Acceptance criteria:**
- STRONG: k_g=8 gate ≥ 95.0% (within 2.3pp of step235, at 35× fewer routing MACs)
- MEDIUM: k_g=8 gate ≥ 93.0% (4pp below baseline, ~10% FLOPs)
- WEAK: k_g=32 gate matches step235 (routing unchanged, gate doesn't hurt)
- ABANDON: k_g=256 gate (ablation control) hurts more than 1pp vs Ref A

**Tier:** T0 (20ep, 50% data) → if MEDIUM+, T1 (75ep, 50% data)

---

### step612 — Group-Level ΔW (Granularity Probe)

**Architecture:**
Replace per-neuron ΔW projection (N=2048 direction vectors, one per neuron) with per-group ΔW
(g=256 direction vectors, one per group of 8 neurons).

In step235: each neuron i has its own ΔW_i ∈ R^D. Routing coefficient for edge (i,j):
```
c_ij = abs(ΔW_i · Z_j)    # per-neuron dot product
```
In step612: all 8 neurons in group g share ΔW_g ∈ R^D:
```
c_ij = abs(ΔW_g[i] · Z_j)    # g[i] = group index of neuron i
```
Everything else identical: same K_iter=5, same K_hh=2, same W_pos, same architecture.

**Parameter count vs baseline:**
- step235 ΔW matrix: N × D = 2048 × 16 = 32,768 params
- step612 ΔW matrix: g × D = 256 × 16 = 4,096 params
- Reduction: 28,672 params (87.5% of ΔW params freed)
- Total step612 params: 34,976 − 28,672 = ~6,304 params (**6.8× fewer total params**)

**FLOPs per sample:**
Unchanged — same routing ops, same number of message passes. The ΔW dot product is one of the
cheapest ops (N×D MACs per step = 32K per step = 163K total over K_iter=5).
Total routing MACs: ~0.98M (identical to step235).

**What this experiment answers:**
If step612 ≥ step235 − 1pp: routing signal lives at group granularity. Per-neuron ΔW is parameter
waste. 64× ΔW param reduction with no accuracy cost. Paper claim: "group-level routing suffices."

If step612 < step235 − 3pp: per-neuron routing is doing real work that groups cannot replicate.
Fine-grained routing is justified. Paper claim: "neuron-level specialization, not group-level."

If step612 is between (−1pp to −3pp): ambiguous — test at T1, check if accuracy recovers.

**Acceptance criteria:**
- STRONG: step612 ≥ step235 − 0.5pp (within 0.5pp at 6.8× fewer params = huge param efficiency win)
- MEDIUM: step612 ≥ step235 − 2pp (routing is somewhat granularity-dependent)
- WEAK: step612 ≥ step235 − 3pp (marginal loss, worth T1)
- ABANDON: step612 < step235 − 3pp at T0 (granularity matters significantly)

**Tier:** T0 (20ep, 50%) → any result not clearly below ABANDON → T1 (75ep, 50%)
Note: T0 here is NOT a rejection filter — it's a granularity probe. Even a −3pp T0 result should
be T1'd if the variance is high, because this is a mechanism question, not a config sweep.

---

### step613 — Coarse-to-Fine Routing (Iteration Curriculum)

**Architecture:**
K_iter=5 with mixed routing granularity across iterations:
- Iterations 1-2 (coarse): group-level ΔW routing (same as step612, shared ΔW_g per group)
- Iterations 3-5 (fine): per-neuron ΔW routing (same as step235)

Two separate ΔW parameter sets:
- ΔW_coarse ∈ R^{g × D} = 256 × 16 = 4,096 params (shared within groups)
- ΔW_fine ∈ R^{N × D} = 2048 × 16 = 32,768 params (per neuron)

Total params: base(step235) − 32,768 + 32,768 + 4,096 = 34,976 + 4,096 = **39,072 params** (same ballpark as step611).

**Motivation from MoE literature:**
Token routing in LLMs shows that early layers route to "syntax" experts, late layers to "semantic"
experts. The hypothesis here: early K_iter steps establish coarse topology (which group am I in?),
later steps refine within-group signal propagation (which neuron within my neighbors is relevant?).

This tests the "routing curriculum by iteration" hypothesis — separate from the "routing curriculum
by layer depth" finding from transformer MoE.

**FLOPs per sample:**
- Coarse iterations (1-2): g-dim ΔW dot product instead of N-dim: 256×16 vs 2048×16 per step.
  Saves 87.5% on the ΔW component for those 2 steps.
- Net routing MACs: approximately same as step235 (ΔW dot product is minor fraction of total MACs).
- Effective change is negligible in absolute FLOPs (the message-passing dominates).

**What this experiment answers:**
If step613 ≈ step235: coarse early routing is functionally equivalent, meaning the fine routing in
steps 3-5 already erases any coarse initialization from steps 1-2.
If step613 > step235: routing benefits from curriculum (coarse scaffold → fine refinement).
If step613 < step612 and < step235: the mixed granularity is worse than either pure approach —
routing granularity consistency matters.

**Acceptance criteria:**
- STRONG: step613 ≥ step235 + 0.5pp (coarse-to-fine curriculum wins — routing has multi-scale structure)
- MEDIUM: step613 ≈ step235 ± 0.5pp (neutral — routing granularity per iteration doesn't matter)
- WEAK: step613 ≥ step612 (coarse-to-fine better than pure coarse; fine refinement helps)
- ABANDON: step613 < min(step612, step235) − 1pp (mixed granularity actively interferes)

**Tier:** T0 (20ep, 50%) → MEDIUM+ → T1. ABANDON → stop.

---

## Part 3: Decisive Experiment Recommendation

### The core falsification question

> "If per-neuron routing (SGNNET) barely beats MLP_37 at matched FLOPs, does group-level routing
> (MoE-style) give up nothing? If so, the 'neuron-level MoE' framing is falsified."

**step612 is the decisive experiment.** Here is why:

**step612 isolates the routing granularity variable with zero architectural confounds.** It changes
exactly one thing: ΔW from per-neuron to per-group. Same topology, same W_pos, same K_iter, same
K_hh, same everything. The only question it asks is: does the routing signal carry neuron-level
information, or group-level information?

The answer directly determines the paper's mechanism claim:
- step612 STRONG → the framing "neuron-level MoE" is **CONFIRMED wasteful**. Paper pivots to:
  "group-level routing suffices, and SGNNET at g=256 achieves 6.8× ΔW param reduction with no
  accuracy loss." This is actually a stronger paper claim — not "we need fine routing" but "we
  discovered that group-level routing is sufficient, which prior MoE literature (coarse experts)
  failed to show at this granularity."
- step612 ABANDON → neuron-level routing is doing real work. The "per-neuron MoE" framing is
  **CONFIRMED** necessary. Paper claim: "routing signal is neuron-specific, not group-specific —
  this is the key property missing from standard MoE."

step611 (hierarchical) is secondary because it adds architectural complexity on top of the
unanswered granularity question. If we don't know whether group routing suffices, building a
two-level hierarchy is premature.

step613 (coarse-to-fine) is tertiary — it only makes sense to run after we know step612's answer.
If step612 STRONG, step613 tests whether a curriculum of coarse→fine outperforms pure coarse.
If step612 ABANDON, step613 is already partially answered (fine routing dominates throughout).

**Recommended order:** step612 → step611 → step613.
Run step612 immediately (T0, ~2h on one slot). Decision tree branches on result.

### Decision tree

```
step612 T0 result
  ├─ STRONG (≥ −0.5pp): group routing suffices
  │   ├─ T1 step612 to confirm
  │   ├─ step611 (k_g=8 sparse groups, extreme FLOPs reduction)
  │   └─ step613 (coarse-to-fine curriculum adds anything?)
  │
  ├─ MEDIUM (−0.5 to −2pp): partial granularity dependence
  │   ├─ T1 step612 to measure at full training budget
  │   └─ step611 with k_g=32 (larger active fraction to compensate)
  │
  └─ ABANDON (<−3pp): per-neuron routing is real
      ├─ step611 (hierarchical can still reduce FLOPs IF accuracy holds at k_g=32)
      └─ step613 is moot (fine routing wins in all iterations)
```

### Paper pivot depending on outcome

| step612 result | Paper framing | Claim status |
|---|---|---|
| STRONG | "group routing suffices; 6.8× ΔW savings" | Novel: small-group MoE at 8-neuron granularity |
| MEDIUM | "neuron routing marginally better; tradeoff exists" | Weak-moderate |
| ABANDON | "neuron-level specialization confirmed; MoE too coarse" | "Neuron-level MoE" framing CONFIRMED |

Either outcome is paper-worthy. STRONG is the more surprising result (prior MoE never tested at
8-neuron granularity and found group routing sufficient). ABANDON validates the original framing.
The worst outcome is MEDIUM — that requires a params-vs-accuracy framing, which is available but
less crisp.

---

## Appendix: Prior routing failures (context)

- step83: group routing via softmax gate on group centroids → gate-death, 3 failure modes.
- step511-514, step523, step525: all dynamic topology variants failed due to W_pos co-adaptation.
- ReLU routing (RESEARCH_routing_mechanisms.md §6) proposed as alternative to softmax — never tested.
  step612 avoids ALL routing collapse risks because it keeps existing ΔW mechanism, just shares
  the vector within groups. No new gate, no softmax, no auxiliary loss. Zero new failure modes.

step612 is structurally safer than any prior routing experiment precisely because it is NOT a new
routing mechanism — it is a parameter-sharing ablation of the existing one.
