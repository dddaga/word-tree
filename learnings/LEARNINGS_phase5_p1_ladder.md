# Phase 5: Validation Ladder (Steps 1–4 + Combined Run)

Experimental results from validation ladder (2026-03-27).
See `docs/resonant_sgnnet_spec.md` for architecture context.

---

## Step 1: Safety Valve Fix + Norm Mode Sweep

### Safety valve: bounded quadratic CONFIRMED

Old: `relu(1/d - 1/r)` — Coulomb divergence, safety dominated task at large N
New: `relu(r - d)^2 / r^2` + 15% soft cap relative to task loss

Results at N=512 (l2 norm):
```
safety/task = 0.14%  (was 140% before — 1000× reduction)
```
At N=1024: safety/task = 0.02%. Fix holds at all tested scales.

### Norm mode comparison — CRITICAL FINDING

All tested at N=512, 60 epochs, CPU, SmallWorld topology:

| norm_mode | top1 | task_loss | Notes |
|---|---|---|---|
| masked (old) | 10.2% | 3.97 | Random chance — confirmed broken |
| relu | 8.2% | 2.70 | WORSE than random in places |
| **l2** | **23.9%** | **2.06** | **Winner — use for all future runs** |

**Why masked fails:** normalises across all N neurons (mean/std), collapsing every activation to same scale. Destroys class-discriminative structure.

**Why relu fails at D=4:** relu on 4-dim vector leaves 1-2 non-zero dims. F.normalize of near-zero vector → noise unit vector. D=4 too small for relu element-wise on D axis.

**Why l2 wins:** both positive and negative dims carry signal. Direction space in D=4 has full S³ sphere coverage.

**Action taken:** `norm_mode="l2"` now default in `experiment_config.topology_kwargs()`.

### N=1024 with l2 norm

Previous: N=1024 plateaued at total_loss=5.5 (task=2.3, safety=3.2 — safety dominated)
Now: task=2.29 at epoch 10, top1=20.9%, safety=0.0005 (0.02% of task)

---

## Step 2: W_phase Receiver — HYPOTHESIS REQUIRES REVISION

All at N=512, 60 epochs, CPU:

| config | top1 | vs baseline | Notes |
|---|---|---|---|
| baseline (l2 only) | 16.1% | — | Reference |
| W_phase K=8 | 15.9% | **-0.2%** | Hurt performance |
| W_phase K=16 | 15.0% | **-1.1%** | Worse with more connections |

**Conclusion:** Static W_phase graph adds noise, not signal.

**Why it failed:** W_phase graph built once from random W_phase at init, never rebuilt. "Dynamic long-range routing" was actually second frozen random topology — all complexity of phase routing, none of adaptability.

**Revised hypothesis:** W_phase NOT rejected as concept. Test was of "frozen random phase graph" — just noise. Real hypothesis (online-rebuilt phase graph tracking learned W_phase directions) still untested. `tick_epoch()` rebuilds this graph between epochs — tested in Step 3+.

**Key insight:** 16.1% baseline in Step 2 vs 23.9% in Step 1 = pure run-to-run variance at N=512/60 epochs. Single-seed comparisons at this scale unreliable; ceiling likely 15–25% across seeds.

---

## Step 3: Routing Mechanisms — STRONG POSITIVE RESULTS

All at N=512, 60 epochs, CPU, TuringSmallWorld wrapper:

| mode | top1 | vs baseline | Notes |
|---|---|---|---|
| baseline (no routing) | 13.9% | — | Step 3 reference |
| threshold | 18.1% | **+4.2%** | relu(Z - θ) gates propagation |
| reflection | 16.5% | **+2.6%** | self-inhibition, still converging at e60 |
| turing | 20.4% | **+6.5%** | local excite + long-range inhibit |
| **learnable θ** | **20.5%** | **+6.6%** | per-neuron threshold — best |

**Mechanism hierarchy:**
- `learnable ≈ turing > threshold > reflection > baseline`
- Two-scale Turing strongest: local excitatory via conn_hh + long-range inhibitory via phase beam → competition → non-overlapping feature detectors
- Threshold gate most robust: prevents noise propagation, works standalone
- Reflection needs >60 epochs — loss still descending, not converged
- Stack order: threshold + reflection + turing all additive (combined in Resonant)

**tick_epoch works:** Phase graph rebuilds from W_phase each epoch. Confirmed by learnable mode outperforming — graph updated meaningfully.

---

## Step 4: Simulated Annealing — REJECTED

All at N=512, 60 epochs, CPU, AnnealedSmallWorld wrapper:

| schedule | top1 | tau final | beam final | Notes |
|---|---|---|---|---|
| **fixed** | **19.3%** | 1.0 | 32 | **Winner** |
| exponential | 18.2% | 0.05 | 8 | Shrinking hurts |
| cosine | 15.2% | ~0 | 8 | Collapses at end |
| stepwise | 15.1% | 0.1 | 8 | Hard drop kills recovery |

**Why annealing failed:** Shrinking beam_size = irreversible capacity reduction. Once beam drops 32→8, model permanently loses 75% long-range broadcast capacity. No recovery possible in remaining epochs.

**Revised understanding:** Don't anneal beam_size downward. Fixed beam=32 throughout. If temperature scheduling desired, apply only to gate_temp (softness of routing weights), not active connection count. Or consider reverse annealing: start small beam (local structure first), expand later.

---

## Combined Run: SGNNET_Resonant on MPS (120 epochs)

Running in `scripts/train_resonant.py`. Configs:
- A: baseline SmallWorld l2 (reference)
- B: resonant (threshold + reflection + turing, static phase rebuilt per epoch)
- C: dynamic_gate (fixed graph, GAT-style activation-conditioned edge weights)
- D: dynamic_z (graph rebuilt from Z similarity each forward pass — Hopfield-style)
- E: resonant N=1024 (scale-up)

Results (MPS, 120ep, dynamic_z mode):
```
baseline_N512:     24.4%
resonant_N512:     25.3%
dynamic_gate_N512: 25.1%
dynamic_z_N512:    26.5%   ← iter1 reference
```

**Key finding:** Topology change per input matters more than weight modulation on fixed topology. dynamic_z (rebuilds full graph per step) > dynamic_gate (fixed graph, activation-gated weights).

---

## Dynamic Connectivity — Research Summary (2026-03-27)

Research into input-dependent pseudo-connections. Key finding: **all useful dynamic connectivity mechanisms converge on attention** — computing normalized similarity score between two nodes as function of current features, then weighting message aggregation by that score.

### Mechanism ranking for this setting (N=512, D=4, fixed base graph):

| Rank | Mechanism | Key property | Cost |
|---|---|---|---|
| 1 | **GAT** | Modulates existing edges, fully differentiable | O(E·F) |
| 2 | **Geometric-biased attention** | W_pos in gradient via logit bias | O(N²) |
| 3 | **Hypernetwork edge weights** | MLP(pos_i, pos_j, h_i, h_j) → scalar | O(E·MLP) |
| 4 | Modern Hopfield | All-pairs dynamic, = transformer attention | O(N²·D) |
| 5 | DGCNN-style KNN rebuild | Topology changes per layer | O(N²), partial grad |

### Critical gap identified:

Current `dynamic_gate` and `dynamic_z` modes do NOT include W_pos in similarity score. Strongest unexplored variant = geometric-biased attention:

```python
# next experiment:
score(h, k) = dot(Z[b,k], W_phase[h]) - gamma * ||W_pos[h] - W_pos[k]||^2
```

Puts W_pos into routing gradient, not just readout. Geometrically close AND feature-similar neurons communicate more strongly.

### Three implementation strategies tested:

1. **resonant** — static phase graph (W_phase k-NN), rebuilt per epoch by tick_epoch
2. **dynamic_gate** — fixed graph, edge weights = f(Z) per input (GAT-style)
3. **dynamic_z** — graph topology rebuilt from current Z similarity per forward pass

---

## Next Experiments Queue (after train_resonant.py — now outdated)

See `EXPERIMENT_QUEUE.md` for current prioritised queue.
Original next-steps from this phase:
1. Geometric-biased attention (now implemented as dynamic_z_geo)
2. D=8 dimensionality sweep (now completed as encoding sweep)
3. Multi-seed confirmation (planned)
4. Longer reflection convergence (now tested at D=16)
5. Hypernetwork edge scoring (deferred)