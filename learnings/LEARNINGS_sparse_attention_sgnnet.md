# Research: Sparse Attention + SGNNET Implications — Part B: Binding & Design Principles

**Date:** 2026-03-30  
**Source:** Literature review — see Part A for mechanism details

---

## 4. Binding by Synchrony / Oscillatory Networks

- **Core neuroscience claim (Wolf Singer, 1990s):** Neurons representing the same object
  synchronize their gamma-band oscillations (~40-90 Hz). Temporal correlation pattern, not
  just firing rate, encodes binding. Objects distinguished by *which* neurons fire in synchrony.

- **2024 Nature Human Behaviour:** Cortico-cortical co-ripples (~90 Hz, ~100ms) increase
  during reading and semantic decisions. Synchrony is functionally specific, not a global state.

- **Computational form in SGNNET:** The signed coupling operation
  `Z_h += alpha * sum_j cos(Z_h, Z_j) * Z_j` is exactly a continuous-time binding-by-synchrony:
  - cos(Z_h, Z_j) > 0 → neurons aligned (same "phase") → excitatory coupling
  - cos(Z_h, Z_j) < 0 → anti-aligned → inhibitory
  - Network dynamically forms synchronized clusters per input — binding

- **Hopfield connection:** Signed coupling resembles a Hopfield network update rule (stored
  patterns as attractors). Step17 Hopfield variant: 30.52% (vs 29.22% ref), consistent.

- **All-pairs coupling essential:** Step18 all-pairs signed coupling = +10.93pp over baseline.
  Sparse-K variant crashed (scatter gradient bug). Differentiable all-pairs coupling is essential.
  **NOTE: This mechanism is dead at D=64 (step18/23/28/42/44 all ≤32%) — see Part A ARM status.**

---

## 5. SGNNET Implications: 7 Design Principles

### P1. FFN analog is implemented — routing depth (K_iter) is the compute

Standard FFN: `FFN(x) = max(0, xW1 + b1)W2`. SGNNET analog:
- Seed (x → Z_input) = first projection
- Routing iterations with theta gating = "ReLU-equivalent sparsification"
- Readout (Z → logits via W_out) = second projection

Critical difference: SGNNET's activation function is a *multi-step iterative process* (K_iter=8),
not a single nonlinearity. This is closer to MoD's layer-stacking or Hopfield's attractor dynamics.
**Do not compress routing into a single matrix multiply — K_iter IS the compute.**

### P2. Routing sparsity K is analogous to MoE top-K, but neighbors interact

In MoE, K experts are selected independently — they don't interact. In SGNNET, K_hh neighbors
form a graph; routed signals propagate through that graph across K_iter steps.
Effective receptive field grows exponentially with K_iter (bounded by graph diameter ~9 hops).

BigBird shows that random + local + global suffices for universal approximation of full attention.
SGNNET's K_local + K_random structure is already the right pattern. Minimum useful K_hh is set
by mixing time (step13: K_iter=8 optimal, implying mixing is the bottleneck, not capacity).

### P3. D is the routing space dimension, not the output dimension

FlashAttention uses head_dim=64. SGNNET's D is the routing space dimension: neuron directions
live on S^(D-1). The step8 breakthrough (+6.55pp from D=4→D=16) maps to: at D=4, routing
noise dominates (neurons too crowded on S^3); at D=16, routing signal is clean.

**D should scale as log(N) or slightly faster. For N=1024, D=64 is the confirmed ceiling —
above D=64, Fourier encoding produces near-orthogonal seeds that routing cannot align.**

### P4. Signed coupling is the strongest single mechanism — but dead at D=64

Step18 result: all-pairs signed coupling = +10.93pp (D=16 N=512 K_iter=3).
At D=16, E[cos²(Z_i, Z_j)] ≈ 1/16 — meaningful signal.
At D=64, E[cos²(Z_i, Z_j)] ≈ 1/64 — pure noise. **Do not use at D=64.**

AntiHebb (step16/step29) provides the competitive inhibition that signed coupling can't,
without requiring directional alignment: **AntiHebb = +13.86pp at D=64** vs signed = hurts.

### P5. Shared experts (DeepSeekMoE) map to SGNNET's interneurons

DeepSeekMoE: K_s "shared" experts always activate; routing selects from remaining.
SGNNET step20: 50% interneurons + readout=all = +2.83pp at D=16.
Interneurons that never receive direct input are "shared experts" — they integrate signals
from all seeded neurons, capturing cross-feature combinations individual neurons can't.

**Optimal architecture may be: K_s shared interneurons always participating in routing
(always-active attractor cores), plus N-K_s conditionally active neurons.**

### P6. MoD layer-skipping FAILS for SGNNET

MoD routes tokens past layers; they carry residual unchanged. Applied to SGNNET routing
steps (step34): **catastrophic failure (~20%).** All K_iter=8 steps are necessary — there
is no early-exit epoch where routing has "converged." Over-smoothing at K_iter=12 (D=16)
and K_iter>8 (D=64, step48) is from message averaging, not routing depth per se.

### P7. Matformer nested training for N sweep

Train at N=2048; per step sample N_active ∈ {512, 1024, 2048}. One run answers all N
questions, saving ~3x compute. Before the next N sweep, implement nested N training.
The subset graph (lower-index neurons) is already a valid smaller model — formalize this.

---

## Sources

- [Binding of cortical modules by synchronous oscillations: Nature Human Behaviour 2024](https://www.nature.com/articles/s41562-024-01952-2)
- [Sparse Attention in LLMs survey 2024](https://www.clausiuspress.com/assets/default/article/2024/11/12/article_1731408067.pdf)
- [Hopfield networks: Hopfield 1982](https://www.pnas.org/doi/10.1073/pnas.79.8.2554)
