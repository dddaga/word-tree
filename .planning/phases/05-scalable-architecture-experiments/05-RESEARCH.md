# Phase 5: Fast Weights / Intra-Forward W_phase Adaptation - Research

**Researched:** 2026-03-30
**Domain:** Fast weight dynamics, Hebbian learning, Hopfield networks, ISTA sparse coding, in-context learning as gradient descent
**Confidence:** HIGH (core mathematical results from well-established literature, project context from direct code inspection)

---

## Summary

The core problem is that W_phase [N, D] is a per-neuron "resonant direction" that when
learned by gradient descent converges to a single average direction that does not
discriminate between inputs. The key insight from project history confirms this: learned
W_phase gave −0.69% vs static random W_phase (step 10b). The static random W_phase at
D=16 is already competitive (28.59%) because in S¹⁵ a random direction is a genuinely
distinct direction — but it cannot be input-specific.

The proposed solution — adapting W_phase WITHIN the forward pass using Hebbian/fast-weight
rules — has deep connections to five bodies of literature: Ba et al. (2016) fast weights,
modern Hopfield networks (Ramsauer et al. 2020), in-context learning as implicit gradient
descent (Akyürek et al. 2022 / von Oswald et al. 2023), ISTA / sparse coding (Olshausen
& Field 1996), and Oja's rule for online PCA.

The central finding across all five areas is that a biologically-motivated outer-product
or rank-1 Hebbian update on W_phase can create a dynamic "associative memory" that is
input-specific, bounded, and has a well-understood fixed point. The critical design tension
is: (a) does W_phase track Z (attractor follows state), or (b) does Z track W_phase
(state attracted to memory)? These two directions have opposite fixed points and opposite
explosion risks.

**Primary recommendation:** Use rule (d) — the attention-style fast update — as the
theoretically cleanest baseline, because it has the best-understood convergence properties
(modern Hopfield / softmax attention fixed point). Then test rule (b) — Oja's rule variant
— because it has a stable normalized fixed point with direct biological precedent. Avoid
rule (a) without explicit batch separation. Rule (c) is architecturally sound but
introduces a θ hyperparameter that duplicates the existing learnable theta.

---

## Project-Specific Context

From inspection of `src/sgnnet/model_resonant.py` and experiment logs:

- W_phase shape: [N, D] = [512, 16] as `nn.Parameter(torch.rand(N, D))`
- Z shape within routing loop: [B, N, D], l2-normalized after each step
- K_iter = 3 routing steps (currently)
- The routing loop already calls `F.normalize(Z_new, dim=-1)` at each step — this is load-bearing
- W_phase is currently OUTSIDE the routing loop — it is used to build a static graph via k-NN, not updated inside the loop
- The `dynamic_z` mode already builds a per-input graph from Z@Z.T inside the loop — this is the excitatory radiation analog
- Step 10b confirmed: W_phase LEARNED via gradient descent = bad (−0.69% vs static)
- Step 14 (running) is already testing "excitatory radiation" — the truly dynamic analog of what fast weights would do for excitation
- The fast-weight / intra-forward update is a distinct and unexplored axis

---

## Literature Area 1: Fast Weights (Ba et al. 2016)

**Source:** "Using Fast Weights to Attend to the Recent Past" — Ba, Hinton, Mnih, Leibo, Ionescu (NIPS 2016). Confidence: HIGH (widely reproduced, well-understood).

### Core mechanism

Ba et al. maintain two weight matrices:
- **Slow weights** W_s: standard parameters learned across examples by gradient descent
- **Fast weights** A: a matrix reset to zero at the start of each sequence, updated within the sequence by a Hebbian outer product rule

The fast weight update at each step t within a sequence:
```
A(t) ← λ * A(t-1) + η * h(t) ⊗ h(t)
```
where h(t) is the hidden state at step t, λ is a decay constant (λ < 1), η is a learning rate.

The retrieval step applies the fast weights to the current hidden state:
```
h_retrieved = LayerNorm(h + softmax(A @ h))
```
or in the original formulation, inner loop iterations of:
```
h ← f(W_s * x + A * h_prev)
```

### Key properties

1. **Reset per sequence:** A is zeroed at each new input. This is what makes it input-specific: A encodes associations from the CURRENT input's processing history, not a global average.

2. **Explosion prevention:** The λ decay constant ensures A does not grow unboundedly. Without decay, A accumulates outer products indefinitely. With λ < 1, old associations are discounted exponentially. In practice λ=0.95 and η=0.5 are typical.

3. **Slow/fast interaction:** The slow weights W_s capture statistical regularities across the training distribution. The fast weights A capture within-sequence associations that deviate from that average.

4. **Outer product structure:** A = Σ_t η·λ^(T-t) * h(t) ⊗ h(t) is a weighted sum of rank-1 matrices — a positive semidefinite matrix by construction.

### Mapping to W_phase

W_phase is NOT currently in the routing loop at all (in dynamic_z mode). The fast-weight
analog would be: W_phase becomes A, reset at each forward pass, updated inside each of
K_iter=3 routing steps. The "sequence" is the K_iter steps, not a token sequence.

Critical difference from Ba et al.: Ba's sequences have 10-100 steps. K_iter=3 is very
short — at most 3 rank-1 updates. This limits how much structure A can accumulate, but
also limits explosion risk.

---

## Literature Area 2: Modern Hopfield Networks (Ramsauer et al. 2020)

**Source:** "Hopfield Networks is All You Need" — Ramsauer, Schäfl, Lehner, Seidl, Widrich, Adler, Gruber, Holzleitner, Pavlovic, Sandve, Greiff, Kreil, Kopp, Klambauer, Brandstetter, Hochreiter (ICLR 2021). Confidence: HIGH.

### Classical Hopfield energy function

Classical (binary) Hopfield:
```
E = -1/2 * Z^T W Z
```
where W = Σ_μ ξ_μ ⊗ ξ_μ (sum of stored patterns ξ_μ as outer products).
The update rule dE/dZ ← 0 drives Z toward stored patterns (local minima of E).

### Modern Hopfield energy function

Ramsauer et al. extend this to continuous-valued states and exponential interactions:
```
E = -lse(β, Z^T W_patterns) + 1/2 * Z^T Z + ...
```
where lse is the log-sum-exp function. The update rule becomes:
```
Z_new = W_patterns * softmax(β * W_patterns^T * Z)
```

This is exactly softmax attention! The "patterns" W_patterns are the attention keys/values,
and Z is the query. The fixed point of this update is:
```
Z* ← W_patterns * softmax(β * W_patterns^T * Z*)
```

### Storage capacity

Modern Hopfield has exponential storage capacity in D (can store ~exp(D/2) patterns vs
classical Hopfield's linear capacity of 0.14*N). At D=16, capacity ≈ exp(8) ≈ 3000 patterns
— far more than N=512 neurons need.

### Convergence

With sufficient β (inverse temperature), retrieval converges in one step with high probability
(Theorem 3, Ramsauer et al.). For lower β, multiple update steps bring Z closer to the stored
pattern. K_iter=3 routing steps are sufficient for reliable retrieval at moderate β.

### Critical insight for W_phase

If W_phase plays the role of "stored patterns" in a Hopfield network, then the update:
```
Z ← normalize(W_phase * softmax(β * W_phase^T * Z))
```
drives Z toward whichever column of W_phase is most aligned with the current Z. This is
rule (c) from the question (with sigmoid replaced by softmax, and W_phase playing the key role).

The fixed point: Z* is the column of W_phase with highest overlap with the initial Z (winner
if β → ∞, weighted average if β is small). This is a RETRIEVAL operation — Z converges to
a stored direction in W_phase, not to the input.

**Explosion risk:** With normalize at each step, no explosion. The issue is identity collapse:
all Z converge to the same most-popular stored direction. Solution: differentiate patterns
by input via the fast-weight update of W_phase itself.

---

## Literature Area 3: In-Context Learning as Implicit Gradient Descent

**Source:** "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes" (Akyürek et al. 2022); "Transformers Learn In-Context by Gradient Descent" (von Oswald et al. 2023). Confidence: HIGH for the mathematical core; MEDIUM for direct applicability to routing.

### Core result

A single attention layer implements one step of gradient descent on a linear regression:
- Input: context pairs (x_1, y_1), ..., (x_k, y_k) as token sequence
- The self-attention key/query/value matrices W_K, W_Q, W_V encode the gradient update
- The residual stream after attention = W - α * ∇L(W, context), where W is the "implicit" regressor

The gradient descent update for linear regression with data (X, y):
```
W ← W - α * X^T (X W - y) = W + α * X^T residuals
```
matches the structure of an outer-product update when written as:
```
W ← W + α * Σ_i (input_i ⊗ target_i)
```

### Mapping to W_phase outer-product update

The proposal W_phase[h] += α * Z[h] ⊗ Z[h] is a special case where input = target = Z[h]
(Hebbian, no supervision signal). This is the UNSUPERVISED analog: W_phase accumulates
the "auto-correlation" matrix of Z.

The gradient-descent interpretation: W_phase is implicitly minimizing:
```
L = Σ_h ||W_phase[h] * 1 - Z[h]||²  (predict Z from W_phase)
```
or equivalently maximizing the projection:
```
L = Σ_h dot(W_phase[h], Z[h])²
```
The fixed point: W_phase[h] aligns with the principal eigenvector of the auto-correlation
matrix of Z[h] across routing steps. This is Oja's rule.

### Key design implication

The outer product update W_phase ← W_phase + α * Z * Z^T * W_phase is NOT batch-separating
unless W_phase starts different per batch item. If W_phase is shared across the batch,
the update averages over the batch dimension and loses input-specificity.

**Solution:** W_phase must be a batch-local temporary: initialize as the slow-learned
W_phase (from nn.Parameter), then update in-place within the forward pass as a local
variable `A = W_phase.clone()` that is discarded after the forward pass. The nn.Parameter
W_phase is the "slow weight" — the accumulated prior. The local A is the "fast weight" —
the input-specific adaptation.

---

## Literature Area 4: ISTA / Sparse Coding (Olshausen & Field 1996)

**Source:** "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images" (Olshausen & Field, Nature 1996); "LISTA: Learning to solve the Lasso" (Gregor & LeCun, ICML 2010). Confidence: HIGH.

### ISTA formulation

Sparse coding finds a representation s for input x given dictionary D:
```
minimize_s  1/2 ||x - D s||² + λ ||s||_1
```

ISTA (Iterative Shrinkage-Thresholding Algorithm) solves this iteratively:
```
s^{k+1} = shrink(s^k + η * D^T (x - D s^k), η·λ)
```
where shrink is soft-thresholding: sign(z) * relu(|z| - threshold).

### Connection to SGNNET routing

SGNNET's K_iter routing steps can be interpreted as ISTA iterations IF:
- W_phase = D (the dictionary)
- Z = s (the sparse representation / code)
- The routing update = one ISTA step

The ISTA interpretation predicts:
- K_iter convergence requires K_iter ≥ 1/η||D||² steps
- The fixed-point Z* is the minimizer of the sparse coding objective
- Higher K_iter → sparser, more accurate representation

**Convergence condition:** For ISTA to converge, η < 2/||D^T D||₂ (Lipschitz condition on the gradient).
For a normalized dictionary (||D||_F bounded), this is satisfied with small enough η.

### LISTA (learned ISTA)

Gregor & LeCun (2010) showed that unrolling ISTA for K steps and learning the weights
gives a learned sparse encoder. In this interpretation, W_phase IS the dictionary, and
the K_iter routing steps ARE the unrolled ISTA iterations. This provides a theoretical
grounding for WHY K_iter=3 might be sufficient: LISTA converges in ~3-5 steps for many
practical dictionaries.

### Implication for W_phase adaptation

If W_phase is the dictionary and we want it to be input-specific, the natural ISTA
extension is: update the dictionary online within the forward pass to better reconstruct
the current Z. The gradient of the reconstruction loss ||Z - D s||² w.r.t. D is:
```
∇_D L = -( Z - D s ) * s^T
```
A stochastic gradient step inside the routing loop:
```
W_phase ← W_phase + η * (Z - W_phase * Z) * Z^T
```
This is identical to a Hebbian update and connects to Oja's rule.

**Shrinkage as θ-gating:** The existing learnable θ (per-neuron threshold) already
implements shrinkage! `F.relu(Z - theta_pos)` is soft-thresholding without sign flip.
The full ISTA connection is closer than it appears — the architecture already implements
approximate ISTA with a learned threshold.

---

## Literature Area 5: Oja's Rule and Normalized Hebbian Learning

**Source:** Oja (1982) "A simplified neuron model as a principal component analyser", J. Mathematical Biology. The fixed-point analysis is classical; confidence: HIGH.

### Oja's rule

Standard Oja's rule for a single weight vector w:
```
w ← w + η * y * (x - y * w)
```
where y = w^T x (the neuron output). Expanding:
```
w ← w + η * (y * x - y² * w)
     = w + η * ((w^T x) * x - (w^T x)² * w)
     = w + η * w^T x * (x - (w^T x) * w)
```

### Fixed point of Oja's rule

At equilibrium (E[Δw] = 0):
```
E[x * x^T] * w = E[(w^T x)²] * w
```
This is the eigenvalue equation for the covariance matrix E[x * x^T]. The fixed point
is the **principal eigenvector** of the input covariance matrix — Oja's rule converges
to PCA.

### Interpretation for W_phase[h]

If we treat Z[h] across routing steps as the "input" stream for neuron h, Oja's rule
drives W_phase[h] toward the direction that explains the most variance of Z[h] — the
direction Z[h] visits most often during routing. This is a natural "resonant direction":
the attractor for Z's routing trajectory.

**Continuous-time form:** dw/dt = η(y x - y² w). For our discrete K_iter=3 steps,
this is approximated by:
```
W_phase[h] ← W_phase[h] + η * (Z[h] * dot(W_phase[h], Z[h])
                                 - dot(W_phase[h], Z[h])² * W_phase[h])
```
After l2-normalizing W_phase, the decay term (y² * w) is handled implicitly by the
normalization step.

### Simplified normalized variant

A commonly used stable variant (avoiding y² term by normalizing after the update):
```
W_phase[h] ← normalize(W_phase[h] + η * Z[h] * dot(W_phase[h], Z[h]))
```
This is rule (b) from the question. The fixed point is the principal eigenvector of
the auto-covariance of Z[h].

### Explosion prevention

Oja's rule is inherently stable: the y² * w term acts as a decay that prevents ||w||
from growing. The normalized variant achieves the same via explicit normalization.
Unlike raw Hebbian w ← w + η * x ⊗ x (which grows without bound), Oja's rule has a
provably stable fixed point.

---

## Analysis of the Four Candidate Rules

### Shared prerequisites

For ALL four rules: W_phase must be handled as a batch-local variable inside the routing
loop, NOT updated in-place on the `nn.Parameter`. The slow-weight W_phase (parameter)
serves as a prior / initialization that is refined within each forward pass. Call it:

```
A = W_phase_slow.clone()  # shape [N, D] — fast-weight copy, discarded after forward
```

Then A is updated inside the K_iter loop. The `nn.Parameter` self.W_phase gets gradient
signal from the routing loss via A's initialization.

**Important:** If D=16 and B=128, a naive per-batch-item W_phase would require [B, N, D]
= [128, 512, 16] = 1M floats. This is acceptable in FP16 (2MB). However, rules that
operate on the batch average are simpler and still useful — they produce a within-batch
"population average attractor" rather than a per-sample one.

---

### Rule (a): Momentum tracking of Z

```
A ← normalize(β * A + (1-β) * Z_mean)
```
where Z_mean = Z.mean(dim=0) is the batch-mean direction at each routing step.

**Mathematical interpretation:**
This is an exponential moving average of Z — W_phase tracks where the AVERAGE Z is going
across routing steps. At K_iter=3 steps with β=0.9: after 3 steps, A has moved ~27% toward
the current Z trajectory.

**Fixed point:** A* = Z_mean at convergence = the average direction of Z across routing steps.
If Z is l2-normalized and the population is diverse, Z_mean → 0 (cancellation). If Z has
a bias toward certain directions, A tracks that bias.

**Expected routing behavior:** A becomes the "center of mass" of Z. It does NOT make
W_phase input-specific in the per-sample sense — it tracks the batch-mean, which is the
same as the old gradient-descent failure mode (converging to an average).

**Key risk:** If Z is l2-normalized and diverse, Z.mean(dim=0) ≈ 0 → normalize of near-zero
→ numerical instability. Safer: A ← normalize(β * A + (1-β) * Z.detach()), using the full
[B, N, D] tensor (then A is [B, N, D] — per-sample tracking).

**Per-sample version:**
```
A[b] ← normalize(β * A[b] + (1-β) * Z[b])  # shape [B, N, D]
```
Fixed point: A[b] ← Z[b] → trivially input-specific but collapses to the last Z value, not
a stable direction. This is a low-pass filter of Z's trajectory.

**Verdict:** Weak — does not create a stable attractor, just a lag. Input-specificity only in
per-sample form. No theoretical connection to an energy function minimum.

---

### Rule (b): Oja's rule variant

```
A ← normalize(A + α * Z * dot(A, Z))
```
where dot(A, Z) = einsum('nd,bnd->bn', A, Z) gives a [B, N] scalar similarity per neuron per batch item.

Written explicitly for the full batch:
```
sim = (A.unsqueeze(0) * Z).sum(-1)   # [B, N]
A ← normalize(A + α * (Z * sim.unsqueeze(-1)).mean(dim=0))
```
Or per-sample (if A is [B, N, D]):
```
sim = (A * Z).sum(-1, keepdim=True)  # [B, N, 1]
A ← normalize(A + α * Z * sim)       # per-sample update
```

**Mathematical interpretation:**
This is Oja's rule for the unnormalized case. The update η * y * x (where y = w^T x) pulls
w toward x scaled by how much they already agree. The normalization after prevents explosion.

**Fixed point:** A* = principal eigenvector of auto-covariance of Z across routing steps.
Equivalently, A* aligns with the direction Z visits most frequently.

**Expected routing behavior:**
- If Z[h] rotates during routing toward some stable direction, A[h] tracks and amplifies it
- A becomes a "memory" of the dominant Z trajectory for this input
- After 3 steps, A has captured the persistent routing directions

**Key risk:** Per-neuron, the update is local (only A[h] and Z[h] interact). No cross-neuron
amplification risk. Magnitude is bounded by normalization. The danger is that all A[h] collapse
to the same dominant direction — need diversity in initial A (initialization from slow weights).

**Biological plausibility:** High. Oja's rule is a biologically plausible Hebbian rule. It's
local in space (only the synapse's pre/post pair) and time (current activation only).

**Verdict:** Strong candidate. Stable fixed point, biologically motivated, local, bounded.
The per-sample variant gives genuine input-specificity but at 2× memory cost.

---

### Rule (c): W_phase as attractor (Hopfield retrieval)

```
Z ← Z + α * A * sigmoid(einsum('nd,bnd->bn', A, Z) - θ)
```
equivalently:
```
gate = sigmoid((A.unsqueeze(0) * Z).sum(-1) - θ)   # [B, N]
Z ← Z + α * A.unsqueeze(0) * gate.unsqueeze(-1)     # [B, N, D]
```
then Z is l2-normalized at the end of the routing step.

**Mathematical interpretation:**
This is the modern Hopfield retrieval step: if Z is close to a stored pattern A[h], the sigmoid
gates a positive correction pushing Z further toward A[h]. The gate is maximal when Z and A[h]
are aligned, zero when orthogonal, and slightly negative (via the −θ shift) when anti-aligned.

Unlike rules (a) and (b), here Z IS UPDATED, not A. W_phase (A) stays fixed — it is the
"memory" that attracts Z.

**Fixed point:** Z* is close to the stored pattern A[h] for each neuron h. Specifically,
Z* = argmax_{||Z||=1} Σ_h lse(β, A[h]^T Z) (modern Hopfield energy minimum).

**Expected routing behavior:**
- Z is pulled toward stored resonant directions in A during routing
- Neurons with similar initial Z will converge to the same stored pattern in A → clustering
- Acts as a "clean-up" memory: noisy Z is corrected toward the nearest stored direction

**Key risk:** Excitatory amplification. The sigmoid gate is positive when Z aligns with A[h].
Adding α * A[h] increases ||Z|| and alignment simultaneously → runaway positive feedback.
This is why the l2-normalization at each routing step is CRITICAL — it prevents the
amplification loop. Without normalize: Z → ∞ along the A[h] direction.

**Duplicate with θ:** The model already has a learnable theta that gates propagation. Adding
another threshold θ here creates an unexplained double-threshold. Prefer sigmoid without shift
(θ=0) or use tanh to allow inhibition when Z is anti-aligned.

**Verdict:** Architecturally clean and theoretically grounded (modern Hopfield). The MOST
important consideration: Z must be normalized AFTER this addition before the next step,
and A should NOT also be updated by rule (b) at the same time (two interacting Hebbian loops
are harder to analyze). Test this alone first.

---

### Rule (d): Attention-style fast update

```
A ← softmax(Z @ A^T / √D) @ Z
```
expanded:
```
# Z: [B, N, D], A: [N, D]
scores = einsum('bnd,md->bnm', Z, A) / sqrt(D)   # [B, N, M=N] — every neuron attends to every stored pattern
weights = softmax(scores, dim=-1)                  # [B, N, N]
A_new = einsum('bnm,bmd->bnd', weights, Z)        # [B, N, D] — new A is weighted average of Z
A ← F.normalize(A_new, dim=-1)                    # [B, N, D]
```

Wait — this produces A with shape [B, N, D] (per-sample), which is the desired property.

**Mathematical interpretation:**
This is one step of modern Hopfield retrieval, but inverted: instead of using A to update Z
(Hopfield retrieval), here Z updates A. The update is: A[h] becomes the weighted average of
Z-vectors most similar to the current A[h] direction. This is the "key-updated" variant.

Alternatively reading it: this is exactly a self-attention layer where A is the query, Z is
the key, and Z is the value. The output is the new "query" — A updated toward the most
similar Z.

**Fixed point:** A* = the dominant cluster centroid of Z in direction space. If Z has k
clusters, each A[h] converges to the nearest cluster center. This gives genuine input-specificity:
for an input that activates cluster 3, A[h] for neurons near cluster 3 converges to that
cluster's centroid direction.

**Expected routing behavior:**
After 1-3 steps: A specializes to the current input's dominant Z-directions. Then in rule (c),
using A as an attractor will pull Z toward the current input's dominant directions → self-reinforcing
selective routing.

**Key risk:** Collapse — all A[h] converge to a single cluster center (the most dominant
Z direction). Softmax with small denominator (√D = 4 at D=16) concentrates weights heavily.
Use temperature τ > 1 to soften: softmax(scores / τ). Or use β < 1/√D.

**Implementation note:** The all-pairs matrix Z @ A^T is [B, N, N] = O(B·N²·D) — for B=128,
N=512, D=16: 128 * 512 * 512 * 16 ≈ 536M flops. Expensive. Can reduce by using only a
subset (beam_size=32 queries, as in the existing dynamic_z code).

**Biological plausibility:** Lower than rules (a)-(c) — attention is not a local rule. Each
A[h] depends on ALL Z[j] simultaneously. However, the top-K sparse variant restores locality.

**Verdict:** Most theoretically grounded (direct connection to modern Hopfield and transformer
attention). Best-understood convergence properties. Expensive without sparsification. The
natural first experiment because its behavior is predictable.

---

## Comparison Table

| Rule | Fixed Point | Tracks Z or tracks A? | Explosion Risk | Input-Specific | Computational Cost | Biological |
|------|-------------|----------------------|----------------|---------------|-------------------|------------|
| (a) momentum | A ← EMA(Z) — unstable if diverse Z | A tracks Z | None (normalize) | Weak (batch-avg) | O(N·D) | Medium |
| (b) Oja | A = principal eigenvector of Z autocorrel. | A tracks dominant Z direction | None (normalize) | Yes (per-sample) | O(N·D) | HIGH |
| (c) Hopfield attract | Z ← nearest A[h] | Z tracks A | YES — need normalize at each step | Only if A is input-specific | O(N·D) | Medium |
| (d) attention update | A ← cluster centroid of Z | A tracks Z clusters | Collapse (all A same) | YES — per-sample | O(N²·D) | Low |

---

## Critical Design Decisions

### Decision 1: Batch-shared vs per-sample A

**Batch-shared (A: [N, D]):** Fast weights represent the batch-mean Z trajectory. Still provides
some input-specificity if the batch is homogeneous (same class), but averages away for mixed batches.
Computationally cheap. Good for first experiments.

**Per-sample (A: [B, N, D]):** Fully input-specific. Memory cost: B=128 * N=512 * D=16 * 4bytes = 4MB.
Acceptable. This is the theoretically correct form.

**Recommendation:** Start with batch-shared for simplicity, then switch to per-sample if batch-shared
shows any improvement.

### Decision 2: Should A be detached from the computation graph?

The fast weight update inside the forward pass creates higher-order gradients (gradient through
the update step). This is expensive and may be unstable.

**Option A (detach A):** A is computed as a non-differentiable variable. Gradients flow only through
the final Z (which depends on A). The slow weights self.W_phase receive gradients only through the
initialization: the gradient of loss w.r.t. self.W_phase = the gradient of loss w.r.t. A at step 0.
This is the standard fast-weights approach.

**Option B (full gradient through A):** Allow gradients to flow through all update steps. Expensive
(K_iter² gradient terms). Not recommended for first experiments.

**Recommendation:** Use `.detach()` for intermediate A updates. Let self.W_phase receive gradient only
from A_0 initialization and possibly from the final routing output.

### Decision 3: Where in the routing step to apply the update

Current routing step structure:
```
1. routing dropout
2. Z_fwd = relu(Z - theta)
3. Z_struct = structural conduction
4. Z_reflected = self-inhibition
5. Z_inhibitory = phase inhibition (dynamic_z_geo)
6. Z_new = Z_struct + Z_reflected + alpha_turing * Z_inhibitory
7. Z = normalize(Z_new)
```

Insert fast-weight update AFTER step 7 (on normalized Z) and BEFORE next iteration's step 1.
This ensures A is updated with the clean, normalized Z, not the pre-normalization mixture.

For rule (c) (Z update by A): insert BETWEEN step 6 and 7, then normalize includes the A contribution.

### Decision 4: K_iter interactions

At K_iter=3, fast weights accumulate only 3 outer products. This is very little information.
Consider:
- Run K_iter=5 or 8 specifically for the fast-weight experiments (step13 depth results pending)
- Or: apply multiple inner iterations per outer step (inner loop of ISTA / Hopfield)

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Hopfield retrieval with softmax | Custom attention | `F.softmax(scores / τ, dim=-1) @ Z` — this is exactly transformer self-attention |
| Oja's rule normalization | Custom normalization scheme | `F.normalize(A, dim=-1)` — already used everywhere in codebase |
| Outer product | Manual loop | `torch.einsum('bnd,bne->bde', Z, Z)` or `Z.unsqueeze(-1) * Z.unsqueeze(-2)` |
| Fast-weight accumulation with decay | Custom memory | `A = β * A + (1-β) * update` — this is a standard EMA |
| Stable Hopfield temperature search | Grid search | Use β = 1/√D as theoretical baseline (same as attention scaling) |

---

## Common Pitfalls

### Pitfall 1: Updating nn.Parameter in-place during forward pass

**What goes wrong:** If A is the nn.Parameter self.W_phase and you do `self.W_phase = normalize(...)`
or `self.W_phase.data += ...` inside forward(), autograd breaks and gradients do not flow correctly.

**How to avoid:** Always clone first: `A = self.W_phase.clone()`. Update A as a local variable.
The parameter receives gradients only through A's initialization step.

### Pitfall 2: Cross-batch contamination of fast weights

**What goes wrong:** If A is batch-shared [N, D] and updated during training, the fast weight
from item 1 bleeds into item 2 in the same batch.

**How to avoid:** Either use per-sample A [B, N, D], or compute the update using the full batch
simultaneously (all batch items contribute to one update = batch-mean). Never update A sequentially
within a batch.

### Pitfall 3: Missing normalization after rule (c) Z update

**What goes wrong:** Rule (c) adds a positive correction to Z aligned with A[h]. Without re-normalizing,
Z grows without bound over K_iter steps. Each step amplifies the already-amplified direction.

**How to avoid:** Normalize Z after every routing step, including steps that apply rule (c). The
existing codebase already does this (F.normalize at end of each step) — do NOT remove it when
testing rule (c).

### Pitfall 4: Conflating "fast weight update" with "second routing mechanism"

**What goes wrong:** If you test rule (b) (A update) AND rule (c) (Z update from A) simultaneously,
you have two coupled Hebbian loops. Analysis becomes impossible: if it fails, you don't know which
loop caused it.

**How to avoid:** Ablation-first. Test: (1) A-update alone with no Z-update (just different init);
(2) Z-update with frozen A (pure Hopfield retrieval); (3) Combined only if both positive alone.

### Pitfall 5: Softmax collapse at high β in rule (d)

**What goes wrong:** softmax(Z @ A^T / √D) concentrates on one index when scores are large. All
N columns of A collapse to a single Z vector. The model loses all neuron specialization.

**How to avoid:** Start with temperature τ = 1.0 (β = 1/√D) as in standard attention. Monitor
entropy of softmax weights: if H → 0, the model has collapsed. Add regularization penalty on
low softmax entropy if needed.

### Pitfall 6: Applying fast-weight update BEFORE normalization

**What goes wrong:** If A is updated using unnormalized Z (before the F.normalize step), the
update is dominated by high-magnitude neurons, which are NOT the most informative ones after
normalization.

**How to avoid:** Always update A from the l2-normalized Z (the output of step 7 in the routing loop).

---

## Architecture Patterns

### Pattern: Fast-weight wrapper around existing routing loop

```python
# Conceptual structure — do NOT implement yet
def _iterate_hidden_fast(self, Z):  # Z: [B, N, D]
    # Slow weight A initialization (per-sample)
    A = self.W_phase.unsqueeze(0).expand(Z.shape[0], -1, -1).clone()  # [B, N, D]

    for k in range(self.K_iter):
        # --- existing routing (structural + inhibitory) ---
        Z = self._existing_routing_step(Z)  # already normalized at end

        # --- fast weight update (rule b, c, or d) ---
        # Rule (b): Oja on A
        sim = (A * Z).sum(-1, keepdim=True)       # [B, N, 1]
        A = F.normalize(A + self.alpha_fw * Z * sim, dim=-1)

        # OR Rule (c): Z attracted to A (use AFTER existing routing, BEFORE normalize)
        # gate = torch.sigmoid((A * Z).sum(-1, keepdim=True))   # [B, N, 1]
        # Z = F.normalize(Z + self.alpha_fw * A * gate, dim=-1) # overrides existing normalize

    return Z
```

### Pattern: Sparse fast-weight update (top-K variant for rule d)

```python
# Instead of all-pairs Z @ A^T, use only beam neurons as keys
# Reduces O(B*N²*D) → O(B*beam*N*D)
activity = Z.norm(dim=-1)                           # [B, N]
beam_idx = activity.topk(beam_size, dim=-1).indices  # [B, M]
Z_beam = Z.gather(1, beam_idx.unsqueeze(-1).expand(-1,-1,D))  # [B, M, D]

# A[h] attends only to beam neurons
scores = torch.einsum('bnd,bmd->bnm', A, Z_beam) / (D ** 0.5)  # [B, N, M]
weights = F.softmax(scores, dim=-1)                              # [B, N, M]
A_new = torch.einsum('bnm,bmd->bnd', weights, Z_beam)           # [B, N, D]
A = F.normalize(A_new, dim=-1)
```

---

## Excitatory Radiation Connection

The fast-weight mechanism is directly connected to step 14's "excitatory radiation"
(currently running as neuro_j). Step 14 computes:

```
Z_h += alpha_exc * weighted_avg(Z[similar neighbours])
```

where similarity is current Z. This is exactly rule (c) WITH A = Z (i.e., A is not a
separate parameter but the Z itself). The step 14 design is a special case of the
fast-weight framework where W_phase = Z continuously (no persistent memory).

The fast-weight design adds the crucial difference: A is a PERSISTENT direction that
accumulates across routing steps, rather than just using current Z. This persistence
is what makes it a "resonant attractor" rather than just "copy from similar neighbors."

Recommended sequencing: wait for step 14 results before implementing fast-weight W_phase.
If excitatory radiation from Z alone improves accuracy, that confirms the excitatory
pathway hypothesis and motivates testing the persistent A version.

---

## Validation Architecture

### Test mapping

| Behavior | Test approach | Signal |
|----------|--------------|--------|
| A remains bounded after K_iter=3 steps | Assert norm(A) ≈ 1.0 at end of forward | Unit test |
| A is input-specific (different for different inputs) | Forward two different inputs, check that A differs | Unit test |
| Routing output unchanged for zero alpha_fw | Assert fast-weight output == existing routing output when alpha=0 | Regression test |
| No gradient flow through A updates (detach) | Check that A.grad is None; W_phase.grad is not None | Gradient test |
| Accuracy with rule (b) vs baseline | Run 120ep experiment with alpha_fw in {0, 0.1, 0.3, 1.0} | Ablation |

### Quick run command

```bash
python -u scripts/train_step_fast_weights.py --device mps --epochs 40 --configs a_only
```

### Phase gate

Full 120ep run for winning alpha_fw, then multi-seed (3x) confirmation before claiming improvement.

---

## Sources

### Primary (HIGH confidence)
- Oja (1982), J. Mathematical Biology — Oja's rule fixed point derivation (PCA eigenvector)
- Olshausen & Field (1996), Nature — Sparse coding / ISTA formulation
- Ba et al. (2016) NIPS 2016 arXiv:1610.06258 — Fast weights update rule, slow/fast interaction
- Ramsauer et al. (2021) ICLR 2021 arXiv:2008.02217 — Modern Hopfield energy, softmax retrieval, exponential capacity
- Gregor & LeCun (2010) ICML — LISTA, unrolled ISTA, convergence in few steps

### Secondary (MEDIUM confidence)
- von Oswald et al. (2023) "Transformers Learn In-Context by Gradient Descent" — outer product as gradient step
- Akyürek et al. (2022) "What Can Transformers Learn In-Context?" — attention as implicit regression

### Project-internal (HIGH confidence — direct code inspection)
- `src/sgnnet/model_resonant.py` — Current W_phase usage and routing loop structure
- `scripts/train_step10b_wphase_d16.py` — Step 10b: learned W_phase −0.69% vs static
- `learnings/LEARNINGS_phase5_p4_d16.md` — Step 14 excitatory radiation design
- `learnings/LEARNINGS_phase5_p3_breakthrough.md` — Step 10a/10b results and D=16 context

---

## Metadata

**Confidence breakdown:**
- Literature (Oja, ISTA, modern Hopfield): HIGH — classical, well-reproduced results
- Mapping to W_phase / SGNNET architecture: HIGH — based on direct code inspection
- Step 14 interaction analysis: MEDIUM — step 14 results still pending
- Per-sample vs batch-shared recommendation: MEDIUM — depends on batch homogeneity which hasn't been measured
- Computational cost estimates: HIGH — straightforward tensor shape arithmetic

**Research date:** 2026-03-30
**Valid until:** 2026-06-30 (stable mathematical literature; SGNNET-specific claims valid until architecture changes)

---

## Open Questions

1. **Will step 14 results make fast-weight W_phase redundant?**
   - What we know: Step 14 tests excitatory radiation from dynamic Z similarity — this is rule (c) with A=Z
   - What's unclear: Whether adding a persistent A (that outlasts one routing step) adds anything beyond the intra-step Z-similarity
   - Recommendation: Check step 14 results first; if excrad improves, that motivates persistent A as the next step

2. **Should fast-weight update happen on every K_iter step, or only at the end?**
   - What we know: At K_iter=3, there are 3 update opportunities; more updates give A more information but cost more flops
   - What's unclear: Whether 3 is enough or whether K_iter=8 is needed for A to converge
   - Recommendation: Step 13 depth results (K_iter=1/2/5/8/12) will inform this — schedule this experiment AFTER those results

3. **Is the slow W_phase nn.Parameter learning anything useful even if fast-weight A is better?**
   - What we know: Learned W_phase hurts (−0.69%) — it converges to average directions
   - What's unclear: Whether W_phase as initialization for A is better than random initialization
   - Recommendation: Test both: initialize A from W_phase vs initialize A from random unit vectors vs initialize A = Z_0 (seed step output)
