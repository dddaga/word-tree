# Research: Dynamic Excitatory Connection Mechanisms for SGNNET

**Researched:** 2026-03-30
**Domain:** Computational neuroscience — dynamic connectivity, synchrony, Hebbian learning, predictive coding, fast weights
**Confidence:** HIGH (all five topics are well-established; translations to SGNNET are reasoned derivations from architecture source code)

---

## Project State Summary (grounding context)

Current ceiling: **29.22%** top-1 (D=16 Fourier, N=512, dynamic_z_geo, 150ep, step9A).

Key asymmetry in current routing:

| Pathway | Topology | Sign | Selection |
|---------|----------|------|-----------|
| Conduction (conn_hh) | Fixed structural | Excitatory | None — all K_hh=6 contribute |
| Radiation (dynamic_z_geo) | Rebuilt per forward from current Z | **Inhibitory** | Top-beam by magnitude |

What failed: W_phase — a learned D=16 "phase anchor" per neuron — learned a **static** graph that averaged across all training inputs and added noise (−0.69% vs static random W_phase). The lesson: a per-neuron STATIC parameter cannot represent an INPUT-SPECIFIC graph.

What is needed: excitatory connections that are **dynamic** (rebuilt from the current activation state on each forward pass) and **input-specific** (not baked in as learned weights).

Step 14 (currently running) tests the simplest version: excitatory radiation where each neuron h finds its top-K_exc most directionally similar neurons by current Z and adds their Z to itself. This research answers: what are the deeper theoretical frameworks behind this and related mechanisms, and what further experiments do they suggest?

---

## Research Question 1: Kuramoto Oscillators and Neural Synchrony

### Core Mechanism

The Kuramoto model describes N coupled oscillators, each with a scalar phase θ_i(t) and natural frequency ω_i. The coupling term drives phases to align with a population mean:

```
dθ_i/dt = ω_i + (K/N) * Σ_j A_ij * sin(θ_j - θ_i)
```

where A is the adjacency matrix. The key quantity is the **order parameter** r(t) = |(1/N) Σ_j exp(iθ_j)| ∈ [0,1]: r→1 means all oscillators are synchronized (one cluster), r→0 means all phases are incoherent. On a graph (Kuramoto on networks), different subsets synchronize into separate clusters when their natural frequencies are close — this is the **community detection** property. The coupling is inherently excitatory: a neuron near phase 0 and a neighbor near phase π will both be pulled toward each other (the sin term reduces the gap). There is no inhibition in the base Kuramoto model.

### High-Dimensional Generalization to D=16

Kuramoto uses scalar phases on S¹. The natural generalization to S^(D-1) is the **Lohe model** or **D-dimensional Kuramoto**:

```
dZ_i/dt = P_{Z_i^⊥}( Σ_j A_ij * Z_j )
```

where P_{Z_i^⊥} is the projection onto the tangent space of Z_i on the sphere — it removes the component parallel to Z_i (which would change the norm) and keeps only the component that rotates Z_i toward Z_j. In discrete time (suitable for our routing loop):

```
Z_i^(k+1) = normalize( Z_i^(k) + α * Σ_j A_ij * (Z_j^(k) - <Z_j^(k), Z_i^(k)> * Z_i^(k)) )
           = normalize( Z_i^(k) + α * Σ_j A_ij * P_{Z_i^⊥}(Z_j^(k)) )
```

The connection to SGNNET is direct: the excitatory radiation in step 14 computes:

```
Z_h += alpha_exc * weighted_avg(Z[similar neighbours])
Z_h = normalize(Z_h)
```

This is **exactly the D-dimensional Kuramoto update** on a dynamically-constructed similarity graph. The difference is that SGNNET's beam selects top-K similar neurons as the adjacency, whereas Kuramoto uses a fixed A. The order parameter r becomes the mean pairwise cosine similarity among neurons in the beam — a measurable diagnostic.

**Synchrony and functional connectivity**: In the biological Kuramoto literature, neurons that synchronize (high r) form a "functional module." The synchrony graph (who synchronizes with whom) is the functional connectivity. In SGNNET terms: if excitatory radiation runs for K_iter steps, neurons that started pointing in similar directions will be pulled into a cluster. Each cluster represents a "feature module." The number of clusters is set by the competition between the pulling-together (excitation) and the pushing-apart (inhibitory radiation keeps clusters separated).

**Confidence:** HIGH for scalar Kuramoto mechanics; HIGH for the D-dimensional (Lohe) extension; MEDIUM for the specific identification of SGNNET's normalize-then-accumulate as the discrete Lohe update (this is a reasoned derivation, not a cited paper).

### Translation to D=16 Direction-Vector Routing

The step 14 excitatory radiation is the discrete Lohe (spherical Kuramoto) update on a dynamically-built K-NN graph. Each routing step k is one Kuramoto time step. K_iter=3 means 3 time steps of synchronization dynamics.

**Key insight from Kuramoto theory**: the coupling strength α and the number of steps K_iter jointly determine whether the system fully synchronizes (one cluster) or settles into multiple co-existing clusters. In SGNNET terms:
- Too high alpha_exc + too many K_iter → all neurons collapse to the same direction (over-smoothing, analogous to Kuramoto full sync)
- Too low alpha_exc or too few K_iter → no meaningful clustering (neurons stay at initial seed)
- The sweet spot: alpha_exc and K_iter such that neurons within a class cluster, but neurons across classes remain separated (inhibitory radiation maintains inter-cluster separation)

**Measured diagnostics**: At the end of each forward pass, r = mean cosine similarity among neurons = Kuramoto order parameter. Tracking r by class label reveals whether the model is:
- Under-synchronizing (r low → clustering not forming)
- Over-synchronizing (r high, uniform across classes → class information lost)
- Well-calibrated (r moderate, with r_within_class >> r_across_class)

### Concrete Experiment Hypothesis

**Hypothesis K1: Kuramoto-gated excitation with explicit coupling constant annealing.**

Current step 14 sets alpha_exc as a fixed hyperparameter. Kuramoto theory suggests a **critical coupling K_c** above which synchrony emerges. Below K_c, excitation does nothing; above it, clustering occurs. The transition is sharp.

Experiment: instead of fixed alpha_exc, use a per-routing-step coupling that scales with the current order parameter r:
```
alpha_exc(k) = alpha_base * r^(1/2)
```
This creates positive feedback (high r → stronger coupling → higher r) but only within a step, since Z is reset at the start of each forward. Expected effect: faster convergence to stable clusters within a forward pass, sharper class boundaries.

**Expected effect**: +0.5% to +1.5% over fixed alpha_exc, primarily by avoiding the under-synchronization regime in early routing steps.

**Why this is different from W_phase**: No learned parameters. The coupling is entirely determined by the current activation state. Input 1 produces one synchrony pattern; Input 2 produces a different one.

### Risk Factors

1. **Over-smoothing**: If K_iter > 3 and alpha_exc > 0.3, the model may collapse to a single cluster per batch, destroying all discriminative information. Monitor r as a diagnostic.
2. **Gradient pathology**: The top-K selection is not differentiable (same issue as in step 14's current implementation). Gradients flow only through the K selected neighbors. This may create dead neurons that are never selected and receive no gradient. A soft attention alternative (softmax over all N instead of hard top-K) is differentiable but O(N²) per step.
3. **Interaction with inhibitory radiation**: The two pathways (excitation pulls similar Z together, inhibition pushes high-magnitude Z apart) may create oscillations in Z within a single forward pass, preventing convergence. This is the SGNNET analog of the edge-of-chaos in coupled oscillator networks.
4. **Scale sensitivity**: The D=16 space has much larger angular separations than D=4, which means the coupling term (which depends on cosine similarity) will be weaker on average. Alpha_exc values that worked in scalar Kuramoto need to be significantly larger in D=16 due to the reduced average pairwise similarity.

---

## Research Question 2: Binding by Synchrony (von der Malsburg / Singer)

### Core Mechanism

von der Malsburg (1981) proposed that **synchronous oscillation of firing rates** solves the binding problem: neurons responding to features of the same object fire in phase with each other, while neurons responding to different objects fire out of phase. This creates a time-multiplexed "address" — features bound together share a temporal code. Wolf Singer's lab (1989–2000) confirmed experimentally in cat V1 that neurons responding to a single moving bar synchronize (~40Hz gamma oscillations) while neurons responding to two separate bars do not, even when local firing rates are identical. The computational mechanism: a feature neuron emits a spike only within a time window defined by its current "phase." Receiving neurons integrate only spikes within their own window. If the phase difference exceeds π/2, the signal is suppressed; if the phase difference is near 0, the signal is amplified.

**The key point**: synchrony is not about magnitude (firing rate) — it is about PHASE ALIGNMENT. Two neurons can fire at the same rate and be desynchronized (carry different objects' information), or fire at different rates but be synchronized (bound to the same object).

### Translation to D=16 Direction-Vector Routing

In SGNNET, the "phase" of neuron h is its Z direction (unit vector in R^D). Two neurons are "synchronized" (in the binding-by-synchrony sense) if their Z vectors point in the same direction (cosine similarity ≈ 1). They are "desynchronized" if they are orthogonal (cosine ≈ 0) or anti-phase (cosine ≈ -1).

The routing rule that implements binding-by-synchrony is:
- **Excitation if cos(Z_h, Z_j) > θ_bind**: neuron j reinforces neuron h (they are "bound" to the same feature)
- **Inhibition if cos(Z_h, Z_j) < -θ_bind**: neuron j suppresses neuron h (they are bound to different features, competitive inhibition)
- **No interaction if |cos(Z_h, Z_j)| < θ_bind**: orthogonal neurons are irrelevant to each other

This is a **Mexican hat** function in D-dimensional direction space, applied via the current Z similarity:
```
strength(h, j) = cos(Z_h, Z_j)    if |cos| > θ_bind
               = 0                 otherwise
```

The current SGNNET already implements HALF of this: inhibitory radiation suppresses neurons with high cosine similarity (competes within the same "phase group"). What is missing is the EXCITATORY half: neurons with high cosine similarity should also REINFORCE each other.

**The biological analogy in SGNNET terms:**
- "Synchrony group" = neurons whose Z vectors are in the same hemisphere (cosine > θ_bind)
- "Object representation" = the dominant Z direction that emerges after K_iter routing steps
- "Binding" = the convergence of all feature-relevant neurons to the same Z direction
- "Segmentation" = the inhibitory competition separating different objects' Z directions

**What this means for routing design**: The binding mechanism requires a **signed coupling** rather than a purely excitatory or purely inhibitory one. Neurons in the same phase group (similar Z) excite each other AND compete via inhibition for representational dominance. Neurons in different phase groups are mutually inhibitory. This is richer than step 14's "top-K similar = excite" alone.

### Concrete Experiment Hypothesis

**Hypothesis B1: Signed coupling ("Mexican hat" in direction space)**

Current architecture: inhibitory radiation operates on top-beam neurons (top by magnitude). Excitatory radiation (step 14) operates on top-K similar neurons (top by cosine similarity).

Proposed experiment: replace two separate pathways with a SINGLE signed routing term:
```
coupling(h, j) = cos(Z_h, Z_j)                      # positive = excitatory, negative = inhibitory
Z_h += alpha_signed * Σ_j coupling(h,j) * Z_j / Z   # weighted sum by signed coupling
Z_h = normalize(Z_h)
```

This is biologically precise: neurons in the same phase group (cos > 0) reinforce h; neurons in the opposite phase group (cos < 0) suppress h. No separate beam selection is needed — the sign of the cosine similarity selects the direction of influence naturally.

**Expected effect**: Because this replaces BOTH the current inhibitory radiation AND the proposed excitatory radiation with a single unified term, it removes the need to tune two separate alpha parameters (alpha_turing and alpha_exc). The signed coupling learns to allocate excitation/inhibition automatically from the data. Expected: +1-2% by replacing noisy magnitude-based inhibition with direction-based signed coupling.

**Practical concern**: This requires computing a full N×N cosine similarity matrix per routing step (O(B×N²×D)). At N=512, D=16: 512×512×16 = 4.2M ops per batch per step. On MPS this is feasible — it is identical to the bmm in step 14's excitatory radiation computation.

**Hypothesis B2: Phase synchrony diagnostic**

Before implementing B1, instrument the current model to measure "synchrony" as a function of class label: after each routing step k, compute the mean pairwise cosine similarity among neurons grouped by their most-activated output class. If the model is learning to "bind by synchrony," this measure should increase monotonically across routing steps and be higher within-class than across-class. This diagnostic requires no architecture change and can run on any saved checkpoint.

### Risk Factors

1. **Sign flip instability**: If both excitation and inhibition operate simultaneously on the same coupling, the system may oscillate. Neurons that are initially slightly above cos=0 get excited, which pushes them to higher cos, which increases excitation — runaway. The l2-normalization at each step is the only stabilizing force. Need to verify that normalize() is sufficient to prevent this runaway.
2. **Crowding in D=16**: At D=16, 512 neurons on S¹⁵ are well-separated, so on average pairwise cosine similarities are close to zero. The signed coupling will mostly produce near-zero contributions, which is fine for excitation/inhibition balance but means a high alpha is needed to see any effect.
3. **False binding**: If two feature detectors for two different objects happen to have high cosine similarity (both represent "edge-like" features), binding-by-synchrony will erroneously "bind" them. This false binding is the primary criticism of the biological theory, and applies equally to the computational version.
4. **Conflict with existing inhibitory radiation**: The signed coupling already contains an inhibitory component (cos < 0 suppresses). Adding this on top of the existing dynamic_z inhibitory radiation doubles the inhibitory signal and may over-suppress the network.

---

## Research Question 3: STDP (Spike-Timing Dependent Plasticity)

### Core Mechanism

Hebbian plasticity: "neurons that fire together, wire together." STDP makes this timing-precise: if neuron A fires BEFORE neuron B (Δt > 0), the synapse A→B is potentiated (strengthened). If A fires AFTER B (Δt < 0), the synapse A→B is depressed. The learning window is typically:
```
ΔW_AB = A+ * exp(-Δt / τ+)   if Δt > 0   (LTP: A fires before B)
ΔW_AB = -A- * exp(|Δt| / τ-)  if Δt < 0   (LTD: A fires after B)
```

The key property: STDP is **causal** — it strengthens connections that transmitted information that caused subsequent firing, and weakens connections that transmitted information after the fact (those connections were not causally relevant). This creates input-specific connectivity: the synaptic weights W_AB after learning reflect the temporal correlations in the CURRENT input stream, not averages over past inputs. Online STDP (updating on each input) is the biological form of fast in-context adaptation.

In the context of deep learning, STDP analogues appear as **Contrastive Hebbian Learning** (Hopfield 1984, Movellan 1990): run the network on the input (free phase), run it with the target clamped (clamped phase), the weight update is proportional to the difference between the two Hebbian correlation matrices.

### Translation to SGNNET Routing Steps

The "time" in STDP maps to the routing step k ∈ {0, 1, ..., K_iter-1}. The "spike" maps to entering the beam (being in top-beam by magnitude) at step k. The "synapse" maps to the excitatory connection strength between two neurons.

**STDP-in-routing formulation:**

For routing steps k and k', define a "temporal correlation" between neurons h and j as:
```
corr(h, j) = (beam_indicator[h,k] * beam_indicator[j,k']) for k' > k
```
"Neuron h was in the beam at step k, then neuron j entered the beam at step k+1" — this is the STDP potentiation condition (h fired before j). The excitatory weight for h→j should increase.

In a single forward pass (no actual weight update possible via gradient), this translates to a **within-pass fast weight update**:
```
# After step k: record which neurons are in the beam
beam_at_k = top_beam_indices(Z[k])

# At step k+1: neurons that were NOT in beam at k, but are now = "just fired" neurons
newly_active = beam_at_k+1 - beam_at_k  ∩ (beam_at_k+1)

# STDP-inspired excitation: previously active neurons (beam[k]) excite newly active neurons
For each h in newly_active:
    Z[k+1][h] += alpha_stdp * mean(Z[k][j] for j in beam_at_k if cos(Z[k][j], Z[k+1][h]) > 0)
```

The causal structure is preserved: neurons that were active at step k (the "pre-synaptic" event) selectively excite neurons that become active at step k+1 (the "post-synaptic" event). This is strictly causal — no information from future steps is used.

**Why this avoids the W_phase failure mode**: W_phase learned a static synaptic matrix by gradient descent over many training examples. STDP as described above updates the effective connectivity WITHIN a single forward pass based on the temporal ordering of beam entry. No learned parameters are added. The connectivity is fully input-specific: Input A produces a different beam-entry sequence than Input B, producing different STDP potentiation patterns.

### Concrete Experiment Hypothesis

**Hypothesis S1: Cross-step excitation from prior beam to current non-beam**

At each routing step k+1, for each neuron h NOT currently in the beam, compute the cosine similarity between Z_h and each neuron j that WAS in the beam at step k. If cos(Z_h, Z_j[k]) is high, add a fraction of Z_j[k] to Z_h before the step-k+1 routing computation:
```
pre_beam_signal = mean(Z[j][k] for j in top-M_beam if cos(Z[h], Z[j][k]) > theta_stdp)
Z[h][k+1] += alpha_stdp * pre_beam_signal  # before normalize
```

This is the **feed-forward STDP**: the beam at step k excites neurons that are similar to it at step k+1, giving them a boost that may push them into the beam at step k+1. Over K_iter=3 steps, this creates a causal chain of beam propagation: neurons activated early in the routing pass "teach" similar neurons to activate later.

**Implicit hypothesis about why K_iter helps**: If this mechanism is correct, K_iter=3 is the minimum for one full STDP cycle (pre-fire → post-fire → confirmation). K_iter=5 or K_iter=8 allows deeper causal chains. The step 13 depth sweep (currently running) directly tests this: if STDP-like temporal causality is the key mechanism, accuracy should increase sharply with K_iter (matching the "exponential gains with depth" hypothesis from step 13's notes).

**Expected effect**: +0.5-1.5% by making beam-entry causally propagating rather than parallel. The most important property is that this creates a "routing agenda" — early routing steps determine later ones within the same forward pass.

**Alternative simpler formulation (S2)**: Rather than tracking beam membership across steps, simply use step k's Z as an additional input to step k+1's excitatory computation:
```
# At step k+1, neuron h also receives signal from step k's beam via cosine similarity
Z_excit[h] += alpha_stdp * sum(Z[j][k] * relu(cos(Z[h][k], Z[j][k])) for j in beam[k])
```
This is essentially the step 14 excitatory radiation, but using the PREVIOUS step's Z rather than the CURRENT step's Z. This introduces a one-step memory into the routing, analogous to STDP's Δt > 0 window. It is easy to implement by caching Z_prev inside the routing loop.

### Risk Factors

1. **Memory overhead**: Caching Z at each step requires storing K_iter additional [B, N, D] tensors. At K_iter=3, B=128, N=512, D=16: 3 × 128 × 512 × 16 × 4 bytes = 12.6 MB per forward pass. This is negligible on MPS.
2. **Stale signal problem**: At step k+1, the beam from step k may no longer be relevant (routing has moved on). If K_iter > 5, the step-0 beam signal may be stale enough to confuse rather than guide. Use only the IMMEDIATELY prior step (k → k+1) to avoid stale signal.
3. **Confusion with step 14's current excitatory radiation**: Step 14 uses CURRENT Z; STDP S2 uses PREVIOUS Z. These are related but not identical. If S2 is added without removing step 14's excitatory radiation, there will be double-excitation. Should be ablated cleanly: S2 only, vs S2 + step14, vs step14 only.
4. **Not truly STDP**: The biological STDP involves gradient-based weight updates between training examples, not within a single forward pass. The SGNNET analog (cross-step excitation) preserves the causal timing intuition but is not learning — it is a routing rule. The name "STDP-inspired" is more accurate than "STDP."

---

## Research Question 4: Predictive Coding (Rao & Ballard 1999 / Friston Free Energy)

### Core Mechanism

Rao & Ballard (1999) proposed that the visual cortex implements a hierarchical predictive coding scheme: each layer generates a **prediction** of what it expects to receive from lower layers. The actual signal from lower layers is compared to the prediction, and only the **prediction error** (residual) is passed upward. Learning minimizes prediction error by updating both the predictions and the generative model weights. Friston's Free Energy Principle (2005–2010) generalizes this: every neuron group generates predictions about its inputs; the system minimizes "surprise" (variational free energy = prediction error + complexity). The key computational property: **top-down connections carry predictions, bottom-up connections carry prediction errors**. When a top-down prediction is good, very little bottom-up signal propagates (the prediction "explains away" the input). When the prediction is poor, a large error signal propagates upward to update higher-level beliefs.

**In the context of dynamic excitation**: The top-down prediction acts as an **excitatory prior** — neurons that match the current prediction are excited (their prediction error is low, they "fit" the expectation), and neurons that do NOT match the prediction are suppressed (high prediction error, they are unexpected). This creates input-specific excitation: the "which neurons are currently predicted to be active" question is answered differently for every input.

### Translation to SGNNET: W_phase Repurposed as Prediction

W_phase failed as a static graph because it averaged over all inputs. But if W_phase is repurposed as a **predictive template** — what each neuron expects its neighborhood to look like GIVEN ITS CURRENT STATE — it becomes input-specific.

**Predictive W_phase mechanism:**

At each routing step k, each neuron h generates a prediction of what it expects to see from its neighborhood:
```
prediction_h = W_phase[h] ⊙ Z_h      # element-wise modulation: W_phase shapes which directions h expects
```
or more precisely:
```
prediction_h = normalize(W_phase[h] + Z_h)   # W_phase biases Z toward a "habitual" pattern
```

Then, for each other neuron j, compare Z_j to prediction_h:
```
match_score(h, j) = cos(Z_j, prediction_h)
if match_score > 0: Z_h += alpha_pred * match_score * Z_j    # excite h if j matches h's prediction
```

The key difference from W_phase as static graph: here the prediction is the COMBINATION of W_phase (a learned prior, which IS static) and Z_h (the current activation, which IS input-specific). The composition is input-specific even if W_phase is static. This is exactly how predictive coding works: the prior is static (learned over many inputs), but the prediction = prior + current belief update is dynamic.

**Why this might avoid the W_phase failure**: W_phase failed because it was used ALONE as the adjacency matrix (step 2, step 10b). In the predictive coding formulation, W_phase is only a PRIOR that gets combined with the current Z. The network has learned a "prior belief about what neighbors should look like" (W_phase), but the EFFECTIVE prediction is always modulated by the current input (Z). For an unusual input that activates neurons in a completely different pattern, Z deviates from W_phase and the prediction is dominated by Z. For familiar inputs, W_phase correctly predicts the neighborhood and reinforces efficient routing.

### Concrete Experiment Hypothesis

**Hypothesis P1: Prediction-error gated excitation**

For each neuron h at routing step k:
1. Generate prediction: pred_h = normalize(beta * W_phase[h] + (1-beta) * Z_h)  [beta ∈ {0.1, 0.3, 0.5}]
2. For each neuron j in the current beam, compute match: m = cos(Z_j, pred_h)
3. Excitation if m > 0: Z_h += alpha_pred * m * Z_j
4. W_phase is not updated during the forward pass (no fast weight update), but its gradient now comes from whether predictions were accurate

The ablation:
- beta=0: prediction = Z_h (pure bottom-up, equivalent to step 14 excitatory radiation)
- beta=0.5: equal mix of prior and current state
- beta=1: prediction = W_phase only (pure static prior = W_phase failure mode)

**Expected effect**: beta=0.3 should outperform both extremes. When beta=0 (pure Z_h), the "prediction" is simply the current Z, so we excite neurons similar to current Z — this is step 14 exactly. When beta=1 (pure W_phase), we are back to the failed static graph. At beta=0.3, W_phase shapes the prediction while Z_h keeps it input-specific.

**Why W_phase learns useful priors in this setting**: With prediction-error gated excitation, W_phase[h] receives a gradient signal telling it "which direction in D-space did neurons tend to come from when h was active and was correctly routing." This is a meaningful signal: it corresponds to the habitual neighborhood of neuron h across the training set. The gradient is now meaningful (not just "minimize KL loss indirectly"), which is why W_phase may learn something useful rather than noise.

**Hypothesis P2: Top-down K-iter prediction loop**

Use the FIRST routing step to generate predictions and the SUBSEQUENT steps to compute prediction errors. Step 0: forward pass with seed Z, identify beam neurons, generate predictions pred_h = f(Z_h, W_phase[h]) for all h. Steps 1..K_iter: at each step, excite neurons that match their predictions from step 0, suppress neurons that do not. This creates a top-down / bottom-up alternation even without a hierarchical architecture: step 0 is the "top-down prior," steps 1..K provide the "bottom-up updates."

### Risk Factors

1. **W_phase gradient conflict**: In the current model, W_phase receives gradient from the phase graph (conn_phase, which is rebuilt from W_phase's K-NN). In the predictive coding variant, W_phase additionally receives gradient from prediction accuracy. These two gradient sources may conflict.
2. **Beta as a new hyperparameter**: Adding beta to the existing hyperparameter space (beam_size, alpha_turing, alpha_reflect, geo_gamma, resonance_threshold) increases tuning complexity. Should be ablated carefully at 3 values before combining with other mechanisms.
3. **Training time for W_phase to learn**: The predictive coding advantage depends on W_phase having learned meaningful priors. Early in training (epoch 0-20), W_phase is random, so the prediction is random, and the excitation is noise. The model may underperform step 14 during warm-up. Check performance curves epoch by epoch.
4. **Mismatch between W_phase dimension and prediction use**: W_phase is currently initialized as torch.rand(N, D), so its direction is uniform on S^(D-1). In the predictive coding formulation, W_phase needs to converge to the habitual activation direction of each neuron. With 9,469 training samples and 120 epochs, this may be sufficient for convergence, but is uncertain.

---

## Research Question 5: Fast In-Context Hebbian Learning (Fast Weights)

### Core Mechanism

Ba et al. (2016) "Using Fast Weights to Attend to the Recent Past" and the subsequent literature on fast weights (Hinton & Plaut 1987 original concept): neural networks have TWO types of weights — **slow weights** (updated by gradient descent over many examples, stored between sessions) and **fast weights** (updated by a Hebbian rule within a single input or sequence, discarded after the sequence ends). The fast weight matrix A accumulates Hebbian associations within the current context:
```
A(t+1) = λ * A(t) + η * h(t) ⊗ h(t)    # outer product of current hidden state with itself
```
At each time step, the slow-weight network computes its hidden state h, and the fast weight matrix A modulates the hidden state:
```
h'(t) = LayerNorm(W_slow * h(t-1) + A(t-1) * h(t))
```

In transformers, this mechanism is EXACTLY what attention implements: the key-value pairs from the current context form an associative memory (the fast weight matrix), and queries retrieve from this memory. This is the "transformers implement fast weights" insight from Schmidhuber (1993) and rediscovered by Katharopoulos et al. (2020) and others.

For **in-context learning specifically** (large LLMs learning from few-shot examples in the prompt), recent analyses (e.g., von Oswald et al. 2023, Akyürek et al. 2022) show that transformer attention layers can implement gradient-descent-in-context: each attention head computes an implicit gradient step on the key-value examples in the context window, updating an effective "in-context weight matrix" without modifying the actual model weights. This is fast Hebbian learning at inference time.

### Translation to SGNNET Routing

The SGNNET routing loop K_iter iterates over k=0..K_iter-1. This is analogous to a sequence of T time steps in the fast-weight framework. The "context" is the current input (the 25,088-dim VGG feature vector), which creates a specific Z configuration at step 0 (via seeding). The fast weight matrix A is accumulated from Z states across routing steps.

**Fast-weight SGNNET formulation:**

```
A_0 = 0   # empty fast weight matrix, size [D, D]
For k in range(K_iter):
    # Fast weight update: accumulate outer product of current Z (averaged over neurons)
    Z_mean = Z.mean(dim=1)                          # [B, D]
    A_k = lambda_fw * A_{k-1} + eta_fw * Z_mean.unsqueeze(-1) * Z_mean.unsqueeze(-2)  # [B, D, D]

    # Use fast weights to modulate next routing step
    Z_excit = torch.bmm(Z, A_k.transpose(1,2))     # [B, N, D] — each neuron queries the fast weight memory
    Z_excit = F.normalize(Z_excit, dim=-1)

    # Combine with structural routing
    Z = normalize(Z_struct + alpha_fw * Z_excit + alpha_turing * Z_inhibitory)
```

**What the fast weight matrix A captures**: A_k accumulates the outer product of the population mean activation at each routing step. This means A_k encodes "what directions are currently dominant in the population" as a D×D associative matrix. A neuron h at step k retrieves from A_k by multiplying Z_h @ A_k — this projects Z_h through the current population state. If Z_h is already aligned with the current population direction, this reinforces it. If Z_h is orthogonal to the population, it receives no signal. This is a form of **resonance with the current collective state** — the fast weight implements a "population context" that excites neurons aligned with the current global state.

**Key property: input-specificity**: A_k is computed entirely from the current forward pass's Z states. For Input A (dog image), the population Z settles on a different direction than for Input B (tench image). A_k is different for each input. No learned parameters are added. This directly avoids the W_phase failure.

**Connection to transformers**: The fast weight update rule (outer product accumulation) is equivalent to an attention head where queries and keys are both Z_h vectors. This is "self-attention within the routing loop." In fact, step 14's excitatory radiation already implements a simplified version of this: for each neuron h, it finds top-K similar neurons and adds their Z. The fast weight formulation generalizes this to a dense, continuously updating associative memory.

### Concrete Experiment Hypothesis

**Hypothesis F1: Routing-step fast weight accumulation**

Implement a [D, D] fast weight matrix A that accumulates across routing steps:
```
A_k = decay * A_{k-1} + (1/N) * Z.transpose(-1,-2) @ Z   # [B, D, D] = Z^T * Z = Gram matrix
```
The Gram matrix Z^T Z is the covariance of the current Z population. Each routing step's Z is added (with decay) to build up the "current population covariance." Neurons then query this matrix:
```
Z_fw = Z @ A_k   # [B, N, D] — each neuron's Z projected through population covariance
Z_fw = normalize(Z_fw)
Z += alpha_fw * Z_fw
Z = normalize(Z)
```

**Why this is the Gram matrix**: The Gram matrix Z^T Z at entry (d1, d2) measures how much dimensions d1 and d2 co-activate across the neuron population. If many neurons have high values in both d1 and d2, these dimensions are "correlated in the current input." When neuron h's Z is multiplied by this Gram matrix, it amplifies dimensions that co-activate with its current direction — a kind of "resonance with the current feature correlation structure."

**Expected effect**: The Gram matrix is O(B × N × D) to compute (one bmm), which is negligible. The key test is whether accumulating the population Gram matrix over K_iter steps provides additional excitatory signal beyond step 14's pairwise top-K similarity. Hypothesis: it adds roughly +0.5-1.5%, primarily by giving every neuron access to the population structure (not just its K nearest neighbors).

**Hypothesis F2: Linear attention fast weights (numerically stable)**

The outer product Z_mean ⊗ Z_mean version has an instability risk (A grows unboundedly). Use the linear attention form:
```
A_k = softmax(Z @ Z.T / sqrt(D))   # [B, N, N] attention matrix
Z_fw = A_k @ Z                     # [B, N, D] — weighted sum by attention
```
This is exactly the self-attention mechanism, applied to Z at each routing step. It is differentiable, bounded, and numerically stable. The "fast weight" interpretation: for each neuron h, the attention weight A_k[h, j] measures how much neuron j is relevant to h right now. This changes each forward pass because Z changes. Note: this is an O(B × N² × D) operation — at N=512, the same cost as step 14's excitatory radiation. The key question is whether the softmax weighting (all neurons contribute, weighted softly) outperforms the hard top-K selection.

### Risk Factors

1. **The Gram matrix = second-order statistics, not routing signal**: The Gram matrix captures feature correlations in the current population, but this may not translate to useful routing signal. It could be that first-order pairwise similarity (step 14's top-K) is sufficient and the second-order Gram adds complexity without benefit.
2. **Decay parameter lambda_fw**: If lambda is too high (close to 1), A_k becomes a cumulative average over all steps, losing recency. If too low, only the last step matters (no accumulation benefit). Needs its own ablation.
3. **Numerical stability of outer product accumulation**: After K_iter=8 steps, A_k has accumulated 8 outer products. The norm of A_k grows as O(K_iter). Must normalize A_k at each step (divide by K_iter or use layer norm on Z_fw).
4. **Self-attention as the Gram-matrix variant**: If F2 (linear attention) is tested, it should be compared against the existing structural routing directly, since they are operationally similar. The question is whether the attention mechanism (soft routing over ALL N neurons) provides more useful excitation than the structural top-6 (hard routing over fixed neighbors). Given step 14 results showing top-K excitatory radiation helps, F2 is likely to also help but may provide diminishing returns.

---

## Cross-Cutting Analysis: Why W_phase Failed and How Each Framework Addresses It

| Framework | What W_phase does wrong | How the framework fixes it |
|-----------|------------------------|---------------------------|
| Kuramoto | W_phase is a static phase anchor; Kuramoto requires DYNAMIC phase that evolves per input | Replace W_phase with the evolving Z direction (discrete Lohe update from current Z) |
| Binding by synchrony | W_phase defines a fixed "who binds with whom"; synchrony requires per-input binding based on who is CURRENTLY co-active | Use cosine(Z_h, Z_j) as the binding criterion, computed fresh each forward pass |
| STDP | W_phase weight updates are gradient-based over many inputs; STDP updates are Hebbian within a single input's temporal dynamics | Cross-step excitation uses beam-entry order within ONE forward pass — inherently single-input |
| Predictive coding | W_phase alone = pure static prior (failed); predictive coding = prior COMBINED WITH current state | pred_h = beta * W_phase[h] + (1-beta) * Z_h rescues W_phase as a useful prior rather than the entire graph |
| Fast weights | W_phase = slow weight trying to do fast-weight job; fast weights are updated WITHIN the forward pass | Gram matrix A_k accumulates from current Z states, creating an input-specific associative memory without any learned W |

**The unifying principle**: ALL five frameworks predict that the correct excitatory mechanism should use the **current activation state Z** (not a learned parameter) to determine which neurons excite which. The learned parameters (W_phase, theta, W_pos) can act as PRIORS or MODULATORS, but the primary signal for excitation must be computed fresh from Z each forward pass.

This is precisely what step 14's excitatory radiation does. The five frameworks provide five different lenses on why this is the right architecture:

- Kuramoto: it is the discrete Lohe synchrony update
- Binding by synchrony: it implements feature binding by phase alignment
- STDP: it is a one-step causal cross-step Hebbian rule
- Predictive coding: it is the prediction-match excitation (with implicit pred=Z)
- Fast weights: it is the in-context associative retrieval

---

## Prioritized Experiment Designs

### Tier 1: Test immediately (build on step 14, low implementation cost)

**T1A: STDP cross-step excitation (Hypothesis S2)**
- What: cache Z_prev from step k, use it for excitation at step k+1 rather than current Z
- Implementation: 2 lines added to routing loop (cache Z before update, use Z_prev in excitation)
- Ablation: Z_prev only vs current-Z only (step 14F) vs both
- Expected: distinguishes whether temporal causality matters or simultaneous similarity suffices

**T1B: Signed coupling (Hypothesis B1)**
- What: replace separate inhibitory + excitatory terms with a single signed cosine-weighted coupling
- Implementation: modify _phase_inhibit to return signed signal (positive where cos>0, negative where cos<0)
- Ablation: signed vs separate excit/inhib vs inhibit-only (baseline)
- Risk: may over-suppress if combined with existing routing; start with alpha=0.1

**T1C: Gram matrix fast weights (Hypothesis F1)**
- What: accumulate Z^T Z Gram matrix across routing steps, use as excitatory modulator
- Implementation: one bmm per routing step (Z.transpose(-1,-2) @ Z)
- Ablation: gram-excitation vs step14 excitatory radiation vs both
- Expected: Gram captures second-order structure that step14 top-K misses

### Tier 2: Requires careful setup (W_phase dependent)

**T2A: Predictive coding W_phase (Hypothesis P1)**
- What: pred_h = normalize(beta * W_phase[h] + (1-beta) * Z_h); excite neurons that match pred_h
- Implementation: modify W_phase role in forward; add beta as hyperparameter
- Ablation: beta=0.1 / 0.3 / 0.5 at K_iter=3
- Caveat: W_phase gradient semantics change; may require longer training to see benefit

**T2B: Kuramoto order parameter as routing diagnostic**
- What: no architecture change; add logging of r = mean pairwise cosine similarity per step
- Implementation: one line in routing loop (r = Z @ Z.T; log r.mean())
- Value: confirms whether current excitatory radiation is actually creating synchrony clusters

### Tier 3: Requires step 14 results first

**T3A: Adaptive Kuramoto coupling (Hypothesis K1)**
- Depends on: step 14 confirming excitatory radiation helps
- What: alpha_exc(k) = alpha_base * r_k^(1/2) — coupling grows with synchrony order parameter
- Expected: faster in-routing convergence once baseline excitation is confirmed positive

---

## Open Questions

1. **Does step 14 excitatory radiation (currently running) actually produce synchrony clusters?**
   - What we know: the mechanism is sound (discrete Lohe / fast weight / binding by synchrony)
   - What's unclear: whether K_iter=3 is sufficient for cluster formation with alpha_exc=0.1 or 0.3
   - Recommendation: add Kuramoto order parameter logging to step 14 runs before designing next steps

2. **Is the inhibitory radiation actually helping, or is it competing with excitation?**
   - What we know: dynamic_z inhibitory radiation is the CURRENT mechanism and gives 29.22%
   - What's unclear: in the signed coupling (B1) view, excitation and inhibition on the SAME cosine function may interact non-trivially
   - Recommendation: ablate "excitation only (no inhibition)" as a control in any signed coupling experiment

3. **What is the right K_iter for multi-step mechanisms?**
   - What we know: step 13 depth sweep is running (K_iter=1/2/5/8/12)
   - What's unclear: STDP S2 and Kuramoto K1 both predict K_iter matters MORE when excitatory pathways are active (deeper causal chains become possible)
   - Recommendation: combine STDP S2 with the best K_iter from step 13 rather than running K_iter sweep again

4. **Can W_phase be rescued via the predictive coding formulation?**
   - What we know: W_phase learned fails by −0.69% at D=16 (step 10b)
   - What's unclear: whether the failure is (a) wrong gradient signal, (b) wrong functional role, or (c) simply that W_phase is a D-dimensional vector trying to act as a routing prior in D=16 space with insufficient capacity
   - Recommendation: P1 experiment directly tests (b); if P1 also fails at beta=0.3-0.5, conclude that W_phase learned prior is not a useful component regardless of how it is used

---

## Sources

### Primary (HIGH confidence)

- Kuramoto (1984), "Chemical Oscillations, Waves, and Turbulence" — original scalar Kuramoto model
- Lohe (2009), "Non-abelian Kuramoto models and synchronization" — D-dimensional extension to S^(D-1); the dZ/dt = P_{Z^⊥}(ΣA_ij Z_j) formulation
- von der Malsburg (1981), "The correlation theory of brain function" — original binding by synchrony proposal
- Singer & Gray (1995), "Visual feature integration and the temporal correlation hypothesis" — experimental evidence from V1 gamma synchrony
- Markram et al. (1997), "Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs" — original STDP characterization
- Rao & Ballard (1999), "Predictive coding in the visual cortex" — hierarchical predictive coding; top-down prediction / bottom-up error framework
- Friston (2010), "The free-energy principle: a unified brain theory?" — free energy / active inference generalization of predictive coding
- Ba et al. (2016), "Using Fast Weights to Attend to the Recent Past" — fast weight formulation for neural networks
- Schmidhuber (1993), "A self-referential weight matrix" — original fast weight / outer-product Hebbian concept
- von Oswald et al. (2023), "Transformers Learn In-Context by Gradient Descent" — transformers as fast-weight learning machines

### Secondary (MEDIUM confidence — trained knowledge, not verified against current source)

- Katharopoulos et al. (2020), "Transformers are RNNs: Fast autoregressive transformers with linear attention" — linear attention as fast weight matrix
- Akyürek et al. (2022), "What learning algorithm is in-context learning?" — in-context learning as implicit gradient steps
- Hopfield (1984), "Neurons with graded response have collective computational properties like those of two-state neurons" — early Hebbian associative memory formulation

### Architecture source verified directly

- `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/src/sgnnet/model_resonant.py` — routing implementation
- `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/scripts/train_step14_topk_cond_excrad.py` — step 14 excitatory radiation implementation
- `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/learnings/LEARNINGS_phase5_p3_breakthrough.md` — W_phase failure (step 10b) confirmed
- `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/learnings/LEARNINGS_phase5_p4_d16.md` — step 14 design rationale

---

## Metadata

**Confidence breakdown:**
- Kuramoto / Lohe spherical extension: HIGH — well-published; the discrete-time Lohe derivation as applied to SGNNET routing is a reasoned derivation (MEDIUM for the specific SGNNET equivalence claim)
- Binding by synchrony (von der Malsburg / Singer): HIGH — classical neuroscience; the D-dimensional generalization is reasoned (MEDIUM)
- STDP: HIGH — well-characterized biology; the within-forward-pass cross-step analog is a novel design (MEDIUM for the experiment hypotheses)
- Predictive coding (Rao & Ballard / Friston): HIGH — well-established; the W_phase-as-prior formulation is a reasoned design (MEDIUM)
- Fast weights / in-context Hebbian: HIGH — well-established ML; the Gram matrix accumulation in routing is a straightforward derivation (MEDIUM for experiment performance estimates)

**Research date:** 2026-03-30
**Valid until:** 2026-06-30 (stable theoretical foundations; experiment designs should be updated after step 14 results arrive)
