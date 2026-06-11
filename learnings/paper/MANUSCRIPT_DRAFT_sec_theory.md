## §3 Theoretical Framework

### §3.1 Complexity

SGNNET's forward pass is O(N·K) where N is node count and K = K_in + K_hh·K_iter is the maximum degree. All operations are sparse: seed projection gathers at most K_in neighbors per node, and each message-passing iteration operates on at most K_hh edges per node. There are no dense matrix multiplications beyond the final readout (O(N·D)·O(N_out) = O(N·D·N_out), constant in N for fixed D, N_out).

This contrasts with an MLP, where doubling N quadruples parameters (two dense layers = O(N²)) and doubles FLOPs per layer (O(N·input_dim)). SGNNET gains capacity with 2× the FLOPs — a fundamentally different scaling regime.

**Parameter count.** For fixed D, K_in, N_out:
- W_in (input → sphere): N × D (one D-dim vector per node, shared input projection structure)
- W_pos (positional encoding): N × D
- W_out (readout): N_out × N × D (softmax over N node embeddings)
- Total: O(N·D)

At D=16: params ≈ 33·N. For N=2048: 67,744 raw params → 34,976 trainable (precomputed spatial structure removes half). VGG FC has 25,088×4096 + 4096×4096 + 4096×1000 = 119.5M params. SGNNET at N=2048 uses **0.029%** of VGG FC parameters.

### §3.2 Hyperspherical Representation

SGNNET encodes each neuron's state as a unit vector on S^{D-1} (the D-dimensional hypersphere). This encoding has several properties that make it suited to classification:

1. **Angular distance as semantic distance.** Cosine similarity between neuron states has a direct geometric interpretation. The ΔW-proj mechanism exploits angular differences between W_pos vectors, which encode the relative spatial arrangement of input features.

2. **Fourier encoding.** Input features are projected via Fourier random features (Rahimi & Recht 2007), providing an implicit kernel approximation. The hyperspherical normalization of these projections preserves angular structure.

3. **Bounded dynamics.** L2 normalization at each routing step prevents exploding/vanishing signal — a property that dense softmax normalization lacks (experiment step958: LayerNorm routing −2.57pp; L2-normalize confirmed load-bearing).

4. **Class separation.** The readout is a softmax over dot products between the final neuron state Z[i] ∈ S^{D-1} and a learned class embedding for each of N_out classes. This is equivalent to geodesic distance classification on the hypersphere.

### §3.3 ΔW-Projection Routing

At each routing iteration t, node i aggregates messages from its K_hh neighbors via:

    dw_{ij} = normalize(W_pos[i] - W_pos[j])    # displacement direction on S^{D-1}
    proj_{ij} = (Z[j] · dw_{ij}) · dw_{ij}       # project Z[j] onto dw direction
    Z_agg[i] = sum_j( |proj_{ij}| · Z[j] )        # magnitude-weighted aggregation

The displacement vector dw_{ij} encodes the geometric relationship between positions i and j in the learned positional space. The absolute value of the projection coefficient (step939: abs() confirmed load-bearing, −0.64pp without it) ensures that contributions are always positive — collapsing opposing directions onto a common magnitude.

**Interpretation (HYPOTHESIS, not confirmed ablation).** The routing signal acts as a spatial gradient: nodes whose positional vectors differ most contribute the largest displacement signal, guiding the routing toward spatially-structured regions of S^{D-1}. This is analogous to a potential field where ΔW encodes the local gradient of the learned geometry.

**Why routing matters.** step978 confirmed: random projection + mean-pooling = 13.35% (near-chance). Mean-pooling destroys the spatial structure encoded in individual Z[i]. Routing is not the bottleneck — it's the aggregation mechanism that preserves and amplifies this structure.

### §3.4 Anti-Hebbian Suppression

The anti-Hebbian update rule subtracts a neuron's own contribution from its incoming messages:

    Z_ah[i] = Z_agg[i] - alpha_ah · Z[i]

This promotes diversity across neuron states: neurons that align strongly with the population average are penalized. The effect is load-balancing — all neurons remain active (step967 confirmed: node_utilization=100%, all nodes activate for all inputs). Anti-Hebbian suppression prevents mode collapse but, as a consequence, prevents class-selective routing as well.

### §3.5 Reflection Memory

A reflection term accumulates the residual from each routing step:

    Z_ref_{t+1} = alpha_ref · Z_ref_t + (Z_fwd_t - Z_t)

The reflection captures the "momentum" of changes across K_iter routing steps, providing a form of implicit recurrence that allows information from early iterations to persist into later ones. alpha_ref=0.5 is confirmed as the optimal balance (step917/918 ablations).

### §3.6 Connection to Existing Theory

| Concept | SGNNET analog | Key difference |
|---------|--------------|----------------|
| GCN (Kipf & Welling 2017) | Routing loop with mean aggregation | Degree-agnostic (K fixed), spatial structure in W_pos |
| GAT (Veličković et al. 2018) | Routing with ΔW-proj as attention | Attention signal is geometric (position-based), not learned per input |
| Random Features (Rahimi & Recht) | Fourier seed projection | Subsequent routing transforms, not just projection |
| ΔW as potential field | **(HYPOTHESIS)** | Displacement vectors ≈ local gradient of learned geometry |

SGNNET is closer to a GCN with spatial attention than to a dense transformer — the key distinction being that the attention weights are fixed by learned geometry (W_pos), not input-conditioned.
