# Mathematical Foundations of the Neurograph Model

This document describes the mathematics of the Neurograph GNN layer as currently implemented in the `native/` module. The model represents each node as a complex-valued vector and propagates information through the graph via complex arithmetic, softmax-weighted aggregation, and per-node Adam optimization.

---

## 1. Node Representation

Each node $i$ in the graph holds two learnable vectors of dimension $D$ (`vector_dim`):

- **Phase weight** $\phi_i \in \mathbb{R}^D$ -- angular component
- **Magnitude weight** $\mu_i \in \mathbb{R}^D$ -- log-energy component

Together they define a complex-valued vector per node:

$$
z_i = e^{\mu_i} \cdot e^{j\phi_i} = e^{\mu_i + j\phi_i}
$$

where $j = \sqrt{-1}$ and all operations are element-wise across the $D$ dimensions.

### Initialization

- Phase weights are drawn uniformly: $\phi_i \sim \text{Uniform}(0, 2\pi)$ for each dimension.
- Magnitude weights are drawn from a Gaussian: $\mu_i \sim \mathcal{N}(1.0, 0.1)$.

---

## 2. Activation State

During a forward pass, each node maintains three activation tensors (separate from the learnable weights):

- **Phase activation** $\hat{\phi}_i \in \mathbb{R}^D$
- **Magnitude activation** $\hat{\mu}_i \in \mathbb{R}^D$
- **Activation strength** $a_i \in \mathbb{R}$ (scalar per node)

These are initialized from the weights at the start of each forward pass:

$$
\hat{\phi}_i = \phi_i, \quad \hat{\mu}_i = \mu_i
$$

If LayerNorm is enabled, $\hat{\mu}_i = \text{LayerNorm}(\mu_i)$ before computing activation strength.

### Activation Strength

The activation strength is the real-part projection of the complex signal:

$$
a_i = \sum_{d=1}^{D} e^{\hat{\mu}_{i,d}} \cos(\hat{\phi}_{i,d})
$$

This is a scalar that represents how "active" or "energetic" a node is -- it determines routing weights during message passing.

---

## 3. Input Injection

External input $x \in \mathbb{R}^{B \times n_{in} \times D}$ is first mapped to phase space:

$$
x \leftarrow \tanh(x) \cdot \pi
$$

This constrains inputs to the range $(-\pi, \pi)$, suitable for interpretation as phase angles. Input injection is treated as a special message-passing step: for each input node, a virtual source node is created with the input values as its phase activation and zero magnitude. The standard `update_activations` routine (Section 4) runs on these virtual edges, blending the input signal into the existing node activations.

---

## 4. Message Passing (update_activations)

This is the core computation, repeated for `iterations` steps. Given a set of directed edges $(s \to t)$:

### Step 1: Softmax Routing Weights

For each destination node $t$, compute a softmax over the activation strengths of all source nodes $s$ connected to $t$:

$$
w_{s \to t} = \frac{\exp(a_s - \max_{s'} a_{s'})}{\sum_{s'} \exp(a_{s'} - \max_{s'} a_{s'})}
$$

This is a numerically stable softmax computed via `scatter_reduce` (for the max) and `scatter_add` (for the sum) since different destination nodes receive different numbers of incoming edges.

### Step 2: Weighted Complex Superposition

Convert each source's activation to Cartesian form, weighted by routing:

$$
R_t = \sum_{s \to t} w_{s \to t} \cdot e^{\hat{\mu}_s} \cos(\hat{\phi}_s)
$$

$$
I_t = \sum_{s \to t} w_{s \to t} \cdot e^{\hat{\mu}_s} \sin(\hat{\phi}_s)
$$

These sums are accumulated per destination node using `scatter_add`.

### Step 3: Complex Multiplication with Destination Weight

The aggregated complex input is multiplied by the destination node's weight vector (also in complex form):

$$
R_w = \mu_t^{(\text{weight})} \cos(\phi_t^{(\text{weight})}), \quad I_w = \mu_t^{(\text{weight})} \sin(\phi_t^{(\text{weight})})
$$

$$
R_{\text{out}} = R_t R_w - I_t I_w
$$

$$
I_{\text{out}} = R_t I_w + I_t R_w
$$

This is standard complex multiplication $(R_t + jI_t)(R_w + jI_w)$.

### Step 4: Extract New Activations

Convert back to polar form:

$$
\hat{\phi}_t^{(\text{new})} = \text{atan2}(I_{\text{out}}, R_{\text{out}})
$$

$$
\hat{\mu}_t^{(\text{new})} = \frac{1}{2} \ln(R_{\text{out}}^2 + I_{\text{out}}^2)
$$

The magnitude is then mean-centered to prevent drift:

$$
\hat{\mu}_t^{(\text{new})} \leftarrow \hat{\mu}_t^{(\text{new})} - \text{mean}_d(\hat{\mu}_t^{(\text{new})})
$$

### Step 5: New Activation Strength

The new activation strength is computed from the Cartesian output, normalized by the geometric mean of the magnitude:

$$
g = \exp(\text{mean}_d(\hat{\mu}_t^{(\text{new})}))
$$

$$
a_t^{(\text{new})} = \sum_{d=1}^{D} \frac{R_{\text{out},d}}{g}
$$

---

## 5. Output Extraction

After all iterations, the activation strengths of the designated output nodes are collected and scaled:

$$
\text{output}_t = \frac{a_t}{\sqrt{D}}
$$

This gives a vector of size `output_nodes` per sample, which is typically fed into a linear classification head.

---

## 6. Edge Construction and Radiation

### Static Edges

The graph topology is built deterministically from a seed. For each non-input node, a random subset (up to `cardinality`) of non-output nodes are selected as incoming connections. Self-loops are added for every node. These static edges form the backbone of the message-passing graph.

### Radiation Targets (Dynamic Edges)

At each iteration, active nodes can discover new targets via:

1. **Cosine-searched targets**: The node's phase weight is used as a query to find the `radiation_targets` most similar nodes by cosine similarity in phase space (using the conjugate trick: $[\cos\phi; -\sin\phi]$ as query against $[\cos\phi; \sin\phi]$ index). A `radiation_similarity_threshold` can filter low-similarity hits.

2. **Random targets (scattering)**: A fraction of radiation targets are chosen uniformly at random. The fraction is controlled by `scattering_prob`, which decays exponentially during training:

$$
p(t) = p_0 \cdot \frac{e^{-kt/T} - e^{-k}}{1 - e^{-k}}, \quad k = 3.0
$$

where $t$ is the current step, $T = \text{total\_steps} \times \text{stochastic\_radiation\_duration}$, and $p_0$ is `scattering_prob`. After step $T$, scattering drops to zero.

### Progressive Activation

Edges are not all active from the start. An active mask tracks which nodes have been reached. Initially only input nodes are active. At each iteration, outgoing edges from active nodes are followed, expanding the active set. Once all nodes are active, the full static edge set is used directly.

---

## 7. Gradient Checkpointing

Each propagation iteration is wrapped in `torch.utils.checkpoint.checkpoint` to trade compute for memory. Forward activations are recomputed during backward instead of being stored, which is critical for large graphs with many iterations.

---

## 8. Per-Node Adam Optimizer

The `NativeGNNOptimizer` uses a dual-path strategy:

### Path 1: Head Parameters (Linear Layers)
Standard `torch.optim.Adam` -- updated every batch.

### Path 2: GNN Parameters (phase_weight, mag_weight)
Per-node gradient accumulation with Adam:

1. After `loss.backward()`, gradients on `phase_weight` and `mag_weight` are sparse (many nodes receive near-zero gradient in a given batch).
2. Only nodes with gradient norm > $10^{-10}$ are considered "active" for that batch.
3. Active nodes' gradients are accumulated across batches.
4. When a node's accumulation count reaches `accumulation_steps`, the averaged gradient is applied via Adam:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$
$$
\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)
$$
$$
\theta_{t+1} = \theta_t - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
$$

Each node maintains its own step counter $t$, first-moment $m$, and second-moment $v$, independently of other nodes. This means different nodes can be at different optimization steps.

---

## 9. LayerNorm on Magnitude

When enabled (`use_layer_norm=True`, default), `nn.LayerNorm` is applied to the magnitude activations before each message-passing step. This normalizes the magnitude vectors to have zero mean and unit variance across the $D$ dimensions, with learnable affine parameters. It stabilizes training by preventing magnitude drift across iterations.

---

## 10. Summary of the Full Forward Pass

```
Input x: (B, input_nodes, vector_dim)
    |
    v
tanh(x) * pi                              # map to phase range
    |
    v
Replicate weights for B samples           # (N, D) -> (B*N, D)
Initialize activations from weights
    |
    v
Input injection (virtual-source edges)     # blend input into input nodes
    |
    v
For each iteration (1..iterations-1):
    Build/expand edge set (static + radiation)
    update_activations:
        softmax routing -> complex superposition -> complex multiply -> polar decompose
    (wrapped in gradient checkpoint)
    |
    v
Extract output node activation strengths   # (B, output_nodes)
Scale by 1/sqrt(D)
    |
    v
Linear head -> class logits
```
