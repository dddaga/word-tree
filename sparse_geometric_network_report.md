# Sparse Geometric Neural Network (SGNNET) — Research Brief

**Author**: Dhiraj
**Date**: March 2026
**Audience**: A collaborator with a physics background and working knowledge of deep learning. This document explains the hypothesis, every design decision and its reasoning, and where decisions are temporary placeholders vs. intended final design. The goal is to give you enough context to take an independent shot at building this.

---

## 1. The Problem We Are Trying to Solve

Modern large language models (LLMs) like GPT-4 or LLaMA have billions of parameters. A well-known but underexplored property of these models is that they are **overparameterized** — for any given input, only a small fraction of neurons are doing meaningful work. The rest are either redundant or inactive.

The standard solution is **Mixture of Experts (MoE)**: instead of one large network, you train several smaller "expert" sub-networks, and a routing mechanism decides which expert handles each input. This reduces computation per token while keeping total model capacity high. Models like Mixtral use this approach.

But MoE is still coarse. Routing happens at the level of entire sub-networks, not individual neurons. Our hypothesis is that we can go much further — down to the neuron level — and in doing so also remove two other constraints that standard networks impose:

1. **Hard-coded layer order**: in standard networks, information flows strictly from layer 1 → layer 2 → layer 3. There is no mechanism for layer 3 to talk back to layer 1, or for two neurons in the same layer to interact.
2. **Dense connectivity within layers**: every neuron in layer N connects to every neuron in layer N+1. This is computationally expensive and biologically implausible.

**Our three core ideas**:

1. **Sparsity**: most neurons do not talk to most other neurons
2. **Dynamic connectivity**: which neurons communicate changes based on the input at runtime
3. **Recursive computation**: instead of a single forward pass through layers, information cycles through the network for K iterations, allowing different parts of the signal to interact

We are targeting the **feed-forward (FFN) sub-layers of transformers** first. FFN layers are a good starting point because they are the largest component by parameter count (~65% of total parameters in most transformers), they operate independently on each token (no cross-token interaction, unlike attention), and their input-output behavior is well-defined and easy to use as a distillation target.

---

## 2. The Physical Analogy — Neurons as Atoms in a Crystal

To build intuition, consider a crystal lattice. In a crystal:

- **Atoms** sit at specific positions in 3D space
- **Light** entering the crystal diffracts — it interacts most strongly with atoms whose geometric spacing matches the wavelength of the incoming light (Bragg's Law)
- The result is that certain patterns of light are amplified (constructive interference) and others are suppressed

We borrow this analogy:

| Crystal | SGNNET |
|---|---|
| Atom position in 3D space | Neuron weight vector W_i in D-dimensional space |
| Light wave entering | Input activation propagating through network |
| Diffraction (strongest interaction with nearby atoms) | Dynamic connectivity (activations route to geometrically nearby neurons) |
| Multiple scattering through the lattice | Recursive computation loop (K iterations) |
| Emergent diffraction pattern at the exit face | Final activation state read out as prediction |
| Constructive interference at a detector | High self-projection score at an output neuron |

The key insight from the crystal analogy: **the geometry of the medium determines how information flows through it.** We want the same in our network — the spatial arrangement of neuron positions in D-space should govern how activations propagate, and this arrangement should emerge from training data rather than being manually designed.

We are not strictly enforcing this analogy. It is inspiration, not derivation. Where the analogy breaks (diffraction is linear; neural networks are nonlinear), we make pragmatic engineering choices.

---

## 3. The Architecture

### 3.1 Two Tensors Per Neuron

Every neuron i has two associated vectors, both living in R^D:

- **W_i** — the weight/position of the neuron. Where it lives in abstract D-dimensional space. Think of it as the atom's coordinates in the crystal.
- **A_i** — the activation of the neuron. The current state of the neuron for a given input. Think of it as the intensity of the light field at that atom's location.

Stored as matrices across all N neurons:
```
W : shape [N, D]   # neuron positions (partially learned, partially fixed)
A : shape [N, D]   # neuron activations (recomputed every forward pass)
```

**Why D-dimensional vectors rather than scalars?**
Standard neurons have a scalar activation (a single number). Here, each neuron has a D-dimensional vector for both its position and its activation. This serves two purposes:
- It gives the neuron a **geometric identity** in a shared space, which makes proximity-based routing possible
- It makes the space **rich enough** to represent fine-grained distinctions between neuron roles, especially with small N

D is a hyperparameter you choose freely (e.g., 32, 64, 128). Crucially, D does not need to match the transformer hidden dimension or the number of output classes. This decoupling is intentional and explained in Section 3.6.

### 3.2 Three Types of Neurons

Out of N total neurons:

- **Input neurons** (N_in): activations are set directly from the input signal. Their positions W are fixed after initialization and do not change during training.

  *Why fix input neuron positions?* They are anchors. The rest of the network's geometry is learned relative to where the inputs live. If input positions were also learned, the whole space could rotate arbitrarily and there would be no stable reference frame.

- **Output neurons** (N_out): their final activations are read out as the network's prediction. Their positions are initialized from data (K-means, explained in Section 6) and can be learned.

  *Why allow output positions to be learned?* The output neurons need to be reachable from the intermediate computation. If their positions were fixed, the network might learn a geometry where activations never pass near the output neurons, making readout poor. Letting them move allows them to drift toward the regions where the network's computation naturally terminates.

- **Hidden neurons** (N - N_in - N_out): fully learnable positions. The internal computation fabric.

### 3.3 The Static Connectivity Matrix (The Crystal Lattice)

```
C : shape [N, N]    # C[i][j] = connection weight from neuron i to neuron j
```

C is a **sparse directed matrix** — most entries are zero, meaning no connection exists. The nonzero entries are learned weights.

**Why a directed graph?** Information in neural computation is generally asymmetric — neuron A influencing neuron B does not imply the reverse. Directed connections allow the network to learn asymmetric information flows, which is expressive.

**Why random sparse initialization?** *(v1 placeholder)*
In v1, the sparsity pattern (which entries are zero vs. nonzero) is set randomly at initialization and kept fixed throughout training. Only the values of existing connections are learned.

This is a deliberate simplification. Random sparsity is easy to implement and lets us verify the core ideas before adding complexity. The intended future design is **learned sparsity**: the pattern of connections should itself be learnable and change during training. Specifically, connections that carry little information should disappear, and new connections should form between neurons that frequently exchange information. This is described in Section 9.

```python
# v1: random sparse init, pattern fixed
C_dense = torch.randn(N, N)
mask = (torch.rand(N, N) > sparsity).float()
mask.fill_diagonal_(0)           # no self-connections
C_values = nn.Parameter(C_dense * mask)
C_mask = mask                    # frozen — only C_values is learned
```

### 3.4 The Recursive Loop (Light Bouncing Through the Crystal)

After loading the input into input neuron activations, we run K iterations:

```
for k in range(K):
    A = normalize( A @ C  +  dynamic_connections(A, W) )
```

Each iteration: each neuron collects signals from all neurons that connect to it via C, plus signals from dynamically connected neighbors (explained next), then normalizes.

**Why a recursive loop instead of stacked layers?**
Stacked layers process information in one direction. A recursive loop allows information from different parts of the network to mix over multiple passes. Neuron B might be influenced by neuron A in iteration 1, then neuron C (which was influenced by A in iteration 1) might influence B in iteration 2 — an indirect interaction that a single pass cannot capture.

**Why K iterations?** *(v1 placeholder)*
K is a fixed hyperparameter in v1, found by grid search or genetic algorithm (non-gradient search, since K is discrete). This is a placeholder. The intended future design is an adaptive stopping criterion — the network decides when activations have settled (converged below some change threshold) and stops early. This is similar to Adaptive Computation Time (Graves, 2016). The cost of a fixed K is that some inputs may need more iterations than others, and a fixed K wastes compute on simple inputs and may be insufficient for complex ones.

**Why layer normalization at each step?**
Without normalization, activations grow or shrink exponentially through the recursive loop (the same problem as vanishing/exploding gradients in RNNs). Layer norm keeps activations in a stable range. This is not a placeholder — normalization at each step is the intended design.

### 3.5 Dynamic Connectivity (The Diffraction Step)

This is the core novel mechanism.

At each recursive iteration, in addition to the static connections in C, we add **temporary pseudo-connections** based on proximity in weight space. Rather than connecting a fixed number of nearest neighbors (which is arbitrary), we connect all neurons within a principled distance threshold r* — the **personal volume radius**.

**Why a distance threshold rather than a fixed top-K?**
Top-K always connects exactly K neighbors regardless of whether they are meaningfully close or not. The 4th nearest neighbor might be very far away and semantically irrelevant. A distance threshold means: only neurons within a genuine neighborhood get connected. The number of connections per neuron becomes variable and input-dependent, which is the right behavior — neurons in dense regions have more neighbors; isolated neurons have fewer.

**Why proximity in weight space, not activation space?**
Finding neighbors of A_i in activation space would route based on which neurons are currently firing similarly — a purely dynamic criterion. Finding neighbors in *weight space* routes based on which neurons' structural identities (W) are close to the current activation value. This is the diffraction analogy: the activation (wave) interacts most strongly with the neurons (atoms) whose structural position resonates with it. It couples the dynamic state (A) to the learned structure (W).

#### Deriving the Threshold: Personal Volume Radius

We need a threshold that is principled, dimension-independent, and guaranteed to stay within the bounded space.

**Setup**: all N neuron positions W live inside a D-dimensional unit hypercube [0, 1]^D, confined there by the boundary repulsion forces described in Section 7. Due to mutual repulsion between neurons, they distribute approximately uniformly inside the space.

**The key insight**: rather than dividing the hypercube volume by N to get personal volume (which leads to a sphere radius that can exceed the box size in high dimensions due to the curse of dimensionality), we use the **inscribed sphere** — the sphere that just touches all walls — as the reference volume.

Why the inscribed sphere? In high dimensions, almost all of the hypercube's volume sits in the corners, regions a sphere cannot reach. The inscribed sphere captures the "usable" central volume. This is not just a mathematical trick — it reflects the actual distribution of neurons under mutual repulsion, which naturally pushes them away from corners.

The inscribed sphere has radius R = box_size / 2. Its volume is:
```
V_inscribed = V_unitball × R^D
```

Personal volume per neuron:
```
V_personal = V_inscribed / N = V_unitball × R^D / N
```

Now solve for r* such that V_sphere(r*) = V_personal:
```
V_unitball × r*^D  =  V_unitball × R^D / N

V_unitball cancels:

r*^D  =  R^D / N

r*  =  R / N^(1/D)
```

**V_unitball drops out entirely.** The threshold is purely a function of R (box size) and N (number of neurons). No gamma functions, no π, no dimension-dependent constants.

**Is r* always within the box?** Yes. Since N ≥ 1, N^(1/D) ≥ 1, therefore r* ≤ R ≤ box_size always. The threshold is guaranteed to be within the confined space in any dimension.

**Expected neighbors per neuron** (by construction):
```
Expected neighbors = N × V_sphere(r*) / V_inscribed
                   = N × r*^D / R^D
                   = N × (R/N^(1/D))^D / R^D
                   = N × (1/N)
                   = 1
```

On average, each neuron has exactly 1 dynamic neighbor. In practice some will have 0, some 2–3, depending on local density. This is the correct sparse regime.

```python
def personal_volume_radius(N, D, box_size=1.0):
    """
    r* = R / N^(1/D)   where R = box_size / 2

    Derivation: use inscribed sphere volume (not hypercube volume) as
    reference. V_unitball cancels when solving V_sphere(r*) = V_inscribed/N.

    Properties:
      - r* <= R always: guaranteed within confined space
      - Expected neighbors per neuron = exactly 1 in any dimension
      - No free hyperparameter: determined entirely by N and D
    """
    R = box_size / 2.0
    return R / (N ** (1.0 / D))
```

#### Dynamic Connectivity Implementation

Connection strength uses a **Gaussian kernel** — smooth decay from maximum at distance 0 to near-zero at r*. This avoids the sharp gradient discontinuity of a binary gate.

```python
def dynamic_connectivity(A, W, N, D, box_size=1.0):
    """
    A        : [N, D] current activations
    W        : [N, D] neuron weight positions
    N, D     : network dimensions (used to compute r*)
    box_size : size of the confining hypercube

    Returns: [N, D] activation contribution from dynamic neighbors

    Threshold r* is computed from N and D — no free hyperparameter.
    Neurons within r* of A_i in W-space receive A_i's signal,
    weighted by Gaussian kernel (closer = stronger).
    """
    r_star = personal_volume_radius(N, D, box_size)

    # Pairwise distances: dists[i][j] = ||A_i - W_j||_2
    dists = torch.cdist(A, W)  # [N, N]

    # Gaussian kernel: strength peaks at 0, decays smoothly to ~0 at r*
    strength = torch.exp(-dists ** 2 / (r_star ** 2 + 1e-8))

    # Hard gate: zero beyond r* (neurons outside range get no signal)
    gate = (dists < r_star).float()
    strength = strength * gate
    strength.fill_diagonal_(0)  # no self-connections

    # Normalize: incoming weights per neuron sum to 1
    strength = strength / (strength.sum(dim=0, keepdim=True) + 1e-8)

    # Propagate: output[j] = weighted sum of activations routing to j
    return torch.einsum('ij,id->jd', strength, A)  # [N, D]
```

**A note on differentiability** *(known limitation)*: the gate `(dists < r_star)` is non-differentiable. The routing decision — which neurons are within range — does not receive gradients. The Gaussian strength values are differentiable, so W learns to move toward activation values that should route to it, but cannot directly learn to position itself outside a neighbor's range. This is accepted in v1. Future directions: Gumbel-softmax for differentiable discrete routing.

**Computational cost** *(v1 placeholder)*: brute-force cdist is O(N² · D) per iteration per sample. For v1 with small N (~512–1024) this is fine. For larger N, the intended replacement is HNSW or FAISS approximate nearest neighbor search, which scales to O(N · log(N) · D).

### 3.6 Reading the Output — Self-Projection

After K recursive iterations, each output neuron i produces a single scalar score:

```
score_i = dot(A_i, W_i) / ||W_i||
```

This is the projection of the activation vector onto the neuron's own weight direction. It measures: *how much does the current activation align with what this neuron represents?*

In the crystal analogy: this is constructive interference at a lattice point — how much does the outgoing wave resonate with the atom's own orientation?

**Why this, and not just the L2 norm of A_i?**
The L2 norm discards all directional information. Two neurons firing strongly in opposite directions would look identical. Self-projection keeps directional information: it asks whether the activation is pointing *toward* this neuron's identity, not just whether it is large.

**Why not a learned linear readout?**
A learned vector r_i per output neuron (`dot(A_i, r_i)`) is more expressive but introduces extra parameters outside the geometric framework. Self-projection uses only quantities already defined in the architecture (A and W), with no extra parameters. This internal consistency is intentional.

**Why does this decouple D from N_out?**
Each output neuron produces exactly one scalar regardless of D. So:
- For 14-class classification: 14 output neurons → 14 scalars → softmax → CrossEntropy
- For FFN distillation (hidden_dim = 768): 768 output neurons → 768 scalars → MSE against FFN output

D can be anything independently. You are not forced to choose D = hidden_dim or D = number_of_classes.

---

## 4. The Full Forward Pass

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SGNNET(nn.Module):
    def __init__(self, N, D, N_in, N_out, sparsity=0.95, K=3, box_size=1.0):
        """
        N        : total number of neurons
        D        : dimension of each neuron (weight and activation space)
        N_in     : number of input neurons (positions fixed after init)
        N_out    : number of output neurons (positions initialized from data, learned)
        sparsity : fraction of zero entries in static connectivity matrix C
        K        : number of recursive iterations (v1: fixed; future: adaptive)
        box_size : side length of the confining hypercube for neuron positions

        Note: no k_dyn hyperparameter — dynamic connectivity threshold r* is
        derived from N and D automatically (r* = box_size/2 / N^(1/D)).
        """
        super().__init__()
        self.N, self.D = N, D
        self.N_in, self.N_out = N_in, N_out
        self.K, self.box_size = K, box_size

        # Neuron weight positions
        # Input neuron gradients are zeroed in the training loop (kept fixed)
        # Hidden and output neuron positions are fully learned
        self.W = nn.Parameter(torch.rand(N, D) * box_size)  # init inside box

        # Static sparse connectivity matrix
        # v1: sparsity pattern fixed at init, only values are learned
        # Future (v2): pattern itself evolves via Hebbian prune-and-grow (Section 9)
        C_init = torch.randn(N, N)
        mask = (torch.rand(N, N) > sparsity).float()
        mask.fill_diagonal_(0)
        self.register_buffer('C_mask', mask)
        self.C_values = nn.Parameter(C_init * mask)

        # Layer norm applied at each recursive step to prevent activation explosion
        self.norm = nn.LayerNorm(D)

    def forward(self, x):
        """
        x      : [batch, N_in, D] — input activations loaded into input neurons
        Returns: [batch, N_out]   — scalar score per output neuron
        """
        batch_size = x.shape[0]

        # Initialize all activations to zero; load input neurons
        A = torch.zeros(batch_size, self.N, self.D, device=x.device)
        A[:, :self.N_in, :] = x

        C = self.C_values * self.C_mask  # enforce sparsity pattern

        for _ in range(self.K):
            static  = torch.einsum('bnd,nm->bmd', A, C)  # [batch, N, D]
            dynamic = torch.stack([
                dynamic_connectivity(A[b], self.W, self.N, self.D, self.box_size)
                for b in range(batch_size)
            ])
            A = F.relu(self.norm(static + dynamic))

        # Read out output neurons (last N_out rows)
        A_out   = A[:, -self.N_out:, :]       # [batch, N_out, D]
        W_out   = self.W[-self.N_out:, :]     # [N_out, D]

        # Self-projection: score_i = dot(A_i, W_i) / ||W_i||
        W_norm  = F.normalize(W_out, dim=-1)
        scores  = (A_out * W_norm.unsqueeze(0)).sum(dim=-1)  # [batch, N_out]

        return scores
```

---

## 5. Training Setup: Knowledge Distillation

Rather than training from scratch, we use **knowledge distillation**. The idea:

1. Take a pretrained transformer (e.g., BERT, LLaMA)
2. For every input, record the input to a specific FFN layer (`x_ffn`) and the output (`y_ffn`)
3. Train SGNNET to map `x_ffn → y_ffn`

**Why distillation instead of training from scratch?**
Training from scratch requires a full training pipeline, a large dataset, and many GPU-days. Distillation from a single FFN layer gives us a clean, well-defined supervised signal in hours on a single GPU. It also provides a direct comparison baseline: how close can SGNNET get to the original FFN's behavior, and does it do so with fewer active neurons per forward pass?

Distillation is a research tool for v1. The long-term goal is to train SGNNET end-to-end as part of a full model.

```python
# Step 1: collect distillation data

ffn_inputs, ffn_outputs = [], []

def hook_fn(module, input, output):
    ffn_inputs.append(input[0].detach().cpu())
    ffn_outputs.append(output.detach().cpu())

handle = pretrained_model.transformer.layer[6].ffn.register_forward_hook(hook_fn)
for batch in dataloader:
    pretrained_model(batch['input_ids'])
handle.remove()

X = torch.cat(ffn_inputs)   # [total_tokens, hidden_dim]
Y = torch.cat(ffn_outputs)  # [total_tokens, hidden_dim]

# Step 2: train SGNNET

model = SGNNET(N=512, D=64, N_in=hidden_dim, N_out=hidden_dim, K=3, box_size=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        x_in   = x_batch.unsqueeze(-1).expand(-1, -1, model.D)
        scores = model(x_in)

        loss = total_loss(scores, y_batch, model.W,
                          box_size=model.box_size, N=model.N, D=model.D)

        optimizer.zero_grad()
        loss.backward()
        model.W.grad[:model.N_in] = 0   # keep input neuron positions fixed
        optimizer.step()

        # Hard clamp: keep all positions inside the box
        with torch.no_grad():
            model.W.clamp_(0.0 + 1e-4, model.box_size - 1e-4)
```

---

## 6. Initialization Strategy

**Why initialization matters more here than in standard networks:**
In a standard fully-connected network, every path exists from the start — gradient descent can rearrange representations freely. In SGNNET, most connections are sparse, and dynamic connectivity is proximity-based. If neurons start at random positions unrelated to the data, the dynamic connectivity step finds meaningless neighbors, gradients are weak, and learning is slow. Good initialization seeds the geometry with real data structure.

### 6.1 K-means Initialization

Run K-means clustering on a sample of training data. Use cluster centers as initial neuron positions.

- **Hidden neurons**: K-means on `X` (FFN inputs) — neurons start near actual input distribution
- **Output neurons**: K-means on `Y` (FFN outputs) — output neurons start near output space they must represent

```python
from sklearn.cluster import KMeans

def kmeans_init(data, n_clusters, D):
    """
    data      : [n_samples, D]
    n_clusters: number of neurons to initialize
    """
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    km.fit(data.numpy())
    return torch.tensor(km.cluster_centers_, dtype=torch.float32)

with torch.no_grad():
    model.W[model.N_in : model.N_in + N_hidden] = kmeans_init(X_sample, N_hidden, D)
    model.W[-model.N_out:]                       = kmeans_init(Y_sample, N_out, D)
```

**Why not orthogonal initialization?**
Real data manifolds are not orthogonal. The "formal tone" direction and "legal language" direction in language embedding space are correlated, not perpendicular. Forcing orthogonality would impose a geometric structure that misrepresents the data. K-means respects the actual geometry.

### 6.2 Input Neuron Initialization

Input neuron positions are the coordinate frame the rest of the network is organized around. In v1, they are initialized as evenly spaced anchors inside the box and frozen.

*(Future: learn input neuron positions during a warmup phase, then freeze. This might find a better coordinate frame than a uniform grid.)*

---

## 7. Loss Functions

### 7.1 Task Loss

For distillation:
```python
task_loss = F.mse_loss(scores, y_ffn_batch)
```

For classification:
```python
task_loss = F.cross_entropy(scores, labels)
```

### 7.2 The Safety Valve — Dead-Zone Coulomb Repulsion

**Design principle**: neuron positions should be driven almost entirely by what the training data requires. Coulomb forces exist only as a **safety valve** — they are literally zero during normal operation and activate steeply only when neurons approach a dangerous configuration (two neurons colliding, or a neuron escaping through the boundary).

**Why this is better than always-on Coulomb repulsion:**
An always-on Coulomb term requires a very small λ (e.g., 1e-4) so it does not interfere with data-driven gradients. But then when a collision is actually happening, the signal is too weak to prevent it. With a dead-zone design, λ can be large (0.5–1.0) because the loss is zero most of the time. When it activates, it activates hard. Data gradients are never competed with during normal training.

#### Two Radii, Two Roles

The design uses two distinct thresholds derived from r*:

```
r*        — interaction radius: dynamic connectivity fires (neurons exchange activation)
r* / 2    — repulsion radius:   safety valve fires (neurons get pushed apart)
```

The zone between r*/2 and r* is the **interaction zone** — neurons here are actively communicating via the dynamic connectivity step but are not being repelled. They can come close, exchange information strongly (Gaussian kernel peaks near zero distance), and coexist at this range. Only when they enter the inner r*/2 sphere does the repulsion kick in.

This is analogous to the **bond length** in molecular physics: atoms have an equilibrium separation where they interact maximally before core repulsion pushes them apart. Here r*/2 plays the role of the hard core radius.

**Why r*/2 specifically?**
It gives neurons room to communicate before being repelled — the interaction zone (r*/2 to r*) has meaningful volume. Choosing a smaller repulsion radius (e.g., r*/4) would allow near-collisions before the safety valve fires; choosing a larger one (e.g., r* itself) would repel neurons that should be communicating. The half-radius is the natural midpoint.

**A high-dimensional bonus**: the volume of the repulsion sphere relative to the interaction sphere is (r*/2)^D / r*^D = (1/2)^D. For D=16 this is ~1/65000. The repulsion zone contains almost none of the interaction zone's volume, so the safety valve fires extremely rarely in high dimensions — exactly the behavior we want.

**The potential form**: Coulomb-like 1/distance repulsion from each wall, treating each face of the confining cube as a **virtual point charge**. This gives distance-dependent repulsion — the closer a neuron to a wall, the stronger the force. Each wall is treated independently (not as a net field), so opposite walls do not cancel. The boundary repulsion radius is also r*/2, consistent with the mutual repulsion radius.

```python
def safety_valve_loss(W, box_size=1.0, N=None, D=None):
    """
    Dead-zone Coulomb repulsion. Exactly zero during normal operation.
    Activates steeply only when neurons enter a danger zone.

    Two thresholds, both derived from r* = (box_size/2) / N^(1/D):

      r*      — interaction radius (dynamic connectivity threshold)
      r* / 2  — repulsion radius   (safety valve activation threshold)

    The zone between r*/2 and r* is the interaction zone: neurons here
    communicate strongly via dynamic connectivity but are not repelled.
    This allows neurons to come close and exchange information without
    collapsing onto each other — analogous to bond length in chemistry.

    No free hyperparameter: both thresholds derived from N and D.
    lambda can be large (0.5-1.0) because this loss is zero in normal operation.

    Wall repulsion: each cube face is a virtual point charge (Coulomb,
    distance-dependent). Each wall is independent — no field cancellation.
    """
    R        = box_size / 2.0
    r_star   = R / (N ** (1.0 / D))
    r_repel  = r_star / 2.0          # repulsion activates at half the interaction radius

    # --- Mutual repulsion: activates when two neurons are within r_repel ---
    dists   = torch.cdist(W, W)                                 # [N, N]
    mask    = ~torch.eye(W.shape[0], dtype=torch.bool, device=W.device)
    d_pairs = dists[mask].clamp(min=1e-8)
    # relu(1/d - 1/r_repel): exactly 0 for d > r_repel, grows steeply below
    mutual  = F.relu(1.0/d_pairs - 1.0/r_repel).mean()

    # --- Boundary repulsion: activates when neuron is within r_repel of any wall ---
    # Each wall treated independently as a virtual point charge (distance-dependent)
    dist_lower = W.clamp(min=1e-8)                              # [N, D] dist to wall at 0
    dist_upper = (box_size - W).clamp(min=1e-8)                 # [N, D] dist to wall at L
    d_wall     = torch.minimum(dist_lower, dist_upper)
    boundary   = F.relu(1.0/d_wall - 1.0/r_repel).mean()

    return mutual + boundary


def total_loss(scores, targets, W, box_size=1.0, N=None, D=None,
               lambda_safety=0.5, lambda_lb=0.01):
    """
    Task loss dominates. Safety valve is zero unless constraint violated.
    lambda_safety can be large because it rarely fires.
    """
    task   = F.mse_loss(scores, targets)
    safety = safety_valve_loss(W, box_size=box_size, N=N, D=D)
    return task + lambda_safety * safety
```

**What the two radii look like:**

```
         r_repel = r*/2       r* (interaction radius)
              |                |
Repulsion     |                |
strength      |                |
    ^         |                |
    |  \      |                |
    |   \     |                |
    |    \____|________________|______ 0 (inactive beyond r*)
    +----+----+----------------+-----> distance between neurons
         r*/2                  r*

  [repulsion] [  interaction zone  ] [  no interaction  ]
              neurons communicate     neurons are independent
              but are NOT repelled
```

### 7.3 Load Balancing (Anti-Dead-Neuron)

In any sparse routing system, some neurons may never be selected by the dynamic connectivity step. These neurons receive no gradient signal and contribute nothing. Load balancing penalizes high variance in selection frequency.

```python
def load_balance_loss(selection_counts):
    """
    selection_counts : [N] — how often each neuron was selected this batch
    Minimize variance = encourage all neurons to participate.
    """
    freq = selection_counts.float() / selection_counts.sum()
    return freq.var()
```

*(Track selection counts by recording which neurons fall within r* during dynamic_connectivity.)*

---

## 8. Self-Projection and Model Collapse — A Detailed Analysis

The self-projection readout:
```
score_i = dot(A_i, W_i) / ||W_i||
```

The gradient on W_i pushes it to align with A_i. The gradient on A_i (through the recursive loop) pushes it toward W_i. Left unconstrained, they converge toward each other.

**For classification (14 classes)**: each output neuron has a different class to serve. Class 1 examples pull W_1 toward class-1 activations. Class 2 examples pull W_2 toward its own manifold. Task diversity is the natural anti-collapse mechanism. Output collapse is unlikely.

**For distillation**: each output neuron represents one dimension of the FFN output vector. Different dimensions capture different statistical patterns. Again, task diversity prevents collapse.

**Where collapse IS a real risk**: hidden neurons have no direct task assignment. Nothing in the task loss prevents two hidden neurons from converging to the same position. The safety valve's mutual repulsion term handles this — it activates as soon as two neurons come within d_danger of each other.

**Magnitude explosion**: the self-projection score increases if either A_i or W_i grows in magnitude. The denominator `||W_i||` stabilizes W, but not A. Layer normalization at each recursive step prevents A from growing unboundedly.

---

## 9. Roadmap: Dynamic Topology (The Next Major Feature)

In v1, the sparsity pattern of C is fixed at initialization. Only connection values are learned. This is a deliberate placeholder.

The intended v2 feature is **dynamic topology**: the pattern itself changes during training based on information flow statistics. Connections that carry no information disappear. New connections form between neurons that frequently exchange information.

**The Hebbian principle**: in neuroscience, "neurons that fire together, wire together." We adapt this directly: if neuron i's activation repeatedly routes to neuron j's weight position via the dynamic connectivity step, they should develop a permanent static connection. Conversely, static connections with small weights and low co-activation should be pruned.

```python
def update_topology(model, co_activation_counts,
                    prune_fraction=0.05, grow_fraction=0.05):
    """
    Run every M training steps.

    co_activation_counts : [N, N] — how often neuron i dynamically
                           selected neuron j as a neighbor across training steps.
    Prunes weakest static connections and grows new ones where
    dynamic co-activation is consistently high.
    """
    C = model.C_values.data * model.C_mask

    # Prune weakest existing connections
    existing  = model.C_mask.bool()
    scores    = C.abs()
    n_prune   = int(existing.sum() * prune_fraction)
    threshold = scores[existing].kthvalue(n_prune).values
    prune_mask = (scores < threshold) & existing
    model.C_mask[prune_mask]        = 0
    model.C_values.data[prune_mask] = 0

    # Grow new connections where dynamic co-activation is consistently high
    absent     = ~existing
    n_grow     = int(absent.sum() * grow_fraction)
    if n_grow > 0:
        grow_scores = co_activation_counts[absent]
        threshold   = grow_scores.kthvalue(
            max(1, grow_scores.numel() - n_grow)
        ).values
        grow_mask = (co_activation_counts > threshold) & absent
        model.C_mask[grow_mask] = 1
        model.C_values.data[grow_mask] = torch.randn_like(
            model.C_values.data[grow_mask]
        ) * 0.01
```

This creates a network where topology **emerges from information flow statistics**. The crystal lattice restructures itself based on which paths the light actually travels.

---

## 10. Known Open Problems

### 10.1 Routing is Not Fully Differentiable
The gate `(dists < r*)` is non-differentiable. The routing decision — which neurons are within range — does not receive gradients. The network learns how strongly to connect but not who to connect to. Potential directions: Gumbel-softmax, straight-through estimators.

### 10.2 Relationship to Attention
Dynamic connectivity based on proximity in embedding space is mechanistically similar to self-attention. The key differences are hard sparsity (vs. soft weighted sum) and the recursive loop (vs. single pass). Worth investigating: does SGNNET with soft routing and K=1 reduce to a known attention variant? The answer determines what is genuinely novel.

### 10.3 Information Path to Output Neurons
With sparse random connectivity, there is no guarantee that activations reach output neurons within K iterations early in training. One mitigation: guarantee a shortest path from input to output by construction in the initial sparsity pattern (a fixed backbone, with random additional connections).

### 10.4 Sensitivity to K
We do not know how sensitive performance is to K. Systematic ablation (K = 1, 3, 5, 10) is needed before choosing a default.

### 10.5 Variable-Length Input
The current design assumes fixed N_in. If SGNNET replaces an FFN layer, the input is one token at a time (fixed-size), so this is fine for the current scope. Future sequence-level versions would need a mechanism for variable N_in.

---

## 11. Quick Reference — Hyperparameters

| Hyperparameter | What it controls | Starting value |
|---|---|---|
| N | Total neurons | 512 |
| D | Neuron embedding dimension | 64 |
| N_in | Input neurons | = transformer hidden dim (e.g., 768) |
| N_out | Output neurons | = transformer hidden dim (e.g., 768) |
| K | Recursive iterations (v1: fixed) | 3 |
| box_size | Side length of confining hypercube | 1.0 |
| sparsity | Fraction of zeros in C | 0.95 |
| λ_safety | Safety valve loss weight (can be large — usually zero) | 0.5 |
| λ_lb | Load balance loss weight | 0.01 |
| M | Topology update interval (v2) | every 500 steps |

Note: **r* and r_repel are not hyperparameters** — they are derived automatically:
- `r* = (box_size/2) / N^(1/D)` — interaction radius (dynamic connectivity threshold)
- `r_repel = r* / 2` — repulsion radius (safety valve activation threshold)

---

## 12. Suggested Implementation Roadmap

### Phase 1 — Core Verification
- [ ] Implement SGNNET with static C + threshold-based dynamic connectivity + K iterations
- [ ] Verify forward pass produces valid gradients (`loss.backward()` runs cleanly)
- [ ] Visualize W positions before and after training on a toy dataset — do neurons move toward data clusters?
- [ ] Verify load balancing: plot neuron selection frequency histogram
- [ ] Verify safety valve: confirm it is zero during normal training, fires only on near-collisions

### Phase 2 — Distillation Experiment
- [ ] Extract FFN (input, output) pairs from BERT-base layer 6
- [ ] Train SGNNET to replicate that FFN layer; track MSE vs. steps
- [ ] Baseline comparison: MLP with similar parameter count, same training data
- [ ] Measure active neurons per forward pass (effective sparsity)

### Phase 3 — Ablation Studies
- [ ] Static C only vs. dynamic connectivity only vs. both
- [ ] K = 1, 3, 5, 10 — how does performance and compute scale?
- [ ] K-means init vs. random init — does init quality matter?
- [ ] Safety valve fires: how often, and does it prevent collapse?

### Phase 4 — Dynamic Topology (v2)
- [ ] Implement co-activation tracking in the dynamic connectivity step
- [ ] Implement prune-and-grow topology update
- [ ] Visualize how the connectivity graph changes over training
- [ ] Compare final topology to initial random topology — how much restructuring occurs?

---

## 13. Useful References

- **Mixture of Experts**: Shazeer et al. (2017) — "Outrageously Large Neural Networks" — closest existing approach to neuron-level routing
- **Center Loss**: Wen et al. (2016) — "A Discriminative Feature Learning Approach for Deep Face Recognition" — pulling activations toward class centers; implicit in self-projection
- **Adaptive Computation Time**: Graves (2016) — principled approach to adaptive K
- **Self-Organizing Maps**: Kohonen (1990) — neurons self-organizing spatial positions from input distribution; closest precedent to W self-organization
- **Radial Basis Function Networks**: Powell (1987) — neurons with positions in a shared space, activation based on proximity; conceptually similar to dynamic connectivity
- **Hebbian Learning**: Hebb (1949) — "The Organization of Behavior" — biological principle behind dynamic topology in v2
- **Thomson's Problem**: equilibrium positions of N charged particles under mutual repulsion; directly analogous to neuron position equilibrium under the safety valve
- **Reformer**: Kitaev et al. (2020) — locality-sensitive hashing for efficient approximate nearest-neighbor attention; relevant to scaling dynamic connectivity

---

*This is an early-stage research hypothesis. Nothing described here has been experimentally validated. The purpose of this document is to align understanding and enable independent exploration and implementation.*
