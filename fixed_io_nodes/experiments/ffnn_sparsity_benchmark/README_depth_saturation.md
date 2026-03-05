# Depth Saturation Experiment — Dry Run & Beginner's Guide

**Script:** `run_experiment_depth_saturation.py`
**Question being answered:** *If you keep the total number of neurons and the sparsity fixed, but split those neurons across more and more hidden layers — at what depth does the network stop learning and saturate?*

---

## Table of Contents

1. [What Is This Experiment?](#1-what-is-this-experiment)
2. [Core Concepts You Need First](#2-core-concepts-you-need-first)
   - 2a. What is a Hidden Layer?
   - 2b. What is Sparsity?
   - 2c. What is Node-Level Masking?
3. [The Architecture: Variable-Depth FFNN](#3-the-architecture-variable-depth-ffnn)
4. [Toy Dry Run — Step by Step](#4-toy-dry-run--step-by-step)
   - Step 1: Define the toy problem
   - Step 2: Build the network for 2 layers
   - Step 3: Create the node-level masks
   - Step 4: Apply the masks to weights and biases
   - Step 5: Run the forward pass with real numbers
   - Step 6: Now increase depth to 3 layers
   - Step 7: Extreme depth — 6 layers × 1 node each
5. [Real Experiment: 240 Nodes, 90% Sparsity](#5-real-experiment-240-nodes-90-sparsity)
6. [What Happens Layer by Layer as Depth Grows](#6-what-happens-layer-by-layer-as-depth-grows)
7. [Why Saturation Happens](#7-why-saturation-happens)
8. [Reading the Output Files](#8-reading-the-output-files)

---

## 1. What Is This Experiment?

Imagine you have exactly **240 neurons** and a **90% sparsity constraint** (meaning 90% of neurons in every hidden layer are switched off). You cannot change either of these.

The only thing you can change is **how many hidden layers** you use.

| Configuration     | Neurons per layer | Active neurons per layer (at 90% sparsity) |
|-------------------|------------------:|-------------------------------------------:|
| 2 layers × 120    | 120               | 12                                         |
| 4 layers × 60     | 60                | 6                                          |
| 8 layers × 30     | 30                | 3                                          |
| 20 layers × 12    | 12                | 1                                          |
| 40 layers × 6     | 6                 | 0 (fully dead layer!)                      |

As you add layers, each layer becomes thinner. At some point, the layers are so thin that after 90% masking, almost nothing survives. The network hits a wall — this is the **saturation point**.

---

## 2. Core Concepts You Need First

### 2a. What is a Hidden Layer?

A feedforward neural network passes data through a sequence of transformations:

```
Input → [Hidden Layer 1] → [Hidden Layer 2] → … → Output
```

Each hidden layer does this:

```
output = ReLU( W · input + b )
```

Where:
- `W` is a **weight matrix** — each row is one neuron's incoming connections
- `b` is a **bias vector** — one value per neuron
- `ReLU` clips negative values to zero: `ReLU(x) = max(0, x)`

### 2b. What is Sparsity?

Sparsity = the fraction of neurons you **turn off** (deactivate completely).

```
sparsity = 0.0  →  0% of neurons are off  →  full network
sparsity = 0.5  →  50% of neurons are off →  half the network
sparsity = 0.9  →  90% of neurons are off →  only 10% survive
```

Number of dead neurons per layer:

```
n_dead = floor(sparsity × layer_size)
```

Example: layer_size=30, sparsity=0.9 → n_dead = floor(27) = 27 → only **3 neurons active**.

### 2c. What is Node-Level Masking?

"Node-level" means an entire neuron is either **fully ON** or **fully OFF**.

When a neuron is turned off:
- **ALL its incoming weight connections** are set to zero
- **Its bias** is also set to zero

This is different from weight-level sparsity (where individual weights inside a neuron can be zeroed independently). Here, a neuron is atomic — everything connected to it lives or dies together.

Visually for a layer with 4 neurons where neurons 2 and 4 are dead:

```
         Input features
         x1   x2   x3
          │    │    │
    ┌─────┼────┼────┼─────┐  ← Weight matrix (4 rows × 3 cols)
    │  w  │  w │  w │  b  │  neuron 1 → ALIVE  ✓
    │  0  │  0 │  0 │  0  │  neuron 2 → DEAD   ✗  (all zeros)
    │  w  │  w │  w │  b  │  neuron 3 → ALIVE  ✓
    │  0  │  0 │  0 │  0  │  neuron 4 → DEAD   ✗  (all zeros)
    └─────┴────┴────┴─────┘
```

The mask that does this looks like:

```
weight_mask = [[1, 1, 1],   ← neuron 1 alive
               [0, 0, 0],   ← neuron 2 dead
               [1, 1, 1],   ← neuron 3 alive
               [0, 0, 0]]   ← neuron 4 dead

bias_mask   = [1, 0, 1, 0]
```

---

## 3. The Architecture: Variable-Depth FFNN

```
                   ┌─────────────────────────────────────────────────┐
                   │  SparseFFNNDepth                                 │
                   │                                                  │
  Input (dim=11)   │   hidden[0]   hidden[1]   …   hidden[N-1]       │  Output (dim=11)
  ──────────────►  │  ──────────► ──────────► ─── ──────────────►    │  ──────────────►
  wine features    │  ReLU+mask   ReLU+mask       ReLU+mask           │  logits (no mask)
                   │                                                  │
                   │  layer_size = total_neurons // num_layers        │
                   └─────────────────────────────────────────────────┘
```

**Key rule:** Every hidden layer has the same width: `layer_size = 240 // num_layers`

The output layer is **never masked** — all 11 class logits always stay active.

---

## 4. Toy Dry Run — Step by Step

We will use a simplified toy problem to trace every number through the model.

### Step 1: Define the Toy Problem

```
input_dim    = 3    (3 input features instead of 11)
total_neurons = 6   (6 total hidden neurons instead of 240)
output_dim   = 2    (2 classes instead of 11)
sparsity     = 0.5  (50% of neurons die — easier to see than 90%)
```

We will try **2 configurations** of depth:
- Config A: 2 layers × 3 neurons each
- Config B: 3 layers × 2 neurons each

---

### Step 2: Build the Network for 2 Layers (Config A)

```
layer_size = total_neurons // num_layers = 6 // 2 = 3
```

The network has these linear layers:

```
hidden[0]:  Linear(in=3, out=3)   ← takes the 3 input features
hidden[1]:  Linear(in=3, out=3)   ← takes output of hidden[0]
output:     Linear(in=3, out=2)   ← no mask, always full
```

Each `Linear(in, out)` contains:
- A weight matrix `W` of shape `(out, in)` — one row per neuron
- A bias vector `b` of shape `(out,)` — one value per neuron

---

### Step 3: Create the Node-Level Masks (Config A, sparsity=0.5)

```python
n_dead = floor(0.5 × 3) = floor(1.5) = 1
```

So **1 out of 3 neurons dies** in each hidden layer.

Suppose the random generator picks neuron index 2 to die in hidden[0], and neuron index 0 to die in hidden[1]:

**Mask for hidden[0]** (3 neurons, input from 3 features):

```
node_active = [1, 1, 0]   ← neuron 0 alive, neuron 1 alive, neuron 2 dead

weight_mask[0] = [[1, 1, 1],   ← neuron 0: all incoming weights kept
                  [1, 1, 1],   ← neuron 1: all incoming weights kept
                  [0, 0, 0]]   ← neuron 2: ALL incoming weights zeroed

bias_mask[0]   = [1, 1, 0]     ← neuron 2's bias also zeroed
```

**Mask for hidden[1]** (3 neurons, input from 3 features — from previous layer's output):

```
node_active = [0, 1, 1]   ← neuron 0 dead, neuron 1 alive, neuron 2 alive

weight_mask[1] = [[0, 0, 0],   ← neuron 0: dead
                  [1, 1, 1],   ← neuron 1: alive
                  [1, 1, 1]]   ← neuron 2: alive

bias_mask[1]   = [0, 1, 1]
```

**Mask for output layer** (always all ones — no sparsity):

```
weight_mask_out = [[1, 1, 1],
                   [1, 1, 1]]

bias_mask_out   = [1, 1]
```

---

### Step 4: Apply the Masks to Weights and Biases

Suppose the actual learned weights in `hidden[0]` are (random initialization):

```
W[0] (raw) = [[ 0.5, -0.3,  0.8],   ← neuron 0
              [-0.2,  0.6, -0.1],   ← neuron 1
              [ 0.4,  0.9, -0.7]]   ← neuron 2

b[0] (raw) = [0.1, -0.2, 0.3]
```

After multiplying element-wise with the mask:

```
W[0] × weight_mask[0]
    = [[ 0.5, -0.3,  0.8],   ← kept (neuron 0 alive)
       [-0.2,  0.6, -0.1],   ← kept (neuron 1 alive)
       [ 0.0,  0.0,  0.0]]   ← zeroed (neuron 2 dead)

b[0] × bias_mask[0]
    = [0.1, -0.2, 0.0]       ← neuron 2's bias also zeroed
```

This is what actually gets used in the computation — the original weights still exist in memory but are multiplied by zero on every forward pass, so they have no effect.

---

### Step 5: Run the Forward Pass With Real Numbers (Config A)

Let our input sample be:

```
x = [1.2, -0.5, 0.8]   ← one wine sample (3 features)
```

**Layer 0 forward:**

```
pre_activation[0] = W_masked[0] · x + b_masked[0]

  neuron 0:  0.5×1.2 + (-0.3)×(-0.5) + 0.8×0.8 + 0.1
           = 0.60 + 0.15 + 0.64 + 0.10 = 1.49

  neuron 1:  (-0.2)×1.2 + 0.6×(-0.5) + (-0.1)×0.8 + (-0.2)
           = -0.24 - 0.30 - 0.08 - 0.20 = -0.82

  neuron 2:  0×1.2 + 0×(-0.5) + 0×0.8 + 0.0   ← all zeros because masked
           = 0.00

After ReLU (clip negatives to 0):
  h[0] = [max(0, 1.49), max(0, -0.82), max(0, 0.00)]
       = [1.49, 0.00, 0.00]
                ↑           ↑
          neuron 1       neuron 2 dead
          goes to 0       (masked out)
          naturally
```

Notice that neuron 2's output is **forced to 0** by the mask. Neuron 1 also happened to output 0 because its pre-activation was negative — but that's natural (ReLU), not masking.

**Layer 1 forward:**

```
h[0] = [1.49, 0.00, 0.00]   ← this flows into layer 1

Suppose W[1] (raw, after mask) looks like:
  Masked W[1] = [[ 0.0,  0.0,  0.0],   ← neuron 0 dead (masked)
                 [ 0.3, -0.4,  0.7],   ← neuron 1 alive
                 [-0.5,  0.2,  0.1]]   ← neuron 2 alive
  b[1] (masked) = [0.0, 0.15, -0.05]

  neuron 0:  0 + 0 + 0 + 0 = 0.00  (dead, forced by mask)

  neuron 1:  0.3×1.49 + (-0.4)×0.00 + 0.7×0.00 + 0.15
           = 0.447 + 0 + 0 + 0.15 = 0.597

  neuron 2:  (-0.5)×1.49 + 0.2×0.00 + 0.1×0.00 + (-0.05)
           = -0.745 + 0 + 0 - 0.05 = -0.795

After ReLU:
  h[1] = [0.00, 0.597, 0.00]
```

**Output layer forward (no mask):**

```
h[1] = [0.00, 0.597, 0.00]

Suppose output weights W_out and biases b_out are:
  W_out = [[0.4, -0.3, 0.6],
           [-0.2,  0.5, 0.3]]
  b_out = [0.05, -0.1]

  class 0:  0.4×0 + (-0.3)×0.597 + 0.6×0 + 0.05 = -0.179 + 0.05 = -0.129
  class 1: -0.2×0 +  0.5×0.597  + 0.3×0 + (-0.1) =  0.299 - 0.10 =  0.199

logits = [-0.129, 0.199]
→ predicted class = argmax = class 1
```

The loss (cross-entropy) is computed from these logits, backpropagation updates the weights, but the masks are **fixed** — they never change during training.

---

### Step 6: Now Increase Depth to 3 Layers (Config B)

```
layer_size = 6 // 3 = 2   ← only 2 neurons per layer now!
n_dead = floor(0.5 × 2) = 1   ← 1 out of 2 neurons dies per layer
active neurons per layer = 1
```

Network structure:

```
hidden[0]:  Linear(in=3, out=2)
hidden[1]:  Linear(in=2, out=2)
hidden[2]:  Linear(in=2, out=2)
output:     Linear(in=2, out=2)  ← no mask
```

Mask for each hidden layer (1 neuron dies):

```
hidden[0]:  node_active = [1, 0]   ← neuron 0 alive, neuron 1 dead
  weight_mask = [[1, 1, 1],
                 [0, 0, 0]]
  bias_mask   = [1, 0]

hidden[1]:  node_active = [0, 1]   ← neuron 0 dead, neuron 1 alive
  weight_mask = [[0, 0],
                 [1, 1]]
  bias_mask   = [0, 1]

hidden[2]:  node_active = [1, 0]   ← neuron 0 alive, neuron 1 dead
  weight_mask = [[1, 1],
                 [0, 0]]
  bias_mask   = [1, 0]
```

Forward pass (same input x = [1.2, -0.5, 0.8]):

```
Layer 0:
  neuron 0 (alive):   w·x + b  → some value → ReLU → h0[0] ∈ ℝ≥0
  neuron 1 (dead):    0         → ReLU(0) = 0
  h[0] = [h0[0], 0]

Layer 1:
  neuron 0 (dead):   0         → 0
  neuron 1 (alive):  w·h[0] + b → ReLU → h1[1] ∈ ℝ≥0
  h[1] = [0, h1[1]]

Layer 2:
  neuron 0 (alive):  w·h[1] + b → ReLU → h2[0] ∈ ℝ≥0
  neuron 1 (dead):   0          → 0
  h[2] = [h2[0], 0]
```

Diagram of the information flow:

```
Input:  [1.2, -0.5, 0.8]
           │
           ▼
Layer 0: [ALIVE, DEAD]  →  only 1 value passes through
           │
           ▼
Layer 1: [DEAD, ALIVE]  →  the single value is relayed (or dies again via ReLU)
           │
           ▼
Layer 2: [ALIVE, DEAD]  →  same situation
           │
           ▼
Output:  [logit_0, logit_1]
```

Each layer has only **one active neuron** acting as a single wire. The network is now essentially a **chain of scalar multiplications** — very little representational capacity.

---

### Step 7: Extreme Depth — 6 Layers × 1 Neuron Each (Edge Case)

```
layer_size = 6 // 6 = 1
n_dead = floor(0.5 × 1) = floor(0.5) = 0   ← INTEGER FLOOR: 0 neurons die!
```

Even at 50% sparsity, with only 1 neuron per layer, **no neuron ever dies** (you cannot kill half a neuron — floor rounds down to 0).

This is an important edge case in the real experiment. At 90% sparsity:
```
layer_size = 2 → n_dead = floor(0.9 × 2) = 1 → 1 neuron active
layer_size = 1 → n_dead = floor(0.9 × 1) = 0 → 1 neuron active (nothing to kill)
```

So very thin layers paradoxically "survive" sparsity — but with only 1 active neuron per layer, the capacity is near zero regardless.

---

## 5. Real Experiment: 240 Nodes, 90% Sparsity

Now apply the same logic at the actual experiment scale:

```
total_neurons = 240
sparsity      = 0.90
input_dim     = 11   (UCI Wine Quality features)
output_dim    = 11   (quality scores 0–10)
```

| num_layers | layer_size | n_dead per layer | active per layer | total active |
|:----------:|:----------:|:----------------:|:----------------:|:------------:|
|     2      |    120     |       108        |        12        |      24      |
|     4      |     60     |        54        |         6        |      24      |
|     8      |     30     |        27        |         3        |      24      |
|    10      |     24     |        21        |         3        |      30      |
|    12      |     20     |        18        |         2        |      24      |
|    20      |     12     |        10        |         2        |      40      |
|    24      |     10     |         9        |         1        |      24      |
|    40      |      6     |         5        |         1        |      40      |
|    60      |      4     |         3        |         1        |      60      |
|    80      |      3     |         2        |         1        |      80      |
|   120      |      2     |         1        |         1        |     120      |

> **Note:** total_active = active_per_layer × num_layers. Even though total active *count* may be similar, the arrangement changes — deeper networks have fewer active neurons per layer but more layers to compose through.

---

## 6. What Happens Layer by Layer as Depth Grows

### Shallow (2 layers × 120 nodes):

```
Input (11)
   │
   ▼
┌──────────────────────────────────────────────────────┐
│  Layer 0:  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  12 alive / 120
└──────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────┐
│  Layer 1:  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  12 alive / 120
└──────────────────────────────────────────────────────┘
   │
   ▼
Output (11) — no mask
```

12 active neurons per layer gives reasonable capacity. The network can form complex non-linear combinations.

### Deep (24 layers × 10 nodes):

```
Input (11)
   │   ▼  Layer 0:  █░░░░░░░░░  1 alive / 10
   │   ▼  Layer 1:  █░░░░░░░░░  1 alive / 10
   │   ▼  Layer 2:  █░░░░░░░░░  1 alive / 10
   │   ▼  ...
   │   ▼  Layer 23: █░░░░░░░░░  1 alive / 10
   │
   ▼
Output (11) — no mask
```

Each layer has only 1 active neuron. The signal is passed through a long chain of single neurons — like telephone game through 24 relays. High risk of vanishing gradient and information bottleneck.

### Visualizing the squeeze:

```
Total active neurons per layer at 90% sparsity:

 2 layers:  ████████████  (12 active neurons per layer)
 4 layers:  ██████        (6)
 8 layers:  ███           (3)
12 layers:  ██            (2)
24 layers:  █             (1) ← information bottleneck
40 layers:  █             (1)
120 layers: █             (1)
            └────────────── the width of each layer in the signal path
```

---

## 7. Why Saturation Happens

There are three independent failure modes that kick in as depth grows:

### Failure Mode 1 — Capacity Collapse

Each active neuron computes: `output = ReLU(w₁a₁ + w₂a₂ + … + wₙaₙ + b)`

With only 1 active input from the previous layer:

```
output = ReLU(w₁ × a₁ + b)   ← just a scaled version of the input, plus a bias
```

This is a linear transform followed by ReLU — equivalent to a single leaky multiplier. No matter how many such layers you stack, the function you can learn is at best:

```
f(x) = ReLU(c₁ · ReLU(c₂ · ReLU(c₃ · x + b₃) + b₂) + b₁)
```

Which is far less expressive than a shallow network with many active neurons combining many features.

### Failure Mode 2 — Gradient Vanishing

During backpropagation, gradients flow backward through every layer. If a neuron is dead (masked to zero), its gradient is also zero — that layer contributes nothing to learning. With 90% of neurons dead, most gradient paths are blocked:

```
Gradient flowing backward:

Output ← Layer N-1 ← Layer N-2 ← … ← Layer 0 ← Input

At each layer, 90% of neurons are masked → gradient passes through
only the 10% alive. With 1 active neuron per layer, 100% of the
gradient must squeeze through a single bottleneck neuron at each step.
```

### Failure Mode 3 — Dead Layer Problem

At extreme depths (e.g., 40 layers × 6 nodes, 90% sparsity):

```
layer_size = 6
n_dead = floor(0.9 × 6) = 5
active = 1 neuron per layer
```

If that 1 active neuron receives only zeros as input from the previous layer (because the previous layer's 1 active neuron output a negative value that ReLU killed), then this layer also outputs all zeros — and so does every subsequent layer:

```
h[k] = ReLU(w × 0 + b) = ReLU(b)   ← only the bias survives
       if b < 0: h[k] = 0 entirely

h[k+1] = ReLU(w × 0 + b) = same problem
```

The network can become a zero-propagation chain — no signal reaches the output.

---

## 8. Reading the Output Files

After running the script, a timestamped folder is created under `runs_depth_saturation/`:

```
runs_depth_saturation/
└── run_YYYYMMDD_HHMMSS/
    ├── results.csv                    ← all metrics, one row per depth config
    ├── findings.txt                   ← human-readable summary with saturation analysis
    ├── saturation_curve.png           ← THE main result: accuracy + active nodes vs depth
    ├── metrics_vs_depth.png           ← 3×3 grid: all metrics vs num_layers
    ├── active_nodes_per_layer.png     ← bar chart per depth config showing alive nodes
    ├── loss_curves.png                ← train/val loss for all 14 configs overlaid
    └── tradeoffs.png                  ← accuracy vs compute / time, coloured by depth
```

**How to identify the saturation point from `results.csv`:**

```
1. Find the row with maximum accuracy → this is the "sweet spot" depth
2. Find the deepest row where accuracy >= 0.95 × max_accuracy → this is the saturation threshold
3. Everything beyond that row = diminishing returns zone
```

**What to look for in `saturation_curve.png`:**

```
Accuracy
  │ ▲
  │  \
  │   \___
  │       \___________  ← flat zone (saturation)
  │                   \
  │                    \___  ← collapse zone
  └──────────────────────────► num_layers
      ↑           ↑
   sweet spot  saturation point
```

The dual-axis plot overlays "total active nodes" on the right axis, so you can directly see the correlation between active capacity and accuracy degradation.

---

*This experiment is part of the FFNN Sparsity Benchmark series.*
*See also: `run_experiment_bias_weight_mask.py` (single-layer sweep) and `run_experiment_2layer_bias_weight_mask.py` (two-layer sweep).*
