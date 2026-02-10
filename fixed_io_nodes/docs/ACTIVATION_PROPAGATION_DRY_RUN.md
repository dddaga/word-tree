# End-to-End Mathematical Dry Run: Forward and Backward Propagation

This document gives a **complete, step-by-step mathematical dry run** of how activation is initialized, propagated, and how gradients flow in the GNN layer. It is written so that anyone can follow the model’s behaviour without reading the code first.

---

## 1. Purpose and Scope

- **Forward:** How input is injected into input nodes, how activation (phase and magnitude) propagates over steps via direct (conduction) and radiation connections, and how the output is read.
- **Backward:** How the loss gradient flows back through the output signal into node activations and node weights (phase_weight, mag_weight), and how that ties to the optimizer.

The run uses a **minimal toy setup** (scalar dimension, small graph) so every number and formula can be written explicitly. The same formulas apply to the full model (vector dimension \(D > 1\), many nodes, radiation, etc.).

---

## 2. Toy Setup

### 2.1 Parameters

- **Vector dimension:** \(D = 1\) (one phase and one magnitude value per node).
- **Gamma:** \(\gamma = 1\).
- **Graph:** 2 input nodes (0, 1), 1 hidden node (2), 1 output node (3).  
  Edges: \(0 \to 2\), \(1 \to 2\), \(2 \to 3\). No radiation in this example.
- **Iterations:** 2 propagation steps after input injection (inject → step 1 → step 2 → read output).
- **Single sample:** one input vector for the two input nodes.

### 2.2 Notation

- \(\phi\) = phase (radians), \(m\) = magnitude.
- Node \(i\) learnable parameters: \(\theta_i\) (phase_weight), \(\mu_i\) (mag_weight).
- **Activation strength** (scalar per node):
  \[
  s = f(\phi, m) = \cos(\phi)\, \exp(\gamma\sin(m)).
  \]
  For \(D > 1\), \(s = \sum_{d=1}^{D} \cos(\phi_d)\, \exp(\gamma\sin(m_d))\).

### 2.3 Concrete Numbers

| Node | Role   | \(\theta\) (phase_weight) | \(\mu\) (mag_weight) |
|------|--------|---------------------------|------------------------|
| 0    | input  | 0.5                       | 0                     |
| 1    | input  | 0.8                       | 0                     |
| 2    | hidden | 0.2                       | 0.1                   |
| 3    | output | 0.0                       | 0                     |

**Input (one sample):** \(x_0 = 0.5\), \(x_1 = 1.0\) (phase values for nodes 0 and 1). Input magnitudes = 0.

---

## 3. Forward Pass (Step by Step)

### 3.1 Initialization (Before Any Input)

When a node is loaded, activation is set from its weights:

- \(\phi_{\mathrm{act}} \leftarrow \theta\), \(\quad m_{\mathrm{act}} \leftarrow \mu\), \(\quad s \leftarrow f(\phi_{\mathrm{act}}, m_{\mathrm{act}})\).

So:

- Node 0: \(\phi_0 = 0.5\), \(m_0 = 0\), \(s_0 = \cos(0.5) \approx 0.8776\).
- Node 1: \(\phi_1 = 0.8\), \(m_1 = 0\), \(s_1 \approx 0.6967\).
- Node 2: \(\phi_2 = 0.2\), \(m_2 = 0.1\), \(s_2 = \cos(0.2)\exp(\sin(0.1)) \approx 1.078\).
- Node 3: \(\phi_3 = 0\), \(m_3 = 0\), \(s_3 = 1\).

### 3.2 Input Injection (Step 0)

Input nodes are updated from the sample: **phase** = given input, **magnitude** = 0, **strength** = \(f(\mathrm{phase}, 0)\).

- **Node 0:** \(\phi_0 \leftarrow 0.5\), \(m_0 \leftarrow 0\), \(s_0 = \cos(0.5) = c_0 \approx 0.8776\).
- **Node 1:** \(\phi_1 \leftarrow 1.0\), \(m_1 \leftarrow 0\), \(s_1 = \cos(1.0) = c_1 \approx 0.5403\).

In the code, injection is done by calling `update_activations` with a single incoming “input” (the data), then appending the node’s current state. After that update, node 0 ends with e.g. \(\phi_0 = 1.0\), \(m_0 = 0\), \(s_0 = c_1\).

**Active set after injection:** \(\{0, 1\}\).

### 3.3 Propagation Step 1

- **Who receives:** Node 2 has incoming from 0 and 1. So \(\mathrm{incoming}(2) = [0, 1]\).
- **Node 2 – update_activations:**
  - Inputs: \(\Phi = [\phi_0, \phi_1, \phi_2]^\top = [1.0, 1.0, 0.2]^\top\), \(M = [0, 0, 0.1]^\top\), \(S = [s_0, s_1, s_2]^\top\).
  - Scaled strengths: \(\tilde{s}_k = s_k / \sqrt{D}\) (here \(D=1\) so unchanged).
  - Weights: \(w = \mathrm{softmax}(\tilde{S})\). Numerically \(w \approx [0.2694, 0.2694, 0.4612]^\top\).
  - Phase update (with node 2’s \(\theta_2 = 0.2\)): \(z_k = \phi_k + \theta_2\); \(\phi_2^{\mathrm{new}} = \sum_k w_k z_k \approx 0.8311\); then \(\phi_2 \leftarrow \mathrm{mod}(0.8311, 2\pi) = 0.8311\).
  - Magnitude update (with \(\mu_2 = 0.1\)): \(y_k = \sin(m_k + \mu_2)\); \(\mathrm{mag\_sum} = \sum_k w_k y_k \approx 0.1454\); \(m_2^{\mathrm{new}} = \arcsin(\mathrm{clamp}(\mathrm{mag\_sum})) \approx 0.1459\).
  - New strength: \(s_2 = \cos(0.8311)\,\exp(\sin(0.1459)) \approx 0.7455\).

**Active set:** \(\{0, 1, 2\}\).

### 3.4 Propagation Step 2

- **Who receives:** Node 3 has incoming from 2 only: \(\mathrm{incoming}(3) = [2]\).
- **Node 3 – update_activations:** Same recipe: stack \((\phi_2, \phi_3)\), \((m_2, m_3)\), \((s_2, s_3)\); append self; softmax on scaled strengths; weighted combination with node 3’s \(\theta_3 = 0\), \(\mu_3 = 0\); phase and magnitude updates; then \(s_3 = f(\phi_3, m_3) \approx 0.9937\).

**Output (readout):** The layer output is the activation strength of the output node(s), optionally scaled. Here \(\mathrm{output} = s_3 \approx 0.9937\) (with \(D=1\), scaling by \(1/\sqrt{D}\) leaves it unchanged).

### 3.5 Loss

Example squared-error loss with target \(y = 1.0\):

\[
\mathcal{L} = \frac{1}{2}(\mathrm{output} - y)^2 = \frac{1}{2}(s_3 - 1)^2 \approx 0.000020.
\]

---

## 4. Backward Pass (Gradients, Step by Step)

We use \(\gamma = 1\) and \(f(\phi, m) = \cos(\phi)\,\exp(\sin(m))\).

### 4.1 Gradient at Output Node 3

- \(\displaystyle\frac{\partial \mathcal{L}}{\partial s_3} = s_3 - 1 \approx -0.0063\).
- \(\displaystyle\frac{\partial s}{\partial \phi} = -\sin(\phi)\,\exp(\sin(m)),\quad \frac{\partial s}{\partial m} = \cos(\phi)\,\exp(\sin(m))\,\cos(m)\).
- So \(\displaystyle\frac{\partial \mathcal{L}}{\partial \phi_3}\) and \(\displaystyle\frac{\partial \mathcal{L}}{\partial m_3}\) are obtained by chain rule from \(\partial\mathcal{L}/\partial s_3\).

Node 3’s \((\phi_3, m_3)\) were produced by **update_activations** from predecessor activations and node 3’s weights \(\theta_3\), \(\mu_3\). So gradients propagate back to:

- Predecessor activations (node 2): \(\partial\mathcal{L}/\partial \phi_2\), \(\partial\mathcal{L}/\partial m_2\).
- Node 3’s parameters: \(\partial\mathcal{L}/\partial \theta_3\), \(\partial\mathcal{L}/\partial \mu_3\) (these are the `.grad` on `phase_weight` and `mag_weight` collected by `get_grads()`).

### 4.2 Gradient at Node 2

Node 2’s \((\phi_2, m_2)\) were produced by its own **update_activations** from nodes 0, 1 and self, with weights \(w\) and node 2’s \(\theta_2\), \(\mu_2\). Backprop through the update gives:

- **Phase:** \(\phi_2 = \sum_k w_k(\phi_k + \theta_2)\) \(\Rightarrow\) \(\displaystyle\frac{\partial \phi_2}{\partial \phi_0} = w_0\), \(\displaystyle\frac{\partial \phi_2}{\partial \phi_1} = w_1\), \(\displaystyle\frac{\partial \phi_2}{\partial \theta_2} = 1\). So \(\partial\mathcal{L}/\partial \phi_0\), \(\partial\mathcal{L}/\partial \phi_1\), and \(\partial\mathcal{L}/\partial \theta_2\) follow by chain rule.
- **Magnitude:** \(m_2 = \arcsin(\sum_k w_k \sin(m_k + \mu_2))\); backprop through this and through \(w\) (softmax) gives \(\partial\mathcal{L}/\partial m_k\) and \(\partial\mathcal{L}/\partial \mu_2\) as needed.

The same idea extends to all nodes: gradients flow through every `update_activations` and `activation_strength_forward` in the graph.

### 4.3 Gradients at Input Nodes and Input Gradient

Input nodes got their \((\phi_0, \phi_1)\) from **input injection** via **update_activations** using the **input** \((x_0, x_1)\). So:

- \(\partial\mathcal{L}/\partial x_0\), \(\partial\mathcal{L}/\partial x_1\) are obtained from \(\partial\mathcal{L}/\partial \phi_0\), \(\partial\mathcal{L}/\partial \phi_1\) and the injection update. These are the gradients passed back to the MLP (or whatever produced the GNN input).
- \(\partial\mathcal{L}/\partial \theta_i\), \(\partial\mathcal{L}/\partial \mu_i\) for all nodes that participated are collected by `get_grads()` and then used by the gradient accumulator / GNNAdam to update the stored GNN parameters.

---

## 5. Summary Table (What Is Computed Where)

| Stage | Action | Main quantities |
|--------|--------|------------------|
| Init | Each node: \(\phi \leftarrow \theta\), \(m \leftarrow \mu\), \(s = f(\phi,m)\) | \(s_i = \cos(\theta_i)\exp(\sin(\mu_i))\) |
| Inject | Input nodes: set \(\phi,m\) from data; then update_activations(incoming=data, append self) | \(w = \mathrm{softmax}(\tilde{s})\), \(\phi_{\mathrm{new}} = \sum w_k(\phi_k+\theta)\), \(m_{\mathrm{new}} = \arcsin(\sum w_k\sin(m_k+\mu))\) |
| Step 1 | Node 2: update_activations from nodes 0, 1 and self | New \(\phi_2\), \(m_2\), \(s_2\) |
| Step 2 | Node 3: update_activations from node 2 and self | New \(\phi_3\), \(m_3\), \(s_3\) |
| Readout | output = \(s_3/\sqrt{D}\) (or stack over output nodes) | Scalar or vector over output nodes |
| Loss | \(\mathcal{L} = \tfrac{1}{2}(\mathrm{output}-y)^2\) | \(\partial\mathcal{L}/\partial s_3 = \mathrm{output}-y\) |
| Backward | Chain rule through \(s = f(\phi,m)\), then through each update_activations (softmax, weighted sums, arcsin) | \(\partial\mathcal{L}/\partial \phi_i\), \(\partial\mathcal{L}/\partial m_i\), \(\partial\mathcal{L}/\partial \theta_i\), \(\partial\mathcal{L}/\partial \mu_i\), \(\partial\mathcal{L}/\partial x\) |

---

## 6. Formula Reference

### 6.1 Activation Strength (Unquantized)

\[
s = \sum_{d=1}^{D} \cos(\phi_d)\, \exp\bigl(\gamma\sin(m_d)\bigr).
\]

### 6.2 Node Update (update_activations)

For a node with \(K\) incoming \((\phi^{(k)}, m^{(k)}, s^{(k)})\) plus its own current state (index \(K+1\)):

1. **Scale strengths:** \(\tilde{s}_k = s_k / \sqrt{D}\), then clamp to \([-20, 20]\).
2. **Weights:** \(w = \mathrm{softmax}(\tilde{s})\).
3. **Phase:** \(\phi_{\mathrm{new}} = \sum_k w_k(\phi_k + \theta)\), then reduce mod \(2\pi\).
4. **Magnitude:** \(m_{\mathrm{new}} = \arcsin\bigl(\mathrm{clamp}\bigl(\sum_k w_k\sin(m_k + \mu),\, -1+\epsilon,\, 1-\epsilon\bigr)\bigr)\).
5. **Strength:** \(s_{\mathrm{new}} = f(\phi_{\mathrm{new}}, m_{\mathrm{new}})\).

### 6.3 Gradients for Activation Strength

\[
\frac{\partial s}{\partial \phi_d} = -\sin(\phi_d)\,\exp(\gamma\sin(m_d)),\qquad
\frac{\partial s}{\partial m_d} = \cos(\phi_d)\,\exp(\gamma\sin(m_d))\,\gamma\cos(m_d).
\]

---

## 7. Relation to the Code

- **Activation strength:** `core/custom_functions.py` — `activation_strength_forward_unquantized` (cos(phase), exp(γ·sin(mag)), sum over \(d\)).
- **Node update:** `core/node.py` — `UnquantizedNode.update_activations` (softmax on scaled strengths, weighted phase/mag combination, arcsin clamp, then `calculate_activation_strength`).
- **Forward flow:** `core/gnn_model.py` — input injection, then each step: radiation targets + direct edges, fetch nodes, build incoming_connections, call `update_activations` for each node, optional decay; readout = activation_strength of output nodes, scaled.
- **Backward:** Same graph is backpropagated via PyTorch autograd; `get_grads()` in `gnn_model.py` collects `phase_weight.grad` and `mag_weight.grad` for active nodes; these are fed to the gradient accumulator and then to the optimizer (e.g. GNNAdam).

This dry run, together with the formulas above, gives a complete mathematical picture of how activation is initialised, how it propagates, and how gradients flow from the loss back to the inputs and node weights.
