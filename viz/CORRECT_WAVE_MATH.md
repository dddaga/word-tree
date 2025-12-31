# Correct Wave Interference Mathematics (Reference Implementation)

## Source
Reference implementation: `/Volumes/T9/work/word-tree/fixed_io_nodes/`
- `node.py`: Node class with wave interference logic
- `custom_functions.py`: Activation strength calculation
- `gnn_model.py`: Graph propagation logic

---

## Core Principle: Wave Propagation Model

The network mimics **optical/quantum wave propagation** where:
1. **Each edge carries a phase shift** (like optical path length)
2. **Signals interfere** at nodes (constructive/destructive interference)
3. **Intensity emerges** from the magnitude of the resultant wave

This is fundamentally different from standard neural networks!

---

## Mathematical Formulas

### 1. Activation Strength (Signal Intensity)

**Formula** (`custom_functions.py:3-28`):
```python
activation_strength = Σ cos(phase_i) × exp(γ × sin(mag_i))
```

**Code**:
```python
def activation_strength_forward(phases, mags, gamma=1.0):
    phase_values = torch.cos(phases)
    mag_exponent = gamma * torch.sin(mags)
    mag_exponent = torch.clamp(mag_exponent, min=-10.0, max=10.0)
    mag_values = torch.exp(mag_exponent)
    signal = (phase_values * mag_values).sum(dim=-1)
    return signal + 1e-8
```

**Breakdown**:
- **Phase Component**: `cos(φ)` projects wave onto real axis
- **Magnitude Component**: `exp(γ × sin(m))` creates non-linear amplitude scaling
  - `γ` (gamma) controls sensitivity (default 1.0)
  - `sin(m)` maps magnitude to [-1, 1] range
  - `exp(...)` makes it always positive, with dynamic range
- **Sum over vector_dim**: Each dimension is treated as a separate wave component

**Physical Interpretation**:
- Each dimension represents a wave oscillating in phase space
- `cos(φ)` gives the real part of `e^(iφ)`
- Magnitude modulates the wave amplitude exponentially
- Total signal = coherent sum across all dimensions

---

### 2. Wave Propagation with Phase Shift

**Formula** (`node.py:139-140`):
```python
phase_arriving = phase_source + phase_shift_edge
mag_arriving = mag_source + mag_shift_edge
```

**Code**:
```python
# node.py:139-140
phase_activations = weights * (phase_activations + phase_weight_clamped.reshape(1, -1))
mag_activations = weights * (mag_activations + mag_weight_clamped.reshape(1, -1))
```

**Key Insight**: 
- **Edge weights ARE phase shifts**, not amplitude scalers!
- When a wave travels through an edge, the edge's `phase_weight` is **added** to the incoming phase
- This mimics optical path length difference: `Δφ = 2π × (path_length / wavelength)`

**Physical Analogy**:
```
Light travels from Node A to Node B through fiber optic cable
→ Optical path introduces phase delay based on cable length/refractive index
→ In the network: phase_B = phase_A + weight_edge
```

---

### 3. Multi-Input Wave Interference (Aggregation)

**Formula** (`node.py:92-152`):
```python
# Step 1: Calculate attention weights from signal intensities
weights_i = softmax(activation_strengths / √d)

# Step 2: Apply weighted phase shifts and sum
phase_result = Σ_i weights_i × (phase_i + phase_weight)
mag_result = Σ_i weights_i × (mag_i + mag_weight)

# Step 3: Wrap phase to [0, 2π]
phase_result = phase_result mod 2π
```

**Code**:
```python
# node.py:119-144
# Concatenate incoming activations with node's own state
phase_activations = torch.cat((phase_activations, self.phase_activation.reshape(1, -1)), dim=0)
mag_activations = torch.cat((mag_activations, self.mag_activation.reshape(1, -1)), dim=0)
activation_strengths = torch.cat((activation_strengths.flatten(), self.activation_strength.reshape(1)), dim=0)

# Calculate attention weights based on signal strengths
scaled_strengths = activation_strengths / (vector_dim ** 0.5)
scaled_strengths = torch.clamp(scaled_strengths, min=-20.0, max=20.0)
weights = F.softmax(scaled_strengths, dim=-1).reshape(-1, 1)

# Weighted sum with phase shifts
phase_weight_clamped = torch.clamp(self.phase_weight, min=-3*math.pi, max=3*math.pi)
mag_weight_clamped = torch.clamp(self.mag_weight, min=-3*math.pi, max=3*math.pi)

phase_activations = weights * (phase_activations + phase_weight_clamped.reshape(1, -1))
mag_activations = weights * (mag_activations + mag_weight_clamped.reshape(1, -1))

# Sum to get resultant wave
self.phase_activation = phase_activations.sum(dim=0)
self.mag_activation = mag_activations.sum(dim=0)

# Wrap phase to valid range
self.phase_activation = torch.remainder(self.phase_activation, 2 * math.pi)
self.mag_activation = torch.clamp(self.mag_activation, min=-3*math.pi, max=3*math.pi)
```

**Detailed Breakdown**:

1. **Attention Mechanism**: Uses softmax over activation strengths to weight contributions
   - Stronger signals get more weight (like constructive interference being stronger)
   - Scaled by `1/√d` for numerical stability (similar to Transformer attention)

2. **Phase Shift Addition**: Each incoming wave's phase is shifted by the node's learned phase weight
   - This is the **key difference from standard NNs**!
   - Weights are additive in phase space, not multiplicative in amplitude space

3. **Weighted Interference**: 
   ```
   phase_result = w₁(φ₁ + Δφ) + w₂(φ₂ + Δφ) + ... + wₙ(φₙ + Δφ)
   ```
   where `wᵢ = softmax(strength_i / √d)`

4. **Phase Wrapping**: Keeps phase in [0, 2π] for numerical stability
   - Mathematically correct since phase is circular

---

### 4. Radiation (Phase-Based Dynamic Connectivity)

**Formula** (`gnn_model.py:89-121`):
```python
# Find top-K nodes with most similar phase vectors
similarity = cosine_similarity(phase_query, phase_database)
radiation_targets = top_k(similarity)
```

**Code**:
```python
# gnn_model.py:108-120
query_vectors = [node.phase_activation for node in nodes]
batch_results = self.node_store.search_nodes_batch(
    query_vectors, 
    vector_name='phase', 
    limit=k,
    with_payload=False, 
    with_vectors=False
)
```

**Physical Interpretation**:
- Nodes with similar phase vectors have **coherent oscillations**
- Radiation creates dynamic connections between phase-aligned nodes
- Like resonance: systems with matching frequencies couple more strongly
- Uses vector database (Qdrant) to efficiently find nearest neighbors in phase space

---

## Comparison Table: Viz vs Reference Implementation

| Aspect | Current Viz (`viz/manager.py`) | Reference Implementation (`fixed_io_nodes/`) |
|--------|-------------------------------|-------------------------------------------|
| **Signal Calculation** | `cos(φ) × m` | `Σ cos(φᵢ) × exp(γ×sin(mᵢ))` |
| **Edge Weights** | Multiplicative amplitude scalers | **Additive phase shifts** |
| **Aggregation** | `act_new = 0.6×act_old + abs(input)` | Weighted sum with attention, phase shifts added |
| **Activation Update** | Scalar activation value | Vector of phases + vector of magnitudes |
| **Phase Wrapping** | Random walk `φ + 0.05×rand()` | Proper modulo 2π wrapping |
| **Magnitude Learning** | Not learned | Learned parameter with gradients |
| **Interference** | None (signals just add) | **True wave interference** via weighted phase-shifted sums |
| **Vector Dimension** | Scalar (1D) | Vector (e.g., 784D for MNIST) |

---

## Key Implementation Differences

### 1. **Edge Weight Interpretation**

**Viz (INCORRECT)**:
```python
strength = source_signal × edge.weight  # Multiplicative
updates[target] += strength
```

**Reference (CORRECT)**:
```python
phase_arriving = phase_source + phase_weight_edge  # Additive!
mag_arriving = mag_source + mag_weight_edge
# Then weighted sum with attention
```

### 2. **Activation Update**

**Viz (INCORRECT)**:
```python
activation_new = 0.6 × activation_old + abs(Σ inputs)
```

**Reference (CORRECT)**:
```python
# Calculate attention weights
weights = softmax(activation_strengths / √d)

# Apply weighted phase shifts
phase_new = Σ weights_i × (phase_i + phase_weight)
mag_new = Σ weights_i × (mag_i + mag_weight)

# Compute new intensity
activation_strength_new = Σ cos(phase_new) × exp(γ×sin(mag_new))
```

### 3. **Node State**

**Viz (INCORRECT)**:
```python
class Node:
    phase: float  # Single scalar
    magnitude: float  # Single scalar
    activation: float  # Single scalar
```

**Reference (CORRECT)**:
```python
class Node:
    phase_weight: torch.Tensor  # Shape: (vector_dim,)
    mag_weight: torch.Tensor    # Shape: (vector_dim,)
    phase_activation: torch.Tensor  # Shape: (vector_dim,)
    mag_activation: torch.Tensor    # Shape: (vector_dim,)
    activation_strength: torch.Tensor  # Scalar, computed from above
```

---

## Physics Analogy: Mach-Zehnder Interferometer

The network operates like a **Mach-Zehnder interferometer** with multiple paths:

```
Input → [Path 1: phase shift φ₁] → ⟩
     → [Path 2: phase shift φ₂] → ⟩ → Combiner → Output Intensity
     → [Path 3: phase shift φ₃] → ⟩

Output Intensity ∝ |Σ Aᵢ e^(iφᵢ)|²
```

In the network:
- **Paths** = Static edges + radiation connections
- **Phase shifts** = Learned edge weights
- **Combiner** = Node aggregation with attention
- **Output intensity** = Activation strength

**Key Insight**: Weights control **interference pattern**, not just signal magnitude!

---

## Implications for Visualization

To correctly visualize the wave dynamics:

1. **Show Phase Shifts on Edges**:
   - Edge color/label shows phase shift amount
   - Animate phase accumulation as signal travels

2. **Visualize Interference**:
   - At nodes receiving multiple inputs, show:
     - Incoming phases (as vectors on complex plane)
     - Resultant phase (vector sum)
     - Intensity (magnitude squared)

3. **Phasor Diagram**:
   - Display complex plane with phase vectors
   - Show how they add to create resultant

4. **Wave Animation**:
   - Animate sinusoidal waves traveling along edges
   - Show constructive/destructive interference at nodes

5. **Intensity Heatmap**:
   - Node color = activation strength
   - Edge thickness = phase shift magnitude

---

## Recommended Fixes for Viz Implementation

### 1. Update Signal Propagation

**Replace** (`viz/manager.py:238-264`):
```python
source_signal = self._get_signal(source) * source.activation

# Static Conductance
strength = source_signal * edge.weight
updates[edge.target_id] += strength

# Radiation
strength = source_signal * alignment_score * 0.5
updates[target_id] += strength
```

**With**:
```python
# Propagate phase and magnitude separately
for edge in self.edges:
    if edge.source_id == source.id:
        # Phase shift applied to signal
        arriving_phase = (source.phase + edge.weight) % (2 * math.pi)
        arriving_mag = source.magnitude  # Or also shift magnitude
        
        # Store phase-shifted contributions
        if edge.target_id not in phase_contributions:
            phase_contributions[edge.target_id] = []
            mag_contributions[edge.target_id] = []
            strengths[edge.target_id] = []
        
        phase_contributions[edge.target_id].append(arriving_phase)
        mag_contributions[edge.target_id].append(arriving_mag)
        strengths[edge.target_id].append(source.activation)
```

### 2. Update Node Activation

**Replace** (`viz/manager.py:267-276`):
```python
node.activation *= 0.6  # Decay
if node.id in updates:
    input_str = updates[node.id]
    if abs(input_str) > 0.05:
        node.activation = min(1.0, node.activation + abs(input_str))
```

**With**:
```python
if node.id in phase_contributions:
    # Calculate attention weights
    strengths_tensor = torch.tensor(strengths[node.id])
    weights = F.softmax(strengths_tensor, dim=0)
    
    # Weighted sum of phase-shifted inputs
    phases = torch.tensor(phase_contributions[node.id])
    mags = torch.tensor(mag_contributions[node.id])
    
    node.phase = (weights @ phases) % (2 * math.pi)
    node.magnitude = torch.clamp(weights @ mags, 0.1, 1.0)
    
    # Compute new activation strength (intensity)
    node.activation = (torch.cos(node.phase) * node.magnitude).item()
```

---

## Summary

**The Key Difference**:
- ❌ **Viz (Wrong)**: Weights scale amplitude, signals add linearly
- ✅ **Reference (Correct)**: Weights shift phase, waves interfere with attention-weighted contributions

This is not just a minor detail—it fundamentally changes how the network learns and represents information! The correct model allows for:
- **Constructive interference**: Aligned phases amplify
- **Destructive interference**: Opposite phases cancel
- **Phase-based computation**: Information encoded in wave timing, not just amplitude
- **Learnable interference patterns**: Weights control phase relationships

Would you like me to create an updated plan that fixes the core mathematics FIRST, then adds the visualization features on top of the correct physics?

