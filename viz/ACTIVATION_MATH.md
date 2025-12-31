# Neurograph Activation Transition Mathematics

## Overview
This document details all mathematical formulas used in the current Neurograph visualization implementation for signal propagation, activation updates, and gradient computation.

---

## 1. Node Signal Computation

### Signal Formula
```python
signal = cos(phase) × magnitude
```

**Code Location**: Line 204-205 in `manager.py`
```python
def _get_signal(self, node: Node) -> float:
    return math.cos(node.phase) * node.magnitude
```

**Rationale**:
- `phase` ∈ [0, 2π): Encodes position/time information
- `magnitude` ∈ [0, 1]: Encodes content/strength
- `cos(phase)` maps phase to [-1, 1], creating oscillatory signal
- Product allows phase to modulate magnitude

**Properties**:
- Signal oscillates between [-magnitude, +magnitude]
- Phase = 0 → signal = +magnitude (maximum positive)
- Phase = π → signal = -magnitude (maximum negative)
- Phase = π/2 or 3π/2 → signal = 0 (null)

---

## 2. Phase Alignment Score (Radiation)

### Alignment Formula
```python
alignment_score = (cos(phase₁ - phase₂) + 1) / 2
```

**Code Location**: Line 200-202 in `manager.py`
```python
def _phase_alignment(self, phase1: float, phase2: float) -> float:
    diff = phase1 - phase2
    return (math.cos(diff) + 1) / 2
```

**Rationale**:
- `cos(Δphase)` ∈ [-1, 1] measures phase similarity
- Adding 1 and dividing by 2 normalizes to [0, 1]
- Score = 1.0 when phases perfectly aligned (Δphase = 0)
- Score = 0.0 when phases opposite (Δphase = π)
- Score = 0.5 when orthogonal (Δphase = π/2)

**Used For**:
- Determining radiation neighbors (top-K nodes with highest alignment)
- Weighting radiation signal strength

---

## 3. Forward Pass: Activation Updates

### 3.1 Signal Propagation

#### Static Conductance
```python
strength_conductance = signal_source × weight_edge
```

**Code Location**: Line 238-251 in `manager.py`
```python
source_signal = self._get_signal(source) * source.activation
# ...
strength = source_signal * edge.weight
updates[edge.target_id] = updates.get(edge.target_id, 0) + strength
```

**Formula Breakdown**:
```
signal_source = cos(phase_source) × magnitude_source × activation_source
strength = signal_source × weight_edge
```

#### Dynamic Radiation
```python
strength_radiation = signal_source × alignment_score × 0.5
```

**Code Location**: Line 254-264 in `manager.py`
```python
radiation_targets = self._get_radiation_neighbors(source)
for target_id, alignment_score in radiation_targets:
    strength = source_signal * alignment_score * 0.5
    updates[target_id] = updates.get(target_id, 0) + strength
```

**Formula Breakdown**:
```
signal_source = cos(phase_source) × magnitude_source × activation_source
alignment = (cos(phase_source - phase_target) + 1) / 2
strength_radiation = signal_source × alignment × 0.5
```

**Note**: The 0.5 multiplier reduces radiation strength relative to static conductance.

### 3.2 Activation Update Rule

```python
activation_new = decay × activation_old + abs(input_strength)
activation_new = min(1.0, activation_new)
```

**Code Location**: Line 267-276 in `manager.py`
```python
for node in self.nodes.values():
    node.activation *= 0.6  # Decay
    if node.id in updates:
        input_str = updates[node.id]
        if abs(input_str) > 0.05:
            node.activation = min(1.0, node.activation + abs(input_str))
            # Phase adaptation
            node.phase = (node.phase + 0.05 * random.uniform(-1, 1)) % (2 * math.pi)
    
    node.active = node.activation > 0.1
```

**Full Update Formula**:
```
activation(t+1) = 0.6 × activation(t) + abs(Σ input_strength)
activation(t+1) = min(1.0, activation(t+1))

where:
    input_strength = Σ (static_conductance + radiation)
    decay_rate = 0.6
    activation_threshold = 0.1 (for "active" status)
    signal_threshold = 0.05 (minimum to trigger update)
```

**Properties**:
- **Exponential Decay**: Without input, activation decays by 40% per step
- **Additive Integration**: Multiple inputs sum linearly
- **Saturation**: Capped at 1.0 to prevent explosion
- **Threshold**: Node becomes "active" when activation > 0.1
- **Signal Gating**: Inputs < 0.05 are ignored (noise suppression)

### 3.3 Phase Adaptation (Hebbian-like)

```python
phase_new = (phase_old + 0.05 × random(-1, 1)) mod 2π
```

**Code Location**: Line 274 in `manager.py`
```python
node.phase = (node.phase + 0.05 * random.uniform(-1, 1)) % (2 * math.pi)
```

**Interpretation**:
- Small random walk in phase space (±0.05 radians = ±2.86°)
- Only occurs when node receives significant input (> 0.05)
- Simulates unsupervised phase adjustment based on activity
- Modulo 2π ensures phase stays in valid range

**Note**: This is a simplified plasticity rule. In full Neurograph, phase would adapt based on input phase relationships.

---

## 4. Backward Pass: Gradient Computation

### 4.1 Loss Calculation (Output Nodes)

#### Phase Error (Circular Distance)
```python
error = target_phase - current_phase
# Shortest path on circle:
if error > π:  error -= 2π
if error < -π: error += 2π
```

**Code Location**: Line 295-303 in `manager.py`
```python
diff = node.target_phase - node.phase
# Shortest path on circle
if diff > math.pi: diff -= 2*math.pi
if diff < -math.pi: diff += 2*math.pi

grad = diff
```

**Rationale**:
- Phase is circular: 0 and 2π are equivalent
- Error must account for wrap-around (e.g., target=0.1, current=6.2 → error = 0.2, not -6.1)
- Gradient points toward closest path to target

**Loss Function** (implicit):
```
L = (target_phase - current_phase)²_circular
∂L/∂phase = 2 × error = 2 × (target - current)_circular
```

**Simplified Gradient**:
```
gradient = error (proportional to 2×error, constant absorbed)
```

### 4.2 Gradient Propagation

#### Static Conductance (Backward)
```python
grad_source = grad_target × weight_edge × 0.5
```

**Code Location**: Line 325-338 in `manager.py`
```python
for edge in self.edges:
    if edge.target_id == target_id:
        src_grad = grad * edge.weight * 0.5
        next_gradients[edge.source_id] = next_gradients.get(edge.source_id, 0) + src_grad
```

**Formula**:
```
∂L/∂source = ∂L/∂target × ∂target/∂source
            = grad_target × weight_edge × 0.5
```

**Note**: The 0.5 factor reduces gradient magnitude during backprop (stabilization).

#### Dynamic Radiation (Backward)
```python
grad_source = grad_target × alignment_score × 0.3
```

**Code Location**: Line 343-358 in `manager.py`
```python
if self.config.use_radiation:
    sources = self._get_radiation_neighbors(target_node)
    for src_id, score in sources:
        src_grad = grad * score * 0.3
        next_gradients[src_id] = next_gradients.get(src_id, 0) + src_grad
```

**Formula**:
```
∂L/∂source = ∂L/∂target × alignment(source, target) × 0.3
```

**Note**: The 0.3 factor further reduces radiation gradient (weaker than static 0.5).

### 4.3 Gradient Accumulation

```python
accumulated_grad += current_grad × 0.1
```

**Code Location**: Line 310, 364 in `manager.py`
```python
node.accumulated_grad_phase += grad * 0.1
```

**Purpose**:
- Track cumulative gradient over multiple steps
- Used for visualization (intensity indicates learning activity)
- Slow accumulation (0.1 factor) provides smoothed view

### 4.4 Parameter Update (Gradient Descent)

```python
phase_new = (phase_old + gradient × learning_rate) mod 2π
```

**Code Location**: Line 367 in `manager.py`
```python
node.phase = (node.phase + grad * learning_rate) % (2 * math.pi)
```

**Formula**:
```
θ(t+1) = θ(t) + α × ∇L
       = θ(t) + learning_rate × gradient

where:
    learning_rate (α) = 0.05 (default)
    gradient points toward target (positive = increase phase)
```

**Note**: This is **gradient ascent** on the error (moving toward target), not descent on loss. Equivalent to gradient descent with negated loss.

---

## 5. Summary of Key Parameters

| Parameter | Value | Location | Purpose |
|-----------|-------|----------|---------|
| Decay Rate | 0.6 | Line 268 | Activation exponential decay per step |
| Activation Threshold | 0.1 | Line 276 | Minimum activation to be "active" |
| Signal Threshold | 0.05 | Line 271 | Minimum input to trigger update |
| Radiation Multiplier | 0.5 | Line 256 | Forward radiation strength reduction |
| Phase Adaptation Rate | 0.05 | Line 274 | Random phase walk magnitude |
| Static Backprop Factor | 0.5 | Line 328 | Gradient reduction through static edges |
| Radiation Backprop Factor | 0.3 | Line 349 | Gradient reduction through radiation |
| Accumulator Rate | 0.1 | Line 310, 364 | Gradient accumulation smoothing |
| Learning Rate | 0.05 | Config | Phase update step size |
| Radiation K | 3 | Config | Number of top phase-aligned neighbors |

---

## 6. Mathematical Properties

### 6.1 Activation Dynamics

**Without Input**:
```
a(t) = a(0) × (0.6)^t
```
- Exponential decay with half-life ≈ 1.7 steps
- Activation < 0.1 after ~4.5 steps (becomes inactive)

**With Constant Input I**:
```
a(∞) = I / (1 - 0.6) = 2.5 × I
```
- Equilibrium activation is 2.5× input strength (capped at 1.0)
- Convergence time constant ≈ 2.5 steps

### 6.2 Phase Space

**Signal Manifold**:
```
signal = cos(φ) × m ∈ [-m, +m]
```
- 1D manifold embedded in 2D (phase, magnitude) space
- Iso-signal curves: phase = arccos(signal/magnitude)

**Phase Gradient Flow**:
```
dφ/dt = learning_rate × error
```
- Linear flow toward target (no curvature correction for circular space)
- Can overshoot if learning_rate × error > π

### 6.3 Radiation Connectivity

**Expected Radiation Edges**:
```
E[radiation_edges] = N × K
```
- For N nodes, K neighbors each
- Dynamic (changes each step based on phase)
- Total connectivity: static + radiation ≈ E + N×K

**Phase Clustering**:
- Nodes with similar phases form dynamic clusters
- Cluster membership changes as phases evolve
- Radiation creates "soft" community structure

---

## 7. Potential Issues & Considerations

### 7.1 Current Implementation Limitations

1. **Phase Adaptation Randomness** (Line 274):
   - Random walk lacks directional learning
   - Should adapt toward input phases (Hebbian)
   - Current version: exploratory noise only

2. **Magnitude Not Learned**:
   - Magnitude is fixed after initialization
   - No gradient for magnitude in backward pass
   - Limits representational capacity

3. **Activation Saturation**:
   - Hard clipping at 1.0 (Line 272)
   - Could use soft saturation (tanh, sigmoid)
   - May cause gradient issues near boundary

4. **Linear Gradient Propagation**:
   - No nonlinearity in backward pass
   - Gradients scale linearly with distance
   - Deep networks may suffer vanishing gradients

5. **Circular Phase Gradient**:
   - Current implementation uses Euclidean gradient
   - Should use Riemannian gradient on circle manifold
   - May cause wraparound issues for large learning rates

### 7.2 Suggested Mathematical Improvements

1. **Hebbian Phase Adaptation**:
```python
# Replace Line 274 with:
phase_shift = learning_rate_local × Σ(input_phase - node.phase) × input_strength
node.phase = (node.phase + phase_shift) % (2 * math.pi)
```

2. **Magnitude Learning**:
```python
grad_magnitude = Σ(grad_target × weight × cos(phase_diff))
node.magnitude = clip(node.magnitude + learning_rate × grad_magnitude, 0.1, 1.0)
```

3. **Soft Activation**:
```python
# Replace Line 272 with:
node.activation = tanh(node.activation + input_str)
```

4. **Nonlinear Backward Pass**:
```python
# Multiply gradient by activation derivative:
src_grad = grad * weight * activation_derivative(node.activation)
```

---

## 8. Temporal Encoding Math (Proposed)

For the upcoming temporal sequence feature:

### Position + Time Encoding
```python
phase = position_encoding + temporal_encoding
position_encoding = (position_index / num_positions) × 2π
temporal_encoding = sin(timestep / period) × π
```

**Properties**:
- Position cycles once over input sequence (0 → 2π)
- Time oscillates slowly (period = 40 steps default)
- Combined phase = periodic function in 2D (pos, time) space
- Different positions maintain relative phase offsets over time

### Content Encoding
```python
magnitude = abs(input_value)
```

**Separates**:
- What (content) → magnitude
- Where (position) → phase component 1
- When (time) → phase component 2

---

## References

**Code File**: [`viz/manager.py`](viz/manager.py)

**Key Functions**:
- `_get_signal()` (Line 204): Signal computation
- `_phase_alignment()` (Line 200): Radiation score
- `step_forward()` (Line 229): Forward pass
- `step_backward()` (Line 280): Backward pass

**External Resources**:
- Neurograph paper: [Add citation if available]
- Phase-based neural networks: Hopfield networks, oscillatory networks
- Circular statistics: Directional Statistics (Mardia & Jupp)

