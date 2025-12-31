# Neurograph Visualization: Correct Wave Physics Implementation Plan

## Overview

This plan implements the visualization with **correct wave interference mathematics** based on the reference implementation in `fixed_io_nodes/`. The implementation proceeds in two phases:

**Phase 1**: Fix core wave physics (PRIORITY)
**Phase 2**: Add visualization features (table view, temporal encoding, architecture toggle)

---

## Phase 1: Fix Core Wave Physics

### Problem Statement

Current `viz/manager.py` uses incorrect signal propagation:
- ❌ Edge weights multiply amplitude: `strength = signal × weight`
- ❌ Scalar activations: `activation_new = 0.6×activation_old + abs(input)`
- ❌ No wave interference: signals just add linearly

Reference implementation uses correct wave propagation:
- ✅ Edge weights shift phase: `phase_arriving = phase_source + phase_weight`
- ✅ Vector activations with attention-weighted aggregation
- ✅ True interference: `strength = Σ cos(φᵢ) × exp(γ×sin(mᵢ))`

### 1.1 Update Node Data Structure

**File**: [`viz/manager.py`](viz/manager.py)

**Current** (Lines 19-38):
```python
@dataclass
class Node:
    id: str
    phase: float          # [0, 2π) - SCALAR
    magnitude: float      # [0, 1] - SCALAR
    role: str
    activation: float = 0.0  # SCALAR
    active: bool = False
    gradient_phase: float = 0.0
    gradient_magnitude: float = 0.0
```

**New**:
```python
@dataclass
class Node:
    id: str
    role: str  # "input", "input_connected", "middle", "output"
    
    # Learnable weights (vectors)
    phase_weight: np.ndarray  # Shape: (vector_dim,) in [0, 2π]
    mag_weight: np.ndarray    # Shape: (vector_dim,) 
    
    # Activation state (vectors)
    phase_activation: np.ndarray  # Shape: (vector_dim,)
    mag_activation: np.ndarray    # Shape: (vector_dim,)
    activation_strength: float = 0.0  # Computed intensity (scalar)
    
    # Training state
    target_phase: Optional[np.ndarray] = None
    gradient_phase: np.ndarray = None
    gradient_magnitude: np.ndarray = None
    accumulated_grad_phase: np.ndarray = None
    accumulated_grad_magnitude: np.ndarray = None
    
    # Metadata
    active: bool = False
    version: int = 0
```

**Config Addition**:
```python
@dataclass
class NetworkConfig:
    # ... existing fields ...
    vector_dim: int = 16  # NEW: Dimensionality of phase/mag vectors
    gamma: float = 1.0    # NEW: Magnitude exponential scaling
```

### 1.2 Implement Correct Activation Strength

**File**: [`viz/manager.py`](viz/manager.py)

**Add new method**:
```python
def _compute_activation_strength(phase: np.ndarray, mag: np.ndarray, gamma: float = 1.0) -> float:
    """
    Compute activation strength from phase and magnitude vectors.
    
    Formula: strength = Σ cos(φᵢ) × exp(γ × sin(mᵢ))
    
    This represents wave interference intensity.
    """
    phase_component = np.cos(phase)
    
    # Magnitude component with exponential scaling
    mag_exponent = gamma * np.sin(mag)
    mag_exponent = np.clip(mag_exponent, -10.0, 10.0)  # Prevent overflow
    mag_component = np.exp(mag_exponent)
    
    # Dot product gives total signal strength
    signal = np.sum(phase_component * mag_component)
    
    return signal + 1e-8  # Small epsilon for numerical stability
```

### 1.3 Update Edge Representation

**File**: [`viz/manager.py`](viz/manager.py)

**Current** (Lines 42-46):
```python
@dataclass
class Edge:
    source_id: str
    target_id: str
    weight: float = 1.0  # SCALAR
```

**New**:
```python
@dataclass
class Edge:
    source_id: str
    target_id: str
    phase_weight: np.ndarray  # Shape: (vector_dim,) - Phase shift
    mag_weight: np.ndarray    # Shape: (vector_dim,) - Magnitude shift
```

### 1.4 Fix Signal Propagation

**File**: [`viz/manager.py`](viz/manager.py)

**Replace `step_forward()` method** (Lines 229-278):

**New Implementation**:
```python
def step_forward(self) -> StepResult:
    """
    Forward pass with correct wave interference.
    """
    self.step_count += 1
    self.active_signals = []
    self.radiation_paths = []
    
    active_nodes = [n for n in self.nodes.values() if n.active]
    
    # Collect phase-shifted contributions for each node
    phase_contributions = {}  # target_id -> List[np.ndarray]
    mag_contributions = {}    # target_id -> List[np.ndarray]
    contribution_strengths = {}  # target_id -> List[float]
    
    for source in active_nodes:
        # 1. Static Conductance
        for edge in self.edges:
            if edge.source_id == source.id:
                # Apply phase shift (KEY: additive, not multiplicative!)
                arriving_phase = (source.phase_activation + edge.phase_weight) % (2 * np.pi)
                arriving_mag = source.mag_activation + edge.mag_weight
                
                # Store for visualization
                self.active_signals.append(SignalPacket(
                    source_id=source.id,
                    target_id=edge.target_id,
                    connection_type=ConnectionType.CONDUCTANCE,
                    signal_strength=source.activation_strength,
                    phase_value=arriving_phase[0] if len(arriving_phase) > 0 else 0.0
                ))
                
                # Collect contributions
                if edge.target_id not in phase_contributions:
                    phase_contributions[edge.target_id] = []
                    mag_contributions[edge.target_id] = []
                    contribution_strengths[edge.target_id] = []
                
                phase_contributions[edge.target_id].append(arriving_phase)
                mag_contributions[edge.target_id].append(arriving_mag)
                contribution_strengths[edge.target_id].append(source.activation_strength)
        
        # 2. Radiation (phase-aligned connections)
        radiation_targets = self._get_radiation_neighbors(source)
        for target_id, alignment_score in radiation_targets:
            # Radiation also applies phase shift (scaled by alignment)
            arriving_phase = (source.phase_activation + alignment_score * 0.5) % (2 * np.pi)
            arriving_mag = source.mag_activation
            
            self.radiation_paths.append(SignalPacket(
                source_id=source.id,
                target_id=target_id,
                connection_type=ConnectionType.RADIATION,
                signal_strength=source.activation_strength * alignment_score,
                phase_value=arriving_phase[0] if len(arriving_phase) > 0 else 0.0
            ))
            
            if target_id not in phase_contributions:
                phase_contributions[target_id] = []
                mag_contributions[target_id] = []
                contribution_strengths[target_id] = []
            
            phase_contributions[target_id].append(arriving_phase)
            mag_contributions[target_id].append(arriving_mag)
            contribution_strengths[target_id].append(source.activation_strength * alignment_score * 0.5)
    
    # 3. Aggregate with attention-weighted interference
    for node in self.nodes.values():
        if node.id not in phase_contributions:
            # No inputs - apply decay
            node.activation_strength *= 0.6
            node.active = node.activation_strength > 0.1
            continue
        
        # Add node's own state to contributions
        phase_contributions[node.id].append(node.phase_activation)
        mag_contributions[node.id].append(node.mag_activation)
        contribution_strengths[node.id].append(node.activation_strength)
        
        # Calculate attention weights (softmax over strengths)
        strengths_array = np.array(contribution_strengths[node.id])
        scaled_strengths = strengths_array / np.sqrt(self.config.vector_dim)
        scaled_strengths = np.clip(scaled_strengths, -20.0, 20.0)
        
        # Softmax
        exp_strengths = np.exp(scaled_strengths - np.max(scaled_strengths))
        weights = exp_strengths / np.sum(exp_strengths)
        
        # Weighted sum (wave interference)
        phases = np.array(phase_contributions[node.id])  # Shape: (num_inputs, vector_dim)
        mags = np.array(mag_contributions[node.id])
        
        node.phase_activation = np.sum(phases * weights[:, np.newaxis], axis=0)
        node.mag_activation = np.sum(mags * weights[:, np.newaxis], axis=0)
        
        # Wrap phase to [0, 2π]
        node.phase_activation = node.phase_activation % (2 * np.pi)
        node.mag_activation = np.clip(node.mag_activation, -3*np.pi, 3*np.pi)
        
        # Compute new activation strength (intensity)
        node.activation_strength = self._compute_activation_strength(
            node.phase_activation, 
            node.mag_activation, 
            self.config.gamma
        )
        
        node.active = node.activation_strength > 0.1
    
    return self._get_state("forward")
```

### 1.5 Update Initialization

**File**: [`viz/manager.py`](viz/manager.py)

**Update `_create_node()` method**:
```python
def _create_node(self, node_id: str, role: str) -> Node:
    """Create node with random vector initialization."""
    n = Node(
        id=node_id,
        role=role,
        phase_weight=np.random.uniform(0, 2*np.pi, self.config.vector_dim),
        mag_weight=np.random.uniform(-np.pi, np.pi, self.config.vector_dim),
        phase_activation=np.random.uniform(0, 2*np.pi, self.config.vector_dim),
        mag_activation=np.random.uniform(-np.pi, np.pi, self.config.vector_dim),
        gradient_phase=np.zeros(self.config.vector_dim),
        gradient_magnitude=np.zeros(self.config.vector_dim),
        accumulated_grad_phase=np.zeros(self.config.vector_dim),
        accumulated_grad_magnitude=np.zeros(self.config.vector_dim),
    )
    
    # Initialize activations from weights
    n.phase_activation = n.phase_weight.copy()
    n.mag_activation = n.mag_weight.copy()
    n.activation_strength = self._compute_activation_strength(
        n.phase_activation, n.mag_activation, self.config.gamma
    )
    
    self.nodes[node_id] = n
    return n
```

**Update edge initialization**:
```python
# In _initialize_network(), when creating edges:
self.edges.append(Edge(
    source_id=src.id, 
    target_id=target.id,
    phase_weight=np.random.uniform(-np.pi/4, np.pi/4, self.config.vector_dim),
    mag_weight=np.random.uniform(-0.5, 0.5, self.config.vector_dim)
))
```

### 1.6 Update Backward Pass

**File**: [`viz/manager.py`](viz/manager.py)

**Replace `step_backward()` method** (Lines 280-369):

```python
def step_backward(self) -> StepResult:
    """
    Backward pass with vector gradients.
    """
    self.active_signals = []
    learning_rate = self.config.learning_rate
    
    # 1. Compute gradients at output nodes
    gradients_phase = {}
    gradients_mag = {}
    
    for node in self.nodes.values():
        if node.role == "output" and node.target_phase is not None:
            # Circular distance for each dimension
            diff = node.target_phase - node.phase_activation
            
            # Shortest path on circle (element-wise)
            diff = np.where(diff > np.pi, diff - 2*np.pi, diff)
            diff = np.where(diff < -np.pi, diff + 2*np.pi, diff)
            
            gradients_phase[node.id] = diff
            gradients_mag[node.id] = np.zeros_like(node.mag_activation)  # Simplified
            
            node.gradient_phase = diff
            node.gradient_magnitude = gradients_mag[node.id]
            node.accumulated_grad_phase += diff * 0.1
    
    # 2. Propagate gradients backward
    next_gradients_phase = {}
    next_gradients_mag = {}
    
    for target_id, grad_phase in gradients_phase.items():
        target_node = self.nodes[target_id]
        grad_mag = gradients_mag[target_id]
        
        # Find incoming edges
        for edge in self.edges:
            if edge.target_id == target_id:
                # Backprop through phase shift
                src_grad_phase = grad_phase * 0.5  # Scaled gradient
                src_grad_mag = grad_mag * 0.5
                
                if edge.source_id not in next_gradients_phase:
                    next_gradients_phase[edge.source_id] = np.zeros(self.config.vector_dim)
                    next_gradients_mag[edge.source_id] = np.zeros(self.config.vector_dim)
                
                next_gradients_phase[edge.source_id] += src_grad_phase
                next_gradients_mag[edge.source_id] += src_grad_mag
                
                # Visualization
                self.active_signals.append(SignalPacket(
                    source_id=target_id,
                    target_id=edge.source_id,
                    connection_type=ConnectionType.CONDUCTANCE,
                    signal_strength=np.linalg.norm(src_grad_phase),
                    is_backward=True
                ))
        
        # Radiation backward
        if self.config.use_radiation:
            sources = self._get_radiation_neighbors(target_node)
            for src_id, score in sources:
                src_grad_phase = grad_phase * score * 0.3
                src_grad_mag = grad_mag * score * 0.3
                
                if src_id not in next_gradients_phase:
                    next_gradients_phase[src_id] = np.zeros(self.config.vector_dim)
                    next_gradients_mag[src_id] = np.zeros(self.config.vector_dim)
                
                next_gradients_phase[src_id] += src_grad_phase
                next_gradients_mag[src_id] += src_grad_mag
    
    # 3. Apply gradients
    for node_id, grad_phase in next_gradients_phase.items():
        node = self.nodes[node_id]
        grad_mag = next_gradients_mag.get(node_id, np.zeros(self.config.vector_dim))
        
        node.gradient_phase = grad_phase
        node.gradient_magnitude = grad_mag
        node.accumulated_grad_phase += grad_phase * 0.1
        node.accumulated_grad_magnitude += grad_mag * 0.1
        
        # Update weights
        node.phase_weight = (node.phase_weight + grad_phase * learning_rate) % (2 * np.pi)
        node.mag_weight = np.clip(
            node.mag_weight + grad_mag * learning_rate,
            -3*np.pi, 3*np.pi
        )
    
    return self._get_state("backward")
```

### 1.7 Update Serialization for Frontend

**File**: [`viz/manager.py`](viz/manager.py)

**Update `_get_state()` method**:
```python
def _get_state(self, mode: str) -> StepResult:
    """Serialize state for frontend (aggregate vectors to scalars for viz)."""
    return StepResult(
        nodes={
            n.id: {
                "id": n.id,
                "role": n.role,
                "phase": float(np.mean(n.phase_activation)),  # Average for viz
                "magnitude": float(np.mean(np.abs(n.mag_activation))),
                "activation": float(n.activation_strength),
                "active": n.active,
                "gradient": float(np.mean(n.gradient_phase)) if n.gradient_phase is not None else 0.0,
                "accumulator": float(np.mean(n.accumulated_grad_phase)),
                "target_phase": float(np.mean(n.target_phase)) if n.target_phase is not None else None,
                "label": f"{n.role[:3].upper()}\nφ:{np.mean(n.phase_activation):.2f}",
                "group": n.role,
                # NEW: Full vectors for detailed inspection
                "phase_vector": n.phase_activation.tolist(),
                "mag_vector": n.mag_activation.tolist(),
            }
            for n in self.nodes.values()
        },
        edges=[{
            "source": e.source_id, 
            "target": e.target_id, 
            "weight": float(np.mean(e.phase_weight))  # Average phase shift for viz
        } for e in self.edges],
        active_signals=[
            {
                "source": s.source_id,
                "target": s.target_id,
                "type": "conductance" if s.connection_type == ConnectionType.CONDUCTANCE else "radiation",
                "strength": s.signal_strength,
                "is_backward": s.is_backward
            }
            for s in self.active_signals + self.radiation_paths
        ],
        radiation_paths=[],
        step_number=self.step_count,
        mode=mode
    )
```

---

## Phase 2: Visualization Features

### 2.1 Temporal Sequence Injection

**File**: [`viz/manager.py`](viz/manager.py)

**Add method**:
```python
def inject_temporal_sequence(self, input_values: List[float], timestep: int):
    """
    Inject temporal sequence with positional encoding.
    
    Encoding:
    - Position -> Phase component (high frequency)
    - Time -> Phase component (low frequency)
    - Value -> Magnitude
    """
    temporal_period = 40  # Can be config parameter
    
    for i, value in enumerate(input_values):
        if f"in_{i}" not in self.nodes:
            continue
        
        input_node = self.nodes[f"in_{i}"]
        
        # Position encoding (cycles once per sequence)
        pos_phase = (i / len(input_values)) * 2 * np.pi
        
        # Temporal encoding (slow oscillation)
        temp_phase = np.sin(timestep / temporal_period) * np.pi
        
        # Broadcast to vector
        base_phase = (pos_phase + temp_phase) % (2 * np.pi)
        input_node.phase_activation = np.full(self.config.vector_dim, base_phase)
        
        # Value -> Magnitude
        input_node.mag_activation = np.full(self.config.vector_dim, abs(value))
        
        # Recompute strength
        input_node.activation_strength = self._compute_activation_strength(
            input_node.phase_activation,
            input_node.mag_activation,
            self.config.gamma
        )
        input_node.active = True
```

### 2.2 Architecture Mode Toggle

**File**: [`viz/manager.py`](viz/manager.py)

**Add to NetworkConfig**:
```python
@dataclass
class NetworkConfig:
    # ... existing ...
    architecture_mode: str = "hierarchical"  # "flat" or "hierarchical"
```

**Modify `_initialize_network()`** to conditionally create Input-Connected layer based on mode.

### 2.3 Table View

**Files**: [`viz/static/index.html`](viz/static/index.html), [`viz/static/script.js`](viz/static/script.js)

Add table showing per-node:
- ID, Role
- Phase (mean), Magnitude (mean)
- Activation strength
- Gradient (mean), Accumulator (mean)
- Expandable: full phase/mag vectors

### 2.4 Timeline Sparklines

**File**: [`viz/static/script.js`](viz/static/script.js)

Track history and render sparklines for:
- Activation strength over time
- Mean phase over time
- Gradient magnitude over time

### 2.5 Phasor Diagram (Advanced)

**File**: [`viz/static/index.html`](viz/static/index.html)

Add complex plane visualization showing:
- Phase vectors as arrows
- Vector sum (interference result)
- Color by magnitude

---

## Implementation Order

### Sprint 1: Core Math Fix (Priority)
1. ✅ Update Node dataclass to vectors
2. ✅ Implement `_compute_activation_strength()`
3. ✅ Update Edge to store vector weights
4. ✅ Fix `step_forward()` with attention + interference
5. ✅ Fix `step_backward()` with vector gradients
6. ✅ Update `_create_node()` and edge initialization
7. ✅ Update `_get_state()` serialization
8. ✅ Test basic forward/backward pass

### Sprint 2: Temporal Encoding
1. ✅ Implement `inject_temporal_sequence()`
2. ✅ Add API endpoint `/api/inject_sequence`
3. ✅ Add frontend controls for sequence injection
4. ✅ Test temporal propagation

### Sprint 3: Visualization
1. ✅ Add table view UI
2. ✅ Implement table rendering with vector inspection
3. ✅ Add timeline sparklines
4. ✅ Add architecture mode toggle
5. ✅ Add phasor diagram (stretch goal)

---

## Testing Strategy

### Unit Tests
- `_compute_activation_strength()`: verify against reference
- Phase wrapping: verify `% 2π` correctness
- Attention weights: verify softmax sums to 1

### Integration Tests
- Forward pass: compare output to reference implementation
- Backward pass: verify gradients flow correctly
- Temporal injection: verify phase encoding formula

### Visual Verification
- Watch interference patterns in network
- Verify constructive/destructive interference occurs
- Check phase shifts visualized on edges

---

## Files to Modify

### Core Math (Phase 1)
- [`viz/manager.py`](viz/manager.py): Complete rewrite of propagation logic
- [`viz/server.py`](viz/server.py): Update to handle vector serialization

### Visualization (Phase 2)
- [`viz/static/index.html`](viz/static/index.html): Add table, timeline, controls
- [`viz/static/script.js`](viz/static/script.js): Rendering logic
- [`viz/server.py`](viz/server.py): Add temporal injection endpoint

---

## Success Criteria

### Phase 1 Complete When:
- ✅ Activation strength matches formula: `Σ cos(φᵢ) × exp(γ×sin(mᵢ))`
- ✅ Phase shifts are additive on edges
- ✅ Multi-input nodes use attention-weighted aggregation
- ✅ Gradients backprop through vector dimensions
- ✅ No errors/warnings in forward/backward pass

### Phase 2 Complete When:
- ✅ Temporal sequences inject with positional encoding
- ✅ Table shows all node states with vector inspection
- ✅ Timeline tracks network evolution
- ✅ Architecture toggle switches between flat/hierarchical
- ✅ User can "feel" the wave interference dynamics

---

## Notes

- **NumPy vs PyTorch**: Current plan uses NumPy for simplicity. Can switch to PyTorch if GPU acceleration needed.
- **Vector Dimension**: Start with `vector_dim=16` for visualization. Reference uses 784 (MNIST pixels).
- **Visualization Simplification**: Frontend shows mean(phase_vector) for clarity. Detailed view shows full vectors.
- **Performance**: Vector operations may be slower. Consider batching if needed.

---

## References

- Reference implementation: [`fixed_io_nodes/node.py`](../fixed_io_nodes/node.py)
- Wave math documentation: [`viz/CORRECT_WAVE_MATH.md`](CORRECT_WAVE_MATH.md)
- Current (incorrect) math: [`viz/ACTIVATION_MATH.md`](ACTIVATION_MATH.md)

