# Radiation Mechanism: Complex Wave Decomposition

## Critical Insight: Real vs Imaginary Components

The wave at each node has **two orthogonal components**:

### 1. Real Component → Conductance (Guided Wave)
```
Real part = cos(activation_phase) × magnitude
```
- Travels through **static edges only**
- Like light in a fiber optic cable (guided)
- Deterministic path following graph topology

### 2. Imaginary Component → Radiation (Broadcast Wave)
```
Imaginary part = sin(activation_phase) × magnitude
```
- **Broadcasts** to nodes with matching phase weights
- Like electromagnetic radiation from an antenna
- Can reach any node in the network (active or inactive)

---

## Radiation Alignment: Activation Phase ↔ Weight Phase

**Key Difference from Previous Understanding**:

❌ **WRONG**: Radiation aligns `phase_activation[source]` with `phase_activation[target]`

✅ **CORRECT**: Radiation aligns `phase_activation[source]` with `phase_weight[target]`

### Why This Matters

**Target node's phase weight** = Its **resonant frequency**
- Like a radio tuned to a specific frequency
- Node "listens" for radiation at its weight phase
- Whether the node is currently active doesn't matter!

**Source node's activation phase** = The **broadcast frequency**
- The imaginary component radiates at this phase
- Only nodes whose weight phase matches will "hear" it

### Physical Analogy: Radio Broadcasting

```
Radio Station (Source Node):
  Broadcasting at 100.5 MHz (activation phase)
  Real signal: Travels via cable to local towers (conductance)
  Imaginary signal: Radiates through air (radiation)

Radio Receiver (Target Node):
  Tuned to 100.5 MHz (phase weight)
  Receives radiation regardless of whether it's currently powered on
  Resonates when broadcast frequency matches tuning
```

---

## Mathematical Formulation

### Complex Wave Representation

Each node emits a complex signal:
```
S = cos(φ_activation) × m + i × sin(φ_activation) × m

Real[S] = cos(φ_activation) × m        → Conductance
Imag[S] = sin(φ_activation) × m        → Radiation
```

### Radiation Reception

Target node `j` receives radiation from source `i` if:
```
alignment = cos(φ_activation[i] - φ_weight[j])
```

High alignment (≈ 1) → Strong reception
Low alignment (≈ 0) → Weak reception

**Received signal**:
```
radiation_signal[j] = sin(φ_activation[i]) × m[i] × alignment
                    = sin(φ_activation[i]) × m[i] × cos(φ_activation[i] - φ_weight[j])
```

### Why sin(θ) for Radiation?

The imaginary component `sin(θ)` represents the **quadrature phase**:
- Orthogonal to the real component `cos(θ)`
- In physics: electric field (cos) vs magnetic field (sin) in EM waves
- In quantum: position vs momentum (complementary variables)
- Here: Guided vs broadcast transmission

---

## Comparison Table

| Aspect | Conductance (Real) | Radiation (Imaginary) |
|--------|-------------------|----------------------|
| **Wave Component** | cos(φ_activation) | sin(φ_activation) |
| **Transmission** | Guided (static edges) | Broadcast (all nodes) |
| **Path** | Deterministic (graph) | Dynamic (phase matching) |
| **Alignment Check** | Edge existence | φ_activation[src] ↔ φ_weight[tgt] |
| **Target State** | Must be connected | Can be active or inactive |
| **Physical Analogy** | Fiber optic cable | Radio waves |
| **Range** | Local (neighbors) | Global (entire network) |

---

## Implementation Corrections

### Current Incorrect Implementation

```python
# WRONG: Compares activation to activation
def _get_radiation_neighbors(self, source: Node):
    for target in self.nodes.values():
        # ❌ This compares current states, not resonant frequencies
        similarity = cosine_similarity(
            source.phase_activation,
            target.phase_activation
        )
```

### Correct Implementation

```python
def _get_radiation_neighbors(self, source: Node, k: int = None) -> List[Tuple[str, float]]:
    """
    Find nodes whose WEIGHT phase aligns with source's ACTIVATION phase.
    
    This is the resonance condition:
    - Source broadcasts at φ_activation
    - Target receives if φ_weight matches
    """
    if not self.config.use_radiation:
        return []
    
    if k is None:
        k = self.config.radiation_k
    
    # Exclude static neighbors and self
    static_neighbors = {e.target_id for e in self.edges if e.source_id == source.id}
    static_neighbors.add(source.id)
    
    candidates = []
    for node_id, node in self.nodes.items():
        if node_id in static_neighbors:
            continue
        
        # KEY CHANGE: Compare activation[source] with weight[target]
        # Cosine similarity between source's activation and target's weight
        dot_product = np.dot(
            source.phase_activation,  # Current broadcast frequency
            node.phase_weight          # Target's resonant frequency
        )
        norm_source = np.linalg.norm(source.phase_activation)
        norm_target = np.linalg.norm(node.phase_weight)
        
        similarity = dot_product / (norm_source * norm_target + 1e-8)
        
        # Normalize to [0, 1]
        alignment_score = (similarity + 1) / 2
        
        candidates.append((node_id, alignment_score))
    
    # Sort by alignment and take top-K
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:k]
```

### Signal Propagation

```python
def step_forward(self):
    # ... existing setup ...
    
    for source in active_nodes:
        source_phase = source.phase_activation
        source_mag = source.magnitude
        
        # 1. REAL COMPONENT → Conductance (static edges)
        real_component = np.cos(source_phase) * source_mag
        
        for edge in self.edges:
            if edge.source_id == source.id:
                # Real part conducts through edge with phase shift
                arriving_phase = (source_phase + edge.phase_weight) % (2 * np.pi)
                arriving_mag = source_mag + edge.mag_weight
                
                phase_contributions[edge.target_id].append(arriving_phase)
                mag_contributions[edge.target_id].append(arriving_mag)
                contribution_strengths[edge.target_id].append(
                    np.sum(real_component)  # Real component strength
                )
        
        # 2. IMAGINARY COMPONENT → Radiation (phase-aligned nodes)
        imaginary_component = np.sin(source_phase) * source_mag
        
        radiation_targets = self._get_radiation_neighbors(source)  # Uses weight matching!
        
        for target_id, alignment_score in radiation_targets:
            # Imaginary part radiates to aligned nodes
            arriving_phase = source_phase  # No edge-based phase shift
            arriving_mag = source_mag
            
            phase_contributions[target_id].append(arriving_phase)
            mag_contributions[target_id].append(arriving_mag)
            contribution_strengths[target_id].append(
                np.sum(imaginary_component) * alignment_score  # Imaginary component
            )
    
    # ... rest of aggregation ...
```

---

## Why This Design?

### 1. Separation of Local vs Global

**Real Component (Conductance)**:
- Short-range, strong connections
- Structured by graph topology
- Carries precise information along known paths

**Imaginary Component (Radiation)**:
- Long-range, weak connections  
- Self-organized by phase alignment
- Broadcasts to matching resonant frequencies

### 2. Inactive Node Activation

**Critical Feature**: Radiation can activate dormant nodes!

```
Example:
  Node A (active): φ_activation = π/4
  Node B (inactive): φ_weight = π/4 (matches!)
  
  → A's imaginary component radiates to B
  → B receives signal even though inactive
  → B becomes active if signal strong enough
  → Spreads activation through phase resonance
```

This is like:
- A radio receiver turning on when it detects a signal
- A quantum system being driven to resonance
- A neuron firing when it receives matching input patterns

### 3. Learning Dynamics

**Forward Pass**:
- Nodes broadcast at their activation phase
- Reception determined by weight phase alignment

**Backward Pass**:
- Gradients flow backward through both conductance and radiation
- Weight phases learn to:
  - Match activation phases of useful sources (increase reception)
  - Avoid noisy sources (decrease reception)

**Result**: Network self-organizes "frequency bands"
- Output nodes learn to tune to informative inputs
- Middle nodes form resonance clusters
- Emergent communication channels

---

## Reference Implementation Analysis

Looking at `fixed_io_nodes/gnn_model.py:108-120`:

```python
query_vectors = [node.phase_activation for node in nodes]
batch_results = self.node_store.search_nodes_batch(
    query_vectors, 
    vector_name='phase',  # <-- Searches in 'phase' vectors
    limit=k,
)
```

**Question**: Does `vector_name='phase'` refer to:
- Option A: `phase_activation` (current state)?
- Option B: `phase_weight` (resonant frequency)?

**Evidence from `nodestore.py`** (likely):
The database stores nodes with their **weight vectors**, not activation vectors (which are transient).

Therefore:
- Database contains `phase_weight` for all nodes
- Query uses `phase_activation` from active nodes
- Search finds nodes whose **weights** match the **activations**

**Conclusion**: Reference implementation likely DOES use activation→weight alignment! ✅

---

## Visualization Implications

### 1. Show Real vs Imaginary Components

**In Frontend** (`viz/static/script.js`):

```javascript
// Separate edge rendering for conductance vs radiation

// Conductance edges (solid)
const conductanceEdges = state.active_signals
    .filter(s => s.type === 'conductance')
    .map(sig => ({
        from: sig.source,
        to: sig.target,
        color: '#00d4ff',  // Cyan for real component
        width: 2,
        dashes: false,
        label: 'cos(θ)'
    }));

// Radiation edges (dashed)
const radiationEdges = state.active_signals
    .filter(s => s.type === 'radiation')
    .map(sig => ({
        from: sig.source,
        to: sig.target,
        color: '#ff00aa',  // Magenta for imaginary component
        width: 1,
        dashes: [5, 5],
        label: 'sin(θ)'
    }));
```

### 2. Node Resonance Visualization

Show whether a node can receive radiation:

```javascript
// Node border color indicates resonance state
const borderColor = (() => {
    if (node.active) return '#ffffff';  // Active
    
    // Check if any active node's activation matches this node's weight
    const isResonant = checkResonanceMatch(node.phase_weight, activeNodes);
    
    if (isResonant) return '#ffaa00';  // Ready to receive radiation
    return '#333333';  // Dormant
})();
```

### 3. Phase Wheel with Dual Rings

```
Outer Ring: Activation phases (current state)
Inner Ring: Weight phases (resonant frequencies)

When activation matches weight:
  → Draw connecting arc
  → Highlight both nodes
  → Show radiation path
```

### 4. Complex Plane Visualization

Plot nodes in complex plane:
```
x-axis: cos(phase) (real component)
y-axis: sin(phase) (imaginary component)

Two plots side-by-side:
  Left: Activation phases (where signals broadcast from)
  Right: Weight phases (where signals can be received)
```

---

## Updated Configuration

```python
@dataclass
class NetworkConfig:
    # ... existing ...
    
    # Radiation parameters
    radiation_k: int = 3                    # Top-K resonant nodes
    radiation_strength: float = 0.5         # Imaginary component multiplier
    resonance_threshold: float = 0.7        # Min alignment to create connection
    
    # Component separation
    conductance_uses_cos: bool = True       # Real component via static edges
    radiation_uses_sin: bool = True         # Imaginary component via radiation
```

---

## Summary

### Key Corrections

1. ❌ **WRONG**: Radiation aligns activation↔activation
   ✅ **RIGHT**: Radiation aligns activation[source]↔weight[target]

2. ❌ **WRONG**: Both real and imaginary components use cos(θ)
   ✅ **RIGHT**: Real=cos(θ) for conductance, Imaginary=sin(θ) for radiation

3. ❌ **WRONG**: Target must be active to receive radiation
   ✅ **RIGHT**: Target can be inactive; radiation activates it if resonant

### Physical Model

```
Each node emits: A·e^(iφ) = A·cos(φ) + i·A·sin(φ)

Real part A·cos(φ):
  → Conducts through static edges
  → Guided transmission
  → Local propagation

Imaginary part A·sin(φ):
  → Radiates to phase-aligned nodes
  → Broadcast transmission
  → Global propagation
  
Reception at target: 
  alignment = cos(φ_source_activation - φ_target_weight)
  
Strong alignment → resonance → signal received
Weak alignment → off-resonance → signal ignored
```

This is analogous to:
- **Photons** (guided in fiber) vs **EM waves** (broadcast through space)
- **Action potentials** (axonal conduction) vs **Neuromodulators** (volume transmission)
- **Wired network** (Ethernet) vs **Wireless** (WiFi)

The network learns which frequencies to broadcast at (activations) and which frequencies to listen for (weights), enabling self-organized communication!

