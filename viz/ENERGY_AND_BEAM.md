# Energy Conservation and Beam Width Mechanisms

## 1. Energy Conservation: Activation Depletion

### Physical Principle

**Energy cannot be created, only transferred.**

When a node conducts and radiates signals, it **loses activation strength** proportional to the energy transmitted:

```
Energy_out = Energy_conducted + Energy_radiated
Activation_new = Activation_old - Energy_out - Decay
```

### Mathematical Formulation

#### Energy Conducted (Real Component)

```python
# For each outgoing static edge
for edge in node.outgoing_edges:
    conducted_signal = cos(phase_activation) × magnitude × edge_weight
    energy_conducted += |conducted_signal|

# Energy cost
conductance_cost = α × energy_conducted
```

where `α` is the conductance efficiency (e.g., 0.1 = 10% energy loss per unit transmitted)

#### Energy Radiated (Imaginary Component)

```python
# For each radiation target
for target, alignment in radiation_targets:
    radiated_signal = sin(phase_activation) × magnitude × alignment
    energy_radiated += |radiated_signal|

# Energy cost
radiation_cost = β × energy_radiated
```

where `β` is the radiation efficiency (e.g., 0.05 = 5% energy loss per unit radiated)

**Note**: Radiation is more efficient (lower β) because it's broadcast—one emission reaches multiple targets.

#### Temporal Decay

Even if the node doesn't transmit anything:

```python
temporal_decay = γ × activation_strength
```

where `γ` is the decay rate (e.g., 0.6 = 40% decay per timestep)

#### Total Update Rule

```python
activation_strength_new = activation_strength_old × (1 - γ)  # Temporal decay
                        - α × energy_conducted                # Conduction cost
                        - β × energy_radiated                 # Radiation cost

# Clamp to [0, max_activation]
activation_strength_new = max(0, min(activation_strength_new, 1.0))

# Deactivate if below threshold
node.active = (activation_strength_new > activation_threshold)
```

### Implementation

**File**: `viz/manager.py`

```python
def step_forward(self) -> StepResult:
    # ... existing setup ...
    
    # Track energy expenditure per node
    energy_conducted = {}  # node_id -> float
    energy_radiated = {}   # node_id -> float
    
    for source in active_nodes:
        energy_conducted[source.id] = 0.0
        energy_radiated[source.id] = 0.0
        
        # 1. Conductance (Real Component)
        real_component = np.cos(source.phase_activation) * source.mag_activation
        
        for edge in self.edges:
            if edge.source_id == source.id:
                signal_strength = np.sum(np.abs(real_component))
                energy_conducted[source.id] += signal_strength
                
                # ... propagate signal ...
        
        # 2. Radiation (Imaginary Component)
        imaginary_component = np.sin(source.phase_activation) * source.mag_activation
        
        radiation_targets = self._get_radiation_neighbors(source)
        for target_id, alignment in radiation_targets:
            signal_strength = np.sum(np.abs(imaginary_component)) * alignment
            energy_radiated[source.id] += signal_strength
            
            # ... propagate signal ...
    
    # 3. Update activation strengths (energy conservation)
    for node in self.nodes.values():
        if node.id in energy_conducted or node.id in energy_radiated:
            # Apply costs
            conductance_cost = self.config.conductance_efficiency * energy_conducted.get(node.id, 0.0)
            radiation_cost = self.config.radiation_efficiency * energy_radiated.get(node.id, 0.0)
            
            # Energy depletion
            node.activation_strength -= conductance_cost
            node.activation_strength -= radiation_cost
        
        # Always apply temporal decay
        node.activation_strength *= (1 - self.config.temporal_decay)
        
        # Clamp and check threshold
        node.activation_strength = max(0.0, min(1.0, node.activation_strength))
        node.active = node.activation_strength > self.config.activation_threshold
    
    return self._get_state("forward")
```

### Configuration Parameters

```python
@dataclass
class NetworkConfig:
    # ... existing ...
    
    # Energy conservation
    conductance_efficiency: float = 0.9   # 90% efficient (10% loss)
    radiation_efficiency: float = 0.95    # 95% efficient (5% loss)
    temporal_decay: float = 0.4           # 40% decay per step
    activation_threshold: float = 0.1     # Min to stay active
```

### Example Dynamics

**Scenario 1: High Conductance, No Radiation**
```
Initial: activation = 1.0
Step 1:
  - Conducts 5.0 units of energy
  - Energy cost: 0.1 × 5.0 = 0.5
  - Temporal decay: 0.4 × 1.0 = 0.4
  - New activation: 1.0 - 0.5 - 0.4 = 0.1 (just above threshold!)
```

**Scenario 2: Mixed Conductance + Radiation**
```
Initial: activation = 1.0
Step 1:
  - Conducts 3.0 units
  - Radiates 2.0 units
  - Conduction cost: 0.1 × 3.0 = 0.3
  - Radiation cost: 0.05 × 2.0 = 0.1
  - Temporal decay: 0.4 × 1.0 = 0.4
  - New activation: 1.0 - 0.3 - 0.1 - 0.4 = 0.2 (still active)
```

**Scenario 3: No Transmission (Idle Node)**
```
Initial: activation = 0.5
Step 1:
  - No conductance
  - No radiation
  - Temporal decay only: 0.4 × 0.5 = 0.2
  - New activation: 0.5 - 0.2 = 0.3 (decays naturally)
```

### Physical Interpretation

This models **signal attenuation** and **energy dissipation**:

1. **Conductance Loss**: Like resistance in electrical circuits
   - More signal conducted → more energy lost
   - Resistive heating, impedance matching losses

2. **Radiation Loss**: Like antenna efficiency
   - Broadcasting consumes energy
   - But more efficient because one transmission reaches many

3. **Temporal Decay**: Like capacitor discharge
   - Nodes naturally lose energy over time
   - Represents metabolic costs, leakage currents

4. **Threshold Deactivation**: Like neural refractory period
   - Depleted nodes become inactive
   - Must be re-energized to participate again

---

## 2. Beam Width: Computational Pruning

### Motivation

**Problem**: Exponential growth of active nodes

```
Step 0: 10 input nodes active
Step 1: Each radiates to 3 nodes → 30 active
Step 2: Each radiates to 3 nodes → 90 active
Step 3: Each radiates to 3 nodes → 270 active
...
→ O(branching_factor^depth) complexity!
```

**Solution**: Beam search pruning

Keep only top-K most active nodes at each step.

### Algorithm

**File**: `viz/manager.py`

```python
def step_forward(self) -> StepResult:
    # ... propagation logic ...
    
    # After computing new activations, prune to beam width
    if self.config.beam_width is not None:
        self._prune_to_beam_width()
    
    return self._get_state("forward")

def _prune_to_beam_width(self):
    """
    Keep only top-K most active nodes.
    
    This ensures linear time complexity: O(beam_width) per step.
    """
    # Get all active nodes sorted by activation strength
    active_candidates = [
        (node.id, node.activation_strength) 
        for node in self.nodes.values() 
        if node.activation_strength > 0.0
    ]
    
    # Sort by strength (descending)
    active_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top beam_width nodes
    beam_width = self.config.beam_width
    
    if len(active_candidates) > beam_width:
        # Nodes to keep
        kept_ids = set(node_id for node_id, _ in active_candidates[:beam_width])
        
        # Deactivate pruned nodes
        for node_id, strength in active_candidates[beam_width:]:
            node = self.nodes[node_id]
            node.active = False
            # Optional: Keep activation strength for debugging
            # node.activation_strength = strength  
            # Or zero it out:
            node.activation_strength = 0.0
        
        if self.verbose:
            print(f"Beam pruning: {len(active_candidates)} → {beam_width} active nodes")
```

### Configuration

```python
@dataclass
class NetworkConfig:
    # ... existing ...
    
    # Beam width (computational efficiency)
    beam_width: Optional[int] = None  # None = no pruning, int = max active nodes
```

**Recommended Values**:
- Small networks (<100 nodes): `beam_width=None` (no pruning needed)
- Medium networks (100-1000): `beam_width=50`
- Large networks (>1000): `beam_width=20`

### Time Complexity Analysis

**Without Beam Width**:
```
Worst case: All N nodes activate
  → Radiation: O(N × K × log N) for K neighbors
  → Conductance: O(N × avg_degree)
  → Total per step: O(N²) in dense graphs
```

**With Beam Width B**:
```
At most B nodes active
  → Radiation: O(B × K × log N)
  → Conductance: O(B × avg_degree)
  → Pruning: O(N log B) for top-K selection
  → Total per step: O(B × K × log N) = O(B) if K, log N constant
```

**Result**: Linear in beam width, independent of total graph size!

### Trade-offs

#### Benefits
1. **Constant time per step**: O(beam_width) regardless of graph size
2. **Memory efficient**: Only track B active nodes
3. **Prevents activation explosion**: Bounded growth
4. **Focus on strong signals**: Weak activations filtered out

#### Drawbacks
1. **Information loss**: Pruned nodes may carry useful signals
2. **Non-deterministic**: Ties in activation strength create randomness
3. **Hyperparameter tuning**: Beam width affects accuracy
4. **No global propagation**: Very distant nodes may never activate

### Visualization

**Show beam pruning in UI**:

```javascript
// In script.js
function renderState(state) {
    // ... existing rendering ...
    
    // Highlight beam-pruned nodes
    if (state.beam_info) {
        const prunedNodes = state.beam_info.pruned_ids;
        
        for (const nodeId of prunedNodes) {
            nodes.update({
                id: nodeId,
                borderWidth: 3,
                borderColor: '#ff0000',  // Red border for pruned
                title: 'PRUNED: Below beam threshold'
            });
        }
        
        // Display beam stats
        document.getElementById('beam-stats').innerHTML = `
            Active: ${state.beam_info.active_count} / ${state.beam_info.candidates}
            (Beam: ${state.beam_info.beam_width})
        `;
    }
}
```

**Frontend controls**:

```html
<div class="section">
    <h3>Beam Width</h3>
    <label>
        <input type="checkbox" id="enable-beam"> Enable Beam Pruning
    </label>
    <label>Beam Size: <span id="val-beam">20</span></label>
    <input type="range" id="inp-beam" min="5" max="100" value="20">
</div>
```

---

## Combined Dynamics Example

### Scenario: Temporal Sequence Processing

**Setup**:
```
Network: 100 nodes
Beam width: 20
Input: 5 nodes receive sequence
```

**Step 0**: Input injection
```
Active: 5 input nodes (activation = 1.0)
```

**Step 1**: First propagation
```
Conductance: 5 → 15 connected nodes (conducted)
Radiation: 5 → 30 phase-aligned nodes (radiated)
Total candidates: 35 nodes

Energy depletion:
  - Input nodes: 1.0 → 0.3 (high transmission cost)
  - New nodes: 0.0 → 0.5 (received signals)

Beam pruning: Keep top 20
  → 15 kept, 20 pruned

Active: 20 nodes
```

**Step 2**: Second propagation
```
Conductance: 20 → 40 nodes
Radiation: 20 → 60 nodes
Total candidates: 80 nodes

Energy depletion:
  - Previous active: Decay + transmission costs
  - Some drop below threshold naturally

Beam pruning: Keep top 20
  → 20 kept, 60 pruned

Active: 20 nodes (different ones from step 1)
```

**Step T**: Equilibrium
```
Active nodes stabilize at beam_width
Activation flows like a wave front through the graph
Beam follows the strongest signal paths
```

### Visualization of Wave Front

```
Step 1: ■■■■■ (input layer)
Step 2: □■■■■■■ (wave spreads)
Step 3: □□■■■■■■ (beam moves forward)
Step 4: □□□■■■■■■ (old nodes deactivate)
Step 5: □□□□■■■■■ (steady state: beam width)

Legend:
■ = Active (in beam)
□ = Inactive (below threshold or pruned)
```

---

## Implementation Priority

### Phase 1: Energy Conservation
1. Track energy conducted/radiated per node
2. Apply depletion based on transmission
3. Apply temporal decay
4. Deactivate nodes below threshold
5. Test: Verify nodes deactivate naturally

### Phase 2: Beam Width
1. Add beam_width config parameter
2. Implement `_prune_to_beam_width()`
3. Track pruning statistics
4. Test: Verify linear time complexity

### Phase 3: Visualization
1. Show energy levels as node brightness
2. Highlight pruned nodes
3. Display beam statistics
4. Animate wave front propagation

---

## Configuration Summary

```python
@dataclass
class NetworkConfig:
    # ... existing topology, radiation ...
    
    # Energy conservation
    conductance_efficiency: float = 0.9      # 90% efficient (10% loss per unit)
    radiation_efficiency: float = 0.95       # 95% efficient (5% loss per unit)
    temporal_decay: float = 0.4              # 40% decay per timestep
    activation_threshold: float = 0.1        # Min activation to stay active
    
    # Computational efficiency
    beam_width: Optional[int] = None         # Max active nodes (None = unlimited)
```

---

## Reference Implementation Check

Looking at `fixed_io_nodes/gnn_model.py:230-235`:

```python
# Apply temporal decay to all active nodes (except input nodes on first step)
if input_values is None:  # Only decay when not receiving new input
    for node in self.active_nodes.values():
        node.decay_activations(self.temporal_decay)
```

And `node.py:170-175`:

```python
def decay_activations(self, decay_factor: float):
    """
    Apply temporal decay to activation strength.
    """
    self.activation_strength = self.activation_strength * decay_factor
```

**Observations**:
- ✅ Temporal decay is implemented
- ❓ Energy depletion from transmission: Not explicitly visible in reference
  - May be implicit in how `activation_strength` is computed
  - Or may not be implemented (simplified model)
- ❓ Beam width: Not in reference implementation
  - Reference may handle this differently (early stopping, or no pruning)

**For Viz**: We should implement both mechanisms for realism and efficiency.

---

## Testing Strategy

### Energy Conservation Tests

1. **Idle node decay**: Single active node, no connections
   - Should decay by `temporal_decay` each step
   - Should deactivate after ~log(threshold)/log(1-decay) steps

2. **Transmission cost**: Active node with many outgoing edges
   - Should deplete faster than idle node
   - Cost should be proportional to signal strength

3. **Radiation cost**: Active node radiating to many targets
   - Should have moderate depletion (less than conductance)

### Beam Width Tests

1. **Pruning correctness**: 100 active nodes, beam=20
   - Top 20 by activation strength should remain active
   - Others should be pruned

2. **Time complexity**: Measure step duration vs active nodes
   - With beam: Should be O(beam_width)
   - Without beam: Should be O(N)

3. **Wave front**: Inject signal, watch propagation
   - Should see moving wave of active nodes
   - Width should be bounded by beam_width

---

## Summary

### Energy Conservation
- **Conductance depletes energy**: Proportional to transmitted signal
- **Radiation depletes energy**: Less than conductance (broadcast efficiency)
- **Temporal decay always applies**: Even idle nodes lose activation
- **Threshold deactivation**: Nodes below threshold become inactive
- **Physical realism**: Models signal attenuation and metabolic costs

### Beam Width
- **Computational pruning**: Keep only top-K active nodes
- **Linear time complexity**: O(beam_width) per step
- **Prevents explosion**: Bounds active node growth
- **Trade-off**: Efficiency vs information retention
- **Like beam search**: Focus computational resources on best paths

Both mechanisms are essential:
- Energy conservation → Physical realism, natural deactivation
- Beam width → Computational efficiency, scalability

Together they create a system that:
- Behaves like real physical waves (energy dissipation)
- Scales to large networks (bounded complexity)
- Shows emergent wave-like propagation dynamics
- Is practical to visualize and understand

