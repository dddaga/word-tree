# Radiation Mechanism: Phase-Based Dynamic Connectivity

## Overview

**Radiation** is a dynamic, attention-like mechanism that creates temporary connections between nodes based on **phase alignment**. Unlike static edges (conductance), radiation connections change every step based on the current phase states of nodes.

## Physical Analogy

Think of it like **resonance** in physics:
- Tuning forks with similar frequencies resonate strongly
- Atoms with matching energy levels couple via photon absorption/emission
- Antennas tuned to the same frequency communicate efficiently

In the Neurograph:
- Nodes with **similar phase vectors** are "in phase" and can radiate to each other
- This creates dynamic pathways that adapt to the current computational state
- Like light diffracting to reach distant points without direct line-of-sight

---

## Implementation Details

### 1. Computing Radiation Targets

**Source**: `gnn_model.py:89-121`

```python
def _compute_radiation_targets(self, nodes: List[Node], k: int = None):
    """
    Find top-K nodes with most similar phase vectors.
    
    Arguments:
        nodes: List of source nodes
        k: Number of radiation targets per node (default: self.radiation_targets)
    
    Returns:
        topk_indices: Dict[node_id -> List[target_node_ids]]
    """
    if k is None:
        k = self.radiation_targets
    
    topk_indices = {}
    
    # Extract phase activation vectors as query vectors
    query_vectors = [node.phase_activation for node in nodes]
    
    # Batch search in vector database (Qdrant)
    batch_results = self.node_store.search_nodes_batch(
        query_vectors, 
        vector_name='phase',  # Search in phase vector space
        limit=k,
        with_payload=False, 
        with_vectors=False
    )
    
    # Map results to node IDs
    for i, node in enumerate(nodes):
        topk_indices[node.id] = [found_point.id for found_point in batch_results[i]]
    
    return topk_indices
```

### 2. Similarity Metric

The vector database (Qdrant) uses **cosine similarity** by default:

```
similarity(phase_A, phase_B) = (phase_A · phase_B) / (||phase_A|| × ||phase_B||)
```

**Why cosine similarity?**
- Measures angle between vectors, not magnitude
- Returns value in [-1, 1]: 
  - +1 = perfectly aligned (same direction)
  - 0 = orthogonal (no alignment)
  - -1 = opposite (anti-aligned)
- Natural for phase vectors where direction matters more than magnitude

**Visualization**:
```
phase_A = [cos(φ₁), cos(φ₂), ..., cos(φₙ)]
phase_B = [cos(ψ₁), cos(ψ₂), ..., cos(ψₙ)]

If phases are similar:
  cos(φᵢ) ≈ cos(ψᵢ) for most i
  → High cosine similarity
  → Node B becomes radiation target of A
```

### 3. Exclusion Rules

**Source**: `gnn_model.py:210-212` (in current viz implementation)

```python
# Exclude static neighbors and self from radiation targets
static_neighbors = {e.target_id for e in self.edges if e.source_id == source.id}
static_neighbors.add(source.id)

# Only consider nodes NOT in static_neighbors
candidates = [node for node in all_nodes if node.id not in static_neighbors]
```

**Rationale**:
- Don't radiate to nodes already connected via static edges (redundant)
- Don't radiate to self (no self-loops)
- Focus radiation on creating **bridges between distant parts of the graph**

### 4. Radiation Signal Propagation

**Source**: `gnn_model.py:254-264` (from viz, similar in reference)

```python
radiation_targets = self._get_radiation_neighbors(source)

for target_id, alignment_score in radiation_targets:
    # Radiation strength scales with alignment
    strength = source_signal * alignment_score * 0.5
    
    # Phase shift applied (like static edges)
    arriving_phase = source.phase_activation + radiation_phase_shift
    arriving_mag = source.mag_activation
    
    # Store contribution for aggregation
    updates[target_id] += (arriving_phase, arriving_mag, strength)
```

**Key Points**:
- Radiation strength proportional to phase alignment
- 0.5 multiplier makes radiation weaker than static conductance
- Same interference rules apply: phase shifts add, weighted aggregation

---

## Comparison: Conductance vs Radiation

| Aspect | Static Conductance | Dynamic Radiation |
|--------|-------------------|-------------------|
| **Connectivity** | Fixed edges, never change | Changes every forward step |
| **Selection** | Graph topology (initialization) | Phase similarity (runtime) |
| **Strength** | Full weight (1.0 multiplier) | Scaled by alignment (0.5 multiplier) |
| **Range** | Local (graph neighbors) | Global (any node in network) |
| **Purpose** | Structured information flow | Long-range dependencies, shortcuts |
| **Analogy** | Wires in a circuit | Wireless radio transmission |

---

## Why Radiation Matters

### 1. **Overcomes Graph Bottlenecks**

Consider this topology:
```
Input → Layer1 → [Bottleneck] → Layer2 → Output
```

**Without radiation**:
- All information must flow through bottleneck
- Limited capacity = information loss

**With radiation**:
- Input can directly radiate to Layer2 if phases align
- Creates bypass routes around bottlenecks
- Emergent shortcut connections

### 2. **Adaptive Routing**

The network learns which nodes should communicate:
- If Output needs information from specific Input
- Backprop adjusts phases to increase alignment
- Next forward pass: stronger radiation connection
- **Self-organizing communication paths**

### 3. **Temporal Dynamics**

For temporal sequences:
```
t=0: Input₀ → Middle (phase φ₀)
t=1: Input₁ → Middle (phase φ₁)
t=2: Input₂ → Middle (phase φ₂)

If φ₀ ≈ φ₂ but φ₁ different:
  → Radiation connects t=0 and t=2 states
  → Captures long-range temporal correlations
  → Like attention mechanism in Transformers
```

### 4. **Isolated Layer Communication**

In the proposed architecture:
- Input layer (isolated)
- Input-Connected layer (isolated)
- Middle layer (isolated, recurrent)
- Output layer (isolated)

**Without radiation**: Layers can't communicate!

**With radiation**: 
- Phase alignment creates bridges
- Information flows between islands
- Network decides which cross-layer connections matter

---

## Mathematical Formulation

### Effective Connectivity Matrix

At each timestep `t`, the network has:

**Static adjacency**: `A_static` (fixed)

**Dynamic adjacency**: `A_radiation(t)` (depends on current phases)

```
A_radiation(t)[i,j] = {
    alignment(phase_i(t), phase_j(t))  if j ∈ top-K(i)
    0                                   otherwise
}

where alignment(φ_i, φ_j) = cosine_similarity(φ_i, φ_j)
```

**Total connectivity**:
```
A_total(t) = A_static + α × A_radiation(t)
```
where `α ≈ 0.5` (radiation strength multiplier)

### Information Flow

Signal from node `i` to node `j`:
```
s_ij = {
    w_ij × signal_i              if (i,j) ∈ static edges (conductance)
    alignment_ij × signal_i       if j ∈ radiation_targets(i)
    0                             otherwise
}
```

Node `j` receives:
```
phase_j(t+1) = weighted_sum( phase_i(t) + shift_ij ) for all i→j
mag_j(t+1) = weighted_sum( mag_i(t) + mag_shift_ij ) for all i→j

where weights = softmax(signal_strengths)
```

---

## Visualization of Radiation

### Current Viz Implementation

**Source**: `viz/static/script.js:207-225`

```javascript
// Remove old radiation edges
const oldRadiation = edges.get({ filter: e => e.radiation === true });
edges.remove(oldRadiation);

// Add new radiation edges (transient, dashed)
const newRadiation = state.active_signals
    .filter(s => s.type === 'radiation')
    .map((rad, idx) => ({
        id: `rad-${idx}`,
        from: rad.source,
        to: rad.target,
        color: { color: rad.is_backward ? '#ff00aa' : '#aa00ff', opacity: 0.6 },
        width: 1 + rad.strength * 3,
        dashes: [5, 5],  // Dashed line to distinguish from static
        radiation: true,
        physics: false,
        arrows: rad.is_backward ? { from: true } : { to: true }
    }));
edges.add(newRadiation);
```

**Visual Properties**:
- **Dashed lines**: Distinguish from solid static edges
- **Purple color**: `#aa00ff` for forward, `#ff00aa` for backward
- **Dynamic**: Removed and re-added every frame
- **Width scales with strength**: Stronger alignment = thicker line
- **No physics**: Fixed positions (don't affect layout)

### Suggested Enhanced Visualization

1. **Pulsing Animation**: Radiation edges pulse/fade to show dynamic nature
2. **Alignment Heatmap**: Color gradient based on phase similarity
3. **Flow Particles**: Animate particles flowing through radiation paths
4. **Phase Wheel**: Show phase vectors on unit circle, highlight aligned nodes
5. **Temporal Trace**: Show history of radiation connections as faint trails

---

## Radiation Parameters

### Configuration

**Source**: `NetworkConfig` in `manager.py`

```python
@dataclass
class NetworkConfig:
    radiation_k: int = 3          # Top-K radiation neighbors
    use_radiation: bool = True     # Enable/disable radiation
    radiation_strength: float = 0.5  # Multiplier for radiation signals
```

**Tuning Guidelines**:
- **`radiation_k` (K value)**:
  - Small (K=1-3): Sparse, focused connections
  - Medium (K=5-10): Balanced
  - Large (K=20+): Dense, expensive computation
  - Default: 3 (good for small networks <100 nodes)

- **`radiation_strength`**:
  - 0.0: Disabled (static graph only)
  - 0.3-0.5: Weaker than static (typical)
  - 1.0: Equal to static
  - >1.0: Stronger than static (unusual)

### Computational Cost

**Time Complexity**:
- Vector search: `O(N × d)` where N=nodes, d=vector_dim
- With Qdrant (HNSW index): `O(log N × d)` (much faster!)
- Per step: `O(active_nodes × log N × K)`

**Space Complexity**:
- Radiation edges: `O(active_nodes × K)` (transient, recreated each step)
- Qdrant index: `O(N × d)` (persistent)

**Optimization**:
- Only compute radiation for active nodes
- Batch vector searches (done in reference implementation)
- Cache radiation targets if phases don't change much (not yet implemented)

---

## Radiation in Visualization Plan

### Phase 1: Core Implementation

**File**: `viz/manager.py`

```python
def _get_radiation_neighbors(self, source: Node, k: int = None) -> List[Tuple[str, float]]:
    """
    Find top-K nodes with highest phase alignment to source.
    
    Returns:
        List of (target_id, alignment_score) tuples
    """
    if not self.config.use_radiation:
        return []
    
    if k is None:
        k = self.config.radiation_k
    
    # Exclude static neighbors and self
    static_neighbors = {e.target_id for e in self.edges if e.source_id == source.id}
    static_neighbors.add(source.id)
    
    # Compute cosine similarity with all other nodes
    candidates = []
    for node_id, node in self.nodes.items():
        if node_id in static_neighbors:
            continue
        
        # Cosine similarity between phase vectors
        dot_product = np.dot(source.phase_activation, node.phase_activation)
        norm_source = np.linalg.norm(source.phase_activation)
        norm_target = np.linalg.norm(node.phase_activation)
        
        similarity = dot_product / (norm_source * norm_target + 1e-8)
        
        # Normalize to [0, 1] range
        alignment_score = (similarity + 1) / 2
        
        candidates.append((node_id, alignment_score))
    
    # Sort by alignment score (descending) and take top-K
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:k]
```

### Phase 2: Enhanced Visualization

**File**: `viz/static/script.js`

Add controls:
```javascript
// Radiation Toggle
<button id="btn-toggle-radiation">Toggle Radiation</button>

// K Value Slider
<label>Radiation K: <span id="val-k">3</span></label>
<input type="range" id="inp-k" min="0" max="10" value="3">

// Alignment Threshold
<label>Min Alignment: <span id="val-align-thresh">0.5</span></label>
<input type="range" id="inp-align-thresh" min="0" max="100" value="50">
```

Visual enhancements:
```javascript
// Animate radiation edges with pulsing effect
function animateRadiation() {
    const radiationEdges = edges.get({ filter: e => e.radiation });
    
    radiationEdges.forEach(edge => {
        const pulse = Math.sin(Date.now() / 200) * 0.3 + 0.7;
        edges.update({
            id: edge.id,
            color: { opacity: pulse }
        });
    });
    
    requestAnimationFrame(animateRadiation);
}
```

---

## Comparison to Transformer Attention

| Aspect | Transformer Attention | Neurograph Radiation |
|--------|----------------------|---------------------|
| **Similarity Metric** | Query·Key / √d_k | Cosine(phase_A, phase_B) |
| **Selection** | All pairs (softmax) | Top-K only |
| **Values** | Learned projection | Phase-shifted activations |
| **Computation** | O(N²) per layer | O(N log N) per step |
| **Topology** | Fully connected | Sparse dynamic graph |
| **Mechanism** | Attention weights | Wave interference |

**Key Insight**: Radiation is like **sparse, phase-based attention** with:
- Top-K selection (efficient)
- Physical wave dynamics (interpretable)
- Graph structure preservation (inductive bias)

---

## Summary

**Radiation Mechanism**:
1. **Compute phase similarity** between all node pairs (cosine similarity)
2. **Select top-K most aligned** nodes as radiation targets
3. **Propagate signals** through radiation paths (phase-shifted, weighted)
4. **Aggregate with attention** at receiving nodes
5. **Update every step** based on current phase states

**Purpose**:
- Create dynamic shortcuts between distant nodes
- Enable information flow between isolated graph components
- Capture long-range dependencies (spatial and temporal)
- Self-organize communication pathways through learning

**Physical Analogy**:
- Like electromagnetic radiation connecting antennas tuned to the same frequency
- Or quantum entanglement between particles with matching states
- Creates "action at a distance" without direct wiring

This mechanism is what makes the Neurograph unique—it combines the structure of graph neural networks with the dynamics of attention mechanisms, all grounded in wave physics!

