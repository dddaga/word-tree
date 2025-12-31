# Temporal Neurograph with Architecture Toggle

## Overview

Create an interactive temporal sequence visualizer with:
1. **Dual Architecture Support**: Toggle between Direct Radiation (Approach 1) and Hierarchical with IC layer (Approach 2)
2. **Table View**: Real-time node state tracking (phase, magnitude, activation, gradients)
3. **Temporal Injection**: Sequential input with phase-based positional encoding
4. **Timeline View**: Historical sparklines showing network evolution
5. **Comparative Metrics**: Performance tracking for both architectures

## Architecture Comparison

### Approach 1: Direct Radiation (Flat)

```
Input Nodes -.radiation.-> Middle Network -.radiation.-> Output Nodes
                                |
                                v
                           Middle (recurrent)
```

**Characteristics**:
- No Input-Connected layer
- Pure phase-based routing
- 2-hop path: Input → Middle → Output
- Tests radiation sufficiency

### Approach 2: Hierarchical (Current)

```
Input Nodes --> Input-Connected --> Middle Network --> Output Nodes
     |              |                      |
     |              |                      v
     |              +---.radiation.->  Middle (recurrent)
     +-----------.radiation.------------^
```

**Characteristics**:
- Input-Connected layer for aggregation
- Dual pathways (static + radiation)
- Explicit feature composition
- 3-hop path: Input → IC → Middle → Output

## Implementation Plan

### 1. Backend: Dual Topology Support ([`manager.py`](manager.py))

**Add `architecture_mode` parameter**:
```python
@dataclass
class NetworkConfig:
    # ... existing fields ...
    architecture_mode: str = "hierarchical"  # "flat" or "hierarchical"
    
    # Conditional fields (only used in hierarchical mode)
    num_input_connected: int = 6
    input_cardinality: int = 2
```

**Modify `_initialize_network()` to support both modes**:
- In "flat" mode: Skip Input-Connected layer creation, only create Input → Middle → Output
- In "hierarchical" mode: Create full topology with Input-Connected layer

**Add temporal injection**:
```python
def inject_temporal_sequence(self, input_values: List[float], timestep: int):
    """
    Inject sequence with temporal phase encoding.
    Phase = position_encoding + temporal_encoding
    Magnitude = content value
    """
    for i, value in enumerate(input_values):
        input_node = self.nodes[f"in_{i}"]
        
        # Content -> Magnitude
        input_node.magnitude = abs(value)
        
        # Position + Time -> Phase
        pos_phase = (i / len(input_values)) * 2 * math.pi
        temp_phase = math.sin(timestep / 20) * math.pi
        input_node.phase = (pos_phase + temp_phase) % (2 * math.pi)
        
        # Activate
        input_node.activation = 1.0
        input_node.active = True
```

**Add history tracking**:
```python
def _track_history(self):
    """Store current state in history buffer (last 100 steps)."""
    if not hasattr(self, 'history'):
        self.history = {node_id: [] for node_id in self.nodes.keys()}
    
    for node_id, node in self.nodes.items():
        self.history[node_id].append({
            'phase': node.phase,
            'magnitude': node.magnitude,
            'activation': node.activation,
            'gradient': node.gradient_phase,
            'timestep': self.step_count
        })
        
        if len(self.history[node_id]) > 100:
            self.history[node_id].pop(0)
```

### 2. Backend API Updates ([`server.py`](server.py))

**Update config models**:
```python
class InitConfig(BaseModel):
    # ... existing fields ...
    architecture_mode: str = "hierarchical"  # NEW

class ConfigUpdate(BaseModel):
    # ... existing fields ...
    architecture_mode: Optional[str] = None  # NEW
```

**New endpoints**:
```python
@app.post("/api/inject_sequence")
async def inject_sequence(data: dict):
    """Inject temporal sequence."""
    session = get_session()
    session.network.inject_temporal_sequence(
        data["sequence"], 
        data.get("timestep", 0)
    )
    return {"status": "injected", "timestep": data.get("timestep", 0)}

@app.get("/api/history/{node_id}")
async def get_node_history(node_id: str, limit: int = 50):
    """Get historical states for a node."""
    session = get_session()
    return session.network.get_history(node_id, limit)

@app.post("/api/metrics")
async def get_metrics():
    """Get performance metrics."""
    session = get_session()
    return {
        "total_params": len(session.network.edges),
        "active_nodes": sum(1 for n in session.network.nodes.values() if n.active),
        "avg_gradient": sum(abs(n.gradient_phase) for n in session.network.nodes.values()) / len(session.network.nodes),
        "mode": session.network.config.architecture_mode
    }
```

### 3. Frontend UI ([`static/index.html`](static/index.html))

**Three-Panel Layout**:
```
┌─────────────────────────────────────────────┐
│  Network Visualization (60% height)         │
│          Vis.js Graph                       │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│  Node Details Table (25% height)            │
│  ID | Role | Phase | Mag | Act | Grad | Acc │
│  (scrollable, sortable)                     │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│  Timeline View (15% height)                 │
│  Sparklines for phase/activation history    │
└─────────────────────────────────────────────┘
```

**New Controls**:
```html
<div class="section">
    <h3>Architecture Mode</h3>
    <div class="btn-group">
        <button id="btn-mode-flat" class="mode-btn">Direct Radiation</button>
        <button id="btn-mode-hierarchical" class="mode-btn primary">Hierarchical</button>
    </div>
</div>

<div class="section">
    <h3>Temporal Sequence</h3>
    <label>Input Sequence:</label>
    <input type="text" id="inp-sequence" placeholder="0.5, 0.8, 0.3" value="0.5, 0.8, 0.3">
    
    <div class="btn-group">
        <button id="btn-inject-seq" class="primary">Inject & Step</button>
        <button id="btn-auto-seq">Auto Mode</button>
    </div>
    
    <div>Timestep: <span id="timestep-count" class="value">0</span></div>
</div>

<div class="section">
    <h3>Metrics</h3>
    <div>Static Edges: <span id="metric-edges" class="value">--</span></div>
    <div>Active Nodes: <span id="metric-active" class="value">--</span></div>
    <div>Avg Gradient: <span id="metric-grad" class="value">--</span></div>
</div>
```

**Table Structure**:
```html
<div id="table-container">
    <table id="node-table">
        <thead>
            <tr>
                <th>ID</th>
                <th>Role</th>
                <th>Phase (rad)</th>
                <th>Magnitude</th>
                <th>Activation</th>
                <th>Gradient</th>
                <th>Accumulator</th>
            </tr>
        </thead>
        <tbody id="node-table-body"></tbody>
    </table>
</div>

<div id="timeline-container">
    <canvas id="timeline-canvas"></canvas>
</div>
```

### 4. Frontend Logic ([`static/script.js`](static/script.js))

**State Management**:
```javascript
let currentArchMode = "hierarchical";
let timestep = 0;
let nodeHistory = {}; // {node_id: [{phase, mag, act, grad, timestep}, ...]}
let autoSeqInterval = null;
```

**Architecture Toggle**:
```javascript
async function switchToFlat() {
    currentArchMode = "flat";
    await apiCall('/api/config', { architecture_mode: "flat" });
    await initNetwork();
    updateModeUI();
}

async function switchToHierarchical() {
    currentArchMode = "hierarchical";
    await apiCall('/api/config', { architecture_mode: "hierarchical" });
    await initNetwork();
    updateModeUI();
}

function updateModeUI() {
    // Update button states, disable IC controls in flat mode
}
```

**Temporal Injection**:
```javascript
async function injectSequence() {
    const input = document.getElementById('inp-sequence').value;
    const sequence = input.split(',').map(v => parseFloat(v.trim()) || 0);
    
    await apiCall('/api/inject_sequence', {
        sequence: sequence,
        timestep: timestep
    });
    
    timestep++;
    await stepForward();
    updateHistory(lastState);
    renderTable(lastState);
    renderTimeline();
    updateMetrics();
}

function startAutoSequence() {
    autoSeqInterval = setInterval(async () => {
        // Generate sinusoidal pattern
        const numInputs = parseInt(document.getElementById('inp-n-in').value);
        const sequence = Array(numInputs).fill(0).map((_, i) => 
            Math.sin((timestep * 0.1) + (i * 0.5)) * 0.5 + 0.5
        );
        document.getElementById('inp-sequence').value = sequence.map(v => v.toFixed(2)).join(', ');
        await injectSequence();
    }, 600);
}
```

**Table Rendering**:
```javascript
function renderTable(state) {
    const tbody = document.getElementById('node-table-body');
    tbody.innerHTML = '';
    
    // Sort by role: Input → Input-Connected → Middle → Output
    const roleOrder = {'input': 0, 'input_connected': 1, 'middle': 2, 'output': 3};
    const sortedNodes = Object.values(state.nodes).sort((a, b) => {
        const roleA = roleOrder[a.role] || 99;
        const roleB = roleOrder[b.role] || 99;
        if (roleA !== roleB) return roleA - roleB;
        return a.id.localeCompare(b.id);
    });
    
    for (const node of sortedNodes) {
        const row = document.createElement('tr');
        row.className = `role-${node.role}`;
        row.onclick = () => selectNodeFromTable(node.id);
        
        row.innerHTML = `
            <td class="mono">${node.id}</td>
            <td><span class="badge badge-${node.role}">${node.role}</span></td>
            <td class="numeric" style="color: ${phaseToColor(node.phase)}">${node.phase.toFixed(4)}</td>
            <td class="numeric">${node.magnitude.toFixed(4)} ${createMiniBar(node.magnitude, 1.0, '#00d4ff')}</td>
            <td class="numeric ${node.activation > 0.1 ? 'highlight' : ''}">${node.activation.toFixed(4)}</td>
            <td class="numeric" style="color: ${node.gradient > 0 ? '#00ff88' : '#ff0055'}">${node.gradient.toFixed(5)}</td>
            <td class="numeric">${node.accumulator.toFixed(5)} ${createMiniBar(Math.abs(node.accumulator), 0.5, '#ff00aa')}</td>
        `;
        
        tbody.appendChild(row);
    }
}
```

**Timeline Sparklines**:
```javascript
function renderTimeline() {
    const canvas = document.getElementById('timeline-canvas');
    const ctx = canvas.getContext('2d');
    
    // Draw sparklines for representative nodes from each layer
    const selectedNodes = ['in_0', 'inc_0', 'mid_0', 'out_0']
        .filter(id => nodeHistory[id] && nodeHistory[id].length > 1);
    
    // Plot phase/activation/gradient over time
    // Color-coded by node role
}
```

### 5. CSS Styling

```css
/* Three-panel layout */
#mynetwork { height: 60vh; }

#table-container {
    position: absolute;
    bottom: 15vh;
    left: 360px;
    right: 20px;
    height: 25vh;
    overflow-y: auto;
    background: rgba(20, 20, 30, 0.95);
    border: 1px solid #444;
}

#timeline-container {
    position: absolute;
    bottom: 0;
    left: 360px;
    right: 20px;
    height: 14vh;
    background: rgba(20, 20, 30, 0.95);
    border: 1px solid #444;
}

/* Table styles */
#node-table thead { position: sticky; top: 0; background: #1a1a2e; }
#node-table tr:hover { background: rgba(0, 212, 255, 0.15); cursor: pointer; }

.role-input { background: rgba(0, 100, 255, 0.08); }
.role-input_connected { background: rgba(150, 0, 255, 0.06); }
.role-middle { background: rgba(100, 0, 150, 0.05); }
.role-output { background: rgba(0, 255, 100, 0.08); }

.numeric { text-align: right; font-family: monospace; }
.highlight { font-weight: bold; color: #fff; text-shadow: 0 0 5px #00d4ff; }

/* Mini bar visualization */
.mini-bar {
    display: inline-block;
    height: 6px;
    margin-left: 5px;
    border-radius: 2px;
}

/* Badges */
.badge {
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 0.7rem;
    font-weight: bold;
    text-transform: uppercase;
}

.badge-input { background: #0066ff; color: #fff; }
.badge-input_connected { background: #9600ff; color: #fff; }
.badge-middle { background: #640096; color: #fff; }
.badge-output { background: #00ff66; color: #000; }
```

## Key Features Summary

1. **Architecture Toggle**: Switch between flat and hierarchical modes in real-time
2. **Temporal Encoding**: Phase = position + time, Magnitude = content
3. **Table View**: All nodes with sortable columns, color-coded by role
4. **Timeline**: Historical sparklines for phase/activation/gradient evolution
5. **Metrics**: Compare architectures (edges, active nodes, avg gradient)
6. **Auto-Sequence**: Generate sinusoidal patterns for continuous testing

## Files to Modify

- [`manager.py`](manager.py): Dual topology, temporal injection, history tracking
- [`server.py`](server.py): New endpoints and config updates
- [`static/index.html`](static/index.html): UI controls, table, timeline, CSS
- [`static/script.js`](static/script.js): Mode toggle, table/timeline rendering, temporal logic

## Expected Outcome

Users can:
- Toggle between flat and hierarchical architectures in real-time
- Inject temporal sequences with phase-based positional encoding
- View all node states in a sortable table with visual indicators
- Track network evolution through timeline sparklines
- Compare architectures using metrics
- Run auto-sequence mode with generated patterns
- Click table rows to select and inspect nodes
- Export table data for external analysis

