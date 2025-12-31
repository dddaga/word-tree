// Vis.js Graph Setup
const nodes = new vis.DataSet([]);
const edges = new vis.DataSet([]);
const container = document.getElementById('mynetwork');

const data = { nodes, edges };

const options = {
    nodes: {
        shape: 'dot',
        font: { color: '#ffffff', size: 14, face: 'monospace', strokeWidth: 3, strokeColor: '#000' },
        borderWidth: 2,
        shadow: true
    },
    edges: {
        width: 1,
        color: { color: '#444', highlight: '#00d4ff' },
        smooth: { type: 'continuous' },
        arrows: { to: { enabled: true, scaleFactor: 0.5 } } // Directed graph
    },
    physics: {
        enabled: true,
        barnesHut: {
            gravitationalConstant: -4000,
            centralGravity: 0.1,
            springLength: 150,
            springConstant: 0.04,
            damping: 0.09
        },
        stabilization: { iterations: 200 }
    },
    interaction: { hover: true, tooltipDelay: 200 }
};

const network = new vis.Network(container, data, options);

// State
let isPlaying = false;
let playInterval = null;
let currentMode = "forward"; // forward, backward
let selectedNodeId = null;
let currentArchMode = "hierarchical";
let timestep = 0;
let nodeHistory = {}; // {node_id: [{phase, mag, act, grad, timestep}, ...]}
let autoSeqInterval = null;
let temporalPeriod = 40;
let lastState = null;

// === API ===

async function apiCall(endpoint, body = {}) {
    try {
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        return await res.json();
    } catch (e) {
        console.error(`API Error ${endpoint}:`, e);
        return null;
    }
}

async function initNetwork() {
    const config = {
        num_input: parseInt(document.getElementById('inp-n-in').value),
        num_input_connected: parseInt(document.getElementById('inp-n-inc').value),
        num_middle: parseInt(document.getElementById('inp-n-mid').value),
        num_output: parseInt(document.getElementById('inp-n-out').value),
        input_cardinality: parseInt(document.getElementById('inp-c-in').value),
        middle_cardinality: parseInt(document.getElementById('inp-c-mid').value),
        radiation_k: parseInt(document.getElementById('inp-k').value),
        learning_rate: parseInt(document.getElementById('inp-lr').value) / 100.0,
        architecture_mode: currentArchMode
    };
    
    // Reset temporal state
    timestep = 0;
    nodeHistory = {};
    document.getElementById('timestep-count').textContent = '0';
    
    const state = await apiCall('/api/init', config);
    if (state) {
        renderState(state, true);
        updateModeUI();
    }
}

async function stepForward() {
    const state = await apiCall('/api/step');
    if (state) {
        currentMode = "forward";
        renderState(state, false);
    }
}

async function stepBackward() {
    const state = await apiCall('/api/backprop');
    if (state) {
        currentMode = "backward";
        renderState(state, false);
    }
}

// === Rendering ===

function renderState(state, reset = false) {
    if (reset) {
        nodes.clear();
        edges.clear();
    }

    // --- Nodes ---
    const nodeUpdates = [];
    const containerRect = container.getBoundingClientRect();
    const width = containerRect.width || 800;
    const height = containerRect.height || 600;

    for (const [id, n] of Object.entries(state.nodes)) {
        // Visuals
        const hue = (n.phase / (2 * Math.PI)) * 360;
        const color = `hsl(${hue}, ${70 + n.magnitude * 30}%, 50%)`;
        const size = 10 + n.magnitude * 10;
        
        let label = '';
        if (n.role === 'input') label = 'IN';
        else if (n.role === 'output') label = 'OUT';
        
        // Highlight active nodes or gradients
        let borderWidth = 2;
        let borderColor = color;
        
        if (state.mode === 'backward' && Math.abs(n.gradient) > 0.01) {
            borderColor = '#ff00aa'; // Pink for gradient update
            borderWidth = 2 + Math.abs(n.gradient) * 10;
        } else if (n.active) {
            borderColor = '#ffffff'; // White for activation
            borderWidth = 4;
        }

        const nodeData = {
            id: id,
            label: label,
            color: {
                background: color,
                border: borderColor,
                highlight: { background: color, border: '#fff' }
            },
            size: size,
            borderWidth: borderWidth,
            title: `Phase: ${n.phase.toFixed(2)}\nMag: ${n.magnitude.toFixed(2)}\nRole: ${n.role}`
        };

        // Layout Hints (only on reset to let physics take over)
        if (reset) {
            if (n.role === 'input') {
                nodeData.x = -width * 0.4;
                nodeData.y = (Math.random() - 0.5) * height * 0.5;
            } else if (n.role === 'output') {
                nodeData.x = width * 0.4;
                nodeData.y = (Math.random() - 0.5) * height * 0.5;
            } else if (n.role === 'input_connected') {
                nodeData.x = -width * 0.2;
            } else if (n.role === 'middle') {
                nodeData.x = 0;
            }
        }
        
        nodeUpdates.push(nodeData);
        
        // Update inspector if selected
        if (selectedNodeId === id) updateInspector(n);
    }
    nodes.update(nodeUpdates);

    // --- Edges ---
    // 1. Static Edges
    if (reset) {
        const staticEdges = state.edges.map(e => ({
            id: `${e.source}-${e.target}`,
            from: e.source,
            to: e.target,
            color: { color: '#444', opacity: 0.3 },
            width: 1,
            arrows: 'to',
            dashes: false
        }));
        edges.update(staticEdges);
    }

    // 2. Active Signals (Animation)
    // Reset all edge styles first
    edges.forEach(e => {
        if (!e.radiation) { // Don't clear radiation yet
            edges.update({
                id: e.id, 
                width: 1, 
                color: { color: '#444', opacity: 0.3 }
            });
        }
    });

    // Highlight active paths
    state.active_signals.forEach(sig => {
        if (sig.type === 'conductance') {
            const edgeId = `${sig.source}-${sig.target}`;
            // Handle reverse direction for backprop visualization
            const targetEdgeId = sig.is_backward ? `${sig.target}-${sig.source}` : edgeId;
            
            const existing = edges.get(targetEdgeId);
            if (existing) {
                edges.update({
                    id: targetEdgeId,
                    width: 2 + sig.strength * 4,
                    color: sig.is_backward ? '#ff0055' : '#00d4ff', // Red for backward, Cyan for forward
                    opacity: 1.0
                });
            }
        }
    });

    // 3. Radiation (Transient)
    const oldRadiation = edges.get({ filter: e => e.radiation === true });
    edges.remove(oldRadiation);

    const newRadiation = state.active_signals
        .filter(s => s.type === 'radiation')
        .map((rad, idx) => ({
            id: `rad-${idx}`,
            from: rad.source,
            to: rad.target,
            color: { color: rad.is_backward ? '#ff00aa' : '#aa00ff', opacity: 0.6 },
            width: 1 + rad.strength * 3,
            dashes: [5, 5],
            radiation: true,
            physics: false,
            arrows: rad.is_backward ? { from: true } : { to: true }
        }));
    edges.add(newRadiation);

    // Update UI Status
    document.getElementById('step-count').textContent = state.step_number;
    document.getElementById('mode-indicator').textContent = state.mode.toUpperCase();
    document.getElementById('mode-indicator').style.color = state.mode === 'backward' ? '#ff0055' : '#00d4ff';
    
    // Store for history tracking
    lastState = state;
    
    // Update table and timeline
    renderTable(state);
    updateHistory(state);
    renderTimeline();
    updateMetrics(state);
}

// === Inspection ===

function updateInspector(node) {
    const div = document.getElementById('inspect-content');
    const hue = (node.phase / (2 * Math.PI)) * 360;
    
    let html = `
        <div class="section">
            <h3>${node.role.toUpperCase()} Node</h3>
            <div class="inspect-row"><span>ID:</span> <span class="value">${node.id}</span></div>
        </div>
        <div class="section">
            <h3>State</h3>
            <div class="inspect-row"><span>Phase:</span> <span class="inspect-val">${node.phase.toFixed(4)} rad</span></div>
            <div class="inspect-row"><span>Magnitude:</span> <span class="inspect-val">${node.magnitude.toFixed(4)}</span></div>
            <div class="inspect-row"><span>Activation:</span> <span class="inspect-val">${node.activation.toFixed(4)}</span></div>
            <div class="inspect-row"><span>Color:</span> <span style="color: hsl(${hue}, 80%, 50%)">■</span></div>
        </div>
    `;
    
    if (node.target_phase !== null) {
        html += `
            <div class="section">
                <h3>Target</h3>
                <div class="inspect-row"><span>Target Phase:</span> <span class="value">${node.target_phase.toFixed(4)}</span></div>
                <div class="inspect-row"><span>Error:</span> <span class="inspect-grad">${(node.target_phase - node.phase).toFixed(4)}</span></div>
            </div>
        `;
    }
    
    html += `
        <div class="section">
            <h3>Gradients</h3>
            <div class="inspect-row"><span>Current Grad:</span> <span class="inspect-grad">${node.gradient.toFixed(5)}</span></div>
            <div class="inspect-row"><span>Accumulator:</span> <span class="inspect-grad">${node.accumulator.toFixed(5)}</span></div>
        </div>
    `;
    
    div.innerHTML = html;
    document.getElementById('inspector').style.display = 'block';
}

network.on("click", function (params) {
    if (params.nodes.length > 0) {
        selectedNodeId = params.nodes[0];
        // Fetch current state to populate inspector immediately
        const node = nodes.get(selectedNodeId); // This is just viz data
        // We rely on the next render loop to update inspector with full data, 
        // OR we can fetch state. For now, next step updates it.
    } else {
        selectedNodeId = null;
        document.getElementById('inspector').style.display = 'none';
    }
});

// === Controls ===

// Sliders
['inp-n-in', 'inp-n-inc', 'inp-n-mid', 'inp-n-out', 'inp-c-in', 'inp-c-mid', 'inp-k', 'inp-lr'].forEach(id => {
    document.getElementById(id).addEventListener('input', (e) => {
        // Update label
        const labelId = id.replace('inp-', 'val-');
        let val = e.target.value;
        if (id === 'inp-lr') val = (val / 100).toFixed(2);
        document.getElementById(labelId).textContent = val;
    });
    
    // Config update on change
    document.getElementById(id).addEventListener('change', (e) => {
        // If structural, re-init
        if (id.includes('-n-') || id.includes('-c-')) {
            initNetwork();
        } else {
            // Parameter update
            const config = {};
            if (id === 'inp-k') config.radiation_k = parseInt(e.target.value);
            if (id === 'inp-lr') config.learning_rate = parseInt(e.target.value) / 100.0;
            apiCall('/api/config', config);
        }
    });
});

document.getElementById('btn-step-fwd').addEventListener('click', () => {
    if (isPlaying) stopAuto();
    stepForward();
});

document.getElementById('btn-step-bwd').addEventListener('click', () => {
    if (isPlaying) stopAuto();
    stepBackward();
});

document.getElementById('btn-reset').addEventListener('click', initNetwork);

document.getElementById('btn-play').addEventListener('click', () => {
    if (isPlaying) {
        stopAuto();
        stopAutoSequence();
    } else {
        startAuto();
    }
});

function startAuto() {
    isPlaying = true;
    document.getElementById('btn-play').textContent = "Stop Auto";
    document.getElementById('btn-play').classList.add('active');
    
    // Training Loop: Forward -> Forward -> Backward
    playInterval = setInterval(async () => {
        if (currentMode === 'forward') {
            // Sometimes do backward
            if (Math.random() > 0.7) await stepBackward();
            else await stepForward();
        } else {
            await stepForward();
        }
    }, 400);
}

function stopAuto() {
    isPlaying = false;
    document.getElementById('btn-play').textContent = "Auto Train";
    document.getElementById('btn-play').classList.remove('active');
    clearInterval(playInterval);
}

// === Table Rendering ===

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
        
        // ID
        const tdId = document.createElement('td');
        tdId.className = 'mono';
        tdId.textContent = node.id;
        row.appendChild(tdId);
        
        // Role
        const tdRole = document.createElement('td');
        tdRole.innerHTML = `<span class="badge badge-${node.role}">${node.role}</span>`;
        row.appendChild(tdRole);
        
        // Phase
        const tdPhase = document.createElement('td');
        tdPhase.className = 'numeric';
        tdPhase.textContent = node.phase.toFixed(4);
        tdPhase.style.color = phaseToColor(node.phase);
        row.appendChild(tdPhase);
        
        // Magnitude
        const tdMag = document.createElement('td');
        tdMag.className = 'numeric';
        tdMag.innerHTML = `${node.magnitude.toFixed(4)} ${createMiniBar(node.magnitude, 1.0, '#00d4ff')}`;
        row.appendChild(tdMag);
        
        // Activation
        const tdAct = document.createElement('td');
        tdAct.className = 'numeric';
        tdAct.textContent = node.activation.toFixed(4);
        if (node.activation > 0.1) tdAct.classList.add('highlight');
        row.appendChild(tdAct);
        
        // Gradient
        const tdGrad = document.createElement('td');
        tdGrad.className = 'numeric';
        tdGrad.textContent = node.gradient.toFixed(5);
        tdGrad.style.color = node.gradient > 0 ? '#00ff88' : '#ff0055';
        row.appendChild(tdGrad);
        
        // Accumulator
        const tdAcc = document.createElement('td');
        tdAcc.className = 'numeric';
        tdAcc.innerHTML = `${node.accumulator.toFixed(5)} ${createMiniBar(Math.abs(node.accumulator), 0.5, '#ff00aa')}`;
        row.appendChild(tdAcc);
        
        tbody.appendChild(row);
    }
}

function createMiniBar(value, maxValue, color) {
    const pct = Math.min(100, (Math.abs(value) / maxValue) * 100);
    return `<span class="mini-bar" style="width: ${pct}px; background: ${color}"></span>`;
}

function phaseToColor(phase) {
    const hue = (phase / (2 * Math.PI)) * 360;
    return `hsl(${hue}, 70%, 60%)`;
}

function selectNodeFromTable(nodeId) {
    selectedNodeId = nodeId;
    network.selectNodes([nodeId]);
    if (lastState && lastState.nodes[nodeId]) {
        updateInspector(lastState.nodes[nodeId]);
    }
}

// === Timeline Rendering ===

function updateHistory(state) {
    for (const [nodeId, node] of Object.entries(state.nodes)) {
        if (!nodeHistory[nodeId]) nodeHistory[nodeId] = [];
        
        nodeHistory[nodeId].push({
            phase: node.phase,
            magnitude: node.magnitude,
            activation: node.activation,
            gradient: node.gradient,
            accumulator: node.accumulator,
            timestep: state.step_number
        });
        
        // Keep last 100 timesteps
        if (nodeHistory[nodeId].length > 100) {
            nodeHistory[nodeId].shift();
        }
    }
}

function renderTimeline() {
    const canvas = document.getElementById('timeline-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight - 30;
    
    ctx.clearRect(0, 0, width, height);
    
    // Select representative nodes from each layer
    const selectedNodes = ['in_0', 'inc_0', 'mid_0', 'out_0']
        .filter(id => nodeHistory[id] && nodeHistory[id].length > 1);
    
    if (selectedNodes.length === 0) return;
    
    const metric = document.getElementById('timeline-metric').value;
    const rowHeight = height / selectedNodes.length;
    
    selectedNodes.forEach((nodeId, idx) => {
        const history = nodeHistory[nodeId];
        const y = idx * rowHeight + rowHeight / 2;
        
        // Draw sparkline
        ctx.strokeStyle = getNodeColor(nodeId);
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        history.forEach((state, i) => {
            const x = (i / 100) * width;
            let val = state[metric];
            
            // Normalize based on metric
            if (metric === 'phase') {
                val = Math.sin(val); // Convert to [-1, 1] for visualization
            }
            
            const plotY = y - (val * rowHeight * 0.4);
            
            if (i === 0) ctx.moveTo(x, plotY);
            else ctx.lineTo(x, plotY);
        });
        
        ctx.stroke();
        
        // Draw label
        ctx.fillStyle = '#aaa';
        ctx.font = '11px monospace';
        ctx.fillText(nodeId, 5, y - rowHeight * 0.3);
        
        // Draw baseline
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    });
}

function getNodeColor(nodeId) {
    if (nodeId.startsWith('in_')) return '#0066ff';
    if (nodeId.startsWith('inc_')) return '#9600ff';
    if (nodeId.startsWith('mid_')) return '#640096';
    if (nodeId.startsWith('out_')) return '#00ff66';
    return '#ffffff';
}

// === Architecture Toggle ===

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
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('primary');
    });
    
    if (currentArchMode === "flat") {
        document.getElementById('btn-mode-flat').classList.add('primary');
        document.getElementById('mode-desc').innerHTML = 
            '<small>Direct Radiation: Input → Middle → Output (no IC layer)</small>';
        
        document.getElementById('inp-n-inc').disabled = true;
        document.getElementById('inp-c-in').disabled = true;
    } else {
        document.getElementById('btn-mode-hierarchical').classList.add('primary');
        document.getElementById('mode-desc').innerHTML = 
            '<small>Hierarchical: Input → Input-Connected → Middle → Output</small>';
        
        document.getElementById('inp-n-inc').disabled = false;
        document.getElementById('inp-c-in').disabled = false;
    }
}

// === Temporal Sequence Injection ===

async function injectSequence() {
    const input = document.getElementById('inp-sequence').value;
    const sequence = input.split(',').map(v => parseFloat(v.trim()) || 0);
    
    await apiCall('/api/inject_sequence', {
        sequence: sequence,
        timestep: timestep
    });
    
    timestep++;
    document.getElementById('timestep-count').textContent = timestep;
    
    // Step forward
    await stepForward();
}

function startAutoSequence() {
    if (autoSeqInterval) {
        stopAutoSequence();
        return;
    }
    
    document.getElementById('btn-auto-seq').textContent = "Stop Auto";
    document.getElementById('btn-auto-seq').classList.add('danger');
    
    autoSeqInterval = setInterval(async () => {
        const numInputs = parseInt(document.getElementById('inp-n-in').value);
        const sequence = Array(numInputs).fill(0).map((_, i) => {
            return Math.sin((timestep * 0.1) + (i * 0.5)) * 0.5 + 0.5;
        });
        
        document.getElementById('inp-sequence').value = 
            sequence.map(v => v.toFixed(2)).join(', ');
        
        await injectSequence();
    }, 600);
}

function stopAutoSequence() {
    if (autoSeqInterval) {
        clearInterval(autoSeqInterval);
        autoSeqInterval = null;
        document.getElementById('btn-auto-seq').textContent = "Auto Mode";
        document.getElementById('btn-auto-seq').classList.remove('danger');
    }
}

// === Metrics ===

function updateMetrics(state) {
    const numEdges = state.edges.length;
    const activeNodes = Object.values(state.nodes).filter(n => n.active).length;
    const gradients = Object.values(state.nodes).map(n => Math.abs(n.gradient));
    const avgGrad = gradients.length > 0 ? gradients.reduce((a, b) => a + b, 0) / gradients.length : 0;
    
    document.getElementById('metric-edges').textContent = numEdges;
    document.getElementById('metric-active').textContent = activeNodes;
    document.getElementById('metric-grad').textContent = avgGrad.toFixed(5);
}

// === Event Listeners ===

document.getElementById('btn-mode-flat').addEventListener('click', switchToFlat);
document.getElementById('btn-mode-hierarchical').addEventListener('click', switchToHierarchical);

document.getElementById('btn-inject-seq').addEventListener('click', injectSequence);
document.getElementById('btn-auto-seq').addEventListener('click', startAutoSequence);

document.getElementById('inp-period').addEventListener('input', (e) => {
    temporalPeriod = parseInt(e.target.value);
    document.getElementById('val-period').textContent = temporalPeriod;
});

document.getElementById('inp-beam').addEventListener('change', (e) => {
    const beamWidth = parseInt(e.target.value);
    document.getElementById('val-beam').textContent = beamWidth;
    apiCall('/api/config', { beam_width: beamWidth });
});

document.getElementById('timeline-metric').addEventListener('change', renderTimeline);

// Start
initNetwork();
