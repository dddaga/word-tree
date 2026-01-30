// Vis.js Graph Setup
const nodes = new vis.DataSet([]);
const edges = new vis.DataSet([]);
const container = document.getElementById('mynetwork');

const data = { nodes, edges };

const options = {
    nodes: {
        shape: 'dot',
        font: { 
            color: '#ffffff', 
            size: 10, 
            face: 'monospace', 
            strokeWidth: 1, 
            strokeColor: '#000000', 
            align: 'center'
        },
        borderWidth: 2,
        shadow: true,
        // Ensure labels are inside nodes
        labelHighlightBold: false
    },
    edges: {
        width: 1,
        color: { color: '#444', highlight: '#00d4ff' },
        smooth: { type: 'continuous' },
        arrows: { to: { enabled: true, scaleFactor: 0.5 } }
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
let selectedNodeId = null;
let currentIteration = 0;
let totalIterations = 0;
let currentLayer = 0;
let layerCount = 1;
let lastState = null;
let iterationInfo = [];

// === API ===

async function apiCall(endpoint, body = null, forcePost = false) {
    try {
        // Determine method: POST if body provided, endpoint requires POST, or forcePost is true
        const requiresPost = forcePost ||
                           endpoint.includes('/api/step') ||
                           endpoint.includes('/api/step_back') ||
                           endpoint.includes('/api/reset') ||
                           endpoint.includes('/api/set_') ||
                           endpoint.includes('/api/set_layer') ||
                           endpoint.includes('/api/inject');
        
        const hasBody = body && Object.keys(body).length > 0;
        const method = (hasBody || requiresPost) ? 'POST' : 'GET';
        
        const res = await fetch(endpoint, {
            method: method,
            headers: { 'Content-Type': 'application/json' },
            body: method === 'POST' ? JSON.stringify(body || {}) : undefined
        });
        if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        return await res.json();
    } catch (e) {
        console.error(`API Error ${endpoint}:`, e);
        return null;
    }
}

async function initNetwork() {
    const iterations = await apiCall('/api/iterations');
    if (!iterations || iterations.total_iterations === 0) {
        alert('Network not initialized. Please run visualize_forward_pass.py first.');
        return;
    }
    totalIterations = iterations.total_iterations;
    currentIteration = iterations.current_iteration || 0;
    iterationInfo = iterations.iterations || [];
    const layersResp = await apiCall('/api/layers');
    if (layersResp) {
        layerCount = layersResp.layer_count || 1;
        currentLayer = layersResp.current_layer || 0;
        updateLayerSelector();
    }
    updateIterationControls();
    await loadState();
}

function updateLayerSelector() {
    const sel = document.getElementById('layer-select');
    sel.innerHTML = '';
    for (let i = 0; i < layerCount; i++) {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = 'Layer ' + i;
        if (i === currentLayer) opt.selected = true;
        sel.appendChild(opt);
    }
    sel.style.display = layerCount > 1 ? 'block' : 'none';
    document.getElementById('layer-section').style.display = layerCount > 1 ? 'block' : 'none';
}

async function setLayer(layerIndex) {
    if (layerIndex < 0 || layerIndex >= layerCount) return;
    const state = await apiCall('/api/set_layer', { layer_index: layerIndex });
    if (state) {
        currentLayer = layerIndex;
        currentIteration = state.step_number || 0;
        const iterations = await apiCall('/api/iterations');
        if (iterations) {
            totalIterations = iterations.total_iterations;
            iterationInfo = iterations.iterations || [];
        }
        updateLayerSelector();
        updateIterationControls();
        renderState(state, true);
        updateMetrics(state);
    }
}

async function loadState() {
    const state = await apiCall('/api/state');
    if (state) {
        if (state.step_number !== undefined) {
            currentIteration = state.step_number;
        }
        updateIterationControls();
        renderState(state, currentIteration === 0);
        updateMetrics(state);
    }
}

async function stepForward() {
    const state = await apiCall('/api/step');
    if (state) {
        currentIteration = state.step_number || currentIteration + 1;
        updateIterationControls();
        renderState(state, false);
        updateMetrics(state);
    }
}

async function stepBackward() {
    const state = await apiCall('/api/step_back');
    if (state) {
        currentIteration = state.step_number || currentIteration - 1;
        updateIterationControls();
        renderState(state, false);
        updateMetrics(state);
    }
}

async function setIteration(iteration) {
    if (iteration < 0 || iteration >= totalIterations) return;
    
    const state = await apiCall('/api/set_iteration', { iteration });
    if (state) {
        currentIteration = state.step_number;
        updateIterationControls();
        renderState(state, currentIteration === 0);
        updateMetrics(state);
    }
}

function updateIterationControls() {
    document.getElementById('iteration-display').textContent = `${currentIteration} / ${totalIterations - 1}`;
    document.getElementById('iteration-slider').max = totalIterations - 1;
    document.getElementById('iteration-slider').value = currentIteration;
    document.getElementById('step-count').textContent = currentIteration;
    
    document.getElementById('btn-prev').disabled = currentIteration <= 0;
    document.getElementById('btn-next').disabled = currentIteration >= totalIterations - 1;
}

// === Utility Functions ===

function formatVector(vec, decimals = 3, maxDisplay = 8) {
    if (!Array.isArray(vec)) {
        // Handle scalar values (backward compatibility)
        return (typeof vec === 'number' && vec.toFixed) ? vec.toFixed(decimals) : String(vec);
    }
    if (vec.length === 0) return '[]';
    if (vec.length <= maxDisplay) {
        return '[' + vec.map(v => {
            if (typeof v === 'number' && v.toFixed) return v.toFixed(decimals);
            return String(v);
        }).join(', ') + ']';
    }
    // Show first few and last few
    const head = vec.slice(0, Math.floor(maxDisplay / 2));
    const tail = vec.slice(-Math.floor(maxDisplay / 2));
    const formatVal = v => (typeof v === 'number' && v.toFixed) ? v.toFixed(decimals) : String(v);
    return '[' + head.map(formatVal).join(', ') + 
           ' ... ' + tail.map(formatVal).join(', ') + ']';
}

function formatVectorCompact(vec, decimals = 2) {
    if (!Array.isArray(vec)) {
        // Handle scalar values (backward compatibility)
        return (typeof vec === 'number' && vec.toFixed) ? vec.toFixed(decimals) : String(vec);
    }
    if (vec.length === 0) return '[]';
    if (vec.length <= 4) {
        return vec.map(v => {
            if (typeof v === 'number' && v.toFixed) return v.toFixed(decimals);
            return String(v);
        }).join(',');
    }
    // Show first 2, ..., last 2
    const formatVal = v => (typeof v === 'number' && v.toFixed) ? v.toFixed(decimals) : String(v);
    return vec.slice(0, 2).map(formatVal).join(',') + 
           '...' + vec.slice(-2).map(formatVal).join(',');
}

function formatNodeTooltip(node, id) {
    const phaseStr = formatVector(node.phase, 2, 6);
    const magStr = formatVector(node.magnitude, 2, 6);
    return `Node ${id}\nPhase: ${phaseStr}\nMag: ${magStr}\nActivation: ${node.activation.toFixed(3)}\nRole: ${node.role}`;
}

// === Helper Functions for Visualization ===

function getActivationRange(state) {
    const activations = Object.values(state.nodes).map(n => n.activation);
    return {
        min: Math.min(...activations),
        max: Math.max(...activations)
    };
}

function interpolateColor(min, max, value) {
    if (max === min) return 'rgb(128, 128, 0)'; // Yellow for zero range
    const normalized = Math.max(0, Math.min(1, (value - min) / (max - min)));
    const r = Math.round(normalized * 255);
    const g = Math.round((1 - normalized) * 255);
    const b = 0;
    return `rgb(${r}, ${g}, ${b})`;
}

function isNodeActive(nodeId, state) {
    const node = state.nodes[nodeId];
    return node && node.active === true;
}

function getReadableTextColor(backgroundColor) {
    // For green-to-red gradient, use white text for darker colors (red side)
    // and black text for lighter colors (green side)
    // For grey nodes, use white text
    if (backgroundColor.includes('rgb')) {
        const match = backgroundColor.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
        if (match) {
            const r = parseInt(match[1]);
            const g = parseInt(match[2]);
            const b = parseInt(match[3]);
            // Calculate brightness
            const brightness = (r * 299 + g * 587 + b * 114) / 1000;
            return brightness > 128 ? '#000000' : '#ffffff';
        }
    }
    // For grey nodes (#666666 or similar), use white
    if (backgroundColor.includes('666') || backgroundColor.includes('555')) {
        return '#ffffff';
    }
    return '#ffffff'; // Default to white
}

// === Rendering ===

function renderState(state, reset = false) {
    lastState = state;
    
    if (reset) {
        nodes.clear();
        edges.clear();
    }

    // --- Nodes ---
    const nodeUpdates = [];
    const containerRect = container.getBoundingClientRect();
    const width = containerRect.width || 800;
    const height = containerRect.height || 600;
    
    // Calculate activation range for color normalization
    const activationRange = getActivationRange(state);

    for (const [id, n] of Object.entries(state.nodes)) {
        // Determine if node is inactive
        const isInactive = !n.active;
        
        // Node size: reduced significantly - minimum 15px to fit ID, scale with activation
        const size = Math.max(15, 15 + n.activation * 5);
        
        // Font size scales with node size but stays readable
        const fontSize = Math.max(8, Math.round(size * 0.5));
        
        // Label: Keep "IN"/"OUT" for input/output nodes, show ID for others
        let label = '';
        if (n.role === 'input') label = 'IN';
        else if (n.role === 'output') label = 'OUT';
        else label = id;
        
        // Color scheme: grey for inactive, green-to-red for active
        let backgroundColor;
        let textColor;
        let borderColor;
        let borderWidth = 2;
        
        if (isInactive) {
            // Grey out inactive nodes
            backgroundColor = '#666666';
            textColor = '#ffffff';
            borderColor = '#888888';
        } else {
            // Green to red based on activation
            backgroundColor = interpolateColor(activationRange.min, activationRange.max, n.activation);
            textColor = getReadableTextColor(backgroundColor);
            borderColor = '#ffffff';
            borderWidth = 3;
        }

        const nodeData = {
            id: id,
            label: label,
            color: {
                background: backgroundColor,
                border: borderColor,
                highlight: { background: backgroundColor, border: '#fff' }
            },
            font: {
                color: textColor,
                size: fontSize,
                face: 'monospace',
                strokeWidth: 1,
                strokeColor: '#000000',
                align: 'center'
            },
            size: size,
            borderWidth: borderWidth,
            title: formatNodeTooltip(n, id),
            // Ensure label is inside the node
            labelHighlightBold: false
        };

        // Layout Hints (only on reset)
        if (reset) {
            if (n.role === 'input') {
                nodeData.x = -width * 0.4;
                nodeData.y = (Math.random() - 0.5) * height * 0.5;
            } else if (n.role === 'output') {
                nodeData.x = width * 0.4;
                nodeData.y = (Math.random() - 0.5) * height * 0.5;
            } else if (n.role === 'middle') {
                nodeData.x = (Math.random() - 0.5) * width * 0.3;
                nodeData.y = (Math.random() - 0.5) * height * 0.5;
            }
        }
        
        nodeUpdates.push(nodeData);
        
        // Update inspector if selected
        if (selectedNodeId === id) updateInspector(n);
    }
    nodes.update(nodeUpdates);

    // --- Edges ---
    // Helper function to determine edge color based on node activity
    function getEdgeColor(sourceId, targetId, defaultColor, defaultOpacity) {
        const sourceActive = isNodeActive(sourceId, state);
        const targetActive = isNodeActive(targetId, state);
        
        if (sourceActive && targetActive) {
            // Both active: use normal color
            return { color: defaultColor, opacity: defaultOpacity };
        } else {
            // At least one inactive: use muted grey
            return { color: '#555555', opacity: 0.3 };
        }
    }
    
    // 1. Static Edges
    if (reset) {
        const staticEdges = state.edges.map(e => {
            const edgeColor = getEdgeColor(e.source, e.target, '#444', 0.3);
            return {
                id: `${e.source}-${e.target}`,
                from: e.source,
                to: e.target,
                color: edgeColor,
                width: 1,
                arrows: 'to',
                dashes: false
            };
        });
        edges.update(staticEdges);
    }

    // 2. Active Signals (conductance)
    state.active_signals.forEach(sig => {
        if (sig.type === 'conductance') {
            const edgeId = `sig-${sig.source}-${sig.target}`;
            const edgeColor = getEdgeColor(sig.source, sig.target, '#00d4ff', 0.6);
            const existing = edges.get(edgeId);
            if (existing) {
                edges.update({
                    id: edgeId,
                    color: edgeColor,
                    width: 1
                });
            } else {
                edges.add({
                    id: edgeId,
                    from: sig.source,
                    to: sig.target,
                    color: edgeColor,
                    width: 1,
                    arrows: 'to',
                    dashes: false
                });
            }
        }
    });

    // 3. Radiation Paths
    const oldRadiation = edges.get({ filter: e => e.radiation === true });
    edges.remove(oldRadiation);
    
    const newRadiation = state.radiation_paths.map(sig => {
        const edgeColor = getEdgeColor(sig.source, sig.target, '#e67e22', 0.5);
        return {
            id: `rad-${sig.source}-${sig.target}`,
            from: sig.source,
            to: sig.target,
            color: edgeColor,
            width: 1,
            arrows: 'to',
            dashes: [5, 5],
            radiation: true
        };
    });
    edges.add(newRadiation);

    // Update table
    updateNodeTable(state);
}

function updateNodeTable(state) {
    const tbody = document.getElementById('node-table-body');
    tbody.innerHTML = '';
    
    const sortedNodes = Object.values(state.nodes).sort((a, b) => {
        if (a.role !== b.role) {
            const roleOrder = { 'input': 0, 'middle': 1, 'output': 2 };
            return (roleOrder[a.role] || 99) - (roleOrder[b.role] || 99);
        }
        return parseInt(a.id) - parseInt(b.id);
    });
    
    sortedNodes.forEach(n => {
        const row = tbody.insertRow();
        row.className = `role-${n.role}`;
        row.onclick = () => {
            selectedNodeId = n.id;
            updateInspector(n);
            network.selectNodes([n.id]);
        };
        
        row.insertCell(0).textContent = n.id;
        row.insertCell(1).textContent = n.role;
        
        // Phase column - show full vector
        const phaseCell = row.insertCell(2);
        phaseCell.textContent = formatVectorCompact(n.phase, 2);
        phaseCell.title = formatVector(n.phase, 3); // Full vector on hover
        
        // Magnitude column - show full vector
        const magCell = row.insertCell(3);
        magCell.textContent = formatVectorCompact(n.magnitude, 2);
        magCell.title = formatVector(n.magnitude, 3); // Full vector on hover
        
        row.insertCell(4).textContent = n.activation.toFixed(3);
        row.insertCell(5).textContent = n.active ? '✓' : '';
    });
}

function updateInspector(node) {
    const inspector = document.getElementById('inspector');
    const content = document.getElementById('inspect-content');
    
    inspector.style.display = 'block';
    
    const phaseStr = formatVector(node.phase, 4);
    const magStr = formatVector(node.magnitude, 4);
    
    content.innerHTML = `
        <h3>Node ${node.id}</h3>
        <div class="inspect-row">
            <span>Role:</span>
            <span class="inspect-val">${node.role}</span>
        </div>
        <div class="inspect-row" style="flex-direction: column; align-items: flex-start;">
            <span style="margin-bottom: 4px;">Phase:</span>
            <span class="inspect-val" style="font-size: 0.8rem; word-break: break-all; max-width: 100%;">${phaseStr}</span>
        </div>
        <div class="inspect-row" style="flex-direction: column; align-items: flex-start;">
            <span style="margin-bottom: 4px;">Magnitude:</span>
            <span class="inspect-val" style="font-size: 0.8rem; word-break: break-all; max-width: 100%;">${magStr}</span>
        </div>
        <div class="inspect-row">
            <span>Activation:</span>
            <span class="inspect-val">${node.activation.toFixed(4)}</span>
        </div>
        <div class="inspect-row">
            <span>Active:</span>
            <span class="inspect-val">${node.active ? 'Yes' : 'No'}</span>
        </div>
        ${node.role === 'output' && node.output_value !== undefined ? `
        <div class="inspect-row">
            <span>Output Value:</span>
            <span class="inspect-val">${node.output_value.toFixed(4)}</span>
        </div>
        ` : ''}
    `;
}

function updateMetrics(state) {
    document.getElementById('metric-loss').textContent = state.loss.toFixed(4);
    document.getElementById('metric-active').textContent = Object.values(state.nodes).filter(n => n.active).length;
    document.getElementById('metric-edges').textContent = state.edges.length;
}

async function reset() {
    const state = await apiCall('/api/reset');
    if (state) {
        currentIteration = state.step_number || 0;
        updateIterationControls();
        renderState(state, true);
        updateMetrics(state);
    }
}

// Event Listeners
document.getElementById('btn-prev').addEventListener('click', stepBackward);
document.getElementById('btn-next').addEventListener('click', stepForward);
document.getElementById('btn-reset').addEventListener('click', reset);

document.getElementById('layer-select').addEventListener('change', (e) => {
    const layerIndex = parseInt(e.target.value);
    setLayer(layerIndex);
});

document.getElementById('iteration-slider').addEventListener('input', (e) => {
    const iteration = parseInt(e.target.value);
    setIteration(iteration);
});

network.on('click', (params) => {
    if (params.nodes.length > 0) {
        selectedNodeId = params.nodes[0];
        if (lastState && lastState.nodes[selectedNodeId]) {
            updateInspector(lastState.nodes[selectedNodeId]);
        }
    }
});

// Initialize on load
window.addEventListener('load', () => {
    initNetwork();
});
