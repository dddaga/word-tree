"""
Visualization Manager for Neurograph
Enhanced with correct wave interference physics, energy conservation, and beam width pruning.
"""

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum


class ConnectionType(Enum):
    CONDUCTANCE = "conductance"  # Static graph edges
    RADIATION = "radiation"      # Dynamic phase-aligned


@dataclass
class Node:
    """A neuron with vector-based phase and magnitude (wave interference model)."""
    id: str
    role: str  # "input", "input_connected", "middle", "output"
    
    # Learnable weights (vectors) - resonant frequencies
    phase_weight: np.ndarray  # Shape: (vector_dim,) in [0, 2π)
    mag_weight: np.ndarray    # Shape: (vector_dim,)
    
    # Activation state (vectors) - current wave state
    phase_activation: np.ndarray  # Shape: (vector_dim,)
    mag_activation: np.ndarray    # Shape: (vector_dim,)
    activation_strength: float = 0.0  # Computed intensity (scalar)
    
    # Training state
    target_phase: Optional[np.ndarray] = None
    gradient_phase: Optional[np.ndarray] = None
    gradient_magnitude: Optional[np.ndarray] = None
    accumulated_grad_phase: Optional[np.ndarray] = None
    accumulated_grad_magnitude: Optional[np.ndarray] = None
    
    # Metadata
    active: bool = False
    version: int = 0
    
    def __post_init__(self):
        """Initialize gradients if None."""
        if self.gradient_phase is None:
            self.gradient_phase = np.zeros_like(self.phase_weight)
        if self.gradient_magnitude is None:
            self.gradient_magnitude = np.zeros_like(self.mag_weight)
        if self.accumulated_grad_phase is None:
            self.accumulated_grad_phase = np.zeros_like(self.phase_weight)
        if self.accumulated_grad_magnitude is None:
            self.accumulated_grad_magnitude = np.zeros_like(self.mag_weight)


@dataclass
class Edge:
    """Static conductance edge with vector phase/magnitude shifts."""
    source_id: str
    target_id: str
    phase_weight: np.ndarray  # Phase shift vector
    mag_weight: np.ndarray    # Magnitude shift vector


@dataclass
class SignalPacket:
    """A signal traveling through the network (for animation)."""
    source_id: str
    target_id: str
    connection_type: ConnectionType
    signal_strength: float = 0.0
    phase_value: float = 0.0
    is_backward: bool = False


@dataclass
class NetworkConfig:
    """Configuration for a network instance."""
    # Graph Structure
    num_input: int = 2
    num_input_connected: int = 4
    num_middle: int = 8
    num_output: int = 2
    
    # Connectivity
    input_cardinality: int = 2
    middle_cardinality: int = 3
    
    # Wave Physics
    vector_dim: int = 16          # Dimensionality of phase/mag vectors
    gamma: float = 1.0            # Magnitude exponential scaling
    
    # Radiation
    radiation_k: int = 3
    use_radiation: bool = True
    
    # Energy Conservation
    conductance_efficiency: float = 0.9   # 90% efficient (10% loss per unit)
    radiation_efficiency: float = 0.95   # 95% efficient (5% loss per unit)
    temporal_decay: float = 0.4          # 40% decay per timestep
    activation_threshold: float = 0.1    # Min activation to stay active
    
    # Computational Efficiency
    beam_width: Optional[int] = None      # Max active nodes (None = unlimited)
    
    # Training
    learning_rate: float = 0.05
    target_phase_pattern: List[float] = field(default_factory=list)
    
    # Architecture Mode
    architecture_mode: str = "hierarchical"  # "flat" or "hierarchical"


@dataclass  
class StepResult:
    """Result of a single forward step."""
    nodes: Dict[str, dict]
    edges: List[dict]
    active_signals: List[dict]
    radiation_paths: List[dict]
    step_number: int
    mode: str = "forward"  # or "backward"
    loss: float = 0.0  # Average phase error for output nodes
    target_values: Dict[str, float] = field(default_factory=dict)  # node_id -> target phase


class SimpleNeuroGraph:
    """
    Neurograph with correct wave interference physics.
    
    Key Features:
    - Vector-based phase/magnitude (wave components)
    - Real component (cos) → Conductance via static edges
    - Imaginary component (sin) → Radiation to phase-aligned nodes
    - Energy conservation (depletion from transmission)
    - Beam width pruning (computational efficiency)
    """
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.step_count = 0
        self.active_signals: List[SignalPacket] = []
        self.radiation_paths: List[SignalPacket] = []
        self.verbose = False
        
        self._initialize_network()
    
    def _compute_activation_strength(self, phase: np.ndarray, mag: np.ndarray, gamma: float = None) -> float:
        """
        Compute activation strength from phase and magnitude vectors.
        
        Formula: strength = Σ cos(φᵢ) × exp(γ × sin(mᵢ))
        
        This represents wave interference intensity.
        """
        if gamma is None:
            gamma = self.config.gamma
        
        # Phase component: cosine values
        phase_component = np.cos(phase)
        
        # Magnitude component: exponential of sine-transformed values
        mag_exponent = gamma * np.sin(mag)
        mag_exponent = np.clip(mag_exponent, -10.0, 10.0)  # Prevent overflow
        mag_component = np.exp(mag_exponent)
        
        # Dot product gives total signal strength
        signal = np.sum(phase_component * mag_component)
        
        return max(0.0, signal + 1e-8)  # Small epsilon, ensure non-negative
    
    def _initialize_network(self):
        """Create constrained topology."""
        self.nodes.clear()
        self.edges.clear()
        
        # 1. Input Nodes
        inputs = []
        for i in range(self.config.num_input):
            n = self._create_node(f"in_{i}", "input")
            inputs.append(n)
            
        # 2. Input Connected Nodes
        input_connected = []
        for i in range(self.config.num_input_connected):
            n = self._create_node(f"inc_{i}", "input_connected")
            input_connected.append(n)
            
        # Edges: Input -> Input Connected (Cardinality)
        for target in input_connected:
            sources = random.sample(inputs, min(len(inputs), self.config.input_cardinality))
            for src in sources:
                phase_weight = np.random.uniform(-np.pi/4, np.pi/4, self.config.vector_dim)
                mag_weight = np.random.uniform(-0.5, 0.5, self.config.vector_dim)
                self.edges.append(Edge(src.id, target.id, phase_weight, mag_weight))
                
        # 3. Middle Nodes (Isolated Island)
        middle = []
        for i in range(self.config.num_middle):
            n = self._create_node(f"mid_{i}", "middle")
            middle.append(n)
            
        # Edges: Middle <-> Middle (Internal Cardinality)
        for target in middle:
            possible_sources = [m for m in middle if m.id != target.id]
            if possible_sources:
                sources = random.sample(possible_sources, min(len(possible_sources), self.config.middle_cardinality))
                for src in sources:
                    phase_weight = np.random.uniform(-np.pi/4, np.pi/4, self.config.vector_dim)
                    mag_weight = np.random.uniform(-0.5, 0.5, self.config.vector_dim)
                    self.edges.append(Edge(src.id, target.id, phase_weight, mag_weight))
                    
        # 4. Output Nodes
        outputs = []
        for i in range(self.config.num_output):
            n = self._create_node(f"out_{i}", "output")
            # Set target values for training visualization
            target_phase_val = (i / max(1, self.config.num_output)) * 2 * np.pi
            n.target_phase = np.full(self.config.vector_dim, target_phase_val)
            outputs.append(n)
            
        # Edges: Middle -> Output? Or Radiation only?
        # User said "Output nodes (radiation can happen across all three cases)".
        # User constraint "middle part... isolated section".
        # Let's assume Output is also isolated statically, relies on radiation to receive signal?
        # Or maybe InputConnected -> Output?
        # To make it interesting, let's keep islands isolated statically.
        # Radiation is the bridge.
        
    def _create_node(self, node_id: str, role: str) -> Node:
        """Create node with random vector initialization."""
        vector_dim = self.config.vector_dim
        
        # Initialize weights randomly
        phase_weight = np.random.uniform(0, 2*np.pi, vector_dim)
        mag_weight = np.random.uniform(-np.pi, np.pi, vector_dim)
        
        # Initialize activations from weights
        phase_activation = phase_weight.copy()
        mag_activation = mag_weight.copy()
        
        n = Node(
            id=node_id,
            role=role,
            phase_weight=phase_weight,
            mag_weight=mag_weight,
            phase_activation=phase_activation,
            mag_activation=mag_activation,
            activation_strength=0.0,
            active=False
        )
        
        # Compute initial activation strength
        n.activation_strength = self._compute_activation_strength(
            n.phase_activation, n.mag_activation, self.config.gamma
        )
        
        self.nodes[node_id] = n
        return n

    def _get_radiation_neighbors(self, source: Node, k: int = None) -> List[Tuple[str, float]]:
        """
        Find top-K nodes whose WEIGHT phase aligns with source's ACTIVATION phase.
        
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
            
            # KEY: Compare activation[source] with weight[target]
            # Cosine similarity between source's activation and target's weight
            dot_product = np.dot(source.phase_activation, node.phase_weight)
            norm_source = np.linalg.norm(source.phase_activation)
            norm_target = np.linalg.norm(node.phase_weight)
            
            similarity = dot_product / (norm_source * norm_target + 1e-8)
            
            # Normalize to [0, 1]
            alignment_score = (similarity + 1) / 2
            
            candidates.append((node_id, alignment_score))
        
        # Sort by alignment and take top-K
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]
    
    def inject_signal(self, node_id: str, strength: float = 1.0):
        """Inject signal into a node (for manual input)."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            # Set activation strength directly
            node.activation_strength = strength
            node.active = strength > self.config.activation_threshold
    
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
            node_id = f"in_{i}"
            if node_id not in self.nodes:
                continue
            
            input_node = self.nodes[node_id]
            
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
            input_node.active = input_node.activation_strength > self.config.activation_threshold
    
    def _prune_to_beam_width(self):
        """Keep only top-K most active nodes (beam width pruning)."""
        if self.config.beam_width is None:
            return
        
        # Get all nodes sorted by activation strength
        candidates = [
            (node.id, node.activation_strength)
            for node in self.nodes.values()
            if node.activation_strength > 0.0
        ]
        
        if len(candidates) <= self.config.beam_width:
            return
        
        # Sort by strength (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top beam_width nodes
        kept_ids = set(node_id for node_id, _ in candidates[:self.config.beam_width])
        
        # Deactivate pruned nodes
        for node_id, _ in candidates[self.config.beam_width:]:
            node = self.nodes[node_id]
            node.active = False
            node.activation_strength = 0.0
        
        if self.verbose:
            print(f"Beam pruning: {len(candidates)} → {self.config.beam_width} active nodes")
    
    def step_forward(self) -> StepResult:
        """
        Forward pass with correct wave interference physics.
        
        Real component (cos) → Conductance via static edges
        Imaginary component (sin) → Radiation to phase-aligned nodes
        Energy conservation: Depletion from transmission + temporal decay
        """
        self.step_count += 1
        self.active_signals = []
        self.radiation_paths = []
        
        active_nodes = [n for n in self.nodes.values() if n.active]
        
        # Track energy expenditure per node
        energy_conducted = {}  # node_id -> float
        energy_radiated = {}   # node_id -> float
        
        # Collect phase-shifted contributions for each node
        phase_contributions = {}  # target_id -> List[np.ndarray]
        mag_contributions = {}    # target_id -> List[np.ndarray]
        contribution_strengths = {}  # target_id -> List[float]
        
        for source in active_nodes:
            energy_conducted[source.id] = 0.0
            energy_radiated[source.id] = 0.0
            
            # 1. REAL COMPONENT → Conductance (static edges)
            real_component = np.cos(source.phase_activation) * source.mag_activation
            
            for edge in self.edges:
                if edge.source_id == source.id:
                    # Apply phase shift (KEY: additive, not multiplicative!)
                    arriving_phase = (source.phase_activation + edge.phase_weight) % (2 * np.pi)
                    arriving_mag = source.mag_activation + edge.mag_weight
                    
                    # Track energy conducted
                    signal_strength = np.sum(np.abs(real_component))
                    energy_conducted[source.id] += signal_strength
                    
                    # Store for visualization
                    self.active_signals.append(SignalPacket(
                        source_id=source.id,
                        target_id=edge.target_id,
                        connection_type=ConnectionType.CONDUCTANCE,
                        signal_strength=float(signal_strength),
                        phase_value=float(np.mean(arriving_phase))
                    ))
                    
                    # Collect contributions
                    if edge.target_id not in phase_contributions:
                        phase_contributions[edge.target_id] = []
                        mag_contributions[edge.target_id] = []
                        contribution_strengths[edge.target_id] = []
                    
                    phase_contributions[edge.target_id].append(arriving_phase)
                    mag_contributions[edge.target_id].append(arriving_mag)
                    contribution_strengths[edge.target_id].append(float(signal_strength))
            
            # 2. IMAGINARY COMPONENT → Radiation (phase-aligned nodes)
            imaginary_component = np.sin(source.phase_activation) * source.mag_activation
            
            radiation_targets = self._get_radiation_neighbors(source)
            for target_id, alignment_score in radiation_targets:
                # Radiation signal strength
                signal_strength = np.sum(np.abs(imaginary_component)) * alignment_score
                energy_radiated[source.id] += signal_strength
                
                # No edge-based phase shift for radiation (direct broadcast)
                arriving_phase = source.phase_activation
                arriving_mag = source.mag_activation
                
                self.radiation_paths.append(SignalPacket(
                    source_id=source.id,
                    target_id=target_id,
                    connection_type=ConnectionType.RADIATION,
                    signal_strength=float(signal_strength),
                    phase_value=float(np.mean(arriving_phase))
                ))
                
                if target_id not in phase_contributions:
                    phase_contributions[target_id] = []
                    mag_contributions[target_id] = []
                    contribution_strengths[target_id] = []
                
                phase_contributions[target_id].append(arriving_phase)
                mag_contributions[target_id].append(arriving_mag)
                contribution_strengths[target_id].append(float(signal_strength * 0.5))  # Radiation multiplier
        
        # 3. Aggregate with attention-weighted interference
        for node in self.nodes.values():
            if node.id not in phase_contributions:
                # No inputs - apply decay only
                node.activation_strength *= (1 - self.config.temporal_decay)
                node.active = node.activation_strength > self.config.activation_threshold
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
        
        # 4. Energy conservation: Apply depletion
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
        
        # 5. Beam width pruning
        self._prune_to_beam_width()
        
        return self._get_state("forward")

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
                        signal_strength=float(np.linalg.norm(src_grad_phase)),
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
                    
                    self.active_signals.append(SignalPacket(
                        source_id=target_id,
                        target_id=src_id,
                        connection_type=ConnectionType.RADIATION,
                        signal_strength=float(np.linalg.norm(src_grad_phase)),
                        is_backward=True
                    ))
        
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

    def _get_state(self, mode: str) -> StepResult:
        """Serialize state for frontend (aggregate vectors to scalars for viz)."""
        # Compute target values dict for output nodes
        target_values = {}
        for n in self.nodes.values():
            if n.role == "output" and n.target_phase is not None:
                target_values[n.id] = float(np.mean(n.target_phase))
        
        return StepResult(
            nodes={
                n.id: {
                    "id": n.id,
                    "role": n.role,
                    "phase": float(np.mean(n.phase_activation)),  # Average for viz
                    "magnitude": float(np.mean(np.abs(n.mag_activation))),
                    "activation": float(n.activation_strength),
                    "active": bool(n.active),  # FIX: Convert numpy.bool_ to Python bool
                    "gradient": float(np.mean(n.gradient_phase)) if n.gradient_phase is not None else 0.0,
                    "accumulator": float(np.mean(n.accumulated_grad_phase)) if n.accumulated_grad_phase is not None else 0.0,
                    "target_phase": float(np.mean(n.target_phase)) if n.target_phase is not None else None,
                    "label": f"{n.role[:3].upper()}\nφ:{np.mean(n.phase_activation):.2f}",
                    "group": n.role,
                    # Full vectors for detailed inspection
                    "phase_vector": n.phase_activation.tolist(),
                    "mag_vector": n.mag_activation.tolist(),
                    "phase_weight_vector": n.phase_weight.tolist(),
                    "mag_weight_vector": n.mag_weight.tolist(),
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
                    "strength": float(s.signal_strength),  # FIX: Ensure float
                    "is_backward": bool(s.is_backward)  # FIX: Convert to Python bool
                }
                for s in self.active_signals + self.radiation_paths
            ],
            radiation_paths=[],
            step_number=int(self.step_count),  # FIX: Convert to Python int
            mode=mode,
            loss=float(self.compute_loss()),  # Add loss computation
            target_values=target_values  # Add target values
        )
    
    def compute_loss(self) -> float:
        """
        Compute average phase error loss for output nodes.
        
        Returns circular distance between target and actual phase.
        """
        total_error = 0.0
        count = 0
        
        for node in self.nodes.values():
            if node.role == "output" and node.target_phase is not None:
                # Average phase for comparison
                actual = float(np.mean(node.phase_activation))
                target = float(np.mean(node.target_phase))
                
                # Circular distance
                error = abs(target - actual)
                if error > np.pi:
                    error = 2 * np.pi - error
                
                total_error += error
                count += 1
        
        return total_error / count if count > 0 else 0.0
    
    def set_target(self, node_id: str, target_phase: float):
        """Set target phase for a specific node."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.target_phase = np.full(self.config.vector_dim, target_phase)
    
    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Re-init if topology or vector dimension changes
        topo_keys = ["num_input", "num_input_connected", "num_middle", "num_output", 
                     "input_cardinality", "middle_cardinality", "vector_dim", "architecture_mode"]
        if any(k in kwargs for k in topo_keys):
            self._initialize_network()


class VizSession:
    def __init__(self):
        self.network: Optional[SimpleNeuroGraph] = None
    
    def init_network(self, config: Optional[dict] = None) -> StepResult:
        cfg = NetworkConfig(**(config or {}))
        self.network = SimpleNeuroGraph(cfg)
        return self.network._get_state("init")
    
    def step_forward(self) -> StepResult:
        if not self.network: return self.init_network()
        return self.network.step_forward()
        
    def step_backward(self) -> StepResult:
        if not self.network: return self.init_network()
        return self.network.step_backward()
    
    def inject(self, node_id: str, strength: float):
        if self.network:
            self.network.inject_signal(node_id, strength)
    
    def inject_temporal_sequence(self, sequence: List[float], timestep: int):
        if self.network:
            self.network.inject_temporal_sequence(sequence, timestep)
    
    def set_targets(self, targets: Dict[str, float]):
        """Set target phases for output nodes."""
        if self.network:
            for node_id, target_phase in targets.items():
                self.network.set_target(node_id, target_phase)
    
    def update_config(self, **kwargs):
        if self.network:
            self.network.update_config(**kwargs)
    
    def get_state(self) -> Optional[StepResult]:
        return self.network._get_state("state") if self.network else None


_session = VizSession()
def get_session(): return _session
