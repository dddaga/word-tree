"""
Adapter to convert ForwardPassTracer data to StepResult format for visualization.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import torch
import numpy as np
from core.nodestore import NodeStore
from .utils import get_node_role, get_static_edges, extract_node_weights, compute_loss_from_trace


@dataclass
class StepResult:
    """Result of a single forward step for visualization."""
    nodes: Dict[str, dict]
    edges: List[dict]
    active_signals: List[dict]
    radiation_paths: List[dict]
    step_number: int
    mode: str = "forward"
    loss: float = 0.0
    target_values: Dict[str, float] = field(default_factory=dict)


def _tensor_to_scalar(value) -> float:
    """Convert tensor/onay to scalar float."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        else:
            return float(torch.mean(value).item())
    elif isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        else:
            return float(np.mean(value))
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0.0
        return float(np.mean(value))
    else:
        return float(value)


def _tensor_to_list(value) -> List[float]:
    """Convert tensor/array to list of floats."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    else:
        return [float(value)]


def trace_to_step_result(
    trace_data: Dict[str, Any],
    iteration_idx: int,
    node_store: NodeStore,
    config: Optional[Dict] = None,
    target_values: Optional[Dict[str, float]] = None
) -> StepResult:
    """
    Convert ForwardPassTracer trace data to StepResult format for visualization.
    
    Args:
        trace_data: Trace data from ForwardPassTracer.get_trace()
        iteration_idx: Index of iteration to convert (0-based)
        node_store: NodeStore instance for node information
        config: Optional config dict for additional context
        target_values: Optional dict of {node_id: target_value} for output nodes
        
    Returns:
        StepResult object ready for visualization
    """
    iterations = trace_data.get('iterations', [])
    if iteration_idx >= len(iterations):
        raise ValueError(f"Iteration {iteration_idx} not found in trace data (has {len(iterations)} iterations)")
    
    iter_data = iterations[iteration_idx]
    node_details = iter_data.get('node_details', {})
    active_nodes = iter_data.get('active_nodes', [])
    radiation_targets = iter_data.get('radiation_targets', {})
    direct_connections = iter_data.get('direct_connections', {})
    
    # Get ALL nodes from the graph (not just active ones)
    all_node_ids_int = set(range(node_store.total_nodes))
    active_nodes_set = set(int(nid) for nid in active_nodes)
    
    # Build nodes dict - include all nodes
    nodes_dict = {}
    
    # Process all nodes in the graph
    for node_id_int in all_node_ids_int:
        node_id_str = str(node_id_int)
        
        # Get node role
        role = get_node_role(node_id_int, node_store)
        
        # Check if node is active in this iteration
        is_active_in_trace = node_id_int in active_nodes_set
        
        # Get node details if available (only for active nodes)
        detail = node_details.get(node_id_int, {}) if is_active_in_trace else {}
        
        # Extract activation values - use trace data if active, otherwise default to 0
        if is_active_in_trace:
            phase_activation = detail.get('phase_activation', None)
            mag_activation = detail.get('mag_activation', None)
            activation_strength = detail.get('activation_strength', 0.0)
        else:
            phase_activation = None
            mag_activation = None
            activation_strength = 0.0
        
        # Convert to scalars for visual calculations (color, size)
        phase_scalar = _tensor_to_scalar(phase_activation) if phase_activation is not None else 0.0
        mag_scalar = _tensor_to_scalar(mag_activation) if mag_activation is not None else 0.0
        activation_scalar = _tensor_to_scalar(activation_strength)
        
        # Convert to lists for display (full vectors)
        phase_vector = _tensor_to_list(phase_activation) if phase_activation is not None else []
        mag_vector = _tensor_to_list(mag_activation) if mag_activation is not None else []
        
        # Get weights from node_store (for all nodes)
        weights = extract_node_weights(node_store, node_id_int)
        phase_weight_vector = weights.get('phase_weight', [])
        mag_weight_vector = weights.get('mag_weight', [])
        phase_weight_scalar = _tensor_to_scalar(phase_weight_vector) if phase_weight_vector else 0.0
        
        # Determine if node is active
        is_active = is_active_in_trace and activation_scalar > 0.0
        
        # Build node dict
        # Note: phase and magnitude are now vectors for display, but we keep phase_scalar for color calculation
        node_dict = {
            "id": node_id_str,
            "role": role,
            "phase": phase_vector,  # Full vector for display
            "magnitude": mag_vector,  # Full vector for display
            "phase_scalar": phase_scalar,  # Scalar for color calculation
            "magnitude_scalar": mag_scalar,  # Scalar for color calculation
            "activation": activation_scalar,
            "active": is_active,
            "gradient": 0.0,  # Not available in forward pass trace
            "accumulator": 0.0,  # Not available in forward pass trace
            "target_phase": None,
            "target_intensity": None,
            "output_value": activation_scalar if role == "output" else None,
            "label": f"{role[:3].upper()}\nφ:{phase_scalar:.2f}",
            "group": role,
            "phase_vector": phase_vector,  # Keep for backward compatibility
            "mag_vector": mag_vector,  # Keep for backward compatibility
            "phase_weight_vector": phase_weight_vector,
            "mag_weight_vector": mag_weight_vector,
        }
        
        nodes_dict[node_id_str] = node_dict
    
    # Build edges from static connections - get ALL edges in the graph
    static_edges = get_static_edges(node_store, None)
    
    # Build active signals from direct connections and radiation targets
    active_signals = []
    radiation_paths = []
    
    # Direct connections (conductance)
    for source_id, targets in direct_connections.items():
        source_id_int = int(source_id)
        if source_id_int in node_details:
            source_detail = node_details[source_id_int]
            source_strength = _tensor_to_scalar(source_detail.get('activation_strength', 0.0))
            
            for target_id in targets:
                target_id_int = int(target_id)
                active_signals.append({
                    "source": str(source_id_int),
                    "target": str(target_id_int),
                    "type": "conductance",
                    "strength": source_strength,
                    "is_backward": False
                })
    
    # Radiation connections
    for source_id, targets in radiation_targets.items():
        source_id_int = int(source_id)
        if source_id_int in node_details:
            source_detail = node_details[source_id_int]
            source_strength = _tensor_to_scalar(source_detail.get('activation_strength', 0.0))
            
            for target_id in targets:
                target_id_int = int(target_id)
                signal = {
                    "source": str(source_id_int),
                    "target": str(target_id_int),
                    "type": "radiation",
                    "strength": source_strength,
                    "is_backward": False
                }
                active_signals.append(signal)
                radiation_paths.append(signal)
    
    # Loss computation removed with target setting feature
    loss = 0.0
    
    return StepResult(
        nodes=nodes_dict,
        edges=static_edges,
        active_signals=active_signals,
        radiation_paths=radiation_paths,
        step_number=iteration_idx,
        mode="forward",
        loss=loss,
        target_values=target_values or {}
    )


class TraceVisualizer:
    """
    Helper class to manage trace data visualization. Supports single-layer (backward compat)
    or multi-layer via layers list. Each layer has (trace_data, node_store).
    """

    def __init__(
        self,
        trace_data: Optional[Dict[str, Any]] = None,
        node_store: Optional[NodeStore] = None,
        config: Optional[Dict] = None,
        *,
        layers: Optional[List[tuple]] = None,
    ):
        """
        Initialize with trace data (single layer) or layers (multi-layer).

        Single layer (backward compat): trace_data, node_store, config.
        Multi-layer: layers=[(trace_data_0, node_store_0), (trace_data_1, node_store_1), ...], config.
        """
        self.config = config or {}
        self.target_values = {}
        self._converted_states = {}  # (layer_index, iteration_idx) -> StepResult

        if layers is not None:
            self._layers = list(layers)
            self.trace_data = self._layers[0][0] if self._layers else {}
            self.node_store = self._layers[0][1] if self._layers else None
        else:
            self._layers = [(trace_data or {}, node_store)] if trace_data is not None and node_store is not None else []
            self.trace_data = trace_data or {}
            self.node_store = node_store

    def set_targets(self, targets: Dict[str, float]):
        """Set target values for output nodes (deprecated - no-op)."""
        pass

    def get_layer_count(self) -> int:
        """Number of layers (1 for single-layer trace)."""
        return len(self._layers)

    def get_iteration_count(self, layer_index: int = 0) -> int:
        """Get number of iterations for the given layer."""
        if layer_index < 0 or layer_index >= len(self._layers):
            return 0
        trace_data, _ = self._layers[layer_index]
        return len(trace_data.get("iterations", []))

    def get_state(self, iteration_idx: int, layer_index: int = 0) -> StepResult:
        """
        Get StepResult for a specific layer and iteration.

        Args:
            iteration_idx: Index of iteration (0-based)
            layer_index: Index of layer (0-based)

        Returns:
            StepResult for that layer and iteration
        """
        if layer_index < 0 or layer_index >= len(self._layers):
            raise ValueError(f"layer_index {layer_index} out of range [0, {len(self._layers)})")
        trace_data, node_store = self._layers[layer_index]

        cache_key = (layer_index, iteration_idx)
        if cache_key in self._converted_states:
            return self._converted_states[cache_key]

        result = trace_to_step_result(
            trace_data,
            iteration_idx,
            node_store,
            self.config,
            self.target_values,
        )
        self._converted_states[cache_key] = result
        return result

    def get_all_states(self, layer_index: int = 0) -> List[StepResult]:
        """Get all converted states for the given layer."""
        count = self.get_iteration_count(layer_index)
        return [self.get_state(i, layer_index) for i in range(count)]

    def _trace_data_for_layer(self, layer_index: int) -> Dict[str, Any]:
        """Internal: trace_data for a layer (for /api/iterations per-layer)."""
        if layer_index < 0 or layer_index >= len(self._layers):
            return {}
        return self._layers[layer_index][0]
