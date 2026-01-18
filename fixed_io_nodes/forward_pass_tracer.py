"""
Forward Pass Tracer Module

Captures detailed information about node activations, inputs, and connection types
during a forward pass through the GNN network.
"""

from typing import Dict, List, Optional, Any
import torch


class ForwardPassTracer:
    """
    Context manager that traces forward pass execution, capturing:
    - Active nodes per iteration
    - Input sources and types (radiation vs direct) for each node
    - Activation values (phase, magnitude, strength)
    - Connection details
    """
    
    def __init__(self):
        self.trace_data = {
            'iterations': []
        }
        self.current_iteration = -1
        self._is_active = False
    
    def __enter__(self):
        """Activate tracing when entering context."""
        self._is_active = True
        self.clear()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Deactivate tracing when exiting context."""
        self._is_active = False
    
    def clear(self):
        """Clear all trace data."""
        self.trace_data = {'iterations': []}
        self.current_iteration = -1
    
    def start_iteration(self, iteration: int, input_injected: bool = False):
        """Start recording a new iteration."""
        if not self._is_active:
            return
        
        self.current_iteration = iteration
        iteration_data = {
            'iteration': iteration,
            'input_injected': input_injected,
            'active_nodes': [],
            'radiation_targets': {},
            'direct_connections': {},  # source -> list of targets
            'node_details': {}
        }
        self.trace_data['iterations'].append(iteration_data)
    
    def record_radiation_targets(self, radiation_targets: Dict[int, List[int]]):
        """Record radiation targets for current iteration."""
        if not self._is_active or self.current_iteration < 0:
            return
        
        current_iter = self.trace_data['iterations'][-1]
        current_iter['radiation_targets'] = radiation_targets.copy()
    
    def record_direct_connections(self, direct_connections: Dict[int, List[int]]):
        """Record direct connections for current iteration."""
        if not self._is_active or self.current_iteration < 0:
            return
        
        current_iter = self.trace_data['iterations'][-1]
        current_iter['direct_connections'] = direct_connections.copy()
    
    def record_node_update(
        self,
        node_id: int,
        input_sources: List[int],
        input_types: List[str],
        phase_activation: torch.Tensor,
        mag_activation: torch.Tensor,
        activation_strength: torch.Tensor,
    ):
        """
        Record details for a node that was updated.
        
        Args:
            node_id: ID of the node
            input_sources: List of node IDs that provided input
            input_types: List of 'radiation' or 'direct' for each input source
            phase_activation: Current phase activation tensor
            mag_activation: Current magnitude activation tensor
            activation_strength: Current activation strength tensor
        """
        if not self._is_active or self.current_iteration < 0:
            return
        
        current_iter = self.trace_data['iterations'][-1]
        
        # Convert tensors to CPU and detach for storage
        phase_act = phase_activation.detach().cpu().clone() if isinstance(phase_activation, torch.Tensor) else phase_activation
        mag_act = mag_activation.detach().cpu().clone() if isinstance(mag_activation, torch.Tensor) else mag_activation
        act_strength = activation_strength.detach().cpu().item() if isinstance(activation_strength, torch.Tensor) else float(activation_strength)
        
        node_detail = {
            'inputs': input_sources.copy(),
            'input_types': input_types.copy(),
            'phase_activation': phase_act,
            'mag_activation': mag_act,
            'activation_strength': act_strength,
        }
        
        current_iter['node_details'][node_id] = node_detail
        
        # Track active nodes
        if node_id not in current_iter['active_nodes']:
            current_iter['active_nodes'].append(node_id)
    
    def record_active_nodes(self, node_ids: List[int]):
        """Record list of active nodes for current iteration."""
        if not self._is_active or self.current_iteration < 0:
            return
        
        current_iter = self.trace_data['iterations'][-1]
        current_iter['active_nodes'] = list(set(current_iter['active_nodes'] + node_ids))
    
    def get_trace(self) -> Dict[str, Any]:
        """Get the complete trace data."""
        return self.trace_data
    
    def get_iteration(self, iteration: int) -> Optional[Dict[str, Any]]:
        """Get trace data for a specific iteration."""
        for iter_data in self.trace_data['iterations']:
            if iter_data['iteration'] == iteration:
                return iter_data
        return None
    
    def get_num_iterations(self) -> int:
        """Get the number of iterations traced."""
        return len(self.trace_data['iterations'])
    
    def is_active(self) -> bool:
        """Check if tracer is currently active."""
        return self._is_active
