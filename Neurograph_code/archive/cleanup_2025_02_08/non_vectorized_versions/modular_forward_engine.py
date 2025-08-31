"""
Modular Forward Engine for NeuroGraph
Clean implementation without legacy dependencies
"""

import torch
from typing import Dict, List, Tuple, Optional
import pandas as pd

from core.activation_table import ActivationTable
from core.propagation import propagate_step
from core.node_store import NodeStore
from core.modular_cell import ModularPhaseCell


class ModularForwardEngine:
    """
    Clean modular forward engine for NeuroGraph.
    
    Features:
    - Event-driven propagation over multiple timesteps
    - Static DAG conduction + dynamic radiation
    - No legacy dependencies
    - Configurable termination conditions
    """
    
    def __init__(
        self,
        graph_df: pd.DataFrame,
        node_store: NodeStore,
        phase_cell: ModularPhaseCell,
        activation_table: ActivationTable,
        max_timesteps: int = 35,
        top_k_neighbors: int = 4,
        use_radiation: bool = True,
        radiation_batch_size: int = 64,
        min_output_activation_timesteps: int = 2,
        device: str = 'cpu',
        verbose: bool = False
    ):
        """
        Initialize modular forward engine.
        
        Args:
            graph_df: Static topology DataFrame
            node_store: Node parameter storage
            phase_cell: Phase computation cell
            activation_table: Activation tracking table
            max_timesteps: Maximum propagation timesteps
            top_k_neighbors: Top-K neighbors for radiation
            use_radiation: Enable dynamic radiation
            min_output_activation_timesteps: Min timesteps before output check
            device: Computation device
            verbose: Enable verbose logging
        """
        self.graph_df = graph_df
        self.node_store = node_store
        self.phase_cell = phase_cell
        self.activation_table = activation_table
        self.max_timesteps = max_timesteps
        self.top_k_neighbors = top_k_neighbors
        self.use_radiation = use_radiation
        self.radiation_batch_size = radiation_batch_size
        self.min_output_activation_timesteps = min_output_activation_timesteps
        self.device = device
        self.verbose = verbose
        
        # Extract output nodes from node store
        self.output_nodes = set(node_store.output_nodes) if hasattr(node_store, 'output_nodes') else set()
    
    def is_output_node_active(self, active_nodes: List[int]) -> bool:
        """Check if any output nodes are currently active."""
        return any(node_id in self.output_nodes for node_id in active_nodes)
    
    def propagate(self, input_context: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> ActivationTable:
        """
        Perform forward propagation through the network.
        
        Args:
            input_context: Input node activations {node_id: (phase_indices, mag_indices)}
            
        Returns:
            Final activation table
        """
        # Reset activation table
        self.activation_table.clear()
        
        # Inject initial input context (convert integer node IDs to string format)
        for node_id, (phase_idx, mag_idx) in input_context.items():
            # Convert integer node ID to string format expected by graph
            string_node_id = f"n{node_id}"
            self.activation_table.inject(
                string_node_id, 
                phase_idx.to(self.device), 
                mag_idx.to(self.device), 
                strength=1.0
            )
        
        if self.verbose:
            print(f"🚀 Starting forward propagation")
            print(f"   📊 Input nodes: {list(input_context.keys())}")
            print(f"   🎯 Output nodes: {list(self.output_nodes)}")
        
        # Propagation loop
        timestep = 0
        while timestep < self.max_timesteps:
            # Get current active context
            active_context = self.activation_table.get_active_context()
            
            if not active_context:
                if self.verbose:
                    print(f"💀 Network died at timestep {timestep} - no active nodes")
                break
            
            # Check for output activation
            active_nodes = list(active_context.keys())
            current_active_outputs = [n for n in active_nodes if n in self.output_nodes]
            
            if self.verbose:
                print(f"\n⏱️ Timestep {timestep}")
                print(f"   🔹 Active nodes: {active_nodes}")
                if current_active_outputs:
                    print(f"   🎯 Active outputs: {current_active_outputs}")
            
            # Check termination condition
            if (timestep >= self.min_output_activation_timesteps and 
                self.is_output_node_active(active_nodes)):
                if self.verbose:
                    print(f"✅ Output activation detected at timestep {timestep}")
                    print(f"   🎯 Active output nodes: {current_active_outputs}")
                break
            
            # Propagation step
            updates = propagate_step(
                active_nodes=active_context,
                node_store=self.node_store,
                phase_cell=self.phase_cell,
                graph_df=self.graph_df,
                lookup_table=self.phase_cell.lookup if hasattr(self.phase_cell, 'lookup') else None,
                use_radiation=self.use_radiation,
                top_k_neighbors=self.top_k_neighbors,
                radiation_batch_size=self.radiation_batch_size,
                device=self.device
            )
            
            # Clear previous activations (overwrite style)
            self.activation_table.clear()
            
            # Apply updates
            for target_node, new_phase, new_mag, strength in updates:
                self.activation_table.inject(target_node, new_phase, new_mag, strength)
            
            # Decay and prune weak activations
            self.activation_table.decay_and_prune()
            
            timestep += 1
        
        # Final summary
        final_context = self.activation_table.get_active_context()
        final_active_outputs = [n for n in final_context.keys() if n in self.output_nodes]
        
        if self.verbose:
            print(f"\n🏁 Forward pass completed after {timestep} timesteps")
            print(f"   🎯 Final active output nodes: {final_active_outputs}")
            print(f"   🔹 Total active nodes: {len(final_context)}")
        
        return self.activation_table


def create_modular_forward_engine(
    graph_df: pd.DataFrame,
    node_store: NodeStore,
    phase_cell: ModularPhaseCell,
    activation_table: ActivationTable,
    config: Dict,
    device: str = 'cpu'
) -> ModularForwardEngine:
    """
    Factory function to create modular forward engine.
    
    Args:
        graph_df: Static topology DataFrame
        node_store: Node parameter storage
        phase_cell: Phase computation cell
        activation_table: Activation tracking table
        config: Configuration dictionary
        device: Computation device
        
    Returns:
        Configured modular forward engine
    """
    return ModularForwardEngine(
        graph_df=graph_df,
        node_store=node_store,
        phase_cell=phase_cell,
        activation_table=activation_table,
        max_timesteps=config.get('forward_pass', {}).get('max_timesteps', 35),
        top_k_neighbors=config.get('graph_structure', {}).get('top_k_neighbors', 4),
        use_radiation=config.get('graph_structure', {}).get('use_radiation', True),
        radiation_batch_size=config.get('radiation', {}).get('batch_size', 64),
        min_output_activation_timesteps=config.get('forward_pass', {}).get('min_output_activation_timesteps', 2),
        device=device,
        verbose=config.get('forward_pass', {}).get('verbose', False)
    )
