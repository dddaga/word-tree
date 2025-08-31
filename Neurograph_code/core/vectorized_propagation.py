"""
Vectorized Propagation Engine for GPU-First NeuroGraph
Replaces individual node loops with batch GPU tensor operations
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
from core.node_store import NodeStore
from core.modular_cell import ModularPhaseCell


class VectorizedPropagationEngine:
    """
    GPU-first propagation engine using vectorized tensor operations.
    
    Key optimizations:
    - Batch processing of all active nodes simultaneously
    - Vectorized radiation neighbor computation
    - GPU-based graph topology operations
    - Elimination of Python loops over nodes
    """
    
    def __init__(
        self,
        graph_df: pd.DataFrame,
        node_store: NodeStore,
        phase_cell,  # PhaseCell or ModularPhaseCell
        max_nodes: int,
        device: str = 'auto'
    ):
        """
        Initialize vectorized propagation engine.
        
        Args:
            graph_df: Static topology DataFrame
            node_store: Node parameter storage
            phase_cell: Phase computation cell
            max_nodes: Maximum number of nodes in graph
            device: Computation device
        """
        self.graph_df = graph_df
        self.node_store = node_store
        self.phase_cell = phase_cell
        self.max_nodes = max_nodes
        
        # Device setup
        if device == 'auto':
            from utils.device_manager import get_device_manager
            self.device = get_device_manager().device
        else:
            self.device = torch.device(device)
        
        # Pre-process graph topology for GPU operations
        self._preprocess_graph_topology()
        
        # Pre-allocate tensors for batch operations
        self._init_batch_tensors()
        
        # Performance statistics
        self.stats = {
            'total_propagations': 0,
            'vectorized_operations': 0,
            'batch_sizes': [],
            'radiation_computations': 0,
            'static_propagations': 0
        }
        
        print(f"🚀 Vectorized Propagation Engine initialized:")
        print(f"   📊 Max nodes: {max_nodes}")
        print(f"   💾 Device: {self.device}")
        print(f"   🔗 Graph edges: {len(self.edge_list)}")
    
    def _preprocess_graph_topology(self):
        """Pre-process graph topology for efficient GPU operations."""
        # Convert graph to edge list format for vectorized operations
        self.edge_list = []
        self.node_to_index = {}
        self.index_to_node = {}
        
        # Build node index mapping
        all_nodes = set()
        for _, row in self.graph_df.iterrows():
            source_node = row['node_id']
            target_nodes = row['input_connections']
            all_nodes.add(source_node)
            all_nodes.update(target_nodes)
        
        for i, node_id in enumerate(sorted(all_nodes)):
            self.node_to_index[node_id] = i
            self.index_to_node[i] = node_id
        
        # Build edge list: (source_index, target_index)
        for _, row in self.graph_df.iterrows():
            source_node = row['node_id']
            source_idx = self.node_to_index[source_node]
            
            for target_node in row['input_connections']:
                target_idx = self.node_to_index[target_node]
                self.edge_list.append((source_idx, target_idx))
        
        # Convert to GPU tensors for maximum performance
        if self.edge_list:
            edge_array = torch.tensor(self.edge_list, dtype=torch.long, device=self.device)
            self.source_indices = edge_array[:, 0]  # [num_edges]
            self.target_indices = edge_array[:, 1]  # [num_edges]
        else:
            self.source_indices = torch.empty(0, dtype=torch.long, device=self.device)
            self.target_indices = torch.empty(0, dtype=torch.long, device=self.device)
        
        # Create adjacency matrix for efficient neighbor lookups
        num_nodes = len(all_nodes)
        self.adjacency_matrix = torch.zeros(
            (num_nodes, num_nodes), 
            dtype=torch.bool, 
            device=self.device
        )
        
        if len(self.edge_list) > 0:
            self.adjacency_matrix[self.source_indices, self.target_indices] = True
    
    def _init_batch_tensors(self):
        """Initialize pre-allocated tensors for batch operations."""
        # Pre-allocated tensors for batch processing
        vector_dim = getattr(self.phase_cell, 'vector_dim', getattr(self.phase_cell, 'D', 5))
        self.vector_dim = vector_dim
        
        self.batch_source_phases = torch.zeros(
            (self.max_nodes, vector_dim), 
            dtype=torch.long, 
            device=self.device
        )
        self.batch_source_mags = torch.zeros(
            (self.max_nodes, vector_dim), 
            dtype=torch.long, 
            device=self.device
        )
        self.batch_target_phases = torch.zeros(
            (self.max_nodes, vector_dim), 
            dtype=torch.long, 
            device=self.device
        )
        self.batch_target_mags = torch.zeros(
            (self.max_nodes, vector_dim), 
            dtype=torch.long, 
            device=self.device
        )
        self.batch_strengths = torch.zeros(
            self.max_nodes, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Radiation computation tensors
        self.radiation_scores = torch.zeros(
            (self.max_nodes, self.max_nodes), 
            dtype=torch.float32, 
            device=self.device
        )
        self.radiation_mask = torch.zeros(
            (self.max_nodes, self.max_nodes), 
            dtype=torch.bool, 
            device=self.device
        )
    
    def propagate_vectorized(
        self,
        active_indices: torch.Tensor,
        active_phases: torch.Tensor,
        active_mags: torch.Tensor,
        use_radiation: bool = True,
        top_k_neighbors: int = 4,
        radiation_batch_size: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized propagation step processing all active nodes simultaneously.
        
        Args:
            active_indices: Active node indices [num_active]
            active_phases: Active node phases [num_active, vector_dim]
            active_mags: Active node magnitudes [num_active, vector_dim]
            use_radiation: Enable dynamic radiation
            top_k_neighbors: Number of radiation neighbors
            radiation_batch_size: Batch size for radiation computation
            
        Returns:
            Tuple of (target_indices, new_phases, new_mags, strengths)
        """
        num_active = len(active_indices)
        if num_active == 0:
            # Return empty tensors
            empty_tensor = torch.empty(0, dtype=torch.long, device=self.device)
            empty_phases = torch.empty(0, self.vector_dim, dtype=torch.long, device=self.device)
            empty_strengths = torch.empty(0, dtype=torch.float32, device=self.device)
            return empty_tensor, empty_phases, empty_phases, empty_strengths
        
        # Collect all propagation targets
        all_target_indices = []
        all_source_indices = []
        
        # Static propagation - vectorized edge processing
        static_targets, static_sources = self._get_static_targets_vectorized(active_indices)
        all_target_indices.extend(static_targets)
        all_source_indices.extend(static_sources)
        
        # Dynamic radiation - vectorized neighbor selection
        if use_radiation:
            radiation_targets, radiation_sources = self._get_radiation_targets_vectorized(
                active_indices, active_phases, top_k_neighbors, radiation_batch_size
            )
            all_target_indices.extend(radiation_targets)
            all_source_indices.extend(radiation_sources)
        
        if not all_target_indices:
            # No targets found
            empty_tensor = torch.empty(0, dtype=torch.long, device=self.device)
            empty_phases = torch.empty(0, self.vector_dim, dtype=torch.long, device=self.device)
            empty_strengths = torch.empty(0, dtype=torch.float32, device=self.device)
            return empty_tensor, empty_phases, empty_phases, empty_strengths
        
        # Convert to tensors
        target_tensor = torch.tensor(all_target_indices, dtype=torch.long, device=self.device)
        source_tensor = torch.tensor(all_source_indices, dtype=torch.long, device=self.device)
        
        # Batch phase cell computation
        filtered_target_indices, new_phases, new_mags, strengths = self._compute_phase_cell_batch(
            source_tensor, target_tensor, active_indices, active_phases, active_mags, radiation_batch_size
        )
        
        # Update statistics
        self.stats['total_propagations'] += 1
        self.stats['vectorized_operations'] += 1
        self.stats['batch_sizes'].append(len(target_tensor))
        if use_radiation:
            self.stats['radiation_computations'] += 1
        self.stats['static_propagations'] += 1
        
        return filtered_target_indices, new_phases, new_mags, strengths
    
    def _get_static_targets_vectorized(
        self, 
        active_indices: torch.Tensor
    ) -> Tuple[List[int], List[int]]:
        """
        Get static propagation targets using vectorized operations with bounds checking.
        
        Args:
            active_indices: Active node indices [num_active]
            
        Returns:
            Tuple of (target_indices, source_indices)
        """
        target_indices = []
        source_indices = []
        
        # Get adjacency matrix dimensions
        num_nodes = self.adjacency_matrix.size(0)
        
        # Use adjacency matrix for efficient neighbor lookup with bounds checking
        for source_idx in active_indices:
            source_idx_val = source_idx.item()
            
            # Skip if source index is out of bounds for adjacency matrix
            if source_idx_val >= num_nodes:
                continue
                
            # Find all targets for this source
            targets = torch.nonzero(self.adjacency_matrix[source_idx_val], as_tuple=True)[0]
            
            for target_idx in targets:
                target_indices.append(target_idx.item())
                source_indices.append(source_idx_val)
        
        return target_indices, source_indices
    
    def _get_radiation_targets_vectorized(
        self,
        active_indices: torch.Tensor,
        active_phases: torch.Tensor,
        top_k: int,
        batch_size: int
    ) -> Tuple[List[int], List[int]]:
        """
        Get radiation targets using vectorized phase alignment computation.
        
        Args:
            active_indices: Active node indices [num_active]
            active_phases: Active node phases [num_active, vector_dim]
            top_k: Number of radiation neighbors
            batch_size: Batch size for computation
            
        Returns:
            Tuple of (target_indices, source_indices)
        """
        target_indices = []
        source_indices = []
        
        num_active = len(active_indices)
        num_nodes = len(self.node_to_index)
        
        # Process active nodes in batches for memory efficiency
        for batch_start in range(0, num_active, batch_size):
            batch_end = min(batch_start + batch_size, num_active)
            batch_active_indices = active_indices[batch_start:batch_end]
            batch_active_phases = active_phases[batch_start:batch_end]
            
            # Compute radiation scores for this batch
            batch_targets, batch_sources = self._compute_radiation_batch(
                batch_active_indices, batch_active_phases, top_k, num_nodes
            )
            
            target_indices.extend(batch_targets)
            source_indices.extend(batch_sources)
        
        return target_indices, source_indices
    
    def _compute_radiation_batch(
        self,
        batch_active_indices: torch.Tensor,
        batch_active_phases: torch.Tensor,
        top_k: int,
        num_nodes: int
    ) -> Tuple[List[int], List[int]]:
        """
        Compute radiation neighbors for a batch of active nodes.
        
        Args:
            batch_active_indices: Batch of active node indices [batch_size]
            batch_active_phases: Batch of active phases [batch_size, vector_dim]
            top_k: Number of neighbors to select
            num_nodes: Total number of nodes
            
        Returns:
            Tuple of (target_indices, source_indices)
        """
        batch_size = len(batch_active_indices)
        target_indices = []
        source_indices = []
        
        # Get all node phases for comparison
        all_node_phases = self._get_all_node_phases(num_nodes)  # [num_nodes, vector_dim]
        
        for i, source_idx in enumerate(batch_active_indices):
            source_phase = batch_active_phases[i]  # [vector_dim]
            source_idx_val = source_idx.item()
            
            # Exclude static neighbors and self with bounds checking
            excluded_nodes = set([source_idx_val])
            
            # Skip if source index is out of bounds for adjacency matrix
            if source_idx_val < self.adjacency_matrix.size(0):
                static_neighbors = torch.nonzero(self.adjacency_matrix[source_idx_val], as_tuple=True)[0]
                excluded_nodes.update(static_neighbors.cpu().numpy())
            
            # REMOVED: Output node exclusion logic - allow all nodes to be radiation targets
            
            # Create candidate mask
            candidate_mask = torch.ones(num_nodes, dtype=torch.bool, device=self.device)
            for excluded_idx in excluded_nodes:
                if excluded_idx < num_nodes:  # bounds check
                    candidate_mask[excluded_idx] = False
            
            if not candidate_mask.any():
                continue
            
            # Get candidate phases
            candidate_phases = all_node_phases[candidate_mask]  # [num_candidates, vector_dim]
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=True)[0]
            
            # Compute phase alignment scores
            alignment_scores = self._compute_phase_alignment(
                source_phase, candidate_phases
            )  # [num_candidates]
            
            # Select top-k neighbors
            k = min(top_k, len(alignment_scores))
            if k > 0:
                top_k_indices = torch.topk(alignment_scores, k=k, largest=True)[1]
                selected_targets = candidate_indices[top_k_indices]
                
                for target_idx in selected_targets:
                    target_indices.append(target_idx.item())
                    source_indices.append(source_idx.item())
        
        return target_indices, source_indices
    
    def _get_all_node_phases(self, num_nodes: int) -> torch.Tensor:
        """
        Get phase vectors for all nodes in vectorized format.
        
        Args:
            num_nodes: Total number of nodes
            
        Returns:
            All node phases [num_nodes, vector_dim]
        """
        all_phases = torch.zeros(
            (num_nodes, self.vector_dim), 
            dtype=torch.long, 
            device=self.device
        )
        
        for i in range(num_nodes):
            if i in self.index_to_node:
                node_id = self.index_to_node[i]
                node_phase = self.node_store.get_phase(node_id).to(self.device)
                all_phases[i] = node_phase
        
        return all_phases
    
    def _compute_phase_alignment(
        self, 
        source_phase: torch.Tensor, 
        candidate_phases: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute phase alignment scores using vectorized operations.
        
        Args:
            source_phase: Source phase vector [vector_dim]
            candidate_phases: Candidate phase vectors [num_candidates, vector_dim]
            
        Returns:
            Alignment scores [num_candidates]
        """
        # Expand source phase for broadcasting
        source_expanded = source_phase.unsqueeze(0).expand_as(candidate_phases)
        
        # Get phase bins from phase cell
        phase_bins = getattr(self.phase_cell, 'phase_bins', getattr(self.phase_cell, 'N', 32))
        
        # Compute phase sums (mod N)
        phase_sums = (source_expanded + candidate_phases) % phase_bins
        
        # Use lookup table for phase alignment
        if hasattr(self.phase_cell, 'lookup'):
            # Use phase cell's lookup table
            alignment_values = self.phase_cell.lookup.lookup_phase(phase_sums)
            alignment_scores = alignment_values.sum(dim=1)  # Sum over vector_dim
        else:
            # Fallback: simple cosine approximation
            alignment_scores = torch.cos(
                2 * torch.pi * phase_sums.float() / phase_bins
            ).sum(dim=1)
        
        return alignment_scores
    
    def _compute_phase_cell_batch(
        self,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        active_indices: torch.Tensor,
        active_phases: torch.Tensor,
        active_mags: torch.Tensor,
        radiation_batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch computation of phase cell operations with excitatory/inhibitory filtering.
        
        Args:
            source_indices: Source node indices [num_propagations]
            target_indices: Target node indices [num_propagations]
            active_indices: Active node indices [num_active]
            active_phases: Active phases [num_active, vector_dim]
            active_mags: Active magnitudes [num_active, vector_dim]
            
        Returns:
            Tuple of (new_phases, new_mags, strengths) - only excitatory signals
        """
        num_propagations = len(source_indices)
        
        if num_propagations == 0:
            empty_phases = torch.empty(0, self.vector_dim, dtype=torch.long, device=self.device)
            empty_strengths = torch.empty(0, dtype=torch.float32, device=self.device)
            return empty_phases, empty_phases, empty_strengths
        
        # Create mapping from node indices to active data
        active_index_map = {idx.item(): i for i, idx in enumerate(active_indices)}
        
        # Gather source context data
        source_phases_batch = torch.zeros(
            (num_propagations, self.vector_dim), 
            dtype=torch.long, 
            device=self.device
        )
        source_mags_batch = torch.zeros(
            (num_propagations, self.vector_dim), 
            dtype=torch.long, 
            device=self.device
        )
        
        valid_propagations = []
        for i, source_idx in enumerate(source_indices):
            # Only process if source is active
            source_node_id = self.index_to_node.get(source_idx.item())
            if source_node_id and self.node_store.is_node_active(source_node_id):
                if source_idx.item() in active_index_map:
                    active_pos = active_index_map[source_idx.item()]
                    source_phases_batch[i] = active_phases[active_pos]
                    source_mags_batch[i] = active_mags[active_pos]
                    valid_propagations.append(i)
        
        # Gather target self data
        target_phases_batch = torch.zeros(
            (num_propagations, self.vector_dim), 
            dtype=torch.long, 
            device=self.device
        )
        target_mags_batch = torch.zeros(
            (num_propagations, self.vector_dim), 
            dtype=torch.long, 
            device=self.device
        )
        
        for i, target_idx in enumerate(target_indices):
            if target_idx.item() in self.index_to_node:
                node_id = self.index_to_node[target_idx.item()]
                target_phases_batch[i] = self.node_store.get_phase(node_id).to(self.device)
                target_mags_batch[i] = self.node_store.get_mag(node_id).to(self.device)
        
        # Batch phase cell computation
        new_phases_batch = torch.zeros_like(target_phases_batch)
        new_mags_batch = torch.zeros_like(target_mags_batch)
        strengths_batch = torch.zeros(num_propagations, dtype=torch.float32, device=self.device)
        
        # Process in smaller batches to manage memory
        for batch_start in range(0, num_propagations, radiation_batch_size):
            batch_end = min(batch_start + radiation_batch_size, num_propagations)
            
            batch_ctx_phases = source_phases_batch[batch_start:batch_end]
            batch_ctx_mags = source_mags_batch[batch_start:batch_end]
            batch_self_phases = target_phases_batch[batch_start:batch_end]
            batch_self_mags = target_mags_batch[batch_start:batch_end]
            
            # Vectorized phase cell computation
            for i in range(batch_end - batch_start):
                global_idx = batch_start + i
                
                # Skip invalid propagations
                if global_idx not in valid_propagations:
                    continue
                
                ctx_phase = batch_ctx_phases[i]
                ctx_mag = batch_ctx_mags[i]
                self_phase = batch_self_phases[i]
                self_mag = batch_self_mags[i]
                
                # Call phase cell
                phase_out, mag_out, signal, strength, grad_phase, grad_mag = self.phase_cell(
                    ctx_phase, ctx_mag, self_phase, self_mag
                )
                
                # NEW: Only process excitatory signals (strength > 0)
                if strength > 0:
                    # Activate target node
                    target_idx = target_indices[global_idx]
                    target_node_id = self.index_to_node.get(target_idx.item())
                    if target_node_id:
                        self.node_store.activate_node(target_node_id)
                    
                    # Store results
                    new_phases_batch[global_idx] = phase_out
                    new_mags_batch[global_idx] = mag_out
                    strengths_batch[global_idx] = strength.item() if hasattr(strength, 'item') else strength
                # Inhibitory signals (strength <= 0) are discarded
        
        # Filter to only excitatory propagations
        excitatory_mask = strengths_batch > 0
        
        # Also filter target indices to match
        filtered_target_indices = target_indices[excitatory_mask]
        
        return (
            filtered_target_indices,
            new_phases_batch[excitatory_mask],
            new_mags_batch[excitatory_mask],
            strengths_batch[excitatory_mask]
        )
    
    def get_performance_stats(self) -> Dict:
        """Get detailed performance statistics."""
        avg_batch_size = (
            sum(self.stats['batch_sizes']) / max(len(self.stats['batch_sizes']), 1)
        )
        
        return {
            'total_propagations': self.stats['total_propagations'],
            'vectorized_operations': self.stats['vectorized_operations'],
            'radiation_computations': self.stats['radiation_computations'],
            'static_propagations': self.stats['static_propagations'],
            'average_batch_size': avg_batch_size,
            'max_batch_size': max(self.stats['batch_sizes']) if self.stats['batch_sizes'] else 0,
            'graph_edges': len(self.edge_list),
            'device': str(self.device),
            'vectorization_ratio': (
                self.stats['vectorized_operations'] / 
                max(self.stats['total_propagations'], 1)
            )
        }
    
    def print_performance_report(self):
        """Print detailed performance report."""
        stats = self.get_performance_stats()
        
        print(f"\n🚀 Vectorized Propagation Engine Performance")
        print(f"=" * 55)
        print(f"Total propagations: {stats['total_propagations']:,}")
        print(f"Vectorized operations: {stats['vectorized_operations']:,}")
        print(f"Radiation computations: {stats['radiation_computations']:,}")
        print(f"Static propagations: {stats['static_propagations']:,}")
        
        print(f"\nBatch Processing:")
        print(f"  Average batch size: {stats['average_batch_size']:.1f}")
        print(f"  Max batch size: {stats['max_batch_size']:,}")
        print(f"  Vectorization ratio: {stats['vectorization_ratio']:.1%}")
        
        print(f"\nGraph Topology:")
        print(f"  Total edges: {stats['graph_edges']:,}")
        print(f"  Device: {stats['device']}")
        
        print(f"\nPerformance:")
        if stats['vectorization_ratio'] > 0.9:
            print("  🟢 Excellent vectorization (>90%)")
        elif stats['vectorization_ratio'] > 0.7:
            print("  🟡 Good vectorization (>70%)")
        else:
            print("  🔴 Poor vectorization (<70%)")


# Factory function for easy creation
def create_vectorized_propagation_engine(
    graph_df: pd.DataFrame,
    node_store: NodeStore,
    phase_cell,
    max_nodes: int,
    device: str = 'auto'
) -> VectorizedPropagationEngine:
    """
    Factory function to create vectorized propagation engine.
    
    Args:
        graph_df: Static topology DataFrame
        node_store: Node parameter storage
        phase_cell: Phase computation cell
        max_nodes: Maximum number of nodes
        device: Computation device
        
    Returns:
        Configured vectorized propagation engine
    """
    return VectorizedPropagationEngine(
        graph_df=graph_df,
        node_store=node_store,
        phase_cell=phase_cell,
        max_nodes=max_nodes,
        device=device
    )
