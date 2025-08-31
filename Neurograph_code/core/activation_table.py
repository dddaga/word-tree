"""
Vectorized Activation Table for GPU-First NeuroGraph
Replaces Python dictionaries with GPU tensors for maximum performance
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from utils.device_manager import get_device_manager
import numpy as np


class VectorizedActivationTable:
    """
    GPU-first activation table using tensor operations instead of Python dictionaries.
    
    Key optimizations:
    - Integer node indices instead of string IDs
    - All operations vectorized on GPU
    - Persistent tensor allocation
    - Batch processing for inject/decay/prune
    """
    
    def __init__(
        self,
        max_nodes: int = 1200,
        vector_dim: int = 8,
        phase_bins: int = 32,
        mag_bins: int = 512,
        decay_factor: float = 0.95,
        min_strength: float = 0.001,
        device: str = 'auto'
    ):
        """
        Initialize vectorized activation table.
        
        Args:
            max_nodes: Maximum number of nodes that can be active (increased to 1200)
            vector_dim: Dimensionality of phase/mag vectors
            phase_bins: Number of discrete phase values
            mag_bins: Number of discrete magnitude values
            decay_factor: Decay factor per timestep
            min_strength: Minimum activation strength threshold
            device: Computation device
        """
        self.max_nodes = max_nodes
        self.vector_dim = vector_dim
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.decay_factor = decay_factor
        self.min_strength = min_strength
        
        # Adaptive pruning parameters - GENTLE & STABLE
        self.base_min_strength = min_strength  # Original threshold
        self.current_min_strength = min_strength  # Dynamic threshold
        self.target_active_nodes = 800  # Generous target for stable operation
        self.adaptation_rate = 0.1  # Gentle adaptation rate
        self.max_min_strength = 5.0  # Reasonable maximum threshold
        
        # Device management
        if device == 'auto':
            self.device_manager = get_device_manager()
            self.device = self.device_manager.device
        else:
            self.device_manager = None
            self.device = torch.device(device)
        
        # Initialize GPU tensors
        self._init_gpu_tensors()
        
        # Node mapping for compatibility with string-based systems
        self.node_id_to_index: Dict[str, int] = {}
        self.index_to_node_id: Dict[int, str] = {}
        self.next_free_index = 0
        self.free_indices = []  # Pool of recycled indices
        
        # Performance statistics
        self.stats = {
            'total_injections': 0,
            'total_decays': 0,
            'total_prunes': 0,
            'vectorized_operations': 0,
            'peak_active_nodes': 0
        }
        
        print(f"🚀 Vectorized Activation Table initialized:")
        print(f"   📊 Max nodes: {max_nodes}")
        print(f"   💾 Device: {self.device}")
        print(f"   📈 GPU memory: {self._estimate_gpu_memory():.2f} MB")
    
    def _init_gpu_tensors(self):
        """Initialize all GPU tensors for maximum performance."""
        # Core activation storage [max_nodes, vector_dim]
        self.phase_storage = torch.zeros(
            (self.max_nodes, self.vector_dim), 
            dtype=torch.long, 
            device=self.device
        )
        self.mag_storage = torch.zeros(
            (self.max_nodes, self.vector_dim), 
            dtype=torch.long, 
            device=self.device
        )
        
        # Activation strengths [max_nodes]
        self.strength_storage = torch.zeros(
            self.max_nodes, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Active node mask [max_nodes] - key optimization
        self.active_mask = torch.zeros(
            self.max_nodes, 
            dtype=torch.bool, 
            device=self.device
        )
        
        # Pre-allocated tensors for batch operations
        self.batch_indices = torch.zeros(
            self.max_nodes, 
            dtype=torch.long, 
            device=self.device
        )
        self.batch_strengths = torch.zeros(
            self.max_nodes, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Decay tensor for vectorized operations
        self.decay_tensor = torch.full(
            (self.max_nodes,), 
            self.decay_factor, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Minimum strength tensor for vectorized pruning
        self.min_strength_tensor = torch.full(
            (self.max_nodes,), 
            self.min_strength, 
            dtype=torch.float32, 
            device=self.device
        )
    
    def _estimate_gpu_memory(self) -> float:
        """Estimate GPU memory usage in MB."""
        # Phase and mag storage
        phase_mem = self.max_nodes * self.vector_dim * 8  # long = 8 bytes
        mag_mem = self.max_nodes * self.vector_dim * 8
        
        # Strength and mask storage
        strength_mem = self.max_nodes * 4  # float32 = 4 bytes
        mask_mem = self.max_nodes * 1  # bool = 1 byte
        
        # Batch operation tensors
        batch_mem = self.max_nodes * (8 + 4)  # indices + strengths
        
        # Decay and threshold tensors
        util_mem = self.max_nodes * 4 * 2  # decay + min_strength tensors
        
        total_bytes = phase_mem + mag_mem + strength_mem + mask_mem + batch_mem + util_mem
        return total_bytes / (1024 * 1024)
    
    def _get_node_index(self, node_id: str) -> int:
        """Get or allocate tensor index for node ID with index recycling."""
        if node_id in self.node_id_to_index:
            return self.node_id_to_index[node_id]
        
        # Try to reuse a freed index first
        if self.free_indices:
            index = self.free_indices.pop()
            # Reduced logging for cleaner output
            if len(self.node_id_to_index) % 100 == 0:  # Log every 100 allocations
                print(f"♻️ Recycling index {index} for node '{node_id}' (total: {len(self.node_id_to_index)})")
        else:
            # Only allocate new index if no recycled ones available
            if self.next_free_index >= self.max_nodes:
                print(f"🚨 ERROR: Activation table full!")
                print(f"🚨 Max nodes: {self.max_nodes}")
                print(f"🚨 Next free index: {self.next_free_index}")
                print(f"🚨 Total unique nodes seen: {len(self.node_id_to_index)}")
                print(f"🚨 Free indices available: {len(self.free_indices)}")
                raise RuntimeError(f"Activation table full ({self.max_nodes} nodes)")
            
            index = self.next_free_index
            self.next_free_index += 1
            # Reduced logging for cleaner output
            if len(self.node_id_to_index) % 100 == 0:  # Log every 100 allocations
                print(f"🆕 Allocating NEW index {index} for node '{node_id}' (total: {len(self.node_id_to_index)})")
        
        # Map the node to its index
        self.node_id_to_index[node_id] = index
        self.index_to_node_id[index] = node_id
        
        return index
    
    def inject_single(
        self, 
        node_id: str, 
        phase_idx: torch.Tensor, 
        mag_idx: torch.Tensor, 
        strength: float
    ):
        """
        Inject single node activation (compatibility method).
        
        Args:
            node_id: Node identifier
            phase_idx: Phase indices tensor [vector_dim]
            mag_idx: Magnitude indices tensor [vector_dim]
            strength: Activation strength
        """
        # Ensure tensors are on GPU
        phase_idx = phase_idx.to(self.device)
        mag_idx = mag_idx.to(self.device)
        
        # Get tensor index
        index = self._get_node_index(node_id)
        
        # Vectorized injection
        if self.active_mask[index]:
            # Update existing activation
            self.phase_storage[index] = (self.phase_storage[index] + phase_idx) % self.phase_bins
            self.mag_storage[index] = (self.mag_storage[index] + mag_idx) % self.mag_bins
            self.strength_storage[index] += strength
        else:
            # New activation
            self.phase_storage[index] = phase_idx % self.phase_bins
            self.mag_storage[index] = mag_idx % self.mag_bins
            self.strength_storage[index] = strength
            self.active_mask[index] = True
        
        self.stats['total_injections'] += 1
        self.stats['peak_active_nodes'] = max(
            self.stats['peak_active_nodes'], 
            self.active_mask.sum().item()
        )
    
    def inject_batch(
        self,
        node_indices: torch.Tensor,
        phase_indices: torch.Tensor,
        mag_indices: torch.Tensor,
        strengths: torch.Tensor
    ):
        """
        Vectorized batch injection for maximum GPU performance with excitatory filtering.
        
        Args:
            node_indices: Node indices tensor [batch_size]
            phase_indices: Phase indices tensor [batch_size, vector_dim]
            mag_indices: Magnitude indices tensor [batch_size, vector_dim]
            strengths: Activation strengths tensor [batch_size]
        """
        batch_size = node_indices.size(0)
        
        if batch_size == 0:
            return  # Nothing to inject
        
        # Ensure all tensors are on GPU
        node_indices = node_indices.to(self.device)
        phase_indices = phase_indices.to(self.device)
        mag_indices = mag_indices.to(self.device)
        strengths = strengths.to(self.device)
        
        # Note: Excitatory/inhibitory filtering is already done in propagation engine
        # All signals reaching here should be excitatory (strength > 0)
        filtered_batch_size = batch_size
        
        # Check for existing activations
        existing_mask = self.active_mask[node_indices]
        
        # Update existing activations
        if existing_mask.any():
            existing_indices = node_indices[existing_mask]
            existing_phases = phase_indices[existing_mask]
            existing_mags = mag_indices[existing_mask]
            existing_strengths = strengths[existing_mask]
            
            # Vectorized updates
            self.phase_storage[existing_indices] = (
                self.phase_storage[existing_indices] + existing_phases
            ) % self.phase_bins
            self.mag_storage[existing_indices] = (
                self.mag_storage[existing_indices] + existing_mags
            ) % self.mag_bins
            self.strength_storage[existing_indices] += existing_strengths
        
        # Add new activations
        new_mask = ~existing_mask
        if new_mask.any():
            new_indices = node_indices[new_mask]
            new_phases = phase_indices[new_mask]
            new_mags = mag_indices[new_mask]
            new_strengths = strengths[new_mask]
            
            # Vectorized new activations
            self.phase_storage[new_indices] = new_phases % self.phase_bins
            self.mag_storage[new_indices] = new_mags % self.mag_bins
            self.strength_storage[new_indices] = new_strengths
            self.active_mask[new_indices] = True
        
        self.stats['total_injections'] += filtered_batch_size
        self.stats['vectorized_operations'] += 1
        self.stats['peak_active_nodes'] = max(
            self.stats['peak_active_nodes'], 
            self.active_mask.sum().item()
        )
    
    def _adapt_pruning_threshold(self, current_active_count: int):
        """
        Adapt pruning threshold based on current network load.
        
        Args:
            current_active_count: Current number of active nodes
        """
        if current_active_count > self.target_active_nodes:
            # Network overloaded - increase threshold to prune more aggressively
            overage_ratio = current_active_count / self.target_active_nodes
            self.current_min_strength *= (1.0 + self.adaptation_rate * overage_ratio)
        elif current_active_count < self.target_active_nodes * 0.7:
            # Network underloaded - decrease threshold to keep more nodes
            self.current_min_strength *= (1.0 - self.adaptation_rate * 0.5)
        
        # Emergency brake for extreme overload
        if current_active_count > self.max_nodes * 0.9:
            self.current_min_strength *= 2.0  # Double threshold immediately
        
        # Keep within reasonable bounds
        self.current_min_strength = max(0.5, min(self.current_min_strength, self.max_min_strength))
    
    def decay_and_prune_vectorized(self, output_nodes: Optional[Set[str]] = None):
        """
        Enhanced vectorized decay and pruning with adaptive threshold and output protection.
        
        Args:
            output_nodes: Set of output node IDs to protect from pruning
        """
        # Get active nodes mask
        active_indices = torch.nonzero(self.active_mask, as_tuple=True)[0]
        
        if len(active_indices) == 0:
            return
        
        current_active_count = len(active_indices)
        
        # Adapt pruning threshold based on network load
        self._adapt_pruning_threshold(current_active_count)
        
        # Vectorized decay - single GPU operation
        self.strength_storage[active_indices] *= self.decay_factor
        
        # Priority-based pruning with output protection
        weak_mask = self.strength_storage[active_indices] < self.current_min_strength
        
        # Protect output nodes from pruning
        if output_nodes:
            output_protection_mask = torch.zeros_like(weak_mask, dtype=torch.bool)
            for i, idx in enumerate(active_indices):
                node_id = self.index_to_node_id.get(idx.item())
                if node_id and node_id in output_nodes:
                    output_protection_mask[i] = True
            
            # Apply protection: don't prune protected nodes
            weak_mask = weak_mask & ~output_protection_mask
        
        weak_indices = active_indices[weak_mask]
        
        if len(weak_indices) > 0:
            # Vectorized removal
            self.active_mask[weak_indices] = False
            self.strength_storage[weak_indices] = 0.0
            
            # Add freed indices to recycling pool (DON'T delete mappings)
            for idx in weak_indices.cpu().numpy():
                if idx in self.index_to_node_id:
                    node_id = self.index_to_node_id[idx]
                    # Remove from active mappings but keep the permanent mapping
                    del self.node_id_to_index[node_id]
                    del self.index_to_node_id[idx]
                    # Add to free pool for recycling
                    self.free_indices.append(idx)
                    # Verbose recycling output disabled for cleaner training logs
                    # print(f"♻️ Added index {idx} (node '{node_id}') to recycling pool")
            
            self.stats['total_prunes'] += len(weak_indices)
        
        self.stats['total_decays'] += 1
        self.stats['vectorized_operations'] += 1
    
    def get_adaptive_stats(self) -> Dict:
        """Get adaptive pruning statistics."""
        return {
            'base_min_strength': self.base_min_strength,
            'current_min_strength': self.current_min_strength,
            'target_active_nodes': self.target_active_nodes,
            'adaptation_rate': self.adaptation_rate,
            'threshold_ratio': self.current_min_strength / self.base_min_strength,
            'active_nodes': self.get_num_active(),
            'capacity_utilization': self.get_num_active() / self.max_nodes
        }
    
    def get_active_indices(self) -> torch.Tensor:
        """
        Get tensor of active node indices for vectorized operations.
        
        Returns:
            Active node indices tensor [num_active]
        """
        return torch.nonzero(self.active_mask, as_tuple=True)[0]
    
    def get_active_context_vectorized(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get vectorized active context for batch processing.
        
        Returns:
            Tuple of (active_indices, phase_data, mag_data)
            - active_indices: [num_active] 
            - phase_data: [num_active, vector_dim]
            - mag_data: [num_active, vector_dim]
        """
        active_indices = self.get_active_indices()
        
        if len(active_indices) == 0:
            # Return empty tensors
            empty_indices = torch.empty(0, dtype=torch.long, device=self.device)
            empty_data = torch.empty(0, self.vector_dim, dtype=torch.long, device=self.device)
            return empty_indices, empty_data, empty_data
        
        # Vectorized data extraction
        phase_data = self.phase_storage[active_indices]
        mag_data = self.mag_storage[active_indices]
        
        return active_indices, phase_data, mag_data
    
    def get_active_context_dict(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get active context as dictionary (compatibility method).
        
        Returns:
            Dictionary of node_id → (phase_indices, mag_indices)
        """
        active_indices = self.get_active_indices()
        context = {}
        
        for idx in active_indices.cpu().numpy():
            if idx in self.index_to_node_id:
                node_id = self.index_to_node_id[idx]
                context[node_id] = (
                    self.phase_storage[idx].clone(),
                    self.mag_storage[idx].clone()
                )
        
        return context
    
    def clear(self):
        """Clear all activations and reset tensors."""
        # Vectorized clearing
        self.active_mask.fill_(False)
        self.strength_storage.fill_(0.0)
        self.phase_storage.fill_(0)
        self.mag_storage.fill_(0)
        
        # Clear mappings
        self.node_id_to_index.clear()
        self.index_to_node_id.clear()
        self.next_free_index = 0
        
        # Memory cleanup
        if self.device_manager:
            self.device_manager.cleanup_memory()
    
    def get_num_active(self) -> int:
        """Get number of currently active nodes."""
        return self.active_mask.sum().item()
    
    def get_active_node_ids(self) -> List[str]:
        """Get list of active node IDs (compatibility method)."""
        active_indices = self.get_active_indices()
        return [
            self.index_to_node_id[idx.item()] 
            for idx in active_indices 
            if idx.item() in self.index_to_node_id
        ]
    
    def get_active_nodes_at_last_timestep(self) -> List[str]:
        """
        Get active nodes at the last timestep (compatibility method for training).
        
        This method provides compatibility with the training system that expects
        to get active nodes for credit assignment.
        
        Returns:
            List of active node IDs
        """
        return self.get_active_node_ids()
    
    def is_active(self, node_id: str) -> bool:
        """Check if node is currently active."""
        if node_id not in self.node_id_to_index:
            return False
        index = self.node_id_to_index[node_id]
        return self.active_mask[index].item()
    
    def get_performance_stats(self) -> Dict:
        """Get detailed performance statistics."""
        active_count = self.get_num_active()
        
        return {
            'active_nodes': active_count,
            'max_nodes': self.max_nodes,
            'utilization': active_count / self.max_nodes,
            'gpu_memory_mb': self._estimate_gpu_memory(),
            'device': str(self.device),
            'stats': self.stats.copy(),
            'vectorization_ratio': (
                self.stats['vectorized_operations'] / 
                max(self.stats['total_injections'] + self.stats['total_decays'], 1)
            )
        }
    
    def print_performance_report(self):
        """Print detailed performance report."""
        stats = self.get_performance_stats()
        
        print(f"\n🚀 Vectorized Activation Table Performance")
        print(f"=" * 50)
        print(f"Active nodes: {stats['active_nodes']:,} / {stats['max_nodes']:,}")
        print(f"GPU utilization: {stats['utilization']:.1%}")
        print(f"GPU memory: {stats['gpu_memory_mb']:.2f} MB")
        print(f"Device: {stats['device']}")
        
        print(f"\nVectorized Operations:")
        print(f"  Total injections: {stats['stats']['total_injections']:,}")
        print(f"  Total decays: {stats['stats']['total_decays']:,}")
        print(f"  Total prunes: {stats['stats']['total_prunes']:,}")
        print(f"  Vectorized ops: {stats['stats']['vectorized_operations']:,}")
        print(f"  Vectorization ratio: {stats['vectorization_ratio']:.1%}")
        
        print(f"\nPerformance:")
        if stats['vectorization_ratio'] > 0.8:
            print("  🟢 Excellent vectorization (>80%)")
        elif stats['vectorization_ratio'] > 0.5:
            print("  🟡 Good vectorization (>50%)")
        else:
            print("  🔴 Poor vectorization (<50%)")


# Compatibility functions for existing code
def create_vectorized_activation_table(
    max_nodes: int,
    vector_dim: int,
    phase_bins: int,
    mag_bins: int,
    config: Dict,
    device: str = 'auto'
) -> VectorizedActivationTable:
    """
    Factory function to create vectorized activation table.
    
    Args:
        max_nodes: Maximum number of nodes
        vector_dim: Vector dimensionality
        phase_bins: Number of phase bins
        mag_bins: Number of magnitude bins
        config: Configuration dictionary
        device: Computation device
        
    Returns:
        Configured vectorized activation table
    """
    return VectorizedActivationTable(
        max_nodes=max_nodes,
        vector_dim=vector_dim,
        phase_bins=phase_bins,
        mag_bins=mag_bins,
        decay_factor=config.get('activation', {}).get('decay_factor', 0.95),
        min_strength=config.get('activation', {}).get('min_strength', 0.001),
        device=device
    )
