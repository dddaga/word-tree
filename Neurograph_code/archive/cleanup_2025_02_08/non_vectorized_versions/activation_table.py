# core/activation_table.py

import torch
from typing import Dict, List, Tuple, Optional, Set
from utils.device_manager import get_device_manager
import gc

class ActivationTable:
    def __init__(self, vector_dim, phase_bins, mag_bins, max_nodes=2000, 
                 decay_factor=0.95, min_strength=0.001, device='auto'):
        """
        Tracks and updates active neurons over time with optimized tensor storage.

        Args:
            vector_dim (int): Dimensionality of each phase/mag vector
            phase_bins (int): Number of discrete phase values
            mag_bins (int): Number of discrete magnitude values
            max_nodes (int): Maximum number of nodes to track simultaneously
            decay_factor (float): Multiplier applied to activation strength each timestep
            min_strength (float): Threshold below which activation is pruned
            device (str): 'cpu', 'cuda', or 'auto' for device manager
        """
        self.vector_dim = vector_dim
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.max_nodes = max_nodes
        self.decay = decay_factor
        self.min_strength = min_strength
        
        # Device management
        if device == 'auto':
            self.device_manager = get_device_manager()
            self.device = self.device_manager.device
        else:
            self.device_manager = None
            self.device = torch.device(device)
        
        # Pre-allocated tensor storage for performance
        self._init_tensor_storage()
        
        # Node tracking
        self.active_nodes: Set[str] = set()
        self.node_to_index: Dict[str, int] = {}
        self.index_to_node: Dict[int, str] = {}
        self.next_free_index = 0
        
        # Memory management
        self.memory_cleanup_counter = 0
        self.cleanup_frequency = 100  # Clean up every N operations
        
        # Statistics
        self.stats = {
            'total_injections': 0,
            'total_decays': 0,
            'memory_cleanups': 0,
            'peak_active_nodes': 0,
            'tensor_reallocations': 0
        }
        
        print(f"🔧 Optimized Activation Table initialized:")
        print(f"   📊 Max nodes: {max_nodes}")
        print(f"   💾 Device: {self.device}")
        print(f"   📈 Pre-allocated memory: {self._estimate_memory_usage():.2f} MB")
    
    def _init_tensor_storage(self):
        """Initialize pre-allocated tensor storage."""
        # Pre-allocated tensors for maximum performance
        self.phase_storage = torch.zeros(
            (self.max_nodes, self.vector_dim), 
            dtype=torch.long, device=self.device
        )
        self.mag_storage = torch.zeros(
            (self.max_nodes, self.vector_dim), 
            dtype=torch.long, device=self.device
        )
        self.strength_storage = torch.zeros(
            self.max_nodes, dtype=torch.float32, device=self.device
        )
        
        # Active mask for efficient operations
        self.active_mask = torch.zeros(
            self.max_nodes, dtype=torch.bool, device=self.device
        )
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        phase_mem = self.max_nodes * self.vector_dim * 8  # long = 8 bytes
        mag_mem = self.max_nodes * self.vector_dim * 8
        strength_mem = self.max_nodes * 4  # float32 = 4 bytes
        mask_mem = self.max_nodes * 1  # bool = 1 byte
        
        total_bytes = phase_mem + mag_mem + strength_mem + mask_mem
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def _get_node_index(self, node_id: str) -> int:
        """Get or allocate index for node."""
        if node_id in self.node_to_index:
            return self.node_to_index[node_id]
        
        # Check if we have space
        if len(self.active_nodes) >= self.max_nodes:
            # Need to free up space - remove oldest inactive node
            self._cleanup_inactive_nodes()
            
            if len(self.active_nodes) >= self.max_nodes:
                raise RuntimeError(f"Activation table full ({self.max_nodes} nodes)")
        
        # Allocate new index
        index = self.next_free_index
        self.node_to_index[node_id] = index
        self.index_to_node[index] = node_id
        
        # Find next free index
        while (self.next_free_index < self.max_nodes and 
               self.active_mask[self.next_free_index]):
            self.next_free_index += 1
        
        if self.next_free_index >= self.max_nodes:
            # Wrap around and find first free slot
            self.next_free_index = 0
            while (self.next_free_index < self.max_nodes and 
                   self.active_mask[self.next_free_index]):
                self.next_free_index += 1
        
        return index
    
    def _cleanup_inactive_nodes(self):
        """Remove inactive nodes to free up space."""
        # Find nodes with very low strength
        active_indices = torch.nonzero(self.active_mask, as_tuple=True)[0]
        if len(active_indices) == 0:
            return
        
        strengths = self.strength_storage[active_indices]
        min_strength_idx = torch.argmin(strengths)
        weakest_index = active_indices[min_strength_idx].item()
        
        # Remove weakest node
        node_id = self.index_to_node[weakest_index]
        self._remove_node(node_id, weakest_index)
        
        self.stats['memory_cleanups'] += 1

    def _remove_node(self, node_id: str, index: int):
        """Remove node from all tracking structures."""
        self.active_nodes.discard(node_id)
        self.node_to_index.pop(node_id, None)
        self.index_to_node.pop(index, None)
        self.active_mask[index] = False
        self.strength_storage[index] = 0.0
        
        # Update next_free_index if this is earlier
        if index < self.next_free_index:
            self.next_free_index = index

    def inject(self, node_id: str, phase_idx: torch.Tensor, mag_idx: torch.Tensor, strength: float):
        """
        Inject or update node activation with optimized tensor operations.
        
        Args:
            node_id: Node identifier
            phase_idx: Phase indices tensor [vector_dim]
            mag_idx: Magnitude indices tensor [vector_dim]
            strength: Activation strength
        """
        # Ensure tensors are on correct device
        if self.device_manager:
            phase_idx = self.device_manager.to_device(phase_idx)
            mag_idx = self.device_manager.to_device(mag_idx)
        else:
            phase_idx = phase_idx.to(self.device)
            mag_idx = mag_idx.to(self.device)
        
        # Get or allocate index
        index = self._get_node_index(node_id)
        
        if node_id in self.active_nodes:
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
            self.active_nodes.add(node_id)
        
        # Update statistics
        self.stats['total_injections'] += 1
        self.stats['peak_active_nodes'] = max(
            self.stats['peak_active_nodes'], len(self.active_nodes)
        )
        
        # Periodic memory cleanup
        self.memory_cleanup_counter += 1
        if self.memory_cleanup_counter >= self.cleanup_frequency:
            self._periodic_cleanup()

    def decay_and_prune(self):
        """
        Apply decay to strengths and remove weak activations using vectorized operations.
        """
        if not self.active_nodes:
            return
        
        # Get active indices
        active_indices = torch.nonzero(self.active_mask, as_tuple=True)[0]
        
        if len(active_indices) == 0:
            return
        
        # Vectorized decay
        self.strength_storage[active_indices] *= self.decay
        
        # Find nodes to prune
        weak_mask = self.strength_storage[active_indices] < self.min_strength
        weak_indices = active_indices[weak_mask]
        
        # Remove weak nodes
        nodes_to_remove = []
        for idx in weak_indices:
            idx_item = idx.item()
            if idx_item in self.index_to_node:
                node_id = self.index_to_node[idx_item]
                nodes_to_remove.append((node_id, idx_item))
        
        for node_id, index in nodes_to_remove:
            self._remove_node(node_id, index)
        
        self.stats['total_decays'] += 1

    def get_active_context(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns: dict of node_id → (phase_idx [D], mag_idx [D])
        """
        context = {}
        for node_id in self.active_nodes:
            index = self.node_to_index[node_id]
            context[node_id] = (
                self.phase_storage[index].clone(),
                self.mag_storage[index].clone()
            )
        return context

    def is_active(self, node_id: str) -> bool:
        """Check if node is currently active."""
        return node_id in self.active_nodes

    def get_active_nodes_at_last_timestep(self) -> List[str]:
        """
        Get list of all currently active node IDs.
        
        Returns:
            List of active node IDs
        """
        return list(self.active_nodes)

    def clear(self):
        """Clear all activations and reset storage."""
        self.active_nodes.clear()
        self.node_to_index.clear()
        self.index_to_node.clear()
        self.next_free_index = 0
        
        # Reset tensors
        self.active_mask.fill_(False)
        self.strength_storage.fill_(0.0)
        self.phase_storage.fill_(0)
        self.mag_storage.fill_(0)
        
        # Memory cleanup
        if self.device_manager:
            self.device_manager.cleanup_memory()
    
    def _periodic_cleanup(self):
        """Perform periodic memory cleanup."""
        self.memory_cleanup_counter = 0
        
        # Check memory pressure
        if self.device_manager and self.device_manager.check_memory_pressure():
            self.device_manager.cleanup_memory(aggressive=True)
        
        # Python garbage collection for node tracking dicts
        if len(self.node_to_index) > self.max_nodes * 1.5:
            # Clean up orphaned entries
            valid_nodes = set(self.active_nodes)
            orphaned_nodes = set(self.node_to_index.keys()) - valid_nodes
            
            for node_id in orphaned_nodes:
                self.node_to_index.pop(node_id, None)
            
            # Rebuild index_to_node
            self.index_to_node = {
                idx: node_id for node_id, idx in self.node_to_index.items()
            }
            
            gc.collect()
    
    def get_memory_stats(self) -> Dict[str, any]:
        """Get detailed memory and performance statistics."""
        return {
            'active_nodes': len(self.active_nodes),
            'max_nodes': self.max_nodes,
            'utilization': len(self.active_nodes) / self.max_nodes,
            'estimated_memory_mb': self._estimate_memory_usage(),
            'stats': self.stats.copy(),
            'device': str(self.device),
            'next_free_index': self.next_free_index
        }
    
    def print_memory_report(self):
        """Print detailed memory usage report."""
        stats = self.get_memory_stats()
        
        print(f"\n📊 Activation Table Memory Report")
        print(f"=" * 45)
        print(f"Active nodes: {stats['active_nodes']:,} / {stats['max_nodes']:,}")
        print(f"Utilization: {stats['utilization']:.1%}")
        print(f"Memory usage: {stats['estimated_memory_mb']:.2f} MB")
        print(f"Device: {stats['device']}")
        
        print(f"\nStatistics:")
        for key, value in stats['stats'].items():
            print(f"  {key.replace('_', ' ').title()}: {value:,}")
        
        if self.device_manager:
            device_memory = self.device_manager.get_memory_usage()
            print(f"\nDevice Memory:")
            print(f"  Allocated: {device_memory['allocated_gb']:.2f} GB")
            print(f"  Utilization: {device_memory['utilization']:.1%}")
