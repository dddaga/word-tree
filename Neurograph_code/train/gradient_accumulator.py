"""
Gradient Accumulation System for Modular NeuroGraph
Implements N-sample gradient buffering with √N learning rate scaling
"""

import torch
import math
from typing import Dict, Tuple, Optional, List
from collections import defaultdict
import numpy as np

class GradientAccumulator:
    """
    Gradient accumulation buffer for stable discrete parameter updates.
    
    Features:
    - N-sample gradient buffering (default: 8 samples)
    - √N learning rate scaling for stability
    - Per-node gradient tracking
    - Automatic buffer management
    - Statistics tracking for analysis
    """
    
    def __init__(self, accumulation_steps: int = 8, lr_scaling: str = "sqrt", 
                 buffer_size: int = 1500, device: str = 'auto'):  # Increased buffer size
        """
        Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of samples to accumulate (default: 8)
            lr_scaling: Learning rate scaling method ("sqrt", "linear", "none")
            buffer_size: Maximum number of nodes to buffer (increased to 1500)
            device: Computation device ('cpu', 'cuda', or 'auto')
        """
        self.accumulation_steps = accumulation_steps
        self.lr_scaling = lr_scaling
        self.buffer_size = buffer_size
        
        # Device management
        if device == 'auto':
            from utils.device_manager import get_device_manager
            self.device_manager = get_device_manager()
            self.device = self.device_manager.device
        else:
            self.device_manager = None
            self.device = torch.device(device)
        
        # Gradient buffers: node_id -> {'phase': [gradients], 'mag': [gradients], 'count': int}
        self.gradient_buffer = defaultdict(lambda: {
            'phase': [],
            'mag': [],
            'count': 0
        })
        
        # Global step counter
        self.step_count = 0
        self.update_count = 0
        
        # Learning rate scaling factor
        if lr_scaling == "sqrt":
            self.lr_scale = math.sqrt(accumulation_steps)
        elif lr_scaling == "linear":
            self.lr_scale = accumulation_steps
        else:
            self.lr_scale = 1.0
        
        # Statistics tracking
        self.stats = {
            'total_gradients_accumulated': 0,
            'total_updates_applied': 0,
            'average_gradient_norm': 0.0,
            'nodes_updated_per_cycle': [],
            'gradient_variance': 0.0
        }
        
        print(f"🔧 Initializing Gradient Accumulator:")
        print(f"   📊 Accumulation steps: {accumulation_steps}")
        print(f"   📈 Learning rate scaling: {lr_scaling} (factor: {self.lr_scale:.2f})")
        print(f"   💾 Buffer size: {buffer_size} nodes")
        print(f"   🎯 Effective batch size: {accumulation_steps}x")
        
        print(f"✅ Gradient accumulator initialized")
    
    def accumulate_gradients(self, node_id: int, phase_grad: torch.Tensor, mag_grad: torch.Tensor):
        """
        Accumulate gradients for a specific node.
        
        Args:
            node_id: Node identifier
            phase_grad: Phase gradient tensor [vector_dim]
            mag_grad: Magnitude gradient tensor [vector_dim]
        """
        # Check buffer size limit
        if len(self.gradient_buffer) >= self.buffer_size and node_id not in self.gradient_buffer:
            # Buffer full, skip this node (could implement LRU eviction)
            return
        
        # Store gradients
        buffer_entry = self.gradient_buffer[node_id]
        buffer_entry['phase'].append(phase_grad.clone().detach())
        buffer_entry['mag'].append(mag_grad.clone().detach())
        buffer_entry['count'] += 1
        
        # Update statistics
        self.stats['total_gradients_accumulated'] += 1
        
        # Track gradient norms for statistics
        phase_norm = torch.norm(phase_grad).item()
        mag_norm = torch.norm(mag_grad).item()
        total_norm = math.sqrt(phase_norm**2 + mag_norm**2)
        
        # Update running average of gradient norm
        alpha = 0.1  # Exponential moving average factor
        self.stats['average_gradient_norm'] = (
            (1 - alpha) * self.stats['average_gradient_norm'] + 
            alpha * total_norm
        )
    
    def should_update(self) -> bool:
        """
        Check if gradients should be applied based on accumulation steps.
        
        Returns:
            True if update should be performed
        """
        self.step_count += 1
        return self.step_count % self.accumulation_steps == 0
    
    def get_averaged_gradients(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get averaged gradients for all nodes in buffer.
        
        Returns:
            Dictionary mapping node_id -> (averaged_phase_grad, averaged_mag_grad)
        """
        averaged_gradients = {}
        nodes_with_updates = []
        
        for node_id, buffer_entry in self.gradient_buffer.items():
            if buffer_entry['count'] == 0:
                continue
            
            # Average accumulated gradients
            phase_grads = torch.stack(buffer_entry['phase'])  # [count, vector_dim]
            mag_grads = torch.stack(buffer_entry['mag'])      # [count, vector_dim]
            
            avg_phase_grad = torch.mean(phase_grads, dim=0)
            avg_mag_grad = torch.mean(mag_grads, dim=0)
            
            # Apply learning rate scaling
            scaled_phase_grad = avg_phase_grad * self.lr_scale
            scaled_mag_grad = avg_mag_grad * self.lr_scale
            
            averaged_gradients[node_id] = (scaled_phase_grad, scaled_mag_grad)
            nodes_with_updates.append(node_id)
            
            # Compute gradient variance for statistics
            if len(phase_grads) > 1:
                phase_var = torch.var(phase_grads, dim=0).mean().item()
                mag_var = torch.var(mag_grads, dim=0).mean().item()
                total_var = (phase_var + mag_var) / 2
                
                # Update running variance
                alpha = 0.1
                self.stats['gradient_variance'] = (
                    (1 - alpha) * self.stats['gradient_variance'] + 
                    alpha * total_var
                )
        
        # Update statistics
        self.stats['nodes_updated_per_cycle'].append(len(nodes_with_updates))
        if len(self.stats['nodes_updated_per_cycle']) > 100:
            self.stats['nodes_updated_per_cycle'].pop(0)  # Keep last 100 cycles
        
        return averaged_gradients
    
    def apply_accumulated_updates(self, node_store: Dict[int, Dict[str, torch.Tensor]], 
                                base_lr: float, phase_bins: int, mag_bins: int) -> int:
        """
        Apply accumulated gradients to node parameters.
        
        Args:
            node_store: Node parameter storage
            base_lr: Base learning rate (before scaling)
            phase_bins: Number of phase bins for modular arithmetic
            mag_bins: Number of magnitude bins for modular arithmetic
            
        Returns:
            Number of nodes updated
        """
        if not self.should_update():
            return 0
        
        averaged_gradients = self.get_averaged_gradients()
        nodes_updated = 0
        
        for node_id, (phase_grad, mag_grad) in averaged_gradients.items():
            if node_id not in node_store:
                continue
            
            # Get current parameters
            current_phase = node_store[node_id]['phase']  # [vector_dim]
            current_mag = node_store[node_id]['mag']      # [vector_dim]
            
            # Apply discrete updates with modular arithmetic
            # Note: base_lr is already scaled by self.lr_scale in averaged gradients
            new_phase = (current_phase.float() - base_lr * phase_grad) % phase_bins
            new_mag = (current_mag.float() - base_lr * mag_grad) % mag_bins
            
            # Convert back to integer indices
            node_store[node_id]['phase'] = new_phase.long().clamp(0, phase_bins - 1)
            node_store[node_id]['mag'] = new_mag.long().clamp(0, mag_bins - 1)
            
            nodes_updated += 1
        
        # Clear buffer after applying updates
        self.clear_buffer()
        
        # Update statistics
        self.stats['total_updates_applied'] += nodes_updated
        self.update_count += 1
        
        return nodes_updated
    
    def clear_buffer(self):
        """Clear the gradient buffer."""
        self.gradient_buffer.clear()
    
    def get_buffer_status(self) -> Dict[str, any]:
        """
        Get current buffer status.
        
        Returns:
            Dictionary with buffer information
        """
        total_gradients = sum(entry['count'] for entry in self.gradient_buffer.values())
        
        return {
            'nodes_in_buffer': len(self.gradient_buffer),
            'total_gradients_buffered': total_gradients,
            'buffer_utilization': len(self.gradient_buffer) / self.buffer_size,
            'step_count': self.step_count,
            'update_count': self.update_count,
            'steps_until_update': self.accumulation_steps - (self.step_count % self.accumulation_steps)
        }
    
    def get_statistics(self) -> Dict[str, any]:
        """Get accumulator statistics."""
        stats = self.stats.copy()
        
        # Add computed statistics
        if self.stats['nodes_updated_per_cycle']:
            stats['avg_nodes_per_update'] = np.mean(self.stats['nodes_updated_per_cycle'])
            stats['std_nodes_per_update'] = np.std(self.stats['nodes_updated_per_cycle'])
        
        stats['effective_learning_rate'] = f"base_lr × {self.lr_scale:.2f}"
        stats['accumulation_efficiency'] = (
            self.stats['total_updates_applied'] / 
            max(1, self.stats['total_gradients_accumulated'] / self.accumulation_steps)
        )
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.stats = {
            'total_gradients_accumulated': 0,
            'total_updates_applied': 0,
            'average_gradient_norm': 0.0,
            'nodes_updated_per_cycle': [],
            'gradient_variance': 0.0
        }
        self.step_count = 0
        self.update_count = 0
    
    def save_state(self, filepath: str):
        """Save accumulator state."""
        state = {
            'gradient_buffer': dict(self.gradient_buffer),
            'step_count': self.step_count,
            'update_count': self.update_count,
            'stats': self.stats,
            'config': {
                'accumulation_steps': self.accumulation_steps,
                'lr_scaling': self.lr_scaling,
                'lr_scale': self.lr_scale,
                'buffer_size': self.buffer_size
            }
        }
        
        torch.save(state, filepath)
        print(f"💾 Gradient accumulator state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load accumulator state."""
        state = torch.load(filepath, map_location=self.device)
        
        self.gradient_buffer = defaultdict(lambda: {'phase': [], 'mag': [], 'count': 0})
        self.gradient_buffer.update(state['gradient_buffer'])
        self.step_count = state['step_count']
        self.update_count = state['update_count']
        self.stats = state['stats']
        
        print(f"📂 Gradient accumulator state loaded from {filepath}")

class AdaptiveGradientAccumulator(GradientAccumulator):
    """
    Adaptive gradient accumulator with dynamic accumulation steps.
    """
    
    def __init__(self, min_accumulation: int = 4, max_accumulation: int = 16, **kwargs):
        """
        Initialize adaptive accumulator.
        
        Args:
            min_accumulation: Minimum accumulation steps
            max_accumulation: Maximum accumulation steps
            **kwargs: Other arguments for base class
        """
        super().__init__(**kwargs)
        
        self.min_accumulation = min_accumulation
        self.max_accumulation = max_accumulation
        self.base_accumulation = self.accumulation_steps
        
        # Track gradient stability
        self.gradient_stability_window = []
        self.stability_threshold = 0.1
    
    def update_accumulation_steps(self):
        """Adapt accumulation steps based on gradient stability."""
        if len(self.gradient_stability_window) < 10:
            return
        
        # Compute gradient stability (lower variance = more stable)
        recent_variance = np.mean(self.gradient_stability_window[-10:])
        
        if recent_variance < self.stability_threshold:
            # Gradients are stable, can reduce accumulation
            self.accumulation_steps = max(self.min_accumulation, self.accumulation_steps - 1)
        else:
            # Gradients are unstable, increase accumulation
            self.accumulation_steps = min(self.max_accumulation, self.accumulation_steps + 1)
        
        # Update learning rate scaling
        if self.lr_scaling == "sqrt":
            self.lr_scale = math.sqrt(self.accumulation_steps)
        elif self.lr_scaling == "linear":
            self.lr_scale = self.accumulation_steps
    
    def accumulate_gradients(self, node_id: int, phase_grad: torch.Tensor, mag_grad: torch.Tensor):
        """Accumulate gradients with stability tracking."""
        super().accumulate_gradients(node_id, phase_grad, mag_grad)
        
        # Track gradient stability
        grad_norm = torch.norm(phase_grad).item() + torch.norm(mag_grad).item()
        self.gradient_stability_window.append(grad_norm)
        
        if len(self.gradient_stability_window) > 50:
            self.gradient_stability_window.pop(0)
        
        # Update accumulation steps periodically
        if self.step_count % 20 == 0:
            self.update_accumulation_steps()

class BatchController:
    """
    Controller for managing gradient accumulation cycles.
    """
    
    def __init__(self, accumulator: GradientAccumulator):
        """
        Initialize batch controller.
        
        Args:
            accumulator: Gradient accumulator instance
        """
        self.accumulator = accumulator
        self.cycle_count = 0
        self.samples_processed = 0
    
    def process_sample(self, node_gradients: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> bool:
        """
        Process a single sample's gradients.
        
        Args:
            node_gradients: Dictionary mapping node_id -> (phase_grad, mag_grad)
            
        Returns:
            True if update cycle completed
        """
        # Accumulate gradients for all nodes
        for node_id, (phase_grad, mag_grad) in node_gradients.items():
            self.accumulator.accumulate_gradients(node_id, phase_grad, mag_grad)
        
        self.samples_processed += 1
        
        # Check if update should be applied
        if self.accumulator.should_update():
            self.cycle_count += 1
            return True
        
        return False
    
    def get_progress(self) -> Dict[str, any]:
        """Get processing progress information."""
        buffer_status = self.accumulator.get_buffer_status()
        
        return {
            'cycle_count': self.cycle_count,
            'samples_processed': self.samples_processed,
            'samples_per_cycle': self.accumulator.accumulation_steps,
            'current_cycle_progress': buffer_status['step_count'] % self.accumulator.accumulation_steps,
            'buffer_utilization': buffer_status['buffer_utilization']
        }

def create_gradient_accumulator(accumulator_type: str = "standard", **kwargs) -> GradientAccumulator:
    """
    Factory function to create gradient accumulators.
    
    Args:
        accumulator_type: Type of accumulator ("standard", "adaptive")
        **kwargs: Additional arguments for accumulator
        
    Returns:
        Gradient accumulator instance
    """
    if accumulator_type == "standard":
        return GradientAccumulator(**kwargs)
    elif accumulator_type == "adaptive":
        return AdaptiveGradientAccumulator(**kwargs)
    else:
        raise ValueError(f"Unknown accumulator type: {accumulator_type}")
