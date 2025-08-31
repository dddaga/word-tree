"""
High-Resolution Lookup Tables for Modular NeuroGraph
Supports configurable phase bins (default: 512) and magnitude bins (default: 1024)
Optimized with JIT compilation for critical operations
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Optional

# JIT compilation flag - can be disabled for debugging
ENABLE_JIT = True

def jit_compile_if_enabled(func):
    """Decorator to conditionally apply JIT compilation."""
    if ENABLE_JIT:
        try:
            return torch.jit.script(func)
        except Exception as e:
            print(f"Warning: JIT compilation failed for {func.__name__}: {e}")
            return func
    return func

class HighResolutionLookupTables(nn.Module):
    """
    High-resolution lookup tables for discrete phase-magnitude computation.
    
    Features:
    - 64 phase bins (8x increase from 8)
    - 1024 magnitude bins (4x increase from 256)
    - 16x total resolution increase
    - Efficient GPU/CPU computation
    - Gradient computation for discrete updates
    """
    
    
    def setup_phase_tables(self):
        """Setup phase-related lookup tables."""
        # Phase values: [0, 2π) mapped to [0, N-1]
        phase_values = torch.linspace(0, 2 * math.pi, self.N + 1)[:-1]  # Exclude 2π
        
        # Cosine and sine tables for phase computation
        self.register_buffer('phase_cos_table', torch.cos(phase_values))
        self.register_buffer('phase_sin_table', torch.sin(phase_values))
        
        # Phase gradient table: -sin for cosine derivative
        self.register_buffer('phase_grad_table', -torch.sin(phase_values))
    
    def setup_magnitude_tables(self):
        """Setup magnitude-related lookup tables with bounded exp(sin(x)) function."""
        # Magnitude values: map indices to [-π, π] for sin input
        mag_range = torch.linspace(-math.pi, math.pi, self.M)
        
        # NEW: exp(sin(magnitude)) for bounded output [exp(-1), exp(1)] ≈ [0.37, 2.72]
        sin_vals = torch.sin(mag_range)
        self.register_buffer('mag_exp_sin_table', torch.exp(sin_vals))
        
        # Gradient: d/dx[exp(sin(x))] = cos(x) * exp(sin(x))
        cos_vals = torch.cos(mag_range)
        self.register_buffer('mag_exp_sin_grad_table', cos_vals * torch.exp(sin_vals))
        
        # Keep legacy tables for backward compatibility
        self.register_buffer('mag_exp_table', torch.exp(torch.linspace(-3, 3, self.M)))
        self.register_buffer('mag_grad_table', torch.exp(torch.linspace(-3, 3, self.M)))
        
        # Alternative: sigmoid mapping for bounded magnitudes
        sigmoid_range = torch.linspace(-3, 3, self.M)
        self.register_buffer('mag_sigmoid_table', torch.sigmoid(sigmoid_range))
        self.register_buffer('mag_sigmoid_grad_table', 
                           torch.sigmoid(sigmoid_range) * (1 - torch.sigmoid(sigmoid_range)))
    
    def setup_gradient_tables(self):
        """Setup gradient computation tables."""
        # Pre-compute common gradient patterns for efficiency
        
        # Phase gradient scaling factors
        phase_scale = 2 * math.pi / self.N
        self.register_buffer('phase_grad_scale', torch.tensor(phase_scale))
        
        # Magnitude gradient scaling factors
        mag_scale = 6.0 / self.M  # Range [-3, 3] divided by bins
        self.register_buffer('mag_grad_scale', torch.tensor(mag_scale))
    
    def lookup_phase(self, phase_indices: torch.Tensor) -> torch.Tensor:
        """
        Lookup cosine values for phase indices.
        
        Args:
            phase_indices: LongTensor of shape [...] with values in [0, N-1]
            
        Returns:
            FloatTensor of cosine values with same shape as input
        """
        # Clamp indices to valid range
        phase_indices = torch.clamp(phase_indices, 0, self.N - 1)
        return self.phase_cos_table[phase_indices]
    
    def lookup_phase_sin(self, phase_indices: torch.Tensor) -> torch.Tensor:
        """
        Lookup sine values for phase indices.
        
        Args:
            phase_indices: LongTensor of shape [...] with values in [0, N-1]
            
        Returns:
            FloatTensor of sine values with same shape as input
        """
        phase_indices = torch.clamp(phase_indices, 0, self.N - 1)
        return self.phase_sin_table[phase_indices]
    
    def lookup_magnitude(self, mag_indices: torch.Tensor, use_sigmoid: bool = False) -> torch.Tensor:
        """
        Lookup magnitude values for magnitude indices using exp(sin(x)) for boundedness.
        
        Args:
            mag_indices: LongTensor of shape [...] with values in [0, M-1]
            use_sigmoid: If True, use sigmoid mapping instead of exp(sin(x))
            
        Returns:
            FloatTensor of magnitude values with same shape as input
        """
        mag_indices = torch.clamp(mag_indices, 0, self.M - 1)
        
        if use_sigmoid:
            return self.mag_sigmoid_table[mag_indices]
        else:
            # Use new bounded exp(sin(x)) function by default
            return self.mag_exp_sin_table[mag_indices]
    
    def lookup_phase_grad(self, phase_indices: torch.Tensor) -> torch.Tensor:
        """
        Lookup phase gradients for discrete gradient computation.
        
        Args:
            phase_indices: LongTensor of shape [...] with values in [0, N-1]
            
        Returns:
            FloatTensor of phase gradients with same shape as input
        """
        phase_indices = torch.clamp(phase_indices, 0, self.N - 1)
        return self.phase_grad_table[phase_indices] * self.phase_grad_scale
    
    def lookup_magnitude_grad(self, mag_indices: torch.Tensor, use_sigmoid: bool = False) -> torch.Tensor:
        """
        Lookup magnitude gradients for discrete gradient computation using exp(sin(x)) derivatives.
        
        Args:
            mag_indices: LongTensor of shape [...] with values in [0, M-1]
            use_sigmoid: If True, use sigmoid gradients instead of exp(sin(x))
            
        Returns:
            FloatTensor of magnitude gradients with same shape as input
        """
        mag_indices = torch.clamp(mag_indices, 0, self.M - 1)
        
        if use_sigmoid:
            return self.mag_sigmoid_grad_table[mag_indices] * self.mag_grad_scale
        else:
            # Use new exp(sin(x)) gradients by default
            return self.mag_exp_sin_grad_table[mag_indices] * self.mag_grad_scale
    
    @jit_compile_if_enabled
    def get_signal_vector(self, phase_indices: torch.Tensor, mag_indices: torch.Tensor, 
                         use_sigmoid: bool = False) -> torch.Tensor:
        """
        Compute signal vector from phase and magnitude indices.
        [JIT OPTIMIZED - Critical path method]
        
        Args:
            phase_indices: LongTensor of shape [D] with phase indices
            mag_indices: LongTensor of shape [D] with magnitude indices
            use_sigmoid: If True, use sigmoid magnitude mapping
            
        Returns:
            FloatTensor of shape [D] with signal values
        """
        cos_vals = self.lookup_phase(phase_indices)
        mag_vals = self.lookup_magnitude(mag_indices, use_sigmoid)
        
        return cos_vals * mag_vals
    
    def quantize_to_phase(self, continuous_phase: torch.Tensor) -> torch.Tensor:
        """
        Quantize continuous phase values to discrete indices.
        
        Args:
            continuous_phase: FloatTensor with values in [0, 2π)
            
        Returns:
            LongTensor with discrete phase indices in [0, N-1]
        """
        # Normalize to [0, 1) then scale to [0, N)
        normalized = (continuous_phase % (2 * math.pi)) / (2 * math.pi)
        indices = torch.floor(normalized * self.N).long()
        return torch.clamp(indices, 0, self.N - 1)
    
    def quantize_to_magnitude(self, continuous_mag: torch.Tensor, 
                            mag_range: Tuple[float, float] = (-3.0, 3.0)) -> torch.Tensor:
        """
        Quantize continuous magnitude values to discrete indices.
        
        Args:
            continuous_mag: FloatTensor with magnitude values
            mag_range: Tuple of (min, max) values for magnitude range
            
        Returns:
            LongTensor with discrete magnitude indices in [0, M-1]
        """
        min_mag, max_mag = mag_range
        
        # Clamp to range and normalize to [0, 1]
        clamped = torch.clamp(continuous_mag, min_mag, max_mag)
        normalized = (clamped - min_mag) / (max_mag - min_mag)
        
        # Scale to [0, M) and convert to indices
        indices = torch.floor(normalized * self.M).long()
        return torch.clamp(indices, 0, self.M - 1)
    
    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Each table entry is 4 bytes (float32)
        phase_tables = 4 * self.N * 3  # cos, sin, grad
        mag_tables = 4 * self.M * 4    # exp, sigmoid, exp_grad, sigmoid_grad
        total_bytes = phase_tables + mag_tables
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def get_resolution_info(self) -> dict:
        """Get information about resolution and capacity."""
        return {
            'phase_bins': self.N,
            'mag_bins': self.M,
            'total_combinations': self.N * self.M,
            'resolution_vs_legacy': (self.N / 8) * (self.M / 256),
            'memory_mb': self.estimate_memory_usage(),
            'phase_resolution': 2 * math.pi / self.N,
            'mag_resolution': 6.0 / self.M
        }
    
    @jit_compile_if_enabled
    def compute_signal_gradients(self, phase_indices: torch.Tensor, mag_indices: torch.Tensor, 
                               upstream_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients using continuous function derivatives with chain rule.
        [JIT OPTIMIZED - Critical gradient computation method]
        
        This is the core method for continuous gradient approximation:
        - Uses exact derivatives of cos(x), sin(x), exp(x)
        - Applies chain rule with upstream gradients from loss
        - Maps continuous gradients to discrete parameter updates
        
        Args:
            phase_indices: LongTensor of shape [D] with phase indices
            mag_indices: LongTensor of shape [D] with magnitude indices  
            upstream_grad: FloatTensor of shape [D] with gradients from loss
            
        Returns:
            Tuple of (phase_gradients, magnitude_gradients) as FloatTensors [D]
        """
        # Clamp indices to valid range
        phase_indices = torch.clamp(phase_indices, 0, self.N - 1)
        mag_indices = torch.clamp(mag_indices, 0, self.M - 1)
        
        # Get current function values
        cos_vals = self.lookup_phase(phase_indices)  # cos(phase)
        sin_vals = self.lookup_phase_sin(phase_indices)  # sin(phase)
        mag_vals = self.lookup_magnitude(mag_indices)  # exp(magnitude)
        
        # Compute partial derivatives using continuous functions:
        # For signal = cos(phase) * exp(magnitude):
        # ∂signal/∂phase = -sin(phase) * exp(magnitude)
        # ∂signal/∂magnitude = cos(phase) * exp(magnitude)
        
        phase_partial = -sin_vals * mag_vals  # -sin(phase) * exp(mag)
        mag_partial = cos_vals * mag_vals     # cos(phase) * exp(mag)
        
        # Apply chain rule with upstream gradients
        phase_grad = phase_partial * upstream_grad
        mag_grad = mag_partial * upstream_grad
        
        return phase_grad, mag_grad
    
    def __init__(self, phase_bins: int, mag_bins: int, device: str = 'cpu'):
        """
        Initialize high-resolution lookup tables.
        
        Args:
            phase_bins: Number of discrete phase bins (default: 64)
            mag_bins: Number of discrete magnitude bins (default: 1024)
            device: Computation device ('cpu' or 'cuda')
        """
        super().__init__()
        
        self.N = phase_bins
        self.M = mag_bins
        self.device = device
        
        # Initialize discrete gradient accumulator
        self.discrete_grad_accumulator = {}  # node_id -> {'phase': float, 'mag': float}
        self.accumulation_threshold = 0.01  # Threshold for discrete updates (lowered from 0.08)
        
        print(f"🔧 Initializing High-Resolution Lookup Tables:")
        print(f"   📊 Phase bins: {phase_bins} (vs 8 legacy)")
        print(f"   📊 Magnitude bins: {mag_bins} (vs 256 legacy)")
        print(f"   📈 Resolution increase: {(phase_bins/8) * (mag_bins/256):.1f}x")
        print(f"   💾 Memory usage: ~{self.estimate_memory_usage():.1f} MB")
        print(f"   🎯 Discrete gradient accumulation threshold: {self.accumulation_threshold}")
        
        # Initialize lookup tables
        self.setup_phase_tables()
        self.setup_magnitude_tables()
        self.setup_gradient_tables()
        
        # Move to device
        self.to(device)
        
        print(f"✅ High-resolution tables initialized on {device}")

    def quantize_gradients_to_discrete_updates(self, phase_grad: torch.Tensor, 
                                             mag_grad: torch.Tensor,
                                             phase_learning_rate: float = 0.015,
                                             magnitude_learning_rate: float = 0.012,
                                             node_id: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert continuous gradients to discrete parameter updates using gradient accumulation.
        
        This implements discrete-level gradient accumulation:
        - Accumulates small gradients over multiple samples
        - Only makes discrete updates when accumulated gradient exceeds threshold
        - Preserves gradient direction for better update effectiveness
        
        Args:
            phase_grad: Continuous phase gradients [D]
            mag_grad: Continuous magnitude gradients [D]
            learning_rate: Learning rate for gradient descent
            node_id: Node identifier for accumulation tracking
            
        Returns:
            Tuple of (discrete_phase_updates, discrete_mag_updates) as LongTensors
        """
        # Scale gradients by resolution and learning rate
        phase_resolution = 2 * math.pi / self.N
        mag_resolution = 6.0 / self.M
        
        # Convert to discrete space (negative for gradient descent)
        phase_continuous = -phase_learning_rate * phase_grad / phase_resolution
        mag_continuous = -magnitude_learning_rate * mag_grad / mag_resolution
        
        # Initialize accumulator for this node if needed
        if node_id is not None and node_id not in self.discrete_grad_accumulator:
            self.discrete_grad_accumulator[node_id] = {
                'phase': torch.zeros_like(phase_continuous),
                'mag': torch.zeros_like(mag_continuous)
            }
        
        # Apply discrete gradient accumulation if node_id provided
        if node_id is not None:
            # Accumulate gradients
            self.discrete_grad_accumulator[node_id]['phase'] += phase_continuous
            self.discrete_grad_accumulator[node_id]['mag'] += mag_continuous
            
            # Check if accumulated gradients exceed threshold
            accumulated_phase = self.discrete_grad_accumulator[node_id]['phase']
            accumulated_mag = self.discrete_grad_accumulator[node_id]['mag']
            
            # Determine discrete updates based on accumulated gradients
            discrete_phase_updates = torch.zeros_like(accumulated_phase, dtype=torch.long)
            discrete_mag_updates = torch.zeros_like(accumulated_mag, dtype=torch.long)
            
            # Phase updates: apply when accumulated gradient exceeds threshold
            phase_update_mask = torch.abs(accumulated_phase) >= self.accumulation_threshold
            if torch.any(phase_update_mask):
                # Apply discrete updates and reset accumulator for those dimensions
                discrete_phase_updates[phase_update_mask] = torch.sign(
                    accumulated_phase[phase_update_mask]
                ).long()
                self.discrete_grad_accumulator[node_id]['phase'][phase_update_mask] = 0.0
            
            # Magnitude updates: apply when accumulated gradient exceeds threshold
            mag_update_mask = torch.abs(accumulated_mag) >= self.accumulation_threshold
            if torch.any(mag_update_mask):
                # Apply discrete updates and reset accumulator for those dimensions
                discrete_mag_updates[mag_update_mask] = torch.sign(
                    accumulated_mag[mag_update_mask]
                ).long()
                self.discrete_grad_accumulator[node_id]['mag'][mag_update_mask] = 0.0
            
        else:
            # Fallback: immediate quantization without accumulation
            discrete_phase_updates = torch.where(
                torch.abs(phase_continuous) >= 0.1,
                torch.sign(phase_continuous),
                torch.zeros_like(phase_continuous)
            ).long()
            
            discrete_mag_updates = torch.where(
                torch.abs(mag_continuous) >= 0.1,
                torch.sign(mag_continuous),
                torch.zeros_like(mag_continuous)
            ).long()
        
        # Debug information (can be removed in production)
        if torch.any(discrete_phase_updates != 0) or torch.any(discrete_mag_updates != 0):
            print(f"      🔧 Discrete Updates: Phase={torch.sum(torch.abs(discrete_phase_updates)).item():.0f} changes, "
                  f"Mag={torch.sum(torch.abs(discrete_mag_updates)).item():.0f} changes")
        
        return discrete_phase_updates, discrete_mag_updates
    
    def get_accumulator_stats(self, node_id: str) -> dict:
        """Get statistics about gradient accumulation for a specific node."""
        if node_id not in self.discrete_grad_accumulator:
            return {'phase_accumulated': 0.0, 'mag_accumulated': 0.0, 'ready_for_update': False}
        
        acc = self.discrete_grad_accumulator[node_id]
        phase_max = torch.max(torch.abs(acc['phase'])).item()
        mag_max = torch.max(torch.abs(acc['mag'])).item()
        
        return {
            'phase_accumulated': phase_max,
            'mag_accumulated': mag_max,
            'ready_for_update': phase_max >= self.accumulation_threshold or mag_max >= self.accumulation_threshold,
            'threshold': self.accumulation_threshold
        }
    
    def reset_accumulator(self, node_id: str = None):
        """Reset gradient accumulator for specific node or all nodes."""
        if node_id is not None:
            if node_id in self.discrete_grad_accumulator:
                del self.discrete_grad_accumulator[node_id]
        else:
            self.discrete_grad_accumulator.clear()
    
    def apply_discrete_updates(self, current_phase_indices: torch.Tensor,
                             current_mag_indices: torch.Tensor,
                             phase_updates: torch.Tensor,
                             mag_updates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply discrete updates with proper modular arithmetic.
        
        Args:
            current_phase_indices: Current phase indices [D]
            current_mag_indices: Current magnitude indices [D]
            phase_updates: Phase updates to apply [D]
            mag_updates: Magnitude updates to apply [D]
            
        Returns:
            Tuple of (new_phase_indices, new_mag_indices)
        """
        # Apply updates with modular arithmetic for phase (wraps around)
        new_phase_indices = (current_phase_indices + phase_updates) % self.N
        
        # Apply updates with clamping for magnitude (bounded range)
        new_mag_indices = torch.clamp(
            current_mag_indices + mag_updates, 
            0, self.M - 1
        )
        
        return new_phase_indices, new_mag_indices
    
    def compute_loss_gradients_wrt_signals(self, predicted_signals: torch.Tensor,
                                         target_signals: torch.Tensor,
                                         loss_type: str = 'mse') -> torch.Tensor:
        """
        Compute gradients of loss function w.r.t. predicted signals.
        
        This provides the upstream gradients for the chain rule.
        
        Args:
            predicted_signals: Predicted signal vectors [D]
            target_signals: Target signal vectors [D]
            loss_type: Type of loss ('mse', 'cosine', 'l1')
            
        Returns:
            Gradients w.r.t. predicted signals [D]
        """
        if loss_type == 'mse':
            # MSE: ∂L/∂pred = 2 * (pred - target)
            return 2.0 * (predicted_signals - target_signals)
        elif loss_type == 'l1':
            # L1: ∂L/∂pred = sign(pred - target)
            return torch.sign(predicted_signals - target_signals)
        elif loss_type == 'cosine':
            # Cosine similarity loss (more complex, simplified version)
            diff = predicted_signals - target_signals
            return diff / (torch.norm(diff) + 1e-8)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, phase_indices: torch.Tensor, mag_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for compatibility with existing PhaseCell interface.
        
        Args:
            phase_indices: LongTensor of shape [D]
            mag_indices: LongTensor of shape [D]
            
        Returns:
            Tuple of (signal_vector, phase_gradients, mag_gradients)
        """
        signal = self.get_signal_vector(phase_indices, mag_indices)
        phase_grad = self.lookup_phase_grad(phase_indices)
        mag_grad = self.lookup_magnitude_grad(mag_indices)
        
        return signal, phase_grad, mag_grad

class QuantizationUtils:
    """Utility functions for high-resolution quantization."""
    
    @staticmethod
    def adaptive_quantize_phase(values: torch.Tensor, phase_bins: int) -> torch.Tensor:
        """
        Adaptive phase quantization with improved distribution.
        
        Args:
            values: Continuous values to quantize
            phase_bins: Number of phase bins
            
        Returns:
            Quantized phase indices
        """
        # Normalize to [0, 1] using min-max scaling
        min_val, max_val = values.min(), values.max()
        if max_val > min_val:
            normalized = (values - min_val) / (max_val - min_val)
        else:
            normalized = torch.zeros_like(values)
        
        # Apply non-linear mapping for better distribution
        # Use sqrt for more uniform distribution in lower values
        mapped = torch.sqrt(normalized)
        
        # Quantize to bins
        indices = torch.floor(mapped * phase_bins).long()
        return torch.clamp(indices, 0, phase_bins - 1)
    
    @staticmethod
    def adaptive_quantize_magnitude(values: torch.Tensor, mag_bins: int) -> torch.Tensor:
        """
        Adaptive magnitude quantization with improved distribution.
        
        Args:
            values: Continuous values to quantize
            mag_bins: Number of magnitude bins
            
        Returns:
            Quantized magnitude indices
        """
        # Apply log scaling for magnitude values
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        log_values = torch.log(torch.abs(values) + epsilon)
        
        # Normalize log values
        min_log, max_log = log_values.min(), log_values.max()
        if max_log > min_log:
            normalized = (log_values - min_log) / (max_log - min_log)
        else:
            normalized = torch.zeros_like(log_values)
        
        # Quantize to bins
        indices = torch.floor(normalized * mag_bins).long()
        return torch.clamp(indices, 0, mag_bins - 1)

# Backward compatibility alias
ExtendedLookupTableModule = HighResolutionLookupTables
