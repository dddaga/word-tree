"""
Modular PhaseCell for NeuroGraph
Simplified phase-magnitude flow with direct transfer logic
"""

import torch
import torch.nn as nn
from typing import Tuple
from core.high_res_tables import HighResolutionLookupTables

class ModularPhaseCell(nn.Module):
    """
    Simplified phase-magnitude cell with direct transfer logic.
    
    Features:
    - Direct phase-magnitude transfer (no complex interactions)
    - Modular accumulation: (source + target) % bins
    - High-resolution lookup tables (64×1024)
    - Clean gradient computation
    - Biological plausibility maintained
    """
    
    def __init__(self, vector_dim: int, lookup_tables: HighResolutionLookupTables):
        """
        Initialize modular phase cell.
        
        Args:
            vector_dim: Dimensionality of phase/magnitude vectors
            lookup_tables: High-resolution lookup tables
        """
        super().__init__()
        
        self.vector_dim = vector_dim
        self.lookup = lookup_tables
        self.phase_bins = lookup_tables.N
        self.mag_bins = lookup_tables.M
        
        print(f"🔧 Initializing Modular PhaseCell:")
        print(f"   📐 Vector dimension: {vector_dim}")
        print(f"   📊 Resolution: {self.phase_bins}×{self.mag_bins}")
        print(f"   🧠 Mode: Direct transfer (simplified)")
    
    def forward(self, ctx_phase_idx: torch.Tensor, ctx_mag_idx: torch.Tensor,
                self_phase_idx: torch.Tensor, self_mag_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with direct phase-magnitude transfer.
        
        Args:
            ctx_phase_idx: Context phase indices [vector_dim]
            ctx_mag_idx: Context magnitude indices [vector_dim]
            self_phase_idx: Node's own phase indices [vector_dim]
            self_mag_idx: Node's own magnitude indices [vector_dim]
            
        Returns:
            Tuple of:
            - phase_out: Output phase indices [vector_dim]
            - mag_out: Output magnitude indices [vector_dim]
            - signal: Signal vector [vector_dim]
            - strength: Scalar activation strength
            - grad_phase: Phase gradients [vector_dim]
            - grad_mag: Magnitude gradients [vector_dim]
        """
        # Direct modular transfer: phase determines routing, magnitude controls strength
        phase_out = (ctx_phase_idx + self_phase_idx) % self.phase_bins
        mag_out = (ctx_mag_idx + self_mag_idx) % self.mag_bins
        
        # Compute signal using lookup tables
        signal, grad_phase, grad_mag = self.lookup.forward(phase_out, mag_out)
        
        # NEW: Raw sum strength (can be negative for inhibitory signals)
        strength = torch.sum(signal)
        
        return phase_out, mag_out, signal, strength, grad_phase, grad_mag
    
    def compute_routing_strength(self, ctx_phase_idx: torch.Tensor, self_phase_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute routing strength based on phase alignment.
        
        Args:
            ctx_phase_idx: Context phase indices [vector_dim]
            self_phase_idx: Node's phase indices [vector_dim]
            
        Returns:
            Routing strength tensor [vector_dim]
        """
        # Phase alignment determines routing strength
        phase_diff = torch.abs(ctx_phase_idx - self_phase_idx)
        
        # Handle wraparound: min(diff, bins - diff)
        phase_diff = torch.min(phase_diff, self.phase_bins - phase_diff)
        
        # Convert to routing strength (higher alignment = stronger routing)
        # Normalize to [0, 1] where 0 = opposite phases, 1 = aligned phases
        routing_strength = 1.0 - (2.0 * phase_diff.float() / self.phase_bins)
        
        return routing_strength
    
    def compute_signal_strength(self, mag_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute signal strength from magnitude indices.
        
        Args:
            mag_idx: Magnitude indices [vector_dim]
            
        Returns:
            Signal strength tensor [vector_dim]
        """
        return self.lookup.lookup_magnitude(mag_idx)
    
    def get_phase_similarity(self, phase_idx1: torch.Tensor, phase_idx2: torch.Tensor) -> torch.Tensor:
        """
        Compute phase similarity between two phase vectors.
        
        Args:
            phase_idx1: First phase vector [vector_dim]
            phase_idx2: Second phase vector [vector_dim]
            
        Returns:
            Similarity score [0, 1]
        """
        # Convert to continuous phase values
        phase1_continuous = phase_idx1.float() / self.phase_bins * 2 * torch.pi
        phase2_continuous = phase_idx2.float() / self.phase_bins * 2 * torch.pi
        
        # Compute cosine similarity in phase space
        cos_sim = torch.cos(phase1_continuous - phase2_continuous)
        
        # Convert to [0, 1] range
        similarity = (cos_sim + 1.0) / 2.0
        
        return similarity.mean()  # Average across vector dimensions

'''class BiologicalPhaseCell(ModularPhaseCell):
    """
    Biologically-inspired variant with additional constraints.
    """
    
    def __init__(self, vector_dim: int, lookup_tables: HighResolutionLookupTables,
                 refractory_period: int = 3, saturation_threshold: float = 0.8):
        """
        Initialize biological phase cell.
        
        Args:
            vector_dim: Vector dimension
            lookup_tables: Lookup tables
            refractory_period: Timesteps of reduced responsiveness after activation
            saturation_threshold: Threshold for signal saturation
        """
        super().__init__(vector_dim, lookup_tables)
        
        self.refractory_period = refractory_period
        self.saturation_threshold = saturation_threshold
        
        # Track activation history for refractory period
        self.register_buffer('last_activation', torch.zeros(1, dtype=torch.long))
        self.register_buffer('activation_count', torch.zeros(1, dtype=torch.long))
    
    def forward(self, ctx_phase_idx: torch.Tensor, ctx_mag_idx: torch.Tensor,
                self_phase_idx: torch.Tensor, self_mag_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with biological constraints."""
        # Standard forward pass
        phase_out, mag_out, signal, strength, grad_phase, grad_mag = super().forward(
            ctx_phase_idx, ctx_mag_idx, self_phase_idx, self_mag_idx
        )
        
        # Apply refractory period
        if self.activation_count > 0 and (self.activation_count - self.last_activation) < self.refractory_period:
            # Reduce signal during refractory period
            refractory_factor = 0.3
            signal = signal * refractory_factor
            strength = strength * refractory_factor
            grad_phase = grad_phase * refractory_factor
            grad_mag = grad_mag * refractory_factor
        
        # Apply saturation
        if strength > self.saturation_threshold:
            saturation_factor = self.saturation_threshold / strength
            signal = signal * saturation_factor
            strength = strength * saturation_factor
        
        # Update activation tracking
        if strength > 0.1:  # Threshold for counting as activation
            self.last_activation = self.activation_count.clone()
        
        self.activation_count += 1
        
        return phase_out, mag_out, signal, strength, grad_phase, grad_mag

class AdaptivePhaseCell(ModularPhaseCell):
    """
    Adaptive phase cell with dynamic parameter adjustment.
    """
    
    def __init__(self, vector_dim: int, lookup_tables: HighResolutionLookupTables,
                 adaptation_rate: float = 0.01):
        """
        Initialize adaptive phase cell.
        
        Args:
            vector_dim: Vector dimension
            lookup_tables: Lookup tables
            adaptation_rate: Rate of parameter adaptation
        """
        super().__init__(vector_dim, lookup_tables)
        
        self.adaptation_rate = adaptation_rate
        
        # Adaptive parameters
        self.register_buffer('phase_sensitivity', torch.ones(vector_dim))
        self.register_buffer('mag_sensitivity', torch.ones(vector_dim))
        
        # Track usage statistics
        self.register_buffer('usage_count', torch.zeros(vector_dim))
        self.register_buffer('average_signal', torch.zeros(vector_dim))
    
    def forward(self, ctx_phase_idx: torch.Tensor, ctx_mag_idx: torch.Tensor,
                self_phase_idx: torch.Tensor, self_mag_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with adaptive sensitivity."""
        # Standard forward pass
        phase_out, mag_out, signal, strength, grad_phase, grad_mag = super().forward(
            ctx_phase_idx, ctx_mag_idx, self_phase_idx, self_mag_idx
        )
        
        # Apply adaptive sensitivity
        signal = signal * self.phase_sensitivity * self.mag_sensitivity
        grad_phase = grad_phase * self.phase_sensitivity
        grad_mag = grad_mag * self.mag_sensitivity
        
        # Update statistics
        self.usage_count += 1
        alpha = self.adaptation_rate
        self.average_signal = (1 - alpha) * self.average_signal + alpha * torch.abs(signal)
        
        # Adapt sensitivity based on usage
        if self.training:
            # Increase sensitivity for underused dimensions
            underused_mask = self.average_signal < 0.1
            self.phase_sensitivity[underused_mask] *= (1 + alpha)
            self.mag_sensitivity[underused_mask] *= (1 + alpha)
            
            # Decrease sensitivity for overused dimensions
            overused_mask = self.average_signal > 0.8
            self.phase_sensitivity[overused_mask] *= (1 - alpha)
            self.mag_sensitivity[overused_mask] *= (1 - alpha)
            
            # Clamp to reasonable range
            self.phase_sensitivity = torch.clamp(self.phase_sensitivity, 0.1, 2.0)
            self.mag_sensitivity = torch.clamp(self.mag_sensitivity, 0.1, 2.0)
        
        return phase_out, mag_out, signal, strength, grad_phase, grad_mag

class QuantizedPhaseCell(ModularPhaseCell):
    """
    Phase cell with additional quantization for extreme discretization.
    """
    
    def __init__(self, vector_dim: int, lookup_tables: HighResolutionLookupTables,
                 signal_quantization_levels: int = 16):
        """
        Initialize quantized phase cell.
        
        Args:
            vector_dim: Vector dimension
            lookup_tables: Lookup tables
            signal_quantization_levels: Number of quantization levels for signals
        """
        super().__init__(vector_dim, lookup_tables)
        
        self.signal_quantization_levels = signal_quantization_levels
    
    def quantize_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Quantize signal to discrete levels.
        
        Args:
            signal: Continuous signal tensor
            
        Returns:
            Quantized signal tensor
        """
        # Normalize to [0, 1]
        signal_min, signal_max = signal.min(), signal.max()
        if signal_max > signal_min:
            normalized = (signal - signal_min) / (signal_max - signal_min)
        else:
            normalized = torch.zeros_like(signal)
        
        # Quantize to discrete levels
        quantized = torch.floor(normalized * self.signal_quantization_levels)
        quantized = torch.clamp(quantized, 0, self.signal_quantization_levels - 1)
        
        # Convert back to original range
        quantized_normalized = quantized / self.signal_quantization_levels
        quantized_signal = quantized_normalized * (signal_max - signal_min) + signal_min
        
        return quantized_signal
    
    def forward(self, ctx_phase_idx: torch.Tensor, ctx_mag_idx: torch.Tensor,
                self_phase_idx: torch.Tensor, self_mag_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with signal quantization."""
        # Standard forward pass
        phase_out, mag_out, signal, strength, grad_phase, grad_mag = super().forward(
            ctx_phase_idx, ctx_mag_idx, self_phase_idx, self_mag_idx
        )
        
        # Quantize signal for extreme discretization
        signal = self.quantize_signal(signal)
        strength = torch.sum(torch.abs(signal))
        
        return phase_out, mag_out, signal, strength, grad_phase, grad_mag

def create_phase_cell(cell_type: str = "modular", **kwargs) -> ModularPhaseCell:
    """
    Factory function to create phase cells.
    
    Args:
        cell_type: Type of cell ("modular", "biological", "adaptive", "quantized")
        **kwargs: Additional arguments for cell
        
    Returns:
        Phase cell instance
    """
    if cell_type == "modular":
        return ModularPhaseCell(**kwargs)
    elif cell_type == "biological":
        return BiologicalPhaseCell(**kwargs)
    elif cell_type == "adaptive":
        return AdaptivePhaseCell(**kwargs)
    elif cell_type == "quantized":
        return QuantizedPhaseCell(**kwargs)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")

# Backward compatibility alias
PhaseCell = ModularPhaseCell
'''
