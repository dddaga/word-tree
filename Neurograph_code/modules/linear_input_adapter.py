"""
Enhanced Linear Input Adapter for Modular NeuroGraph
Deep neural network architecture with optimized quantization (784 → 1024 → 1024 → 2000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import math
from typing import Dict, Tuple, Optional
from core.high_res_tables import QuantizationUtils

class LinearInputAdapter(nn.Module):
    """
    Enhanced deep neural network input adapter for MNIST processing.
    
    Architecture Improvements:
    - 3-layer deep network: 784 → 1024 → 1024 → 2000
    - Non-linear feature extraction with ReLU activations
    - Layer normalization for stable training
    - Dropout regularization to prevent overfitting
    - Tanh output activation for bounded quantization
    - Optimized quantization for [-1, 1] output range
    - Gradient clipping support for training stability
    
    Performance Targets:
    - 2x parameters: ~3.1M vs previous 1.57M
    - Expected accuracy: 15-25% vs previous 11.5%
    - Better feature extraction from pixel patterns
    """
    
    def __init__(self, input_dim: int, num_input_nodes: int,
                 vector_dim: int, phase_bins: int, mag_bins: int,
                 normalization: str = "layer_norm", dropout: float = 0.1, 
                 learnable: bool = True, device: str = 'cpu',
                 hidden_dims: list = [1024, 1024]):
        """
        Initialize enhanced linear input adapter.
        
        Args:
            input_dim: Input dimension (784 for MNIST)
            num_input_nodes: Number of input nodes (200)
            vector_dim: Vector dimension per node (5)
            phase_bins: Number of discrete phase bins (32)
            mag_bins: Number of discrete magnitude bins (512)
            normalization: Normalization type ("layer_norm", "batch_norm", or None)
            dropout: Dropout probability for regularization
            learnable: Whether projection is learnable or fixed
            device: Computation device
            hidden_dims: Hidden layer dimensions for deep architecture
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_input_nodes = num_input_nodes
        self.vector_dim = vector_dim
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.device = device
        self.learnable = learnable
        self.hidden_dims = hidden_dims
        
        # Calculate output dimension: nodes × vector_dim × 2 (phase + magnitude)
        self.output_dim = num_input_nodes * vector_dim * 2
        
        print(f"🚀 Initializing Enhanced Linear Input Adapter:")
        print(f"   📊 Input dimension: {input_dim}")
        print(f"   🏗️  Architecture: {input_dim} → {' → '.join(map(str, hidden_dims))} → {self.output_dim}")
        print(f"   🎯 Output nodes: {num_input_nodes}")
        print(f"   📐 Vector dimension: {vector_dim}")
        print(f"   📈 Total output dimension: {self.output_dim}")
        print(f"   🧠 Learnable: {learnable}")
        print(f"   📊 Resolution: {phase_bins}×{mag_bins}")
        
        # Enhanced deep neural network architecture
        if learnable:
            self.projection = self._build_enhanced_network(
                input_dim, hidden_dims, self.output_dim, normalization, dropout
            )
        else:
            # Fixed random projection (fallback)
            projection_matrix = torch.randn(input_dim, self.output_dim) * 0.1
            self.register_buffer('fixed_projection', projection_matrix)
            bias = torch.zeros(self.output_dim)
            self.register_buffer('fixed_bias', bias)
        
        # Load MNIST dataset
        self.setup_dataset()
        
        # Enhanced quantization utilities
        self.quantizer = QuantizationUtils()
        
        # Calculate and report parameter count
        if learnable:
            param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"   🔢 Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
            print(f"   📈 Parameter increase: {param_count/1570000:.1f}x vs original")
        
        print(f"✅ Enhanced input adapter initialized")
    
    def _build_enhanced_network(self, input_dim: int, hidden_dims: list, output_dim: int,
                               normalization: str, dropout: float) -> nn.Module:
        """
        Build enhanced deep neural network architecture.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            normalization: Normalization type
            dropout: Dropout probability
            
        Returns:
            Sequential network with enhanced architecture
        """
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            linear = nn.Linear(prev_dim, hidden_dim)
            # Enhanced initialization for deeper networks
            nn.init.xavier_uniform_(linear.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            
            # Activation
            layers.append(nn.ReLU(inplace=True))
            
            # Normalization
            if normalization == "layer_norm":
                layers.append(nn.LayerNorm(hidden_dim))
            elif normalization == "batch_norm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Dropout (not on last hidden layer)
            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer with Tanh activation for bounded output
        output_linear = nn.Linear(prev_dim, output_dim)
        # Smaller initialization for output layer
        nn.init.xavier_uniform_(output_linear.weight, gain=0.5)
        nn.init.zeros_(output_linear.bias)
        layers.append(output_linear)
        
        # Tanh activation for bounded output [-1, 1]
        layers.append(nn.Tanh())
        
        return nn.Sequential(*layers)
    
    def setup_dataset(self):
        """Setup MNIST dataset with proper transforms."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
            transforms.Lambda(lambda x: x.view(-1))  # Flatten to [784]
        ])
        
        self.mnist = datasets.MNIST(
            root="data", train=True, download=True, transform=transform
        )
        
        print(f"📚 MNIST dataset loaded: {len(self.mnist)} samples")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass through deep neural network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim] or [input_dim]
            
        Returns:
            Projected tensor of shape [batch_size, output_dim] or [output_dim]
            Output is bounded in [-1, 1] due to Tanh activation
        """
        # Handle single sample input
        single_sample = x.dim() == 1
        if single_sample:
            x = x.unsqueeze(0)
        
        # Apply enhanced projection network
        if self.learnable:
            projected = self.projection(x)
        else:
            # Fallback to fixed projection
            projected = F.linear(x, self.fixed_projection.t(), self.fixed_bias)
            projected = torch.tanh(projected)  # Apply tanh for consistency
        
        # Return to original shape if single sample
        if single_sample:
            projected = projected.squeeze(0)
        
        return projected
    
    def quantize_to_phase_mag(self, projected: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced quantization optimized for Tanh output range [-1, 1].
        
        This method implements the core quantization process:
        1. Split network output into phase and magnitude components
        2. Map bounded values to appropriate discrete ranges
        3. Apply optimized quantization for each component type
        
        Args:
            projected: Projected tensor of shape [output_dim] with values in [-1, 1]
            
        Returns:
            Tuple of (phase_indices, mag_indices) each of shape [num_nodes, vector_dim]
        """
        # Reshape to [num_nodes, vector_dim, 2] (last dim: phase, magnitude)
        reshaped = projected.view(self.num_input_nodes, self.vector_dim, 2)
        
        # Split into phase and magnitude components
        phase_continuous = reshaped[:, :, 0]  # [num_nodes, vector_dim] - Component 0
        mag_continuous = reshaped[:, :, 1]    # [num_nodes, vector_dim] - Component 1
        
        # Enhanced quantization for Tanh output [-1, 1]
        phase_indices = self._quantize_phase_enhanced(phase_continuous)
        mag_indices = self._quantize_magnitude_enhanced(mag_continuous)
        
        return phase_indices, mag_indices
    
    def _quantize_phase_enhanced(self, phase_continuous: torch.Tensor) -> torch.Tensor:
        """
        Enhanced phase quantization optimized for [-1, 1] input range.
        
        Phase represents the "direction" or "angle" of the signal.
        Maps [-1, 1] → [0, 2π] → [0, phase_bins-1]
        
        Args:
            phase_continuous: Continuous phase values in [-1, 1]
            
        Returns:
            Discrete phase indices in [0, phase_bins-1]
        """
        # Map [-1, 1] to [0, 1]
        phase_normalized = (phase_continuous + 1.0) / 2.0
        
        # Map [0, 1] to [0, 2π]
        phase_radians = phase_normalized * 2 * math.pi
        
        # Quantize to discrete bins [0, phase_bins-1]
        phase_indices = torch.floor(
            phase_radians / (2 * math.pi) * self.phase_bins
        ).long()
        
        # Clamp to valid range
        phase_indices = torch.clamp(phase_indices, 0, self.phase_bins - 1)
        
        return phase_indices
    
    def _quantize_magnitude_enhanced(self, mag_continuous: torch.Tensor) -> torch.Tensor:
        """
        Enhanced magnitude quantization optimized for [-1, 1] input range.
        
        Magnitude represents the "strength" or "amplitude" of the signal.
        Maps [-1, 1] → [-3, 3] → [0, 1] → [0, mag_bins-1]
        Uses expanded range for better dynamic range in exponential space.
        
        Args:
            mag_continuous: Continuous magnitude values in [-1, 1]
            
        Returns:
            Discrete magnitude indices in [0, mag_bins-1]
        """
        # Map [-1, 1] to [-3, 3] for expanded dynamic range
        mag_scaled = mag_continuous * 3.0
        
        # Map [-3, 3] to [0, 1]
        mag_normalized = (mag_scaled + 3.0) / 6.0
        
        # Quantize to discrete bins [0, mag_bins-1]
        mag_indices = torch.floor(mag_normalized * self.mag_bins).long()
        
        # Clamp to valid range
        mag_indices = torch.clamp(mag_indices, 0, self.mag_bins - 1)
        
        return mag_indices
    
    def get_input_context(self, idx: int, input_node_ids: list) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], int]:
        """
        Get input context for a specific MNIST sample using enhanced processing.
        
        Args:
            idx: Sample index
            input_node_ids: List of input node IDs
            
        Returns:
            Tuple of (input_context, label)
            input_context: dict[node_id → (phase_indices, mag_indices)]
            label: Ground truth digit class
        """
        if idx >= len(self.mnist):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self.mnist)}")
        
        # Get MNIST sample
        image, label = self.mnist[idx]
        image = image.to(self.device)
        
        # Forward pass through enhanced projection network
        with torch.no_grad() if not self.training else torch.enable_grad():
            projected = self.forward(image)
        
        # Enhanced quantization to phase-magnitude indices
        phase_indices, mag_indices = self.quantize_to_phase_mag(projected)
        
        # Create input context dictionary
        input_context = {}
        for i, node_id in enumerate(input_node_ids):
            if i >= self.num_input_nodes:
                break
            
            input_context[node_id] = (
                phase_indices[i],  # [vector_dim]
                mag_indices[i]     # [vector_dim]
            )
        
        return input_context, label
    
    def get_batch_input_contexts(self, indices: list, input_node_ids: list) -> Tuple[list, list]:
        """
        Get input contexts for a batch of samples using enhanced processing.
        
        Args:
            indices: List of sample indices
            input_node_ids: List of input node IDs
            
        Returns:
            Tuple of (input_contexts, labels)
        """
        input_contexts = []
        labels = []
        
        for idx in indices:
            context, label = self.get_input_context(idx, input_node_ids)
            input_contexts.append(context)
            labels.append(label)
        
        return input_contexts, labels
    
    def get_dataset_info(self) -> Dict[str, any]:
        """Get information about the dataset and enhanced adapter."""
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad) if self.learnable else 0
        
        return {
            'dataset_size': len(self.mnist),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_input_nodes': self.num_input_nodes,
            'vector_dim': self.vector_dim,
            'phase_bins': self.phase_bins,
            'mag_bins': self.mag_bins,
            'learnable': self.learnable,
            'parameters': param_count,
            'architecture': f"{self.input_dim} → {' → '.join(map(str, self.hidden_dims))} → {self.output_dim}",
            'parameter_increase': param_count / 1570000 if param_count > 0 else 0,
            'enhanced_features': [
                'Deep 3-layer architecture',
                'ReLU non-linear activations', 
                'Layer normalization',
                'Dropout regularization',
                'Tanh bounded output',
                'Optimized quantization'
            ]
        }
    
    def get_projection_stats(self) -> Dict[str, float]:
        """Get statistics about the enhanced projection network."""
        if not self.learnable:
            return {'message': 'Fixed projection - no learnable parameters'}
        
        stats = {}
        
        # Analyze each layer in the network
        for i, layer in enumerate(self.projection):
            if isinstance(layer, nn.Linear):
                layer_name = f'layer_{i//4 + 1}' if i < len(self.projection) - 2 else 'output'
                weight = layer.weight
                bias = layer.bias
                
                stats[f'{layer_name}_weight_mean'] = weight.mean().item()
                stats[f'{layer_name}_weight_std'] = weight.std().item()
                stats[f'{layer_name}_weight_norm'] = torch.norm(weight).item()
                stats[f'{layer_name}_bias_mean'] = bias.mean().item()
                stats[f'{layer_name}_bias_std'] = bias.std().item()
        
        # Overall network statistics
        all_params = torch.cat([p.flatten() for p in self.parameters() if p.requires_grad])
        stats['total_param_mean'] = all_params.mean().item()
        stats['total_param_std'] = all_params.std().item()
        stats['total_param_norm'] = torch.norm(all_params).item()
        stats['total_parameters'] = len(all_params)
        
        return stats
    
    def get_gradient_stats(self) -> Dict[str, float]:
        """Get gradient statistics for monitoring training."""
        if not self.learnable:
            return {'message': 'Fixed projection - no gradients'}
        
        stats = {}
        total_grad_norm = 0.0
        
        for i, layer in enumerate(self.projection):
            if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                layer_name = f'layer_{i//4 + 1}' if i < len(self.projection) - 2 else 'output'
                
                weight_grad_norm = torch.norm(layer.weight.grad).item()
                bias_grad_norm = torch.norm(layer.bias.grad).item()
                
                stats[f'{layer_name}_weight_grad_norm'] = weight_grad_norm
                stats[f'{layer_name}_bias_grad_norm'] = bias_grad_norm
                
                total_grad_norm += weight_grad_norm + bias_grad_norm
        
        stats['total_grad_norm'] = total_grad_norm
        return stats
    
    def clip_gradients(self, max_norm: float) -> float:
        """
        Clip gradients to prevent exploding gradients in deep network.
        
        Args:
            max_norm: Maximum gradient norm
            
        Returns:
            Actual gradient norm before clipping
        """
        if not self.learnable:
            return 0.0
        
        # Compute gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        return total_norm.item()
    
    def visualize_projection_patterns(self, save_path: str = "logs/enhanced/projection_patterns.png"):
        """Visualize learned projection patterns from enhanced network."""
        if not self.learnable:
            print("⚠️  Cannot visualize fixed projection patterns")
            return
        
        import matplotlib.pyplot as plt
        import os
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Get first layer weights (most interpretable)
        first_layer = None
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                first_layer = layer
                break
        
        if first_layer is None:
            print("⚠️  No linear layers found for visualization")
            return
        
        weights = first_layer.weight.detach().cpu().numpy()  # [1024, 784]
        
        # Visualize first 16 hidden units
        num_patterns = min(16, weights.shape[0])
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(num_patterns):
            # Get weights for this hidden unit
            unit_weights = weights[i]  # [784]
            
            # Reshape to 28x28 image
            pattern = unit_weights.reshape(28, 28)
            
            axes[i].imshow(pattern, cmap='RdBu', vmin=-pattern.std(), vmax=pattern.std())
            axes[i].set_title(f'Hidden Unit {i}')
            axes[i].axis('off')
        
        plt.suptitle('Enhanced Input Adapter - First Layer Feature Detectors')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Enhanced projection patterns saved to {save_path}")
    
    def save_adapter(self, filepath: str):
        """Save enhanced adapter state."""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'num_input_nodes': self.num_input_nodes,
                'vector_dim': self.vector_dim,
                'phase_bins': self.phase_bins,
                'mag_bins': self.mag_bins,
                'learnable': self.learnable,
                'hidden_dims': self.hidden_dims,
                'architecture_type': 'enhanced_deep'
            }
        }, filepath)
        print(f"💾 Enhanced adapter saved to {filepath}")
    
    def load_adapter(self, filepath: str):
        """Load enhanced adapter state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        print(f"📂 Enhanced adapter loaded from {filepath}")

class FixedLinearInputAdapter(LinearInputAdapter):
    """Fixed (non-learnable) input adapter for comparison."""
    
    def __init__(self, **kwargs):
        """Initialize with learnable=False."""
        kwargs['learnable'] = False
        super().__init__(**kwargs)

class AdaptiveLinearInputAdapter(LinearInputAdapter):
    """Adaptive input adapter with dynamic quantization statistics."""
    
    def __init__(self, **kwargs):
        """Initialize adaptive adapter with running statistics."""
        super().__init__(**kwargs)
        
        # Track statistics for adaptive quantization
        self.register_buffer('running_phase_mean', torch.zeros(1))
        self.register_buffer('running_phase_std', torch.ones(1))
        self.register_buffer('running_mag_mean', torch.zeros(1))
        self.register_buffer('running_mag_std', torch.ones(1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def update_running_stats(self, phase_vals: torch.Tensor, mag_vals: torch.Tensor):
        """Update running statistics for adaptive quantization."""
        if self.training:
            momentum = 0.1
            
            # Update phase statistics
            phase_mean = phase_vals.mean()
            phase_std = phase_vals.std()
            
            self.running_phase_mean = (1 - momentum) * self.running_phase_mean + momentum * phase_mean
            self.running_phase_std = (1 - momentum) * self.running_phase_std + momentum * phase_std
            
            # Update magnitude statistics
            mag_mean = mag_vals.mean()
            mag_std = mag_vals.std()
            
            self.running_mag_mean = (1 - momentum) * self.running_mag_mean + momentum * mag_mean
            self.running_mag_std = (1 - momentum) * self.running_mag_std + momentum * mag_std
            
            self.num_batches_tracked += 1
    
    def quantize_to_phase_mag(self, projected: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adaptive quantization using running statistics."""
        # Reshape and split as before
        reshaped = projected.view(self.num_input_nodes, self.vector_dim, 2)
        phase_continuous = reshaped[:, :, 0]
        mag_continuous = reshaped[:, :, 1]
        
        # Update running statistics
        self.update_running_stats(phase_continuous, mag_continuous)
        
        # Normalize using running statistics
        if self.num_batches_tracked > 0:
            phase_normalized = (phase_continuous - self.running_phase_mean) / (self.running_phase_std + 1e-8)
            mag_normalized = (mag_continuous - self.running_mag_mean) / (self.running_mag_std + 1e-8)
        else:
            phase_normalized = phase_continuous
            mag_normalized = mag_continuous
        
        # Apply enhanced quantization
        phase_indices = self._quantize_phase_enhanced(phase_normalized)
        mag_indices = self._quantize_magnitude_enhanced(mag_normalized)
        
        return phase_indices, mag_indices

def create_input_adapter(adapter_type: str = "linear", **kwargs) -> LinearInputAdapter:
    """
    Factory function to create enhanced input adapters.
    
    Args:
        adapter_type: Type of adapter ("linear", "fixed", "adaptive")
        **kwargs: Additional arguments for adapter
        
    Returns:
        Enhanced input adapter instance
    """
    if adapter_type == "linear":
        return LinearInputAdapter(**kwargs)
    elif adapter_type == "fixed":
        return FixedLinearInputAdapter(**kwargs)
    elif adapter_type == "adaptive":
        return AdaptiveLinearInputAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

# Backward compatibility alias
MNISTPCAAdapter = LinearInputAdapter
