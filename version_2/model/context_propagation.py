import torch
import torch.nn.functional as F
from .complex_tensor import ComplexTensor

class ContextPropagation:
    def __init__(self, N, M, activation_threshold, max_activation_strength, steps=8, max_clip_value=5.0):
        """
        Initialize context propagation with complex tensor support
        Args:
            N: Number of dimensions for complex tensors
            M: Context vector length
            activation_threshold: Minimum activation strength threshold
            max_activation_strength: Maximum allowed activation strength
            steps: Number of discrete steps for theta values
            max_clip_value: Maximum magnitude value for complex tensors
        """
        self.complex_tensor = ComplexTensor(N, steps=steps, max_clip_value=max_clip_value)
        self.activation_threshold = activation_threshold
        self.max_activation_strength = max_activation_strength

    def propagate_context(self, running_context, node_weights, connected_nodes):
        """
        Propagate context through connected nodes using complex tensor operations
        """
        propagated_activations = []
        for node in connected_nodes:
            # Create complex tensor from node weights
            node_complex = ComplexTensor(init_tensor=node['weight'])
            
            # Calculate alignment using complex tensor operations
            context_alignment = node_complex.forward(running_context).sum()
            context_strength = torch.min(context_alignment, torch.tensor(self.max_activation_strength))
            
            if context_strength > self.activation_threshold:
                # Update context by adding theta and magnitude components
                updated_context = node_complex.update(
                    running_context[0] + node_complex.tensor[0], # updating phase (theta)
                    context_strength * (running_context[1] + node_complex.tensor[1]) # updating magnitude   
                )
                propagated_activations.append(
                    (node['subword'], updated_context, context_strength)
                )
        
        return propagated_activations
