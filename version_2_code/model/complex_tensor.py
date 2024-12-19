import torch
import numpy as np
import random

class ComplexTensor:
    def __init__(self, N=None, init_tensor=None, steps=8, max_clip_value=5.0):
        """
        Initialize complex tensor with N dimensions or from existing tensor
        Args:
            N: Number of dimensions (only used if init_tensor is None)
            init_tensor: Initial tensor values (optional)
            steps: Number of discrete steps for theta (min 2)
            max_clip_value: Maximum absolute value for magnitude components
        """
        self.steps = max(2, steps)
        self.max_clip_value = max_clip_value
        
        if init_tensor is not None:
            self.tensor = init_tensor
            self.N = init_tensor.shape[1]
        else:
            assert N is not None, "Must provide either N or init_tensor"
            self.N = N
            # Initialize 2D tensor (2,N) in euler form
            self.tensor = torch.zeros((2, N), requires_grad=True)
            # Initialize theta with random discrete values from nth roots of unity
            self.tensor.data[0] = torch.tensor([2 * np.pi * random.randint(0, self.steps-1) / self.steps for _ in range(N)])
            # Initialize magnitudes with random values between -max_clip_value and max_clip_value
            self.tensor.data[1] = torch.rand(N) * 2 * max_clip_value - max_clip_value

    def _round_to_nearest_root(self, theta):
        """Round theta values to nearest nth root of unity based on steps"""
        step_size = 2 * np.pi / self.steps
        steps = torch.round(theta / step_size)
        return (steps * step_size) % (2 * np.pi)

    def update(self, grad_theta, grad_magnitude):
        """
        Update the complex tensor with gradients
        Args:
            grad_theta: Gradient for theta components
            grad_magnitude: Gradient for magnitude components
        """
        # Update theta values and round to nearest root of unity
        self.tensor.data[0] = self._round_to_nearest_root(self.tensor.data[0] - grad_theta)
        
        # Update magnitude values with clipping
        new_magnitudes = self.tensor.data[1] - grad_magnitude
        signs = torch.sign(new_magnitudes)
        clipped_magnitudes = torch.min(torch.abs(new_magnitudes), torch.tensor(self.max_clip_value))
        self.tensor.data[1] = signs * clipped_magnitudes
        return self.tensor

    def forward(self, weight_tensor):
        """
        Forward pass for complex tensor operations
        Args:
            weight_tensor: Weight complex tensor (2,N)
        Returns:
            Weighted sum of cosine similarity and magnitudes
        """
        # Compute cosine of theta difference
        theta_sim = torch.cos(self.tensor[0] - weight_tensor[0])
        # Add magnitudes and weight by theta similarity
        return theta_sim * (self.tensor[1] + weight_tensor[1])
