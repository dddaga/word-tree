# core/tables.py

import torch
import math
import torch.nn as nn

def get_cos_table(N: int, device='cpu'):
    indices = torch.arange(N, dtype=torch.float32, device=device)
    return torch.cos(2 * math.pi * indices / N)

def get_sin_table(N: int, device='cpu'):
    indices = torch.arange(N, dtype=torch.float32, device=device)
    return torch.sin(2 * math.pi * indices / N)

def get_exp_sin_table(M: int, device='cpu'):
    indices = torch.arange(M, dtype=torch.float32, device=device)
    scaled = 2 * math.pi * indices / M
    return torch.exp(torch.sin(scaled))

def get_exp_sin_derivative_table(M: int, device='cpu'):
    indices = torch.arange(M, dtype=torch.float32, device=device)
    scaled = 2 * math.pi * indices / M
    return torch.exp(torch.sin(scaled)) * torch.cos(scaled)

class ExtendedLookupTableModule(nn.Module):
    def __init__(self, N: int, M: int, device='cpu'):
        """
        Stores forward and derivative lookup tables for phase and magnitude.
        Args:
            N (int): Number of discrete phase bins
            M (int): Number of discrete magnitude bins
        """
        super().__init__()
        self.N = N
        self.M = M
        self.device = device

        self.register_buffer("cos_table", get_cos_table(N, device))
        self.register_buffer("sin_table", get_sin_table(N, device))  # For gradient
        self.register_buffer("exp_table", get_exp_sin_table(M, device))
        self.register_buffer("exp_deriv_table", get_exp_sin_derivative_table(M, device))

    def lookup_phase(self, theta_indices: torch.Tensor) -> torch.Tensor:
        theta_indices = theta_indices.long()  # Ensure long tensor for indexing
        return self.cos_table[theta_indices % self.N]

    def lookup_phase_grad(self, theta_indices: torch.Tensor) -> torch.Tensor:
        theta_indices = theta_indices.long()  # Ensure long tensor for indexing
        return -self.sin_table[theta_indices % self.N]

    def lookup_magnitude(self, mag_indices: torch.Tensor) -> torch.Tensor:
        mag_indices = mag_indices.long()  # Ensure long tensor for indexing
        return self.exp_table[mag_indices % self.M]

    def lookup_magnitude_grad(self, mag_indices: torch.Tensor) -> torch.Tensor:
        mag_indices = mag_indices.long()  # Ensure long tensor for indexing
        return self.exp_deriv_table[mag_indices % self.M]

    def get_signal_vector(self, phase_indices: torch.Tensor, mag_indices: torch.Tensor) -> torch.Tensor:
        """
        Combines phase and magnitude lookups to create a signal vector.
        
        Args:
            phase_indices: tensor of shape [D] with phase bin indices
            mag_indices: tensor of shape [D] with magnitude bin indices
            
        Returns:
            Signal vector of shape [D] combining phase and magnitude
        """
        phase_values = self.lookup_phase(phase_indices)  # [D]
        mag_values = self.lookup_magnitude(mag_indices)  # [D]
        return phase_values * mag_values  # Element-wise multiplication [D]
