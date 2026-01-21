from torch import nn
import torch

class LookupTable(nn.Module):        
    def __init__(self, phase_bins:int, mag_bins:int, gamma=1,
    device:str='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.N = phase_bins
        self.phase_bins = phase_bins

        self.mag_bins = mag_bins
        self.M = mag_bins
        self.device = device
        self.setup_phase_tables()
        self.setup_magnitude_tables(gamma=gamma)
        

    def setup_phase_tables(self):

        phase_values = torch.linspace(0, 2 * torch.pi, self.N + 1, device=self.device)[:-1]  # Exclude 2π

        self.register_buffer('phase_cos_table', torch.cos(phase_values))
        self.register_buffer('phase_sin_table', torch.sin(phase_values))

        self.register_buffer('phase_grad_table', -torch.sin(phase_values)*torch.pi * 2 / self.N)

    def setup_magnitude_tables(self, gamma:int=1):
        """
        magnitude has discrete values of exp(gamma * sin(x)) where x is in the range [-pi, pi]
        """

        mag_range = torch.linspace(-torch.pi, torch.pi, self.M, device=self.device)        
        mag_exp_sin_table = torch.exp(gamma*torch.sin(mag_range))
        self.register_buffer('mag_exp_sin_table', mag_exp_sin_table)

        self.register_buffer('mag_exp_sin_grad_table', gamma*torch.cos(mag_range)*mag_exp_sin_table *  2*torch.pi/self.M)

    def lookup_phase(self, phase_indices):
        return self.phase_cos_table[torch.tensor(phase_indices, dtype=torch.int64)]

    def lookup_phase_sin(self, phase_indices):
        return self.phase_sin_table[torch.tensor(phase_indices, dtype=torch.int64)]

    def lookup_mag(self, mag_indices):
        return self.mag_exp_sin_table[torch.tensor(mag_indices, dtype=torch.int64)]

    def lookup_phase_grad(self, phase_indices):
        return self.phase_grad_table[torch.tensor(phase_indices, dtype=torch.int64)]

    def lookup_magnitude_grad(self, mag_indices):
        return self.mag_exp_sin_grad_table[torch.tensor(mag_indices, dtype=torch.int64)]
        
