from torch import autograd
import torch
from .lookup_table import LookupTable

_EPSILON = 1e-8

class PhaseLookup(autograd.Function):
    @staticmethod
    def forward(ctx, indices, lookup_table:LookupTable):
        ctx.save_for_backward(indices)
        ctx.lookup_table = lookup_table

        values = lookup_table.lookup_phase(indices.int())
        return values
    
    @staticmethod
    def backward(ctx, grad):
        indices = ctx.saved_tensors[0]
        value_grad = grad * ctx.lookup_table.lookup_phase_grad(indices.int())
        return value_grad, None

class MagLookup(autograd.Function):
    @staticmethod
    def forward(ctx, indices, lookup_table:LookupTable=None):
        ctx.save_for_backward(indices)
        ctx.lookup_table = lookup_table
        values = lookup_table.lookup_mag(indices.int())
        return values
    
    @staticmethod
    def backward(ctx, grad):
        indices = ctx.saved_tensors[0]
        value_grad = grad * ctx.lookup_table.lookup_magnitude_grad(indices.int())
        return value_grad, None

# def activation_strength_forward(phases, mags, lookup_table:LookupTable):

#     phase_values = PhaseLookup.apply(phases, lookup_table)
#     mag_values = MagLookup.apply(mags, lookup_table)
#     signal = (phase_values*mag_values).sum(dim=-1)
#     return signal

def activation_strength_forward_unquantized(phases, mags, gamma=1.0):
    """
    Energy-norm activation strength: sqrt(sum(real² + imag²)) where complex = energy * exp(i*phase), energy = softplus(mag).
    gamma kept for call-site compatibility (unused).
    """
    # Previous: cos(phase) * exp(gamma*sin(mag)) with clamp

    # phase_values = torch.cos(phases)
    # mag_exponent = gamma * torch.sin(mags)
    # mag_exponent = torch.clamp(mag_exponent, min=-10.0, max=10.0)
    # mag_values = torch.exp(mag_exponent)
    # signal = (phase_values * mag_values).sum(dim=-1)
    # return signal

    energy_per_dim = torch.exp(mags)
    real = energy_per_dim * torch.cos(phases)
    return real.sum(dim=-1)
    
#To go back to quantized version, just change the following to activation_strength_forward    
signal_forward = activation_strength_forward_unquantized
activation_strength_forward = activation_strength_forward_unquantized

