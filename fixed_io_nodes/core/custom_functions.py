from torch import autograd
import torch
from .lookup_table import LookupTable

#not used
def phase_forward(x, y, phase_bins:int):
    return (x+y)%phase_bins

#not used
def mag_forward(x, y, mag_bins:int): 
    return (x+y)%mag_bins

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

def activation_strength_forward(phases, mags, lookup_table:LookupTable):

    phase_values = PhaseLookup.apply(phases, lookup_table)
    mag_values = MagLookup.apply(mags, lookup_table)
    signal = (phase_values*mag_values).sum(dim=-1)
    return signal

def activation_strength_forward_unquantized(phases, mags, gamma=1.0):
    """
    Compute activation strength from continuous phase and magnitude values.
    
    phases: tensor of phase values (in radians, typically [0, 2π])
    mags: tensor of magnitude values (in range suitable for exp(gamma*sin(mag)))
    gamma: scaling factor for magnitude exponential
    
    Returns: scalar activation strength = sum(cos(phase) * exp(gamma*sin(mag)))
    """
    # Phase component: cosine values
    phase_values = torch.cos(phases)

    # Magnitude component: exponential of sine-transformed values
    # Clamp the exponent to prevent numerical overflow
    mag_exponent = gamma * torch.sin(mags)
    mag_exponent = torch.clamp(mag_exponent, min=-10.0, max=10.0)  # Prevent exp overflow
    mag_values = torch.exp(mag_exponent)

    # Activation strength is the dot product
    signal = (phase_values * mag_values).sum(dim=-1)

    return signal
    
#To go back to quantized version, just change the following to activation_strength_forward    
signal_forward = activation_strength_forward_unquantized
activation_stength_forward = activation_strength_forward_unquantized
activation_strength_forward = activation_strength_forward_unquantized

