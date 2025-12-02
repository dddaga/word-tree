from torch import autograd
import torch
from lookup_table import LookupTable

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

def activation_stength_forward(phases, mags, lookup_table:LookupTable):

    phase_values = PhaseLookup.apply(phases, lookup_table)
    mag_values = MagLookup.apply(mags, lookup_table)
    signal = (phase_values*mag_values).sum(dim=-1)
    return signal
    
    
#backward compatibility
signal_forward = activation_stength_forward

