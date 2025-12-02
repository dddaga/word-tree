from torch import nn, autograd
import torch
from lookup_table import LookupTable



class PhaseQuantForward(autograd.Function):

    @staticmethod
    def forward(ctx, values, indices, lookup_table:LookupTable):
        ctx.save_for_backward(indices)
        ctx.lookup_table = lookup_table
        return indices

    @staticmethod
    def backward(ctx, grad):

        indices = ctx.saved_tensors[0]
        value_grad = grad * 1/ctx.lookup_table.lookup_phase_grad(indices.int())

        return value_grad, None, None

def quantize_phase(values: torch.Tensor, phase_bins: int, lookup_table: LookupTable) -> torch.Tensor:
    """
    values: any shape
    phase_bins: number of bins for the phase
    lookup_table: lookup table for the phase

    returns: a tensor of same shape as "values", with values being indices of elements from lookup table which are closest to the elements of "values"
    """

    #normalize values to [-1, 1]
    #TODO: .item() is used for detach(). Check if  this is correct.
    max_value, min_value = values.max().item(), values.min().item() 
    normalized = (values - min_value) / (max_value - min_value) * 2 - 1 #this is differentiable


    y = lookup_table.lookup_phase(torch.arange(phase_bins)) #basically get the values from lookup table without directly using lookup_table.phase_table

    y = y.repeat(normalized.numel(), 1).T.reshape(phase_bins, *normalized.shape)

    
    _, indices = torch.abs(normalized-y).min(dim=0)
    

    values = normalized[indices] #as grads can be computed for normalized, it can be done for this too
    indices = indices.to(torch.float16)

    #this allows the gradient to flow from indices to values during backward pass, making use of grad values from lookup_table
    indices = PhaseQuantForward.apply(values, indices, lookup_table)    

    return indices 


class MagQuantForward(autograd.Function):

    @staticmethod
    def forward(ctx, values, indices, lookup_table:LookupTable):
        ctx.save_for_backward(indices)
        return indices

    @staticmethod
    def backward(ctx, grad):
        indices = ctx.saved_tensors[0]

        value_grad = grad * 1/ctx.lookup_table.lookup_magnitude_grad(indices)
        return value_grad, None, None

def quantize_magnitude(values: torch.Tensor, mag_bins: int, lookup_table: LookupTable) -> torch.Tensor:
    """
    values: any shape
    mag_bins: number of bins for the magnitude
    lookup_table: lookup table for the magnitude

    returns: a tensor of same shape as "values", with values being indices of elements from lookup table which are closest to the elements of "values"
    """
    max_value, min_value = values.max(), values.min()
    normalized = (values - min_value) / (max_value - min_value) * 2 - 1 #this is differentiable

    y = lookup_table.lookup_phase(torch.arange(mag_bins))
    y = y.repeat(normalized.numel(), 1).T.reshape(mag_bins, *normalized.shape)

    #This is actually a differentiable operation if you take the "values" tensor
    _, indices = torch.abs(normalized-y).min(dim=0)
    

    values = normalized[indices] #these are the actual values to which we wish to pass gradients
    indices = indices.to(torch.float16)

    #now this is differentiable
    indices = PhaseQuantForward.apply(values, indices, lookup_table)    
    

    return indices 


class Quantizer(nn.Module):
    
    def __init__(self, phase_bins:int, mag_bins:int, lookup_table:LookupTable, vector_dim:int, input_node_count:int):
        """
        phase_bins: number of bins for the phase
        mag_bins: number of bins for the magnitude
        lookup_table: lookup table for the phase and magnitude

        Converts the output of input adapter into the quantized phase and magnitudes, while also ensuring gradient flow
        """
        super().__init__()
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.lookup_table = lookup_table
        self.vector_dim = vector_dim
        self.input_node_count = input_node_count

    def forward(self, x):
        """
        x of shape (node_count * vector_dim)   
        """

        phases = torch.empty((self.input_node_count, self.vector_dim))
        # mags = torch.empty((self.input_node_count, self.vector_dim))

        phases = quantize_phase(x, self.phase_bins, self.lookup_table)
        # mags = quantize_magnitude(x[1], self.mag_bins, self.lookup_table)

        phases = phases.reshape(self.input_node_count, self.vector_dim)
        # mags = mags.reshape(self.input_node_count, self.vector_dim)

        return phases

