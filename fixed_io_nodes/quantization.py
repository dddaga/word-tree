from torch import nn, autograd
import torch
from lookup_table import LookupTable



class PhaseQuantForward(autograd.Function):

    @staticmethod
    def forward(ctx, values, indices, lookup_table:LookupTable):
        #values is not used here, but it is needed for backward pass
        ctx.save_for_backward(indices)
        ctx.lookup_table = lookup_table
        ctx.value_shape = values.shape
        return indices

    @staticmethod
    def backward(ctx, grad):

        indices = ctx.saved_tensors[0]
        value_grad = grad * 1/ctx.lookup_table.lookup_phase_grad(indices.int())
        value_grad = value_grad.reshape(ctx.value_shape)
        return value_grad, None, None


class MagQuantForward(autograd.Function):

    @staticmethod
    def forward(ctx, values, indices, lookup_table:LookupTable):
        #values is not used here, but it is needed for backward pass
        ctx.save_for_backward(indices)
        ctx.lookup_table = lookup_table
        ctx.value_shape = values.shape
        return indices

    @staticmethod
    def backward(ctx, grad):
        indices = ctx.saved_tensors[0]

        value_grad = grad * 1/ctx.lookup_table.lookup_magnitude_grad(indices)
        value_grad = value_grad.reshape(ctx.value_shape)
        return value_grad, None, None



def quantize_phase(values: torch.Tensor, phase_bins: int, lookup_table: LookupTable) -> torch.Tensor:
    """
    values: any shape
    phase_bins: number of bins for the phase
    lookup_table: lookup table for the phase

    returns: a tensor of same shape as "values", with values being indices of elements from lookup table which are closest to the elements of "values"
    """

    input_shape = values.shape
    values = values.flatten().unsqueeze(0)

    #if values are not in the range [-1, 1], normalize them, #TODO: check if this is the right method
    max_value, min_value = values.max().item(), values.min().item() 
    if max_value > 1 or min_value < -1:
        values = (values - min_value) / (max_value - min_value) * 2 - 1 

    y = lookup_table.lookup_phase(torch.arange(phase_bins)).reshape(phase_bins, 1) #basically get the values from lookup table without directly using lookup_table.phase_table

    _, indices = torch.abs(values-y).min(dim=0)
    

    closest_values = y.flatten()[indices] 
    values = (closest_values - values).detach() + values  #this will pass gradients straight from output to input

    indices = indices.to(torch.float16)

    #this allows the gradient to flow from indices to values during backward pass, making use of grad values from lookup_table
    indices = PhaseQuantForward.apply(values, indices, lookup_table)    

    return indices.reshape(*input_shape)


def quantize_magnitude(values: torch.Tensor, mag_bins: int, lookup_table: LookupTable) -> torch.Tensor:
    """
    values: any shape
    mag_bins: number of bins for the magnitude
    lookup_table: lookup table for the magnitude

    returns: a tensor of same shape as "values", with values being indices of elements from lookup table which are closest to the elements of "values"
    """
    input_shape = values.shape
    values = values.flatten().unsqueeze(0)

    #if values are not in the range [-1, 1], normalize them, #TODO: check if this is the right method
    max_value, min_value = values.max().item(), values.min().item() 
    if max_value > 1 or min_value < -1:
        values = (values - min_value) / (max_value - min_value) * 2 - 1 

    y = lookup_table.lookup_mag(torch.arange(mag_bins)).reshape(mag_bins, 1) #basically get the values from lookup table without directly using lookup_table.mag_table

    _, indices = torch.abs(values-y).min(dim=0)
    

    closest_values = y.flatten()[indices] 
    values = (closest_values - values).detach() + values  #this will pass gradients straight from output to input

    indices = indices.to(torch.float16)

    #this allows the gradient to flow from indices to values during backward pass, making use of grad values from lookup_table
    indices = MagQuantForward.apply(values, indices, lookup_table)    

    return indices.reshape(*input_shape)


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

