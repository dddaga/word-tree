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
    
def update_activations(
    phase_activations: torch.Tensor,
    mag_activations: torch.Tensor,
    phase_weights: torch.Tensor,
    mag_weights: torch.Tensor,
    activation_strengths: torch.Tensor,
    edge_index: torch.Tensor,
    weight_real: torch.Tensor = None,
    weight_imag: torch.Tensor = None,
    all_destinations: bool = False,
):
    """
    edge_index: (2, num_edges) - Directed edges from source to destination nodes. 

    Returns: new_phase, new_mag, new_activation_strength - all of shape (num_nodes, vector_dim). If any node was absent 
    in the edge_index, phase/mag activation and act strength will be unchange and returned as is.
    """
    source, dest = edge_index[0], edge_index[1]

    
    # For each edge, we have source->dest
    # We need need to calculate weighted superposition of 
    # input vectors. For this we need following:
    source_phase_activations = phase_activations[source]
    source_mag_activations = mag_activations[source]
    source_activation_strengths = activation_strengths[source]

    
    # Step 1: Calculate routing weights
    # a) Calculate the max activation strength for each destination
    # b) Subtract this from each source's activation strength, grouped according to destination then exponentiate
    # c) Calculate the sum of this for each destination - then divide by sum to normalize 
    # - This is basically softmax operation but can't be done directly using torch.softmax 
    # since we have to do on per-destination basis - and number of incoming edges to a destination
    # destination can be different for each destination
    max_act_strength = torch.full_like(activation_strengths, -10.0**9).scatter_reduce_(0, dest, source_activation_strengths, reduce="amax", include_self=False)
    exp_source_act_strength = torch.exp(source_activation_strengths - max_act_strength[dest])
    sum_exp_source_act_strength = torch.zeros_like(activation_strengths).scatter_add_(0, dest, exp_source_act_strength)
    routing_weights = (exp_source_act_strength / (sum_exp_source_act_strength[dest] + _EPSILON)).unsqueeze(-1)


    # Step 2: Perform the complex transformation for each destination
    # (a) Calculate the weighted complex-vector for each source
    # (b) Sum the weighted complex-vectors for each destination
    # (c) Multiply the sum with the destination's phase/mag weight
    # (d) Calculate the new phase/mag act, and activation strength

    
    # Calculate the weighted superposition of the input vectors for each destination
    weighted_source_mag_activation = routing_weights * torch.exp(source_mag_activations)
    source_real = weighted_source_mag_activation * torch.cos(source_phase_activations)
    source_imaginary = weighted_source_mag_activation * torch.sin(source_phase_activations)

    dest_real_input = torch.zeros_like(phase_weights).scatter_add_(0, dest.unsqueeze(-1).expand_as(source_real), source_real)
    dest_imaginary_input = torch.zeros_like(mag_weights).scatter_add_(0, dest.unsqueeze(-1).expand_as(source_imaginary), source_imaginary)

    #Calculate destination's phase/mag weight vectors (use pre-computed if provided)
    if weight_real is None:
        dest_real_weight = mag_weights * torch.cos(phase_weights)
        dest_imaginary_weight = mag_weights * torch.sin(phase_weights)
    else:
        dest_real_weight = weight_real
        dest_imaginary_weight = weight_imag

    # Perform the complex multiplication b/w input and weight vectors
    dest_real_output = dest_real_input*dest_real_weight - dest_imaginary_input * dest_imaginary_weight
    dest_imaginary_output = dest_real_input*dest_imaginary_weight + dest_imaginary_input * dest_real_weight

    # New phase/mag activation and activation strength
    new_phase = torch.atan2(dest_imaginary_output, dest_real_output + _EPSILON)
    new_mag = 0.5 * torch.log(dest_real_output ** 2 + dest_imaginary_output ** 2 + _EPSILON)
    mean_log_mag = new_mag.mean(dim=-1, keepdim=True)
    new_mag = new_mag - mean_log_mag
    # activation_strength = sum(real / geom_mean) — direct from Cartesian outputs
    geom_mean = torch.exp(mean_log_mag)
    new_activation_strength = (dest_real_output / (geom_mean + _EPSILON)).sum(dim=-1)

    if all_destinations:
        return new_phase, new_mag, new_activation_strength

    mask_1d = torch.zeros(phase_activations.shape[0], dtype=torch.bool, device=phase_activations.device)
    mask_1d[dest] = True
    mask_2d = mask_1d.unsqueeze(-1) # For 2D vector tensors

    # Out-of-place combination (safely builds a brand new tensor for the graph)
    phase_activations = torch.where(mask_2d, new_phase, phase_activations)
    mag_activations = torch.where(mask_2d, new_mag, mag_activations)
    activation_strengths = torch.where(mask_1d, new_activation_strength, activation_strengths)
    

    return phase_activations, mag_activations, activation_strengths




#To go back to quantized version, just change the following to activation_strength_forward    
signal_forward = activation_strength_forward_unquantized
activation_strength_forward = activation_strength_forward_unquantized

