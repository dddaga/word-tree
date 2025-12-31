import torch

def activation_strength_forward(phases, mags, gamma=1.0):
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
    # Maps magnitude to exponential weighting (as in original lookup table)
    mag_values = torch.exp(gamma * torch.sin(mags))
    
    # Activation strength is the dot product
    signal = (phase_values * mag_values).sum(dim=-1)
    return signal

#backward compatibility
signal_forward = activation_strength_forward

