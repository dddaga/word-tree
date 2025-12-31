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
    # Clamp the exponent to prevent numerical overflow
    mag_exponent = gamma * torch.sin(mags)
    mag_exponent = torch.clamp(mag_exponent, min=-10.0, max=10.0)  # Prevent exp overflow
    mag_values = torch.exp(mag_exponent)
    
    # Activation strength is the dot product
    signal = (phase_values * mag_values).sum(dim=-1)
    
    # Add small epsilon to prevent exactly zero signals
    signal = signal + 1e-8
    
    return signal

#backward compatibility
signal_forward = activation_strength_forward

