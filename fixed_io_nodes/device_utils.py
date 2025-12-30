"""Automatically detect and select the best available device."""
import torch

def get_best_device():
    """
    Detect and return the best available device.
    Priority: CUDA > MPS > CPU
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🚀 Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("🚀 Using MPS (Apple Silicon GPU)")
    else:
        device = 'cpu'
        print("💻 Using CPU")
    
    return device


def get_device_info():
    """Get detailed information about available devices."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'cpu_count': torch.get_num_threads()
    }
    
    if info['cuda_available']:
        info['cuda_device'] = torch.cuda.get_device_name(0)
        info['cuda_count'] = torch.cuda.device_count()
    
    return info


if __name__ == "__main__":
    print("Device Detection:")
    print(f"  Best device: {get_best_device()}")
    print("\nDetailed info:")
    for key, value in get_device_info().items():
        print(f"  {key}: {value}")

