"""
Device Management Utilities for NeuroGraph
Handles GPU/CPU device detection and tensor management
"""

import torch
import gc
from typing import Optional, Dict, Any
import psutil
import os

class DeviceManager:
    """
    Centralized device management for NeuroGraph components.
    
    Features:
    - Automatic GPU detection and fallback
    - Memory monitoring and cleanup
    - Device-aware tensor operations
    - Performance optimization hints
    """
    
    def __init__(self, preferred_device: Optional[str] = None, memory_fraction: float = 0.8):
        """
        Initialize device manager.
        
        Args:
            preferred_device: Preferred device ('cuda', 'cpu', or None for auto)
            memory_fraction: Fraction of GPU memory to use (0.1-0.9)
        """
        self.memory_fraction = memory_fraction
        self.device = self._detect_optimal_device(preferred_device)
        self.device_info = self._get_device_info()
        
        # Memory tracking
        self.peak_memory_usage = 0
        self.memory_warnings = []
        
        print(f"🔧 Device Manager Initialized:")
        print(f"   🎯 Selected device: {self.device}")
        print(f"   💾 Available memory: {self.device_info['memory_gb']:.1f} GB")
        print(f"   📊 Memory fraction: {memory_fraction:.1%}")
        
        if self.device.type == 'cuda':
            self._configure_gpu_settings()
    
    def _detect_optimal_device(self, preferred: Optional[str]) -> torch.device:
        """Detect the optimal device for computation."""
        if preferred == 'cpu':
            return torch.device('cpu')
        
        if torch.cuda.is_available():
            # Check GPU memory and capabilities
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            if gpu_memory_gb >= 2.0:  # Minimum 2GB for NeuroGraph
                return torch.device('cuda:0')
            else:
                print(f"⚠️  GPU has insufficient memory ({gpu_memory_gb:.1f} GB < 2.0 GB)")
                return torch.device('cpu')
        else:
            if preferred == 'cuda':
                print("⚠️  CUDA requested but not available, falling back to CPU")
            return torch.device('cpu')
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information."""
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            device_info = {
                'name': props.name,
                'memory_gb': props.total_memory / (1024**3),
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessors': props.multi_processor_count,
            }
            
            # Handle different PyTorch versions
            try:
                device_info['max_threads_per_block'] = props.max_threads_per_block
            except AttributeError:
                device_info['max_threads_per_block'] = getattr(props, 'maxThreadsPerBlock', 'N/A')
            
            return device_info
        else:
            return {
                'name': 'CPU',
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'cores': os.cpu_count(),
                'compute_capability': 'N/A',
                'multiprocessors': os.cpu_count(),
                'max_threads_per_block': 'N/A'
            }
    
    def _configure_gpu_settings(self):
        """Configure optimal GPU settings."""
        if self.device.type == 'cuda':
            # Set memory fraction
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_limit = int(total_memory * self.memory_fraction)
            
            # Enable memory growth (if supported)
            try:
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
                print(f"   ✅ GPU memory limit set to {memory_limit / (1024**3):.1f} GB")
            except:
                print(f"   ⚠️  Could not set memory fraction, using default")
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            print(f"   ✅ CUDNN optimizations enabled")
    
    def to_device(self, tensor: torch.Tensor, non_blocking: bool = True) -> torch.Tensor:
        """Move tensor to managed device."""
        if tensor.device != self.device:
            return tensor.to(self.device, non_blocking=non_blocking)
        return tensor
    
    def create_tensor(self, *args, **kwargs) -> torch.Tensor:
        """Create tensor on managed device."""
        kwargs['device'] = self.device
        return torch.tensor(*args, **kwargs)
    
    def zeros(self, *args, **kwargs) -> torch.Tensor:
        """Create zeros tensor on managed device."""
        kwargs['device'] = self.device
        return torch.zeros(*args, **kwargs)
    
    def ones(self, *args, **kwargs) -> torch.Tensor:
        """Create ones tensor on managed device."""
        kwargs['device'] = self.device
        return torch.ones(*args, **kwargs)
    
    def empty(self, *args, **kwargs) -> torch.Tensor:
        """Create empty tensor on managed device."""
        kwargs['device'] = self.device
        return torch.empty(*args, **kwargs)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            
            self.peak_memory_usage = max(self.peak_memory_usage, allocated)
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'peak_usage_gb': self.peak_memory_usage,
                'utilization': allocated / self.device_info['memory_gb']
            }
        else:
            # CPU memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            allocated = memory_info.rss / (1024**3)
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': allocated,
                'max_allocated_gb': allocated,
                'peak_usage_gb': max(self.peak_memory_usage, allocated),
                'utilization': allocated / self.device_info['memory_gb']
            }
    
    def cleanup_memory(self, aggressive: bool = False):
        """Clean up device memory."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        
        # Python garbage collection
        if aggressive:
            gc.collect()
    
    def check_memory_pressure(self, threshold: float = 0.85) -> bool:
        """Check if memory usage is above threshold."""
        usage = self.get_memory_usage()
        if usage['utilization'] > threshold:
            warning = f"High memory usage: {usage['utilization']:.1%} > {threshold:.1%}"
            if warning not in self.memory_warnings:
                self.memory_warnings.append(warning)
                print(f"⚠️  {warning}")
            return True
        return False
    
    def optimize_for_inference(self):
        """Optimize device settings for inference."""
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = False  # Disable for consistent timing
            torch.backends.cudnn.deterministic = True
    
    def optimize_for_training(self):
        """Optimize device settings for training."""
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True   # Enable for speed
            torch.backends.cudnn.deterministic = False
    
    def get_performance_hints(self) -> Dict[str, str]:
        """Get performance optimization hints."""
        hints = {}
        
        if self.device.type == 'cuda':
            memory_usage = self.get_memory_usage()
            
            if memory_usage['utilization'] > 0.9:
                hints['memory'] = "Consider reducing batch size or model size"
            elif memory_usage['utilization'] < 0.3:
                hints['memory'] = "GPU memory underutilized, consider increasing batch size"
            
            if self.device_info['compute_capability'] < '6.0':
                hints['compute'] = "Older GPU detected, consider mixed precision training"
            
            if not torch.backends.cudnn.benchmark:
                hints['cudnn'] = "Enable CUDNN benchmark for better performance"
        else:
            hints['device'] = "Consider using GPU for significant speedup"
            
            cpu_count = os.cpu_count()
            if cpu_count > 8:
                hints['threading'] = f"High core count ({cpu_count}), ensure proper threading"
        
        return hints
    
    def print_status(self):
        """Print detailed device status."""
        print(f"\n🔧 Device Manager Status")
        print(f"=" * 50)
        print(f"Device: {self.device}")
        print(f"Name: {self.device_info['name']}")
        print(f"Memory: {self.device_info['memory_gb']:.1f} GB")
        
        if self.device.type == 'cuda':
            print(f"Compute Capability: {self.device_info['compute_capability']}")
            print(f"Multiprocessors: {self.device_info['multiprocessors']}")
        
        # Memory usage
        memory = self.get_memory_usage()
        print(f"\nMemory Usage:")
        print(f"  Allocated: {memory['allocated_gb']:.2f} GB")
        print(f"  Peak: {memory['peak_usage_gb']:.2f} GB")
        print(f"  Utilization: {memory['utilization']:.1%}")
        
        # Performance hints
        hints = self.get_performance_hints()
        if hints:
            print(f"\nPerformance Hints:")
            for category, hint in hints.items():
                print(f"  {category.title()}: {hint}")
        
        # Warnings
        if self.memory_warnings:
            print(f"\nWarnings:")
            for warning in self.memory_warnings[-3:]:  # Show last 3 warnings
                print(f"  ⚠️  {warning}")

# Global device manager instance
_global_device_manager: Optional[DeviceManager] = None

def get_device_manager(preferred_device: Optional[str] = None, 
                      memory_fraction: float = 0.8) -> DeviceManager:
    """Get or create global device manager."""
    global _global_device_manager
    
    if _global_device_manager is None:
        _global_device_manager = DeviceManager(preferred_device, memory_fraction)
    
    return _global_device_manager

def get_device() -> torch.device:
    """Get the current managed device."""
    return get_device_manager().device

def to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to managed device."""
    return get_device_manager().to_device(tensor)

def cleanup_memory(aggressive: bool = False):
    """Clean up device memory."""
    get_device_manager().cleanup_memory(aggressive)

def check_memory_pressure(threshold: float = 0.85) -> bool:
    """Check if memory usage is above threshold."""
    return get_device_manager().check_memory_pressure(threshold)
