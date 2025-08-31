"""
GPU Profiling System for NeuroGraph
Real-time GPU performance monitoring and bottleneck detection
"""

import torch
import time
import psutil
import threading
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
from collections import deque
import numpy as np


class CUDAProfiler:
    """
    CUDA-based GPU profiling with event timing and memory monitoring.
    """
    
    def __init__(self, device: str = 'cuda', buffer_size: int = 1000):
        """
        Initialize CUDA profiler.
        
        Args:
            device: CUDA device to monitor
            buffer_size: Size of timing buffer
        """
        self.device = device
        self.buffer_size = buffer_size
        self.is_cuda_available = torch.cuda.is_available()
        
        if self.is_cuda_available:
            self.device_id = torch.cuda.current_device()
            self.device_name = torch.cuda.get_device_name(self.device_id)
            self.device_properties = torch.cuda.get_device_properties(self.device_id)
        else:
            self.device_id = None
            self.device_name = "CPU"
            self.device_properties = None
        
        # Timing buffers
        self.kernel_times = deque(maxlen=buffer_size)
        self.memory_usage = deque(maxlen=buffer_size)
        self.gpu_utilization = deque(maxlen=buffer_size)
        
        # Event pools for efficient timing
        self.start_events = []
        self.end_events = []
        self.event_pool_size = 100
        self._init_event_pools()
        
        # Active timings
        self.active_timings = {}
        self.timing_stats = {}
        
    def _init_event_pools(self):
        """Initialize CUDA event pools for efficient timing."""
        if not self.is_cuda_available:
            return
        
        for _ in range(self.event_pool_size):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            self.start_events.append(start_event)
            self.end_events.append(end_event)
    
    def start_timing(self, operation_name: str) -> int:
        """
        Start timing a GPU operation.
        
        Args:
            operation_name: Name of the operation being timed
            
        Returns:
            Timing ID for this operation
        """
        if not self.is_cuda_available:
            timing_id = len(self.active_timings)
            self.active_timings[timing_id] = {
                'name': operation_name,
                'start_time': time.perf_counter(),
                'start_event': None,
                'end_event': None
            }
            return timing_id
        
        # Get events from pool
        if self.start_events and self.end_events:
            start_event = self.start_events.pop()
            end_event = self.end_events.pop()
        else:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
        
        # Record start event
        start_event.record()
        
        timing_id = len(self.active_timings)
        self.active_timings[timing_id] = {
            'name': operation_name,
            'start_time': time.perf_counter(),
            'start_event': start_event,
            'end_event': end_event
        }
        
        return timing_id
    
    def end_timing(self, timing_id: int) -> float:
        """
        End timing for a GPU operation.
        
        Args:
            timing_id: ID returned by start_timing
            
        Returns:
            Elapsed time in milliseconds
        """
        if timing_id not in self.active_timings:
            return 0.0
        
        timing_info = self.active_timings[timing_id]
        operation_name = timing_info['name']
        
        if not self.is_cuda_available:
            # CPU timing fallback
            elapsed_ms = (time.perf_counter() - timing_info['start_time']) * 1000
        else:
            # Record end event and synchronize
            end_event = timing_info['end_event']
            end_event.record()
            torch.cuda.synchronize()
            
            # Calculate elapsed time
            elapsed_ms = timing_info['start_event'].elapsed_time(end_event)
            
            # Return events to pool
            self.start_events.append(timing_info['start_event'])
            self.end_events.append(end_event)
        
        # Store timing
        self.kernel_times.append({
            'name': operation_name,
            'time_ms': elapsed_ms,
            'timestamp': datetime.now()
        })
        
        # Update statistics
        if operation_name not in self.timing_stats:
            self.timing_stats[operation_name] = {
                'count': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'avg_time': 0.0
            }
        
        stats = self.timing_stats[operation_name]
        stats['count'] += 1
        stats['total_time'] += elapsed_ms
        stats['min_time'] = min(stats['min_time'], elapsed_ms)
        stats['max_time'] = max(stats['max_time'], elapsed_ms)
        stats['avg_time'] = stats['total_time'] / stats['count']
        
        # Clean up
        del self.active_timings[timing_id]
        
        return elapsed_ms
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dictionary with memory statistics in MB
        """
        if not self.is_cuda_available:
            return {
                'allocated': 0.0,
                'cached': 0.0,
                'max_allocated': 0.0,
                'total': 0.0,
                'free': 0.0,
                'utilization': 0.0
            }
        
        allocated = torch.cuda.memory_allocated(self.device_id) / (1024**2)
        cached = torch.cuda.memory_reserved(self.device_id) / (1024**2)
        max_allocated = torch.cuda.max_memory_allocated(self.device_id) / (1024**2)
        
        # Get total memory from device properties
        total_memory = self.device_properties.total_memory / (1024**2)
        free_memory = total_memory - allocated
        utilization = (allocated / total_memory) * 100
        
        memory_info = {
            'allocated': allocated,
            'cached': cached,
            'max_allocated': max_allocated,
            'total': total_memory,
            'free': free_memory,
            'utilization': utilization
        }
        
        # Store in buffer
        self.memory_usage.append({
            **memory_info,
            'timestamp': datetime.now()
        })
        
        return memory_info
    
    def get_gpu_utilization(self) -> float:
        """
        Get current GPU utilization percentage.
        
        Returns:
            GPU utilization as percentage (0-100)
        """
        if not self.is_cuda_available:
            return 0.0
        
        try:
            # Use nvidia-ml-py if available, otherwise estimate from memory usage
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            pynvml.nvmlShutdown()
        except ImportError:
            # Fallback: estimate from memory utilization
            memory_info = self.get_memory_usage()
            gpu_util = min(memory_info['utilization'], 100.0)
        except Exception:
            gpu_util = 0.0
        
        # Store in buffer
        self.gpu_utilization.append({
            'utilization': gpu_util,
            'timestamp': datetime.now()
        })
        
        return gpu_util
    
    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive timing statistics."""
        return self.timing_stats.copy()
    
    def reset_stats(self):
        """Reset all profiling statistics."""
        self.kernel_times.clear()
        self.memory_usage.clear()
        self.gpu_utilization.clear()
        self.timing_stats.clear()
        
        if self.is_cuda_available:
            torch.cuda.reset_peak_memory_stats(self.device_id)
    
    def get_device_info(self) -> Dict[str, any]:
        """Get GPU device information."""
        if not self.is_cuda_available:
            return {
                'device': 'CPU',
                'name': 'CPU',
                'compute_capability': None,
                'total_memory': psutil.virtual_memory().total / (1024**2),
                'multiprocessor_count': psutil.cpu_count()
            }
        
        device_info = {
            'device': f'cuda:{self.device_id}',
            'name': self.device_name,
            'compute_capability': f"{self.device_properties.major}.{self.device_properties.minor}",
            'total_memory': self.device_properties.total_memory / (1024**2),
            'multiprocessor_count': self.device_properties.multi_processor_count
        }
        
        # Add optional properties if available
        if hasattr(self.device_properties, 'max_threads_per_block'):
            device_info['max_threads_per_block'] = self.device_properties.max_threads_per_block
        if hasattr(self.device_properties, 'max_threads_per_multiprocessor'):
            device_info['max_threads_per_multiprocessor'] = self.device_properties.max_threads_per_multiprocessor
        
        return device_info


class ProfiledOperation:
    """Context manager for profiling GPU operations."""
    
    def __init__(self, profiler: CUDAProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
        self.timing_id = None
    
    def __enter__(self):
        self.timing_id = self.profiler.start_timing(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timing_id is not None:
            self.profiler.end_timing(self.timing_id)


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    """
    
    def __init__(
        self,
        profiler: CUDAProfiler,
        monitoring_interval: float = 1.0,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize performance monitor.
        
        Args:
            profiler: CUDA profiler instance
            monitoring_interval: Monitoring interval in seconds
            alert_thresholds: Alert thresholds for various metrics
        """
        self.profiler = profiler
        self.monitoring_interval = monitoring_interval
        self.alert_thresholds = alert_thresholds or {
            'gpu_utilization_low': 20.0,  # Alert if GPU utilization < 20%
            'memory_utilization_high': 90.0,  # Alert if memory > 90%
            'kernel_time_high': 100.0,  # Alert if kernel time > 100ms
        }
        
        self.is_monitoring = False
        self.monitor_thread = None
        self.alerts = deque(maxlen=100)
        self.performance_history = deque(maxlen=1000)
        
    def start_monitoring(self):
        """Start background performance monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print("🔍 GPU Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("⏹️  GPU Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                memory_info = self.profiler.get_memory_usage()
                gpu_util = self.profiler.get_gpu_utilization()
                timing_stats = self.profiler.get_timing_stats()
                
                # Store performance snapshot
                snapshot = {
                    'timestamp': datetime.now(),
                    'memory_info': memory_info,
                    'gpu_utilization': gpu_util,
                    'timing_stats': timing_stats.copy()
                }
                self.performance_history.append(snapshot)
                
                # Check for alerts
                self._check_alerts(memory_info, gpu_util, timing_stats)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _check_alerts(
        self,
        memory_info: Dict[str, float],
        gpu_util: float,
        timing_stats: Dict[str, Dict[str, float]]
    ):
        """Check for performance alerts."""
        current_time = datetime.now()
        
        # Low GPU utilization alert
        if gpu_util < self.alert_thresholds['gpu_utilization_low']:
            self.alerts.append({
                'timestamp': current_time,
                'type': 'low_gpu_utilization',
                'message': f"Low GPU utilization: {gpu_util:.1f}%",
                'severity': 'warning'
            })
        
        # High memory utilization alert
        if memory_info['utilization'] > self.alert_thresholds['memory_utilization_high']:
            self.alerts.append({
                'timestamp': current_time,
                'type': 'high_memory_utilization',
                'message': f"High memory utilization: {memory_info['utilization']:.1f}%",
                'severity': 'critical'
            })
        
        # High kernel time alert
        for op_name, stats in timing_stats.items():
            if stats['avg_time'] > self.alert_thresholds['kernel_time_high']:
                self.alerts.append({
                    'timestamp': current_time,
                    'type': 'high_kernel_time',
                    'message': f"High kernel time for {op_name}: {stats['avg_time']:.1f}ms",
                    'severity': 'warning'
                })
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """Get recent performance alerts."""
        return list(self.alerts)[-count:]
    
    def get_performance_summary(self) -> Dict[str, any]:
        """Get comprehensive performance summary."""
        if not self.performance_history:
            return {}
        
        recent_snapshots = list(self.performance_history)[-10:]  # Last 10 snapshots
        
        # Calculate averages
        avg_gpu_util = np.mean([s['gpu_utilization'] for s in recent_snapshots])
        avg_memory_util = np.mean([s['memory_info']['utilization'] for s in recent_snapshots])
        
        # Get timing statistics
        timing_stats = self.profiler.get_timing_stats()
        
        return {
            'device_info': self.profiler.get_device_info(),
            'avg_gpu_utilization': avg_gpu_util,
            'avg_memory_utilization': avg_memory_util,
            'current_memory_info': self.profiler.get_memory_usage(),
            'timing_stats': timing_stats,
            'recent_alerts': self.get_recent_alerts(),
            'monitoring_active': self.is_monitoring
        }
    
    def print_performance_report(self):
        """Print detailed performance report."""
        summary = self.get_performance_summary()
        
        if not summary:
            print("📊 No performance data available")
            return
        
        print("\n🔍 GPU Performance Report")
        print("=" * 60)
        
        # Device info
        device_info = summary['device_info']
        print(f"Device: {device_info['name']} ({device_info['device']})")
        print(f"Memory: {device_info['total_memory']:.0f} MB")
        if 'compute_capability' in device_info and device_info['compute_capability']:
            print(f"Compute Capability: {device_info['compute_capability']}")
        
        # Performance metrics
        print(f"\nPerformance Metrics:")
        print(f"  GPU Utilization: {summary['avg_gpu_utilization']:.1f}%")
        print(f"  Memory Utilization: {summary['avg_memory_utilization']:.1f}%")
        
        # Memory details
        memory_info = summary['current_memory_info']
        print(f"  Memory Allocated: {memory_info['allocated']:.1f} MB")
        print(f"  Memory Cached: {memory_info['cached']:.1f} MB")
        print(f"  Memory Free: {memory_info['free']:.1f} MB")
        
        # Timing statistics
        timing_stats = summary['timing_stats']
        if timing_stats:
            print(f"\nKernel Timing Statistics:")
            for op_name, stats in timing_stats.items():
                print(f"  {op_name}:")
                print(f"    Count: {stats['count']:,}")
                print(f"    Avg Time: {stats['avg_time']:.2f}ms")
                print(f"    Min/Max: {stats['min_time']:.2f}ms / {stats['max_time']:.2f}ms")
        
        # Recent alerts
        alerts = summary['recent_alerts']
        if alerts:
            print(f"\nRecent Alerts:")
            for alert in alerts[-5:]:  # Show last 5 alerts
                severity_icon = "🔴" if alert['severity'] == 'critical' else "🟡"
                print(f"  {severity_icon} {alert['message']}")


# Factory functions
def create_cuda_profiler(device: str = 'cuda', buffer_size: int = 1000) -> CUDAProfiler:
    """Create a CUDA profiler instance."""
    return CUDAProfiler(device, buffer_size)


def create_performance_monitor(
    profiler: CUDAProfiler,
    monitoring_interval: float = 1.0,
    alert_thresholds: Optional[Dict[str, float]] = None
) -> PerformanceMonitor:
    """Create a performance monitor instance."""
    return PerformanceMonitor(profiler, monitoring_interval, alert_thresholds)


# Convenience function for profiling operations
def profile_operation(profiler: CUDAProfiler, operation_name: str):
    """Context manager for profiling operations."""
    return ProfiledOperation(profiler, operation_name)
