"""
Performance Monitoring Utilities for NeuroGraph
Tracks memory usage, timing, and system performance metrics
"""

import torch
import time
import psutil
import gc
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass
from utils.device_manager import get_device_manager
import threading
import json
from datetime import datetime

@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot."""
    timestamp: float
    epoch: int
    sample: int
    memory_usage_gb: float
    gpu_utilization: float
    cpu_percent: float
    active_nodes: int
    forward_time_ms: float
    backward_time_ms: float
    cache_hit_rate: float

class PerformanceMonitor:
    """
    Comprehensive performance monitoring for NeuroGraph training.
    
    Features:
    - Real-time memory tracking
    - GPU/CPU utilization monitoring
    - Training timing analysis
    - Cache performance tracking
    - Automatic bottleneck detection
    """
    
    def __init__(self, monitoring_interval: float = 1.0, history_size: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            monitoring_interval: Seconds between automatic measurements
            history_size: Maximum number of snapshots to keep
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        # Device management
        self.device_manager = get_device_manager()
        
        # Performance history
        self.snapshots: deque = deque(maxlen=history_size)
        self.timing_data = defaultdict(list)
        
        # Current state tracking
        self.current_epoch = 0
        self.current_sample = 0
        self.epoch_start_time = 0
        self.sample_start_time = 0
        
        # Memory tracking
        self.peak_memory_usage = 0
        self.memory_warnings = []
        self.memory_pressure_threshold = 0.85
        
        # Timing contexts
        self.active_timers = {}
        
        # Background monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Statistics
        self.stats = {
            'total_samples_processed': 0,
            'total_epochs_completed': 0,
            'average_sample_time': 0.0,
            'average_epoch_time': 0.0,
            'memory_pressure_events': 0,
            'performance_warnings': []
        }
        
        print(f"🔧 Performance Monitor initialized:")
        print(f"   📊 Monitoring interval: {monitoring_interval}s")
        print(f"   💾 History size: {history_size} snapshots")
        print(f"   🎯 Device: {self.device_manager.device}")
    
    def start_monitoring(self):
        """Start background performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print("🚀 Background performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("⏹️  Background performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                self._take_snapshot()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                time.sleep(self.monitoring_interval * 2)  # Back off on error
    
    def start_epoch(self, epoch: int):
        """Mark the start of a training epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.current_sample = 0
        
        # Memory cleanup at epoch start
        if epoch > 0 and epoch % 5 == 0:  # Every 5 epochs
            self.device_manager.cleanup_memory(aggressive=True)
    
    def end_epoch(self):
        """Mark the end of a training epoch."""
        if self.epoch_start_time > 0:
            epoch_time = time.time() - self.epoch_start_time
            self.timing_data['epoch_times'].append(epoch_time)
            
            # Update average epoch time
            alpha = 0.1
            self.stats['average_epoch_time'] = (
                (1 - alpha) * self.stats['average_epoch_time'] + 
                alpha * epoch_time
            )
            
            self.stats['total_epochs_completed'] += 1
    
    def start_sample(self, sample: int):
        """Mark the start of processing a sample."""
        self.current_sample = sample
        self.sample_start_time = time.time()
    
    def end_sample(self):
        """Mark the end of processing a sample."""
        if self.sample_start_time > 0:
            sample_time = time.time() - self.sample_start_time
            self.timing_data['sample_times'].append(sample_time)
            
            # Update average sample time
            alpha = 0.1
            self.stats['average_sample_time'] = (
                (1 - alpha) * self.stats['average_sample_time'] + 
                alpha * sample_time
            )
            
            self.stats['total_samples_processed'] += 1
    
    def start_timer(self, name: str):
        """Start a named timer."""
        self.active_timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a named timer and return elapsed time."""
        if name in self.active_timers:
            elapsed = time.time() - self.active_timers[name]
            self.timing_data[f'{name}_times'].append(elapsed)
            del self.active_timers[name]
            return elapsed
        return 0.0
    
    def _take_snapshot(self):
        """Take a performance snapshot."""
        try:
            # Memory usage
            memory_stats = self.device_manager.get_memory_usage()
            memory_gb = memory_stats['allocated_gb']
            
            # Update peak memory
            self.peak_memory_usage = max(self.peak_memory_usage, memory_gb)
            
            # Check for memory pressure
            if memory_stats['utilization'] > self.memory_pressure_threshold:
                self.stats['memory_pressure_events'] += 1
                warning = f"Memory pressure at epoch {self.current_epoch}, sample {self.current_sample}"
                if warning not in self.memory_warnings:
                    self.memory_warnings.append(warning)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # GPU utilization (if available)
            gpu_util = 0.0
            if self.device_manager.device.type == 'cuda':
                try:
                    # This would require nvidia-ml-py, simplified for now
                    gpu_util = memory_stats['utilization'] * 100
                except:
                    gpu_util = 0.0
            
            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=time.time(),
                epoch=self.current_epoch,
                sample=self.current_sample,
                memory_usage_gb=memory_gb,
                gpu_utilization=gpu_util,
                cpu_percent=cpu_percent,
                active_nodes=0,  # Would need activation table reference
                forward_time_ms=0.0,  # Would be set by timing contexts
                backward_time_ms=0.0,
                cache_hit_rate=0.0  # Would need cache reference
            )
            
            self.snapshots.append(snapshot)
            
        except Exception as e:
            print(f"⚠️  Error taking performance snapshot: {e}")
    
    def get_memory_trend(self, window_size: int = 50) -> Dict[str, float]:
        """Get memory usage trend over recent snapshots."""
        if len(self.snapshots) < 2:
            return {'trend': 0.0, 'current': 0.0, 'peak': self.peak_memory_usage}
        
        recent_snapshots = list(self.snapshots)[-window_size:]
        memory_values = [s.memory_usage_gb for s in recent_snapshots]
        
        if len(memory_values) < 2:
            return {'trend': 0.0, 'current': memory_values[0], 'peak': self.peak_memory_usage}
        
        # Simple linear trend
        x = list(range(len(memory_values)))
        n = len(memory_values)
        sum_x = sum(x)
        sum_y = sum(memory_values)
        sum_xy = sum(x[i] * memory_values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        return {
            'trend': trend,
            'current': memory_values[-1],
            'peak': self.peak_memory_usage,
            'average': sum(memory_values) / len(memory_values)
        }
    
    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks."""
        bottlenecks = []
        
        # Memory bottleneck
        memory_trend = self.get_memory_trend()
        if memory_trend['trend'] > 0.01:  # Growing by >10MB per snapshot
            bottlenecks.append({
                'type': 'memory_leak',
                'severity': 'high',
                'description': f"Memory usage growing at {memory_trend['trend']:.3f} GB/snapshot",
                'suggestion': "Check for memory leaks in activation table or caches"
            })
        
        # Slow sample processing
        if self.stats['average_sample_time'] > 5.0:  # >5 seconds per sample
            bottlenecks.append({
                'type': 'slow_processing',
                'severity': 'medium',
                'description': f"Average sample time: {self.stats['average_sample_time']:.2f}s",
                'suggestion': "Consider reducing batch size or optimizing forward pass"
            })
        
        # Memory pressure
        if self.stats['memory_pressure_events'] > 10:
            bottlenecks.append({
                'type': 'memory_pressure',
                'severity': 'high',
                'description': f"{self.stats['memory_pressure_events']} memory pressure events",
                'suggestion': "Increase memory cleanup frequency or reduce model size"
            })
        
        # CPU bottleneck (if recent snapshots show high CPU usage)
        if len(self.snapshots) > 10:
            recent_cpu = [s.cpu_percent for s in list(self.snapshots)[-10:]]
            avg_cpu = sum(recent_cpu) / len(recent_cpu)
            if avg_cpu > 90:
                bottlenecks.append({
                    'type': 'cpu_bottleneck',
                    'severity': 'medium',
                    'description': f"High CPU usage: {avg_cpu:.1f}%",
                    'suggestion': "Consider GPU acceleration or reduce computational complexity"
                })
        
        return bottlenecks
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        memory_trend = self.get_memory_trend()
        bottlenecks = self.detect_bottlenecks()
        
        # Timing statistics
        timing_stats = {}
        for timer_name, times in self.timing_data.items():
            if times:
                timing_stats[timer_name] = {
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }
        
        return {
            'current_state': {
                'epoch': self.current_epoch,
                'sample': self.current_sample,
                'memory_gb': memory_trend['current'],
                'peak_memory_gb': self.peak_memory_usage
            },
            'statistics': self.stats.copy(),
            'memory_trend': memory_trend,
            'timing_stats': timing_stats,
            'bottlenecks': bottlenecks,
            'device_info': self.device_manager.device_info,
            'total_snapshots': len(self.snapshots)
        }
    
    def print_performance_report(self):
        """Print detailed performance report."""
        summary = self.get_performance_summary()
        
        print(f"\n📊 NeuroGraph Performance Report")
        print(f"=" * 50)
        
        # Current state
        current = summary['current_state']
        print(f"Current State:")
        print(f"  Epoch: {current['epoch']}")
        print(f"  Sample: {current['sample']}")
        print(f"  Memory: {current['memory_gb']:.2f} GB")
        print(f"  Peak Memory: {current['peak_memory_gb']:.2f} GB")
        
        # Statistics
        stats = summary['statistics']
        print(f"\nStatistics:")
        print(f"  Samples Processed: {stats['total_samples_processed']:,}")
        print(f"  Epochs Completed: {stats['total_epochs_completed']}")
        print(f"  Avg Sample Time: {stats['average_sample_time']:.3f}s")
        print(f"  Avg Epoch Time: {stats['average_epoch_time']:.1f}s")
        print(f"  Memory Pressure Events: {stats['memory_pressure_events']}")
        
        # Memory trend
        trend = summary['memory_trend']
        print(f"\nMemory Trend:")
        print(f"  Current: {trend['current']:.2f} GB")
        print(f"  Trend: {trend['trend']:+.4f} GB/snapshot")
        print(f"  Average: {trend['average']:.2f} GB")
        
        # Timing breakdown
        timing = summary['timing_stats']
        if timing:
            print(f"\nTiming Breakdown:")
            for name, stats in timing.items():
                print(f"  {name}: {stats['average']:.3f}s avg ({stats['count']} samples)")
        
        # Bottlenecks
        bottlenecks = summary['bottlenecks']
        if bottlenecks:
            print(f"\n⚠️  Performance Bottlenecks:")
            for bottleneck in bottlenecks:
                severity_icon = "🔴" if bottleneck['severity'] == 'high' else "🟡"
                print(f"  {severity_icon} {bottleneck['type'].title()}: {bottleneck['description']}")
                print(f"     💡 {bottleneck['suggestion']}")
        else:
            print(f"\n✅ No performance bottlenecks detected")
        
        # Device info
        device_info = summary['device_info']
        print(f"\nDevice Information:")
        print(f"  Name: {device_info['name']}")
        print(f"  Memory: {device_info['memory_gb']:.1f} GB")
        if 'compute_capability' in device_info:
            print(f"  Compute Capability: {device_info['compute_capability']}")
    
    def save_performance_log(self, filepath: str):
        """Save performance data to JSON file."""
        summary = self.get_performance_summary()
        
        # Add snapshot data
        summary['snapshots'] = [
            {
                'timestamp': s.timestamp,
                'epoch': s.epoch,
                'sample': s.sample,
                'memory_usage_gb': s.memory_usage_gb,
                'gpu_utilization': s.gpu_utilization,
                'cpu_percent': s.cpu_percent,
                'active_nodes': s.active_nodes
            }
            for s in self.snapshots
        ]
        
        # Add metadata
        summary['metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'monitoring_interval': self.monitoring_interval,
            'history_size': self.history_size
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Performance log saved to {filepath}")
    
    def cleanup(self):
        """Clean up monitor resources."""
        self.stop_monitoring()
        self.snapshots.clear()
        self.timing_data.clear()
        self.active_timers.clear()

# Global performance monitor instance
_global_performance_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor(monitoring_interval: float = 1.0, 
                          history_size: int = 1000) -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _global_performance_monitor
    
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor(monitoring_interval, history_size)
    
    return _global_performance_monitor

def start_monitoring():
    """Start global performance monitoring."""
    get_performance_monitor().start_monitoring()

def stop_monitoring():
    """Stop global performance monitoring."""
    if _global_performance_monitor:
        _global_performance_monitor.stop_monitoring()

def print_performance_report():
    """Print global performance report."""
    if _global_performance_monitor:
        _global_performance_monitor.print_performance_report()

class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, name: str, monitor: Optional[PerformanceMonitor] = None):
        self.name = name
        self.monitor = monitor or get_performance_monitor()
        self.start_time = 0
    
    def __enter__(self):
        self.start_time = time.time()
        self.monitor.start_timer(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = self.monitor.end_timer(self.name)
        return False  # Don't suppress exceptions

def timing_context(name: str):
    """Create a timing context manager."""
    return TimingContext(name)
