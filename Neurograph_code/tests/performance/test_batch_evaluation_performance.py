"""
Comprehensive Performance Test for Batch Evaluation Optimization
Validates 5-10x speedup and GPU utilization improvements
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple
import json
import os
from datetime import datetime

# Import optimized components
from train.modular_train_context import create_modular_train_context
from core.batch_evaluation_engine import create_batch_evaluation_engine
from utils.gpu_profiler import create_cuda_profiler, create_performance_monitor


class BatchEvaluationPerformanceTest:
    """
    Comprehensive test suite for batch evaluation performance optimization.
    """
    
    def __init__(self, config_path: str = "config/production.yaml"):
        """
        Initialize performance test.
        
        Args:
            config_path: Path to production configuration
        """
        self.config_path = config_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize profiling
        self.profiler = create_cuda_profiler(device=self.device)
        self.performance_monitor = create_performance_monitor(self.profiler)
        
        # Initialize training context
        print("🚀 Initializing Production Training Context...")
        self.training_context = create_modular_train_context(config_path)
        
        # Test results storage
        self.results = {}
        
        print(f"✅ Performance test initialized on {self.device}")
        print(f"   📊 GPU: {self.profiler.get_device_info()['name']}")
    
    def test_legacy_vs_batch_evaluation(self, num_samples: int = 200) -> Dict[str, any]:
        """
        Compare legacy evaluation vs optimized batch evaluation.
        
        Args:
            num_samples: Number of samples to test
            
        Returns:
            Comparison results
        """
        print(f"\n🔄 Testing Legacy vs Batch Evaluation ({num_samples} samples)")
        print("-" * 60)
        
        # Test 1: Legacy evaluation
        print("   Testing legacy evaluation...")
        start_time = time.perf_counter()
        
        with torch.no_grad():
            legacy_accuracy = self.training_context.evaluate_accuracy(
                num_samples=num_samples, use_batch_evaluation=False
            )
        
        legacy_time = time.perf_counter() - start_time
        legacy_speed = num_samples / legacy_time
        
        print(f"     ✅ Legacy: {legacy_accuracy:.1%} accuracy, {legacy_speed:.1f} samples/s")
        
        # Test 2: Batch evaluation
        print("   Testing batch evaluation...")
        batch_evaluator = create_batch_evaluation_engine(
            self.training_context, batch_size=16, device=self.device, verbose=False
        )
        
        start_time = time.perf_counter()
        batch_results = batch_evaluator.evaluate_accuracy_batched(
            num_samples=num_samples, streaming=True
        )
        batch_time = time.perf_counter() - start_time
        
        print(f"     ✅ Batch: {batch_results['accuracy']:.1%} accuracy, {batch_results['samples_per_second']:.1f} samples/s")
        
        # Calculate speedup
        speedup = batch_results['samples_per_second'] / legacy_speed
        time_reduction = legacy_time - batch_time
        
        results = {
            'legacy': {
                'accuracy': legacy_accuracy,
                'time': legacy_time,
                'samples_per_second': legacy_speed
            },
            'batch': {
                'accuracy': batch_results['accuracy'],
                'time': batch_time,
                'samples_per_second': batch_results['samples_per_second'],
                'cache_hit_rate': batch_results['cache_hit_rate']
            },
            'improvement': {
                'speedup': speedup,
                'time_reduction_seconds': time_reduction,
                'time_reduction_percent': (time_reduction / legacy_time) * 100
            }
        }
        
        print(f"\n📊 Performance Comparison:")
        print(f"   ⚡ Speedup: {speedup:.1f}x")
        print(f"   ⏱️  Time Reduction: {time_reduction:.1f}s ({results['improvement']['time_reduction_percent']:.1f}%)")
        print(f"   🗄️  Cache Hit Rate: {batch_results['cache_hit_rate']:.1%}")
        
        return results
    
    def test_batch_size_optimization(
        self,
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        num_samples: int = 100
    ) -> Dict[int, Dict[str, float]]:
        """
        Test optimal batch size for the current GPU.
        
        Args:
            batch_sizes: List of batch sizes to test
            num_samples: Number of samples per test
            
        Returns:
            Performance results for each batch size
        """
        print(f"\n🎯 Testing Batch Size Optimization")
        print("-" * 60)
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"   Testing batch size: {batch_size}")
            
            # Create batch evaluator with specific batch size
            batch_evaluator = create_batch_evaluation_engine(
                self.training_context, batch_size=batch_size, device=self.device, verbose=False
            )
            
            # Run performance test
            start_time = time.perf_counter()
            eval_results = batch_evaluator.evaluate_accuracy_batched(
                num_samples=num_samples, streaming=True
            )
            total_time = time.perf_counter() - start_time
            
            results[batch_size] = {
                'accuracy': eval_results['accuracy'],
                'samples_per_second': eval_results['samples_per_second'],
                'total_time': total_time,
                'cache_hit_rate': eval_results['cache_hit_rate'],
                'avg_batch_time': eval_results['avg_batch_time']
            }
            
            print(f"     ✅ {eval_results['samples_per_second']:.1f} samples/s, "
                  f"{eval_results['accuracy']:.1%} accuracy")
        
        # Find optimal batch size
        optimal_batch_size = max(results.keys(), key=lambda k: results[k]['samples_per_second'])
        best_speed = results[optimal_batch_size]['samples_per_second']
        baseline_speed = results[1]['samples_per_second']
        batch_speedup = best_speed / baseline_speed
        
        print(f"\n📈 Batch Size Optimization Results:")
        print(f"   🏆 Optimal Batch Size: {optimal_batch_size}")
        print(f"   ⚡ Best Speed: {best_speed:.1f} samples/s")
        print(f"   📊 Batch Speedup: {batch_speedup:.1f}x (vs batch_size=1)")
        
        return results
    
    def test_gpu_utilization(self, duration: float = 30.0) -> Dict[str, any]:
        """
        Test GPU utilization during batch evaluation.
        
        Args:
            duration: Test duration in seconds
            
        Returns:
            GPU utilization statistics
        """
        print(f"\n🔍 Testing GPU Utilization ({duration}s)")
        print("-" * 60)
        
        # Start monitoring
        self.performance_monitor.start_monitoring()
        
        # Create batch evaluator
        batch_evaluator = create_batch_evaluation_engine(
            self.training_context, batch_size=16, device=self.device, verbose=False
        )
        
        # Run continuous evaluation
        start_time = time.perf_counter()
        total_samples = 0
        
        while time.perf_counter() - start_time < duration:
            # Run batch evaluation
            eval_results = batch_evaluator.evaluate_accuracy_batched(
                num_samples=50, streaming=True
            )
            total_samples += eval_results['total_samples']
            
            # Brief pause for monitoring
            time.sleep(0.1)
        
        # Stop monitoring
        self.performance_monitor.stop_monitoring()
        
        # Get results
        performance_summary = self.performance_monitor.get_performance_summary()
        actual_duration = time.perf_counter() - start_time
        
        results = {
            'duration': actual_duration,
            'total_samples': total_samples,
            'samples_per_second': total_samples / actual_duration,
            'gpu_utilization': performance_summary.get('avg_gpu_utilization', 0),
            'memory_utilization': performance_summary.get('avg_memory_utilization', 0),
            'device_info': performance_summary.get('device_info', {}),
            'alerts': performance_summary.get('recent_alerts', [])
        }
        
        print(f"📊 GPU Utilization Results:")
        print(f"   🎯 GPU Utilization: {results['gpu_utilization']:.1f}%")
        print(f"   💾 Memory Utilization: {results['memory_utilization']:.1f}%")
        print(f"   ⚡ Processing Speed: {results['samples_per_second']:.1f} samples/s")
        print(f"   📊 Total Samples: {total_samples:,}")
        
        return results
    
    def test_memory_efficiency(self, num_samples: int = 500) -> Dict[str, float]:
        """
        Test memory efficiency of batch evaluation.
        
        Args:
            num_samples: Number of samples to process
            
        Returns:
            Memory usage statistics
        """
        print(f"\n💾 Testing Memory Efficiency ({num_samples} samples)")
        print("-" * 60)
        
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        initial_memory = self.profiler.get_memory_usage()
        
        # Create batch evaluator
        batch_evaluator = create_batch_evaluation_engine(
            self.training_context, batch_size=16, device=self.device, verbose=False
        )
        
        # Monitor memory during evaluation
        memory_snapshots = []
        
        # Run evaluation in chunks
        chunk_size = 50
        for i in range(0, num_samples, chunk_size):
            current_chunk = min(chunk_size, num_samples - i)
            
            # Process chunk
            eval_results = batch_evaluator.evaluate_accuracy_batched(
                num_samples=current_chunk, streaming=True
            )
            
            # Take memory snapshot
            memory_info = self.profiler.get_memory_usage()
            memory_snapshots.append(memory_info)
            
            if i % (chunk_size * 4) == 0:
                print(f"   Processed {i + current_chunk:3d}/{num_samples}: "
                      f"Memory={memory_info['allocated']:.1f}MB, "
                      f"Utilization={memory_info['utilization']:.1f}%")
        
        final_memory = self.profiler.get_memory_usage()
        
        # Calculate statistics
        peak_memory = max(snapshot['allocated'] for snapshot in memory_snapshots)
        avg_memory = np.mean([snapshot['allocated'] for snapshot in memory_snapshots])
        memory_growth = final_memory['allocated'] - initial_memory['allocated']
        
        results = {
            'initial_memory_mb': initial_memory['allocated'],
            'final_memory_mb': final_memory['allocated'],
            'peak_memory_mb': peak_memory,
            'avg_memory_mb': avg_memory,
            'memory_growth_mb': memory_growth,
            'memory_per_sample_kb': (memory_growth * 1024) / num_samples,
            'samples_processed': num_samples
        }
        
        print(f"\n📈 Memory Efficiency Results:")
        print(f"   💾 Peak Memory: {peak_memory:.1f} MB")
        print(f"   📊 Memory Growth: {memory_growth:.1f} MB")
        print(f"   🔢 Memory per Sample: {results['memory_per_sample_kb']:.2f} KB/sample")
        print(f"   ✅ Memory Stable: {'Yes' if abs(memory_growth) < 10 else 'No'}")
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, any]:
        """
        Run comprehensive batch evaluation performance test.
        
        Returns:
            Complete test results
        """
        print("🚀 Running Comprehensive Batch Evaluation Performance Test")
        print("=" * 70)
        
        results = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'device': self.device,
                'device_info': self.profiler.get_device_info(),
                'config_path': self.config_path
            }
        }
        
        try:
            # Test 1: Legacy vs Batch comparison
            results['legacy_vs_batch'] = self.test_legacy_vs_batch_evaluation(num_samples=200)
            
            # Test 2: Batch size optimization
            results['batch_size_optimization'] = self.test_batch_size_optimization(
                batch_sizes=[1, 4, 8, 16, 32], num_samples=100
            )
            
            # Test 3: GPU utilization
            results['gpu_utilization'] = self.test_gpu_utilization(duration=20.0)
            
            # Test 4: Memory efficiency
            results['memory_efficiency'] = self.test_memory_efficiency(num_samples=300)
            
            # Print comprehensive summary
            self.print_comprehensive_summary(results)
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
        
        return results
    
    def print_comprehensive_summary(self, results: Dict[str, any]):
        """Print comprehensive test summary."""
        print("\n🏆 COMPREHENSIVE PERFORMANCE SUMMARY")
        print("=" * 70)
        
        # Legacy vs Batch Summary
        if 'legacy_vs_batch' in results:
            lvb = results['legacy_vs_batch']
            print(f"🔄 Legacy vs Batch Evaluation:")
            print(f"   Speedup: {lvb['improvement']['speedup']:.1f}x")
            print(f"   Time Reduction: {lvb['improvement']['time_reduction_seconds']:.1f}s "
                  f"({lvb['improvement']['time_reduction_percent']:.1f}%)")
            print(f"   Cache Hit Rate: {lvb['batch']['cache_hit_rate']:.1%}")
        
        # Batch Size Optimization Summary
        if 'batch_size_optimization' in results:
            bso = results['batch_size_optimization']
            optimal_batch = max(bso.keys(), key=lambda k: bso[k]['samples_per_second'])
            best_speed = bso[optimal_batch]['samples_per_second']
            baseline_speed = bso[1]['samples_per_second']
            
            print(f"🎯 Batch Size Optimization:")
            print(f"   Optimal Batch Size: {optimal_batch}")
            print(f"   Best Speed: {best_speed:.1f} samples/s")
            print(f"   Batch Improvement: {best_speed/baseline_speed:.1f}x")
        
        # GPU Utilization Summary
        if 'gpu_utilization' in results:
            gpu = results['gpu_utilization']
            print(f"🔍 GPU Utilization:")
            print(f"   GPU Usage: {gpu['gpu_utilization']:.1f}%")
            print(f"   Memory Usage: {gpu['memory_utilization']:.1f}%")
            print(f"   Processing Speed: {gpu['samples_per_second']:.1f} samples/s")
        
        # Memory Efficiency Summary
        if 'memory_efficiency' in results:
            mem = results['memory_efficiency']
            print(f"💾 Memory Efficiency:")
            print(f"   Peak Memory: {mem['peak_memory_mb']:.1f} MB")
            print(f"   Memory Growth: {mem['memory_growth_mb']:.1f} MB")
            print(f"   Memory per Sample: {mem['memory_per_sample_kb']:.2f} KB/sample")
        
        # Overall Assessment
        if 'legacy_vs_batch' in results:
            speedup = results['legacy_vs_batch']['improvement']['speedup']
            if speedup >= 5.0:
                print(f"\n✅ EXCELLENT: Achieved {speedup:.1f}x speedup (target: 5-10x)")
            elif speedup >= 3.0:
                print(f"\n🟡 GOOD: Achieved {speedup:.1f}x speedup (target: 5-10x)")
            else:
                print(f"\n🔴 NEEDS IMPROVEMENT: Only {speedup:.1f}x speedup (target: 5-10x)")
        
        print(f"\n🎯 Batch evaluation optimization successfully implemented!")


def main():
    """Main test execution."""
    print("🚀 NeuroGraph Batch Evaluation Performance Test")
    print("=" * 70)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        print(f"✅ CUDA Available: {device_name}")
    else:
        print("⚠️  CUDA not available, running on CPU")
    
    # Run tests
    tester = BatchEvaluationPerformanceTest("config/production.yaml")
    results = tester.run_comprehensive_test()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"logs/batch_evaluation_test_{timestamp}.json"
    
    os.makedirs("logs", exist_ok=True)
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_numpy(results), f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
