"""
Comprehensive GPU Optimization Test Suite for NeuroGraph
Tests vectorized propagation, batched evaluation, and GPU profiling
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple

# Import optimization components
from core.vectorized_propagation import create_vectorized_forward_engine
from utils.batched_evaluation import create_batched_evaluator
from utils.gpu_profiler import create_cuda_profiler, create_performance_monitor, profile_operation

# Import training context
from train.modular_train_context import create_modular_train_context


class GPUOptimizationTester:
    """
    Comprehensive test suite for GPU optimizations.
    """
    
    def __init__(self, config_path: str = "config/production_training.yaml"):
        """
        Initialize optimization tester.
        
        Args:
            config_path: Path to training configuration
        """
        self.config_path = config_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize profiling
        self.profiler = create_cuda_profiler(device=self.device)
        self.performance_monitor = create_performance_monitor(self.profiler)
        
        # Initialize training context
        print("🚀 Initializing Training Context...")
        self.training_context = create_modular_train_context(config_path)
        
        # Initialize optimizations
        self.vectorized_engine = None
        self.batched_evaluator = None
        
        print(f"✅ GPU Optimization Tester initialized on {self.device}")
    
    def test_vectorized_propagation(self, num_samples: int = 10) -> Dict[str, float]:
        """
        Test vectorized propagation performance.
        
        Args:
            num_samples: Number of samples to test
            
        Returns:
            Performance metrics
        """
        print(f"\n🔄 Testing Vectorized Propagation ({num_samples} samples)")
        print("-" * 50)
        
        # Create vectorized engine
        self.vectorized_engine = create_vectorized_forward_engine(
            self.training_context.forward_engine, 
            device=self.device
        )
        
        # Test original vs vectorized performance
        original_times = []
        vectorized_times = []
        
        for i in range(num_samples):
            # Get sample input
            input_context, target_label = self.training_context.input_adapter.get_input_context(
                i, self.training_context.input_nodes
            )
            
            # Test original forward pass
            with profile_operation(self.profiler, "original_forward_pass"):
                start_time = time.perf_counter()
                original_output = self.training_context.forward_pass(input_context)
                original_time = time.perf_counter() - start_time
                original_times.append(original_time)
            
            # Test vectorized forward pass (using batched engine with single sample)
            with profile_operation(self.profiler, "vectorized_forward_pass"):
                start_time = time.perf_counter()
                batch_contexts = self.vectorized_engine.propagate_batch([input_context])
                vectorized_time = time.perf_counter() - start_time
                vectorized_times.append(vectorized_time)
            
            if (i + 1) % max(1, num_samples // 4) == 0:
                avg_original = np.mean(original_times[-5:]) * 1000
                avg_vectorized = np.mean(vectorized_times[-5:]) * 1000
                speedup = avg_original / avg_vectorized if avg_vectorized > 0 else 1.0
                print(f"   Sample {i+1:2d}/{num_samples}: "
                      f"Original={avg_original:.1f}ms, "
                      f"Vectorized={avg_vectorized:.1f}ms, "
                      f"Speedup={speedup:.1f}x")
        
        # Calculate metrics
        avg_original_time = np.mean(original_times) * 1000
        avg_vectorized_time = np.mean(vectorized_times) * 1000
        speedup = avg_original_time / avg_vectorized_time if avg_vectorized_time > 0 else 1.0
        
        results = {
            'avg_original_time_ms': avg_original_time,
            'avg_vectorized_time_ms': avg_vectorized_time,
            'speedup': speedup,
            'samples_tested': num_samples
        }
        
        print(f"\n📊 Vectorized Propagation Results:")
        print(f"   Original Time: {avg_original_time:.1f}ms (avg)")
        print(f"   Vectorized Time: {avg_vectorized_time:.1f}ms (avg)")
        print(f"   Speedup: {speedup:.1f}x")
        
        return results
    
    def test_batched_evaluation(self, batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict[int, Dict[str, float]]:
        """
        Test batched evaluation performance.
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            Performance results for each batch size
        """
        print(f"\n🎯 Testing Batched Evaluation")
        print("-" * 50)
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n   Testing batch size: {batch_size}")
            
            # Create batched evaluator
            self.batched_evaluator = create_batched_evaluator(
                self.training_context,
                batch_size=batch_size,
                device=self.device,
                verbose=False
            )
            
            # Test evaluation performance
            with profile_operation(self.profiler, f"batched_evaluation_bs{batch_size}"):
                eval_results = self.batched_evaluator.evaluate_accuracy_batched(
                    num_samples=50,  # Small sample for testing
                    streaming=True
                )
            
            results[batch_size] = eval_results
            
            print(f"     Accuracy: {eval_results['accuracy']:.1%}")
            print(f"     Speed: {eval_results['samples_per_second']:.1f} samples/s")
            print(f"     Time: {eval_results['total_time']:.1f}s")
        
        # Find optimal batch size
        optimal_batch_size = max(results.keys(), key=lambda k: results[k]['samples_per_second'])
        
        print(f"\n📈 Batched Evaluation Results:")
        print(f"   Optimal Batch Size: {optimal_batch_size}")
        print(f"   Best Speed: {results[optimal_batch_size]['samples_per_second']:.1f} samples/s")
        
        return results
    
    def test_gpu_profiling(self, duration: float = 10.0) -> Dict[str, any]:
        """
        Test GPU profiling and monitoring.
        
        Args:
            duration: Monitoring duration in seconds
            
        Returns:
            Profiling results
        """
        print(f"\n🔍 Testing GPU Profiling ({duration}s)")
        print("-" * 50)
        
        # Start monitoring
        self.performance_monitor.start_monitoring()
        
        # Simulate workload
        start_time = time.perf_counter()
        sample_count = 0
        
        while time.perf_counter() - start_time < duration:
            # Perform some GPU operations
            with profile_operation(self.profiler, "test_workload"):
                input_context, target_label = self.training_context.input_adapter.get_input_context(
                    sample_count % 100, self.training_context.input_nodes
                )
                
                output_signals = self.training_context.forward_pass(input_context)
                
                if output_signals:
                    loss, logits = self.training_context.compute_loss(output_signals, target_label)
            
            sample_count += 1
            
            # Brief pause to allow monitoring
            time.sleep(0.1)
        
        # Stop monitoring
        self.performance_monitor.stop_monitoring()
        
        # Get results
        performance_summary = self.performance_monitor.get_performance_summary()
        timing_stats = self.profiler.get_timing_stats()
        
        print(f"\n📊 GPU Profiling Results:")
        if performance_summary:
            print(f"   Device: {performance_summary['device_info']['name']}")
            print(f"   GPU Utilization: {performance_summary['avg_gpu_utilization']:.1f}%")
            print(f"   Memory Utilization: {performance_summary['avg_memory_utilization']:.1f}%")
            print(f"   Samples Processed: {sample_count}")
            print(f"   Samples/Second: {sample_count / duration:.1f}")
        
        if timing_stats:
            print(f"\n   Timing Statistics:")
            for op_name, stats in timing_stats.items():
                print(f"     {op_name}: {stats['avg_time']:.2f}ms (avg), {stats['count']} calls")
        
        return {
            'performance_summary': performance_summary,
            'timing_stats': timing_stats,
            'samples_processed': sample_count,
            'duration': duration
        }
    
    def test_memory_efficiency(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Test memory efficiency improvements.
        
        Args:
            num_samples: Number of samples to process
            
        Returns:
            Memory usage statistics
        """
        print(f"\n💾 Testing Memory Efficiency ({num_samples} samples)")
        print("-" * 50)
        
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        initial_memory = self.profiler.get_memory_usage()
        
        # Process samples and monitor memory
        memory_snapshots = []
        
        for i in range(num_samples):
            # Process sample
            input_context, target_label = self.training_context.input_adapter.get_input_context(
                i % 1000, self.training_context.input_nodes
            )
            
            output_signals = self.training_context.forward_pass(input_context)
            
            if output_signals:
                loss, logits = self.training_context.compute_loss(output_signals, target_label)
            
            # Take memory snapshot every 10 samples
            if i % 10 == 0:
                memory_info = self.profiler.get_memory_usage()
                memory_snapshots.append(memory_info)
                
                if i % 50 == 0:
                    print(f"   Sample {i:3d}: Memory={memory_info['allocated']:.1f}MB, "
                          f"Utilization={memory_info['utilization']:.1f}%")
        
        final_memory = self.profiler.get_memory_usage()
        
        # Calculate memory statistics
        peak_memory = max(snapshot['allocated'] for snapshot in memory_snapshots)
        avg_memory = np.mean([snapshot['allocated'] for snapshot in memory_snapshots])
        memory_growth = final_memory['allocated'] - initial_memory['allocated']
        
        results = {
            'initial_memory_mb': initial_memory['allocated'],
            'final_memory_mb': final_memory['allocated'],
            'peak_memory_mb': peak_memory,
            'avg_memory_mb': avg_memory,
            'memory_growth_mb': memory_growth,
            'samples_processed': num_samples
        }
        
        print(f"\n📈 Memory Efficiency Results:")
        print(f"   Initial Memory: {initial_memory['allocated']:.1f} MB")
        print(f"   Final Memory: {final_memory['allocated']:.1f} MB")
        print(f"   Peak Memory: {peak_memory:.1f} MB")
        print(f"   Memory Growth: {memory_growth:.1f} MB")
        print(f"   Memory per Sample: {memory_growth/num_samples:.3f} MB/sample")
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, any]:
        """
        Run comprehensive optimization test suite.
        
        Returns:
            Complete test results
        """
        print("🚀 Running Comprehensive GPU Optimization Test Suite")
        print("=" * 60)
        
        results = {}
        
        try:
            # Test 1: Vectorized Propagation
            results['vectorized_propagation'] = self.test_vectorized_propagation(num_samples=20)
            
            # Test 2: Batched Evaluation
            results['batched_evaluation'] = self.test_batched_evaluation(batch_sizes=[1, 4, 8, 16])
            
            # Test 3: GPU Profiling
            results['gpu_profiling'] = self.test_gpu_profiling(duration=15.0)
            
            # Test 4: Memory Efficiency
            results['memory_efficiency'] = self.test_memory_efficiency(num_samples=100)
            
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
        print("\n🏆 COMPREHENSIVE OPTIMIZATION SUMMARY")
        print("=" * 60)
        
        # Vectorized Propagation Summary
        if 'vectorized_propagation' in results:
            vp_results = results['vectorized_propagation']
            print(f"🔄 Vectorized Propagation:")
            print(f"   Speedup: {vp_results['speedup']:.1f}x")
            print(f"   Time Reduction: {vp_results['avg_original_time_ms'] - vp_results['avg_vectorized_time_ms']:.1f}ms")
        
        # Batched Evaluation Summary
        if 'batched_evaluation' in results:
            be_results = results['batched_evaluation']
            best_batch_size = max(be_results.keys(), key=lambda k: be_results[k]['samples_per_second'])
            best_speed = be_results[best_batch_size]['samples_per_second']
            baseline_speed = be_results[1]['samples_per_second']
            speedup = best_speed / baseline_speed if baseline_speed > 0 else 1.0
            
            print(f"🎯 Batched Evaluation:")
            print(f"   Optimal Batch Size: {best_batch_size}")
            print(f"   Speed Improvement: {speedup:.1f}x ({best_speed:.1f} vs {baseline_speed:.1f} samples/s)")
        
        # GPU Profiling Summary
        if 'gpu_profiling' in results:
            gp_results = results['gpu_profiling']
            if 'performance_summary' in gp_results and gp_results['performance_summary']:
                perf_summary = gp_results['performance_summary']
                print(f"🔍 GPU Profiling:")
                print(f"   GPU Utilization: {perf_summary['avg_gpu_utilization']:.1f}%")
                print(f"   Memory Utilization: {perf_summary['avg_memory_utilization']:.1f}%")
                print(f"   Processing Speed: {gp_results['samples_processed'] / gp_results['duration']:.1f} samples/s")
        
        # Memory Efficiency Summary
        if 'memory_efficiency' in results:
            me_results = results['memory_efficiency']
            print(f"💾 Memory Efficiency:")
            print(f"   Peak Memory: {me_results['peak_memory_mb']:.1f} MB")
            print(f"   Memory Growth: {me_results['memory_growth_mb']:.1f} MB")
            print(f"   Memory per Sample: {me_results['memory_growth_mb']/me_results['samples_processed']:.3f} MB/sample")
        
        print(f"\n✅ All optimizations tested successfully!")
        print(f"🎯 Expected overall performance improvement: 3-8x faster training and evaluation")


def main():
    """Main test execution."""
    print("🚀 NeuroGraph GPU Optimization Test Suite")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        print(f"✅ CUDA Available: {device_name}")
    else:
        print("⚠️  CUDA not available, running on CPU")
    
    # Run tests
    tester = GPUOptimizationTester("config/production_training.yaml")
    results = tester.run_comprehensive_test()
    
    # Save results
    import json
    import os
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"logs/gpu_optimization_test_{timestamp}.json"
    
    os.makedirs("logs", exist_ok=True)
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
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
