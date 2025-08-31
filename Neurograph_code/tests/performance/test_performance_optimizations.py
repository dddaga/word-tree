#!/usr/bin/env python3
"""
Performance Optimization Test Suite for NeuroGraph
Tests all implemented performance fixes and optimizations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import time
from datetime import datetime
import json

# Import optimized components
from utils.device_manager import get_device_manager, cleanup_memory
from utils.performance_monitor import get_performance_monitor, timing_context
from core.activation_table import ActivationTable
from core.radiation import get_radiation_neighbors, print_cache_performance, clear_radiation_cache
from train.gradient_accumulator import GradientAccumulator
from train.modular_train_context import create_modular_train_context

def test_device_management():
    """Test GPU device management and optimization."""
    print("\n🔧 Testing Device Management")
    print("=" * 50)
    
    device_manager = get_device_manager()
    device_manager.print_status()
    
    # Test tensor operations
    print("\n📊 Testing tensor operations...")
    test_tensor = device_manager.zeros(1000, 100)
    print(f"   ✅ Created tensor on {test_tensor.device}")
    
    # Test memory management
    memory_before = device_manager.get_memory_usage()
    large_tensor = device_manager.zeros(5000, 5000)  # ~100MB
    memory_after = device_manager.get_memory_usage()
    
    print(f"   📈 Memory usage: {memory_before['allocated_gb']:.2f} GB → {memory_after['allocated_gb']:.2f} GB")
    
    # Cleanup test
    del large_tensor
    device_manager.cleanup_memory(aggressive=True)
    memory_cleaned = device_manager.get_memory_usage()
    print(f"   🧹 After cleanup: {memory_cleaned['allocated_gb']:.2f} GB")
    
    return {
        'device': str(device_manager.device),
        'memory_management': 'working',
        'tensor_operations': 'working'
    }

def test_optimized_activation_table():
    """Test optimized activation table with tensor storage."""
    print("\n📊 Testing Optimized Activation Table")
    print("=" * 50)
    
    # Create optimized activation table
    activation_table = ActivationTable(
        vector_dim=5, 
        phase_bins=32, 
        mag_bins=512, 
        max_nodes=1000,
        device='auto'
    )
    
    # Test injection performance
    device_manager = get_device_manager()
    
    print("\n⚡ Performance test...")
    start_time = time.time()
    
    # Inject many activations
    for i in range(500):
        node_id = f"test_node_{i}"
        phase_idx = device_manager.zeros(5, dtype=torch.long)
        mag_idx = device_manager.zeros(5, dtype=torch.long)
        activation_table.inject(node_id, phase_idx, mag_idx, 1.0)
    
    injection_time = time.time() - start_time
    print(f"   📈 Injected 500 nodes in {injection_time:.3f}s ({500/injection_time:.1f} nodes/sec)")
    
    # Test decay performance
    start_time = time.time()
    activation_table.decay_and_prune()
    decay_time = time.time() - start_time
    print(f"   🔄 Decay and prune in {decay_time:.3f}s")
    
    # Memory report
    activation_table.print_memory_report()
    
    return {
        'injection_rate': 500 / injection_time,
        'decay_time': decay_time,
        'memory_usage': activation_table.get_memory_stats()
    }

def test_radiation_optimization():
    """Test radiation system optimizations."""
    print("\n🚀 Testing Radiation System Optimization")
    print("=" * 50)
    
    # Clear cache for clean test
    clear_radiation_cache()
    
    # Create mock components for testing
    device_manager = get_device_manager()
    
    # Mock node store
    class MockNodeStore:
        def __init__(self):
            self.phase_table = {}
            for i in range(1000):
                node_id = f"n{i}"
                self.phase_table[node_id] = device_manager.zeros(5, dtype=torch.long)
        
        def get_phase(self, node_id):
            return self.phase_table.get(node_id, device_manager.zeros(5, dtype=torch.long))
    
    # Mock graph DataFrame
    import pandas as pd
    graph_data = []
    for i in range(100):
        connections = [f"n{j}" for j in range(max(0, i-5), min(1000, i+5))]
        graph_data.append({
            'node_id': f"n{i}",
            'input_connections': connections
        })
    
    graph_df = pd.DataFrame(graph_data)
    
    # Mock lookup table
    class MockLookupTable:
        def __init__(self):
            self.N = 32
        
        def lookup_phase(self, indices):
            return device_manager.ones(indices.shape, dtype=torch.float32)
    
    node_store = MockNodeStore()
    lookup_table = MockLookupTable()
    
    # Test radiation neighbor selection
    print("\n⚡ Testing radiation performance...")
    
    test_phase = device_manager.zeros(5, dtype=torch.long)
    
    # Cold run (cache miss)
    start_time = time.time()
    neighbors1 = get_radiation_neighbors(
        "n50", test_phase, node_store, graph_df, lookup_table, 
        top_k=10, batch_size=128
    )
    cold_time = time.time() - start_time
    
    # Warm run (cache hit)
    start_time = time.time()
    neighbors2 = get_radiation_neighbors(
        "n50", test_phase, node_store, graph_df, lookup_table, 
        top_k=10, batch_size=128
    )
    warm_time = time.time() - start_time
    
    print(f"   🥶 Cold run: {cold_time:.4f}s")
    print(f"   🔥 Warm run: {warm_time:.4f}s")
    print(f"   📈 Speedup: {cold_time/warm_time:.1f}x")
    
    # Cache performance report
    print_cache_performance()
    
    return {
        'cold_time': cold_time,
        'warm_time': warm_time,
        'speedup': cold_time / warm_time,
        'neighbors_found': len(neighbors1)
    }

def test_gradient_accumulator():
    """Test optimized gradient accumulator."""
    print("\n📈 Testing Gradient Accumulator")
    print("=" * 50)
    
    # Create accumulator with increased buffer size
    accumulator = GradientAccumulator(
        accumulation_steps=8,
        lr_scaling="sqrt",
        buffer_size=1500,
        device='auto'
    )
    
    device_manager = get_device_manager()
    
    # Test gradient accumulation
    print("\n⚡ Testing gradient accumulation...")
    
    start_time = time.time()
    
    # Accumulate gradients for many nodes
    for step in range(16):  # 2 accumulation cycles
        for node_id in range(100):
            phase_grad = device_manager.ones(5) * 0.01
            mag_grad = device_manager.ones(5) * 0.01
            accumulator.accumulate_gradients(node_id, phase_grad, mag_grad)
    
    accumulation_time = time.time() - start_time
    
    # Get buffer status
    buffer_status = accumulator.get_buffer_status()
    stats = accumulator.get_statistics()
    
    print(f"   📊 Accumulated gradients in {accumulation_time:.3f}s")
    print(f"   💾 Buffer utilization: {buffer_status['buffer_utilization']:.1%}")
    print(f"   📈 Total gradients: {stats['total_gradients_accumulated']}")
    
    return {
        'accumulation_time': accumulation_time,
        'buffer_utilization': buffer_status['buffer_utilization'],
        'total_gradients': stats['total_gradients_accumulated']
    }

def test_performance_monitoring():
    """Test performance monitoring system."""
    print("\n📊 Testing Performance Monitoring")
    print("=" * 50)
    
    monitor = get_performance_monitor()
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate training activity
    monitor.start_epoch(1)
    
    for sample in range(5):
        monitor.start_sample(sample)
        
        # Simulate work with timing context
        with timing_context("forward_pass"):
            time.sleep(0.1)  # Simulate forward pass
        
        with timing_context("backward_pass"):
            time.sleep(0.05)  # Simulate backward pass
        
        monitor.end_sample()
    
    monitor.end_epoch()
    
    # Wait for some monitoring data
    time.sleep(2)
    
    # Stop monitoring and get report
    monitor.stop_monitoring()
    monitor.print_performance_report()
    
    summary = monitor.get_performance_summary()
    
    return {
        'samples_processed': summary['statistics']['total_samples_processed'],
        'epochs_completed': summary['statistics']['total_epochs_completed'],
        'timing_data_available': len(summary['timing_stats']) > 0
    }

def test_integrated_training():
    """Test integrated training with all optimizations."""
    print("\n🎯 Testing Integrated Training")
    print("=" * 50)
    
    try:
        # Create training context with optimized config
        print("   🔧 Creating training context...")
        trainer = create_modular_train_context('config/production_training.yaml')
        
        print("   ✅ Training context created successfully")
        print(f"   📊 Total parameters: {trainer.count_parameters():,}")
        
        # Test single forward pass
        print("   🚀 Testing forward pass...")
        
        # Get a sample from the dataset
        sample_data, sample_target = next(iter(trainer.train_loader))
        
        with timing_context("integrated_forward"):
            # Process single sample
            input_context = trainer.input_adapter.process_input(sample_data[0])
            activation_table = trainer.forward_engine.propagate(input_context)
        
        print("   ✅ Forward pass completed successfully")
        
        # Test loss computation
        with timing_context("integrated_loss"):
            loss = trainer.compute_loss(activation_table, sample_target[0])
        
        print(f"   📊 Loss computed: {loss:.4f}")
        
        return {
            'training_context': 'created',
            'forward_pass': 'working',
            'loss_computation': 'working',
            'parameters': trainer.count_parameters()
        }
        
    except Exception as e:
        print(f"   ❌ Error in integrated training: {e}")
        return {
            'training_context': 'failed',
            'error': str(e)
        }

def run_comprehensive_test():
    """Run comprehensive performance optimization test suite."""
    print("🚀 NeuroGraph Performance Optimization Test Suite")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    try:
        # Test 1: Device Management
        results['device_management'] = test_device_management()
        
        # Test 2: Optimized Activation Table
        results['activation_table'] = test_optimized_activation_table()
        
        # Test 3: Radiation Optimization
        results['radiation_system'] = test_radiation_optimization()
        
        # Test 4: Gradient Accumulator
        results['gradient_accumulator'] = test_gradient_accumulator()
        
        # Test 5: Performance Monitoring
        results['performance_monitoring'] = test_performance_monitoring()
        
        # Test 6: Integrated Training
        results['integrated_training'] = test_integrated_training()
        
        # Overall summary
        print("\n🎯 Test Suite Summary")
        print("=" * 50)
        
        passed_tests = 0
        total_tests = len(results)
        
        for test_name, test_result in results.items():
            if isinstance(test_result, dict) and 'error' not in test_result:
                status = "✅ PASSED"
                passed_tests += 1
            else:
                status = "❌ FAILED"
            
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        # Save results
        results_file = f"logs/performance_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('logs', exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📄 Detailed results saved to: {results_file}")
        
        # Performance summary
        print(f"\n📊 Performance Improvements Summary")
        print("=" * 50)
        
        if 'activation_table' in results:
            injection_rate = results['activation_table'].get('injection_rate', 0)
            print(f"   Activation Table: {injection_rate:.1f} nodes/sec injection rate")
        
        if 'radiation_system' in results:
            speedup = results['radiation_system'].get('speedup', 1)
            print(f"   Radiation System: {speedup:.1f}x cache speedup")
        
        if 'device_management' in results:
            device = results['device_management'].get('device', 'unknown')
            print(f"   Device Management: {device} optimization active")
        
        print(f"\n🎉 Performance optimization test suite completed!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}
    
    finally:
        # Cleanup
        cleanup_memory(aggressive=True)

if __name__ == "__main__":
    results = run_comprehensive_test()
    
    # Exit with appropriate code
    if 'error' in results:
        sys.exit(1)
    else:
        sys.exit(0)
