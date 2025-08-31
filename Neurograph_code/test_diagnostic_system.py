#!/usr/bin/env python3
"""
Test script to verify backward pass diagnostic system functionality.
Tests both diagnostic initialization and per-sample monitoring.
"""

import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train.modular_train_context import create_modular_train_context
from utils.modular_config import ModularConfig

def test_diagnostic_initialization():
    """Test that diagnostic system initializes properly."""
    print("🔍 Testing Diagnostic System Initialization")
    print("=" * 60)
    
    try:
        # Create training context with diagnostics enabled
        trainer = create_modular_train_context("config/production.yaml")
        
        # Check if diagnostics are properly initialized
        print(f"✅ Training context created successfully")
        print(f"   📊 Diagnostics enabled: {trainer.backward_pass_diagnostics is not None}")
        print(f"   🔍 Gradient flow analyzer: {trainer.gradient_flow_analyzer is not None}")
        print(f"   🔧 Discrete update analyzer: {trainer.discrete_update_analyzer is not None}")
        
        # Check configuration
        config = trainer.config
        diagnostics_config = config.get('diagnostics', {})
        print(f"   ⚙️  Diagnostics config found: {bool(diagnostics_config)}")
        print(f"   🚨 Alerts enabled: {diagnostics_config.get('alerts_enabled', False)}")
        print(f"   📝 Verbose backward pass: {diagnostics_config.get('verbose_backward_pass', False)}")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Failed to initialize diagnostic system: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_single_sample_diagnostics(trainer):
    """Test diagnostic monitoring on a single training sample."""
    print(f"\n🔍 Testing Single Sample Diagnostic Monitoring")
    print("=" * 60)
    
    try:
        # Train on a single sample to test diagnostics
        print(f"📊 Training on sample 0 with full diagnostic monitoring...")
        
        # Enable debug flags for detailed output
        if hasattr(trainer, 'debug_gradients'):
            trainer.debug_gradients = True
        if hasattr(trainer, 'debug_updates'):
            trainer.debug_updates = True
        if hasattr(trainer, 'debug_accumulation'):
            trainer.debug_accumulation = True
        
        # Train single sample
        loss, accuracy = trainer.train_single_sample(0)
        
        print(f"✅ Single sample training completed")
        print(f"   📉 Loss: {loss:.4f}")
        print(f"   🎯 Accuracy: {accuracy:.1%}")
        
        # Check if diagnostic data was collected
        if trainer.backward_pass_diagnostics is not None:
            summary = trainer.get_diagnostic_summary()
            if summary:
                print(f"   📊 Diagnostic data collected: {len(summary)} categories")
                
                # Check gradient statistics
                grad_stats = summary.get('gradient_statistics', {})
                if grad_stats:
                    print(f"   🔍 Gradient statistics: {len(grad_stats)} metrics")
                    for key, stats in list(grad_stats.items())[:3]:  # Show first 3
                        if isinstance(stats, dict) and 'mean' in stats:
                            print(f"      - {key}: {stats['mean']:.6f}")
                
                # Check parameter updates
                param_stats = summary.get('parameter_updates', {})
                if param_stats:
                    print(f"   🔧 Parameter update statistics: {len(param_stats)} metrics")
                    for key, stats in param_stats.items():
                        if isinstance(stats, dict) and 'mean' in stats:
                            print(f"      - {key}: {stats['mean']:.6f}")
                
                # Check loss analysis
                loss_stats = summary.get('loss_analysis', {})
                if loss_stats:
                    print(f"   📉 Loss analysis: {len(loss_stats)} metrics")
                    if 'total_loss' in loss_stats:
                        loss_data = loss_stats['total_loss']
                        print(f"      - Total loss: {loss_data.get('latest', 0):.4f}")
            else:
                print(f"   ⚠️  No diagnostic summary available")
        else:
            print(f"   ❌ Diagnostic system not initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed single sample diagnostic test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_sample_diagnostics(trainer):
    """Test diagnostic monitoring across multiple samples."""
    print(f"\n🔍 Testing Multiple Sample Diagnostic Monitoring")
    print("=" * 60)
    
    try:
        print(f"📊 Training on 5 samples with diagnostic monitoring...")
        
        losses = []
        accuracies = []
        
        for i in range(5):
            print(f"\n--- Sample {i+1}/5 ---")
            loss, accuracy = trainer.train_single_sample(i)
            losses.append(loss)
            accuracies.append(accuracy)
            
            print(f"Sample {i+1}: Loss={loss:.4f}, Acc={accuracy:.1%}")
        
        # Print summary statistics
        avg_loss = sum(losses) / len(losses)
        avg_acc = sum(accuracies) / len(accuracies)
        
        print(f"\n✅ Multiple sample training completed")
        print(f"   📉 Average loss: {avg_loss:.4f}")
        print(f"   🎯 Average accuracy: {avg_acc:.1%}")
        
        # Print comprehensive diagnostic report
        if trainer.backward_pass_diagnostics is not None:
            print(f"\n📊 Comprehensive Diagnostic Report:")
            trainer.print_diagnostic_report()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed multiple sample diagnostic test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_diagnostic_configuration():
    """Test diagnostic configuration loading."""
    print(f"\n🔍 Testing Diagnostic Configuration")
    print("=" * 60)
    
    try:
        config = ModularConfig("config/production.yaml")
        
        # Check diagnostics section
        diagnostics_config = config.get('diagnostics', {})
        print(f"✅ Diagnostics configuration loaded")
        print(f"   📊 Enabled: {diagnostics_config.get('enabled', False)}")
        print(f"   🚨 Alerts enabled: {diagnostics_config.get('alerts_enabled', False)}")
        print(f"   📝 Verbose backward pass: {diagnostics_config.get('verbose_backward_pass', False)}")
        print(f"   💾 Save diagnostic data: {diagnostics_config.get('save_diagnostic_data', False)}")
        print(f"   🔍 Print per sample: {diagnostics_config.get('print_per_sample_diagnostics', False)}")
        
        # Check thresholds
        thresholds = [
            'gradient_explosion_threshold',
            'gradient_vanishing_threshold', 
            'parameter_stagnation_threshold',
            'loss_spike_threshold',
            'memory_usage_threshold'
        ]
        
        print(f"   🎯 Alert thresholds:")
        for threshold in thresholds:
            value = diagnostics_config.get(threshold, 'not set')
            print(f"      - {threshold}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load diagnostic configuration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic system tests."""
    print("🧪 NeuroGraph Diagnostic System Test Suite")
    print("=" * 80)
    
    # Test 1: Configuration loading
    config_success = test_diagnostic_configuration()
    
    # Test 2: Diagnostic initialization
    trainer = test_diagnostic_initialization()
    
    if trainer is None:
        print(f"\n❌ Cannot proceed with further tests - initialization failed")
        return False
    
    # Test 3: Single sample diagnostics
    single_success = test_single_sample_diagnostics(trainer)
    
    # Test 4: Multiple sample diagnostics
    multiple_success = test_multiple_sample_diagnostics(trainer)
    
    # Summary
    print(f"\n🧪 Test Suite Summary")
    print("=" * 80)
    print(f"Configuration loading: {'✅ PASS' if config_success else '❌ FAIL'}")
    print(f"Diagnostic initialization: {'✅ PASS' if trainer is not None else '❌ FAIL'}")
    print(f"Single sample diagnostics: {'✅ PASS' if single_success else '❌ FAIL'}")
    print(f"Multiple sample diagnostics: {'✅ PASS' if multiple_success else '❌ FAIL'}")
    
    all_passed = config_success and trainer is not None and single_success and multiple_success
    print(f"\nOverall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
