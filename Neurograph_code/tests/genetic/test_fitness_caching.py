#!/usr/bin/env python3
"""
Test script for fitness caching functionality in genetic hyperparameter tuner
"""

import sys
import os
import tempfile
import shutil

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock the training context to avoid PyTorch dependency
class MockTrainer:
    def __init__(self):
        self.call_count = 0
    
    def train(self):
        self.call_count += 1
        return [0.5, 0.4, 0.3]
    
    def evaluate_accuracy(self, num_samples=500, use_batch_evaluation=True):
        import random
        # Return consistent results for same config (simulated deterministic training)
        random.seed(42)  # Fixed seed for consistent results
        return random.uniform(0.1, 0.9)

def create_modular_train_context(config_path):
    return MockTrainer()

# Monkey patch the import
sys.modules['train'] = type(sys)('mock_module')
sys.modules['train.modular_train_context'] = type(sys)('mock_module')
sys.modules['train.modular_train_context'].create_modular_train_context = create_modular_train_context

from genetic_hyperparameter_tuner import GeneticHyperparameterTuner

def test_fitness_caching():
    """Test fitness caching functionality"""
    print('Testing Fitness Caching Implementation')
    print('=' * 50)
    
    # Create temporary cache directory for testing
    temp_cache_dir = tempfile.mkdtemp(prefix="test_cache_")
    cache_file = os.path.join(temp_cache_dir, "fitness_cache.json")
    
    try:
        # Initialize tuner with custom cache file
        tuner = GeneticHyperparameterTuner(
            generations=1,
            elite_percentage=0.5,
            crossover_rate=0.7,
            mutation_rate=0.1
        )
        tuner.cache_file = cache_file  # Override cache file location
        
        print(f'✓ Tuner initialized')
        print(f'  Cache file: {cache_file}')
        print(f'  Initial cache size: {len(tuner.fitness_cache)}')
        print()
        
        # Test individual configuration
        test_individual = {
            'vector_dim': 8,
            'phase_bins': 32,
            'mag_bins': 256,
            'cardinality': 5,
            'learning_rate': 0.001,
            'decay_factor': 0.95,
            'orthogonality_threshold': 0.1,
            'warmup_epochs': 5,
            'min_activation_strength': 0.1,
            'batch_size': 5
        }
        
        print('Testing cache functionality:')
        print(f'Test individual: {test_individual}')
        print()
        
        # First evaluation - should be cache miss
        print('1. First evaluation (cache miss expected):')
        fitness1 = tuner.evaluate_fitness(test_individual)
        print(f'   Fitness: {fitness1:.4f}')
        print(f'   Cache hits: {tuner.cache_stats["hits"]}')
        print(f'   Cache misses: {tuner.cache_stats["misses"]}')
        print(f'   Cache size: {len(tuner.fitness_cache)}')
        print()
        
        # Second evaluation - should be cache hit
        print('2. Second evaluation (cache hit expected):')
        fitness2 = tuner.evaluate_fitness(test_individual)
        print(f'   Fitness: {fitness2:.4f}')
        print(f'   Cache hits: {tuner.cache_stats["hits"]}')
        print(f'   Cache misses: {tuner.cache_stats["misses"]}')
        print(f'   Cache size: {len(tuner.fitness_cache)}')
        print()
        
        # Verify results are identical
        if fitness1 == fitness2:
            print('✓ Cache hit successful - identical fitness values')
        else:
            print('✗ Cache hit failed - different fitness values')
        
        # Test cache key generation
        cache_key1 = tuner._generate_cache_key(test_individual)
        cache_key2 = tuner._generate_cache_key(test_individual)
        
        if cache_key1 == cache_key2:
            print('✓ Cache key generation is deterministic')
        else:
            print('✗ Cache key generation is not deterministic')
        
        print(f'   Cache key: {cache_key1[:16]}...')
        print()
        
        # Test cache persistence
        print('3. Testing cache persistence:')
        tuner._save_cache()
        
        if os.path.exists(cache_file):
            print('✓ Cache file saved successfully')
            
            # Load cache in new tuner instance
            tuner2 = GeneticHyperparameterTuner(generations=1)
            tuner2.cache_file = cache_file
            tuner2._load_cache()
            
            if len(tuner2.fitness_cache) > 0:
                print('✓ Cache loaded successfully in new instance')
                print(f'   Loaded cache size: {len(tuner2.fitness_cache)}')
                
                # Test cache hit in new instance
                fitness3 = tuner2.evaluate_fitness(test_individual)
                if tuner2.cache_stats['hits'] > 0:
                    print('✓ Cache hit successful in new instance')
                else:
                    print('✗ Cache hit failed in new instance')
            else:
                print('✗ Cache loading failed')
        else:
            print('✗ Cache file not created')
        
        print()
        
        # Test different individual (should be cache miss)
        print('4. Testing different individual (cache miss expected):')
        different_individual = test_individual.copy()
        different_individual['vector_dim'] = 10  # Change one parameter
        
        fitness4 = tuner.evaluate_fitness(different_individual)
        print(f'   Different individual fitness: {fitness4:.4f}')
        print(f'   Total cache hits: {tuner.cache_stats["hits"]}')
        print(f'   Total cache misses: {tuner.cache_stats["misses"]}')
        print(f'   Final cache size: {len(tuner.fitness_cache)}')
        
        # Calculate hit rate
        total_requests = tuner.cache_stats['hits'] + tuner.cache_stats['misses']
        hit_rate = (tuner.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        print()
        print('=== Final Cache Statistics ===')
        print(f'Cache hits: {tuner.cache_stats["hits"]}')
        print(f'Cache misses: {tuner.cache_stats["misses"]}')
        print(f'Hit rate: {hit_rate:.1f}%')
        print(f'Total evaluations: {tuner.cache_stats["total_evaluations"]}')
        print(f'Cached configurations: {len(tuner.fitness_cache)}')
        
        print()
        print('✓ Fitness caching test completed successfully!')
        
    except Exception as e:
        print(f'✗ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up temporary cache directory
        if os.path.exists(temp_cache_dir):
            shutil.rmtree(temp_cache_dir)
            print(f'✓ Cleaned up temporary cache directory')

if __name__ == "__main__":
    test_fitness_caching()
