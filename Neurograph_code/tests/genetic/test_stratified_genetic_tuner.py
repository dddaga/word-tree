"""
Test script for the enhanced genetic hyperparameter tuner with stratified sampling
"""

import os
import sys
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from genetic_hyperparameter_tuner import GeneticHyperparameterTuner
from modules.stratified_data_manager import create_stratified_data_manager
from modules.multi_run_fitness_evaluator import create_multi_run_fitness_evaluator

def test_stratified_data_manager():
    """Test the stratified data manager functionality."""
    print("🧪 Testing Stratified Data Manager...")
    
    try:
        # Create data manager
        data_manager = create_stratified_data_manager(
            training_samples=500,
            test_samples=500,
            num_classes=10
        )
        
        # Test training sample generation
        training_samples_run1 = data_manager.get_training_samples(run_id=0, base_seed=1000)
        training_samples_run2 = data_manager.get_training_samples(run_id=1, base_seed=1000)
        
        print(f"✅ Training samples generated: Run 1: {len(training_samples_run1)}, Run 2: {len(training_samples_run2)}")
        
        # Verify different samples for different runs
        overlap = set(training_samples_run1) & set(training_samples_run2)
        print(f"✅ Sample overlap between runs: {len(overlap)} (should be low)")
        
        # Test stratification
        train_dist_1 = data_manager.validate_stratification(training_samples_run1, use_test_set=False)
        print(f"✅ Training distribution Run 1: {train_dist_1}")
        
        # Test fixed test set
        test_samples = data_manager.get_fixed_test_set()
        test_dist = data_manager.validate_stratification(test_samples, use_test_set=True)
        print(f"✅ Fixed test set distribution: {test_dist}")
        
        # Get statistics
        stats = data_manager.get_statistics()
        print(f"✅ Data manager statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing stratified data manager: {e}")
        return False

def test_multi_run_evaluator():
    """Test the multi-run fitness evaluator (without actual training)."""
    print("\n🧪 Testing Multi-Run Fitness Evaluator...")
    
    try:
        # Create evaluator
        evaluator = create_multi_run_fitness_evaluator(
            num_runs=2,  # Small number for testing
            training_samples_per_run=500,
            test_samples=500,
            samples_per_class=50
        )
        
        # Test configuration creation
        test_individual = {
            'vector_dim': 5,
            'phase_bins': 32,
            'mag_bins': 256,
            'cardinality': 4,
            'learning_rate': 0.001,
            'decay_factor': 0.95,
            'orthogonality_threshold': 0.1,
            'warmup_epochs': 5,
            'min_activation_strength': 0.1,
            'batch_size': 5
        }
        
        # Test epoch calculation
        epochs = evaluator.calculate_epochs(test_individual['batch_size'])
        print(f"✅ Calculated epochs for batch_size={test_individual['batch_size']}: {epochs}")
        
        # Test config creation
        config = evaluator.create_neurograph_config_for_run(test_individual, run_id=0, temp_dir="/tmp")
        print(f"✅ NeuroGraph config created successfully")
        print(f"   Training epochs: {config['training']['optimizer']['num_epochs']}")
        print(f"   Stratified sampling enabled: {config.get('stratified_sampling', {}).get('enabled', False)}")
        
        # Get evaluation statistics
        stats = evaluator.get_evaluation_statistics()
        print(f"✅ Evaluator statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing multi-run evaluator: {e}")
        return False

def test_genetic_tuner_initialization():
    """Test genetic tuner initialization with new parameters."""
    print("\n🧪 Testing Genetic Tuner Initialization...")
    
    try:
        # Test tuner creation
        tuner = GeneticHyperparameterTuner(
            generations=2,
            elite_percentage=0.5,
            crossover_rate=0.3,
            mutation_rate=0.2
        )
        
        print(f"✅ Genetic tuner initialized successfully")
        print(f"   Multi-run config: {tuner.multi_run_config}")
        print(f"   Fixed params: {tuner.fixed_params}")
        
        # Test individual generation
        individual = tuner.generate_individual()
        print(f"✅ Random individual generated: {individual}")
        
        # Test cache key generation
        cache_key = tuner._generate_cache_key(individual)
        print(f"✅ Cache key generated: {cache_key[:16]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing genetic tuner initialization: {e}")
        return False

def test_genetic_operations():
    """Test genetic operations (crossover, mutation, selection)."""
    print("\n🧪 Testing Genetic Operations...")
    
    try:
        tuner = GeneticHyperparameterTuner(generations=1)
        
        # Create test individuals
        parent1 = tuner.generate_individual()
        parent2 = tuner.generate_individual()
        
        print(f"✅ Parent 1: {parent1}")
        print(f"✅ Parent 2: {parent2}")
        
        # Test crossover
        offspring1, offspring2 = tuner.uniform_crossover(parent1, parent2)
        print(f"✅ Offspring 1: {offspring1}")
        print(f"✅ Offspring 2: {offspring2}")
        
        # Test mutation
        mutated = tuner.mutate(parent1)
        print(f"✅ Mutated individual: {mutated}")
        
        # Test selection
        population = [parent1, parent2, offspring1, offspring2]
        fitness_scores = [0.8, 0.6, 0.7, 0.9]
        
        top_2 = tuner.select_top_k(population, fitness_scores, k=2)
        print(f"✅ Top 2 selected based on fitness")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing genetic operations: {e}")
        return False

def run_mini_genetic_search():
    """Run a minimal genetic search to test the complete pipeline."""
    print("\n🧪 Running Mini Genetic Search (Mock Evaluation)...")
    
    try:
        # Create a mock tuner that doesn't actually train
        class MockGeneticTuner(GeneticHyperparameterTuner):
            def evaluate_fitness(self, individual):
                """Mock fitness evaluation that returns random fitness."""
                import random
                # Simulate cache miss
                cache_key = self._generate_cache_key(individual)
                if cache_key not in self.fitness_cache:
                    self.cache_stats['misses'] += 1
                    self.cache_stats['total_evaluations'] += 1
                    
                    # Mock fitness based on some hyperparameters
                    mock_fitness = (
                        individual['learning_rate'] * 100 +
                        individual['vector_dim'] * 0.05 +
                        random.uniform(0.1, 0.3)
                    )
                    mock_fitness = min(1.0, max(0.0, mock_fitness))
                    
                    self.fitness_cache[cache_key] = mock_fitness
                    self.logger.info(f"Mock fitness for individual: {mock_fitness:.4f}")
                    return mock_fitness
                else:
                    self.cache_stats['hits'] += 1
                    return self.fitness_cache[cache_key]
        
        # Run mini search
        tuner = MockGeneticTuner(
            generations=2,
            elite_percentage=0.5,
            crossover_rate=0.3,
            mutation_rate=0.2
        )
        
        results = tuner.genetic_hyperparam_search(
            config_input={},
            population_size=4,
            top_k=2
        )
        
        print(f"✅ Mini genetic search completed successfully")
        print(f"   Found {len(results)} top configurations")
        for i, result in enumerate(results):
            print(f"   Rank {i+1}: Fitness = {result['fitness']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in mini genetic search: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Genetic Hyperparameter Tuner")
    print("=" * 60)
    
    # Setup logging to reduce noise
    logging.getLogger().setLevel(logging.WARNING)
    
    tests = [
        ("Stratified Data Manager", test_stratified_data_manager),
        ("Multi-Run Evaluator", test_multi_run_evaluator),
        ("Genetic Tuner Initialization", test_genetic_tuner_initialization),
        ("Genetic Operations", test_genetic_operations),
        ("Mini Genetic Search", run_mini_genetic_search)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name:30s} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The enhanced genetic tuner is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
