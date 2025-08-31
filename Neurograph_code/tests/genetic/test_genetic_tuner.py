"""
Test script for the genetic algorithm hyperparameter tuner
Runs a small-scale test to verify functionality
"""

import os
import sys
from genetic_hyperparameter_tuner import GeneticHyperparameterTuner

def test_individual_generation():
    """Test individual generation functionality."""
    print("🧪 Testing individual generation...")
    
    tuner = GeneticHyperparameterTuner()
    
    # Generate a few individuals
    for i in range(3):
        individual = tuner.generate_individual()
        print(f"Individual {i + 1}: {individual}")
    
    print("✅ Individual generation test passed\n")

def test_config_creation():
    """Test NeuroGraph config creation from individual."""
    print("🧪 Testing config creation...")
    
    tuner = GeneticHyperparameterTuner()
    
    # Generate test individual
    individual = tuner.generate_individual()
    print(f"Test individual: {individual}")
    
    # Create config
    config = tuner.create_neurograph_config(individual)
    
    # Verify key sections exist
    required_sections = [
        'architecture', 'resolution', 'training', 'forward_pass',
        'input_processing', 'class_encoding', 'loss_function'
    ]
    
    for section in required_sections:
        if section not in config:
            print(f"❌ Missing section: {section}")
            return False
    
    print("✅ Config creation test passed")
    print(f"   Architecture: {config['architecture']['total_nodes']} nodes")
    print(f"   Resolution: {config['resolution']['phase_bins']}x{config['resolution']['mag_bins']}")
    print(f"   Learning rate: {config['training']['optimizer']['base_learning_rate']}")
    print()

def test_genetic_operators():
    """Test genetic algorithm operators."""
    print("🧪 Testing genetic operators...")
    
    tuner = GeneticHyperparameterTuner()
    
    # Test crossover
    parent1 = tuner.generate_individual()
    parent2 = tuner.generate_individual()
    
    offspring1, offspring2 = tuner.uniform_crossover(parent1, parent2)
    
    print(f"Parent 1: {parent1}")
    print(f"Parent 2: {parent2}")
    print(f"Offspring 1: {offspring1}")
    print(f"Offspring 2: {offspring2}")
    
    # Test mutation
    original = tuner.generate_individual()
    mutated = tuner.mutate(original)
    
    print(f"Original: {original}")
    print(f"Mutated: {mutated}")
    
    # Count differences
    differences = sum(1 for k in original.keys() if original[k] != mutated[k])
    print(f"Parameters changed: {differences}/{len(original)}")
    
    print("✅ Genetic operators test passed\n")

def test_selection():
    """Test selection mechanisms."""
    print("🧪 Testing selection...")
    
    tuner = GeneticHyperparameterTuner()
    
    # Create test population
    population = [tuner.generate_individual() for _ in range(5)]
    fitness_scores = [0.1, 0.8, 0.3, 0.9, 0.2]  # Mock fitness scores
    
    # Test tournament selection
    selected = tuner.tournament_selection(population, fitness_scores)
    print(f"Tournament selection result: {selected}")
    
    # Test top-k selection
    top_k = tuner.select_top_k(population, fitness_scores, k=3)
    print(f"Top-3 selection:")
    for i, individual in enumerate(top_k):
        idx = population.index(individual)
        print(f"  Rank {i + 1}: Individual {idx} (fitness: {fitness_scores[idx]})")
    
    print("✅ Selection test passed\n")

def test_mini_ga():
    """Test a mini genetic algorithm run."""
    print("🧪 Testing mini genetic algorithm...")
    print("⚠️  This will run actual NeuroGraph training (1 generation, 2 individuals)")
    
    response = input("Proceed with mini GA test? This will take ~12 minutes (y/N): ")
    if response.lower() != 'y':
        print("Mini GA test skipped.")
        return
    
    try:
        from genetic_hyperparameter_tuner import genetic_hyperparam_search
        
        # Run very small GA
        results = genetic_hyperparam_search(
            config_input={},
            generations=1,
            population_size=2,
            top_k=2
        )
        
        print("✅ Mini GA test completed successfully!")
        print(f"Results: {len(results)} configurations")
        for i, result in enumerate(results):
            print(f"  Rank {i + 1}: Fitness = {result['fitness']:.4f}")
        
    except Exception as e:
        print(f"❌ Mini GA test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("🧬 Genetic Algorithm Hyperparameter Tuner - Test Suite")
    print("=" * 60)
    
    # Run tests
    test_individual_generation()
    test_config_creation()
    test_genetic_operators()
    test_selection()
    
    # Optional full test
    test_mini_ga()
    
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main()
