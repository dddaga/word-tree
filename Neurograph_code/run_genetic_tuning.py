"""
Example script for running genetic algorithm hyperparameter tuning on NeuroGraph
"""

import os
import sys
from genetic_hyperparameter_tuner import genetic_hyperparam_search

def main():
    """Run genetic algorithm hyperparameter tuning."""
    print("🧬 NeuroGraph Genetic Algorithm Hyperparameter Tuning")
    print("=" * 60)
    
    # Configuration for the GA run
    config = {
        'generations': 5,        # Number of generations to evolve
        'population_size': 20,   # Population size per generation
        'top_k': 5,             # Number of best configs to return
        'crossover_rate': 0.3,   # Crossover probability
        'mutation_rate': 0.2,    # Mutation probability per gene
        'elite_percentage': 0.5  # Elite preservation percentage
    }
    
    print(f"Configuration:")
    print(f"  Generations: {config['generations']}")
    print(f"  Population size: {config['population_size']}")
    print(f"  Top-k results: {config['top_k']}")
    print(f"  Crossover rate: {config['crossover_rate']}")
    print(f"  Mutation rate: {config['mutation_rate']}")
    print(f"  Elite percentage: {config['elite_percentage']:.1%}")
    print(f"  Training epochs per evaluation: 50")
    print(f"  Validation samples: 500")
    print(f"  Max timesteps: 50")
    
    print(f"\nHyperparameters being optimized:")
    print(f"  1. vector_dim: [5, 8, 10]")
    print(f"  2. phase_bins: [16, 32, 64, 128]")
    print(f"  3. mag_bins: [64, 128, 256, 512, 1024]")
    print(f"  4. cardinality: [3, 4, 5, 6, 7, 8]")
    print(f"  5. learning_rate: [0.0001, 0.0005, 0.001, 0.005]")
    print(f"  6. decay_factor: [0.9, 0.925, 0.95, 0.975]")
    print(f"  7. orthogonality_threshold: [0.05, 0.1, 0.15, 0.2]")
    print(f"  8. warmup_epochs: [3, 5, 8, 10]")
    print(f"  9. min_activation_strength: [0.01, 0.05, 0.1, 0.2, 0.5]")
    print(f"  10. batch_size: [3, 5, 8, 10]")
    
    print(f"\nEstimated runtime: ~{config['generations'] * config['population_size'] * 6} minutes")
    print(f"(Assuming ~6 minutes per 50-epoch training run)")
    
    # Confirm before starting
    response = input("\nProceed with genetic algorithm tuning? (y/N): ")
    if response.lower() != 'y':
        print("Tuning cancelled.")
        return
    
    print(f"\n🚀 Starting genetic algorithm hyperparameter search...")
    print(f"This will take a while - check logs/genetic_algorithm/ for progress")
    
    try:
        # Run the genetic algorithm
        best_configs = genetic_hyperparam_search(
            config_input={},  # Not used - we generate configs from scratch
            generations=config['generations'],
            population_size=config['population_size'],
            top_k=config['top_k'],
            crossover_rate=config['crossover_rate'],
            mutation_rate=config['mutation_rate'],
            elite_percentage=config['elite_percentage']
        )
        
        # Display results
        print(f"\n🎉 Genetic algorithm completed successfully!")
        print(f"Found {len(best_configs)} best configurations:")
        
        for i, config_result in enumerate(best_configs):
            print(f"\n--- Rank {i + 1} ---")
            print(f"Fitness (Validation Accuracy): {config_result['fitness']:.4f}")
            print(f"Found in Generation: {config_result['generation']}")
            print(f"Hyperparameters:")
            
            # Display hyperparameters (excluding metadata)
            for param, value in config_result.items():
                if param not in ['fitness', 'generation']:
                    print(f"  {param}: {value}")
        
        print(f"\n📁 Detailed results saved to results/genetic_algorithm/")
        print(f"📊 Logs available in logs/genetic_algorithm/")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Genetic algorithm interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during genetic algorithm: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
