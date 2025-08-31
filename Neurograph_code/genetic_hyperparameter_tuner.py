"""
Genetic Algorithm Hyperparameter Tuner for NeuroGraph
Optimizes discrete neural network hyperparameters through evolutionary search
"""
### Flatten and avoiding nesting params
import os
import sys
import yaml
import json
import random
import tempfile
import shutil
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Union, Any
import logging

# Add current directory to path for NeuroGraph imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# NeuroGraph imports
from train.modular_train_context import create_modular_train_context
from modules.multi_run_fitness_evaluator import create_multi_run_fitness_evaluator


class GeneticHyperparameterTuner:
    """
    Genetic Algorithm for optimizing NeuroGraph hyperparameters.
    
    Optimizes 10 key hyperparameters through evolutionary search with
    actual training runs for fitness evaluation.
    """
    
    def __init__(self, generations=10, elite_percentage=0.5, crossover_rate=0.3, mutation_rate=0.2):
        """
        Initialize the genetic algorithm tuner.
        
        Args:
            generations: Number of evolution cycles to run (1 to 100)
            elite_percentage: Percentage of population that survives to breed (0.0 to 1.0)
            crossover_rate: Probability of crossover between parents (0.0 to 1.0)
            mutation_rate: Probability of mutation per gene (0.0 to 1.0)
        """
        # Validate parameters
        
        if not (1 <= generations <= 100):
            raise ValueError(f"generations must be between 1 and 100, got {generations}")
        if not (0.0 <= elite_percentage <= 1.0):
            raise ValueError(f"elite_percentage must be between 0.0 and 1.0, got {elite_percentage}")
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError(f"crossover_rate must be between 0.0 and 1.0, got {crossover_rate}")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError(f"mutation_rate must be between 0.0 and 1.0, got {mutation_rate}")
        
        # Hyperparameter search spaces
        self.search_space = {
            'vector_dim': [5, 8, 10],
            'phase_bins': [16, 32, 64, 128],
            'mag_bins': [64, 128, 256, 512, 1024],
            'cardinality': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'decay_factor': [0.9, 0.925, 0.95, 0.975],
            'orthogonality_threshold': [0.05, 0.1, 0.15, 0.2],
            'warmup_epochs': [3, 5, 8, 10],
            'min_activation_strength': [0.01, 0.05, 0.1, 0.2, 0.5],
            'batch_size': [3, 5, 8, 10]
        }
        
        # Fixed parameters - UPDATED for stratified sampling
        self.fixed_params = {
            'accumulation_steps': 8,
            'total_training_samples': 500,  # Training samples per run
            'validation_samples': 500,      # Fixed test set size
            'num_evaluation_runs': 5,       # Multiple runs per candidate
            'samples_per_class': 50,        # Stratified sampling
            'stratified_sampling': True     # Enable stratified sampling
        }
        
        # Multi-run fitness evaluator configuration
        self.multi_run_config = {
            'num_runs': self.fixed_params['num_evaluation_runs'],
            'training_samples_per_run': self.fixed_params['total_training_samples'],
            'test_samples': self.fixed_params['validation_samples'],
            'samples_per_class': self.fixed_params['samples_per_class']
        }
        
        # GA parameters (user-configurable)
        self.generations = generations
        self.elite_percentage = elite_percentage  # Now serves as survivor percentage
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        # Initialize fitness cache
        self.fitness_cache = {}  # In-memory cache: {config_hash: fitness_score}
        self.cache_file = "cache/genetic_algorithm/fitness_cache.json"
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'total_evaluations': 0
        }
        
        # Setup logging
        self.setup_logging()
        
        # Load existing cache
        self._load_cache()
        
        # Initialize multi-run fitness evaluator
        self.multi_run_evaluator = create_multi_run_fitness_evaluator(**self.multi_run_config)
        
        # Log GA configuration
        self.logger.info(f"GA Parameters: generations={generations}, elite_percentage={elite_percentage}, "
                        f"crossover_rate={crossover_rate}, mutation_rate={mutation_rate}")
        self.logger.info(f"Multi-run evaluation: {self.fixed_params['num_evaluation_runs']} runs per candidate")
        self.logger.info(f"Stratified sampling: {self.fixed_params['total_training_samples']} samples per run")
        self.logger.info(f"Fitness cache loaded: {len(self.fitness_cache)} cached evaluations")
        
    def setup_logging(self):
        """Setup logging for GA progress tracking."""
        log_dir = "logs/genetic_algorithm"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/ga_tuning_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Genetic Algorithm Hyperparameter Tuner initialized")
    
    def _generate_cache_key(self, individual: Dict[str, Any]) -> str:
        """
        Generate a deterministic cache key from hyperparameter configuration.
        
        Args:
            individual: Dictionary of hyperparameter values
            
        Returns:
            SHA-256 hash string for cache lookup
        """
        # Create a sorted, deterministic representation
        cache_data = {
            'hyperparams': dict(sorted(individual.items())),
            'fixed_params': dict(sorted(self.fixed_params.items()))
        }
        
        # Convert to JSON string with consistent formatting
        cache_str = json.dumps(cache_data, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA-256 hash
        return hashlib.sha256(cache_str.encode('utf-8')).hexdigest()
    
    def _load_cache(self):
        """Load existing fitness cache from JSON file."""
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Validate cache structure
                if 'fitness_cache' in cache_data and 'cache_metadata' in cache_data:
                    # Check if fixed parameters match (cache validation)
                    cached_fixed_params = cache_data['cache_metadata'].get('fixed_params', {})
                    if cached_fixed_params == self.fixed_params:
                        # Load fitness cache
                        self.fitness_cache = {
                            k: v['fitness'] for k, v in cache_data['fitness_cache'].items()
                        }
                        
                        # Update cache stats
                        metadata = cache_data['cache_metadata']
                        self.cache_stats['total_evaluations'] = metadata.get('total_evaluations', 0)
                        
                        self.logger.info(f"Loaded {len(self.fitness_cache)} cached fitness evaluations")
                    else:
                        self.logger.warning("Cache invalidated due to fixed parameter mismatch")
                        self.fitness_cache = {}
                else:
                    self.logger.warning("Invalid cache file format, starting with empty cache")
                    self.fitness_cache = {}
            else:
                self.logger.info("No existing cache found, starting with empty cache")
                self.fitness_cache = {}
                
        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")
            self.fitness_cache = {}
    
    def _save_cache(self):
        """Save current fitness cache to JSON file."""
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            # Prepare cache data structure
            cache_data = {
                'cache_metadata': {
                    'created': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat(),
                    'total_evaluations': self.cache_stats['total_evaluations'],
                    'cache_hits': self.cache_stats['hits'],
                    'cache_misses': self.cache_stats['misses'],
                    'fixed_params': self.fixed_params
                },
                'fitness_cache': {
                    k: {
                        'fitness': v,
                        'timestamp': datetime.now().isoformat()
                    } for k, v in self.fitness_cache.items()
                }
            }
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = self.cache_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            # Atomic rename
            os.rename(temp_file, self.cache_file)
            
            self.logger.info(f"Cache saved: {len(self.fitness_cache)} entries")
            
        except Exception as e:
            self.logger.error(f"Error saving cache: {e}")
        
    def generate_individual(self) -> Dict[str, Any]:
        """
        Generate a random individual (hyperparameter configuration).
        
        Returns:
            Dictionary containing random hyperparameter values
        """
        individual = {}
        for param_name, param_space in self.search_space.items():
            individual[param_name] = random.choice(param_space)
        
        return individual
    
    def create_neurograph_config(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a complete NeuroGraph configuration from GA individual.
        
        Args:
            individual: Dictionary of hyperparameter values
            
        Returns:
            Complete NeuroGraph configuration dictionary
        """
        # Calculate derived parameters
        total_nodes = 200 + 10 + 790  # input + output + intermediate
        input_nodes = 200
        output_nodes = 10
        
        # Build complete configuration
        config = {
            'mode': 'modular',
            'device': 'cuda',
            
            # Architecture parameters
            'architecture': {
                'total_nodes': total_nodes,
                'input_nodes': input_nodes,
                'output_nodes': output_nodes,
                'vector_dim': individual['vector_dim'],
                'seed': 42
            },
            
            # Resolution parameters
            'resolution': {
                'phase_bins': individual['phase_bins'],
                'mag_bins': individual['mag_bins'],
                'resolution_increase': individual['phase_bins'] * individual['mag_bins'] // 64  # Relative to base
            },
            
            # Graph structure
            'graph_structure': {
                'cardinality': individual['cardinality']
            },
            
            # Training parameters
            'training': {
                'gradient_accumulation': {
                    'enabled': True,
                    'accumulation_steps': self.fixed_params['accumulation_steps'],
                    'lr_scaling': 'sqrt',
                    'buffer_size': 1500
                },
                'optimizer': {
                    'base_learning_rate': individual['learning_rate'],
                    'effective_learning_rate': individual['learning_rate'] * (self.fixed_params['accumulation_steps'] ** 0.5),
                    'num_epochs': 50,  # Will be dynamically calculated in multi-run evaluator
                    'warmup_epochs': individual['warmup_epochs'],
                    'batch_size': individual['batch_size']
                }
            },
            
            # Forward pass parameters
            'forward_pass': {
                'max_timesteps': 50,
                'decay_factor': individual['decay_factor'],
                'min_activation_strength': individual['min_activation_strength'],
                'use_radiation': True,
                'top_k_neighbors': 4
            },
            
            # Input processing
            'input_processing': {
                'adapter_type': 'linear_projection',
                'input_dim': 784,
                'normalization': 'layer_norm',
                'dropout': 0.1,
                'learnable': True
            },
            
            # Class encoding
            'class_encoding': {
                'type': 'orthogonal',
                'num_classes': 10,
                'encoding_dim': individual['vector_dim'],
                'orthogonality_threshold': individual['orthogonality_threshold']
            },
            
            # Loss function
            'loss_function': {
                'type': 'categorical_crossentropy',
                'temperature': 1.0,
                'label_smoothing': 0.0
            },
            
            # Batch evaluation
            'batch_evaluation': {
                'enabled': True,
                'batch_size': 16,
                'streaming': True
            },
            
            # Debugging
            'debugging': {
                'evaluation_samples': self.fixed_params['validation_samples'],
                'final_evaluation_samples': self.fixed_params['validation_samples'],
                'log_level': 'INFO'
            },
            
            # Paths (will be set dynamically)
            'paths': {
                'graph_path': None,  # Will be set in evaluate_fitness
                'training_curves_path': None,
                'checkpoint_path': None
            }
        }
        
        return config
    
    def evaluate_fitness(self, individual: Dict[str, Any]) -> float:
        """
        Evaluate fitness of an individual using multi-run stratified sampling with caching.
        
        Args:
            individual: Dictionary of hyperparameter values
            
        Returns:
            Average validation accuracy across multiple runs as fitness score (0.0 to 1.0)
        """
        # Generate cache key for this configuration
        cache_key = self._generate_cache_key(individual)
        
        # Check if fitness is already cached
        if cache_key in self.fitness_cache:
            cached_fitness = self.fitness_cache[cache_key]
            self.cache_stats['hits'] += 1
            
            # Log cache hit
            param_str = ", ".join([f"{k}={v}" for k, v in individual.items()])
            self.logger.info(f"Cache HIT for individual: {param_str}")
            self.logger.info(f"Cached fitness: {cached_fitness:.4f}")
            
            return cached_fitness
        
        # Cache miss - need to evaluate fitness through multi-run training
        self.cache_stats['misses'] += 1
        self.cache_stats['total_evaluations'] += 1
        
        try:
            # Log evaluation start
            param_str = ", ".join([f"{k}={v}" for k, v in individual.items()])
            self.logger.info(f"Cache MISS - Multi-run evaluation for individual: {param_str}")
            
            # Use multi-run fitness evaluator for robust evaluation
            mean_fitness = self.multi_run_evaluator.evaluate_candidate_fitness(individual)
            
            # Cache the result
            self.fitness_cache[cache_key] = mean_fitness
            
            self.logger.info(f"Individual mean fitness: {mean_fitness:.4f} (cached)")
            
            return mean_fitness
            
        except Exception as e:
            self.logger.error(f"Error in multi-run evaluation: {e}")
            # Return low fitness for failed evaluations
            return 0.0
    
    
    def uniform_crossover(self, parent1: Dict[str, Any], 
                         parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform uniform crossover between two parents.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Tuple of two offspring individuals
        """
        offspring1 = {}
        offspring2 = {}
        
        for param_name in self.search_space.keys():
            if random.random() < 0.5:
                # Take from parent1
                offspring1[param_name] = parent1[param_name]
                offspring2[param_name] = parent2[param_name]
            else:
                # Take from parent2
                offspring1[param_name] = parent2[param_name]
                offspring2[param_name] = parent1[param_name]
        
        return offspring1, offspring2
    
    def mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate an individual by randomly changing some parameters.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        for param_name, param_space in self.search_space.items():
            if random.random() < self.mutation_rate:
                # Mutate this parameter
                mutated[param_name] = random.choice(param_space)
        
        return mutated
    
    def select_top_k(self, population: List[Dict[str, Any]], 
                     fitness_scores: List[float], k: int) -> List[Dict[str, Any]]:
        """
        Select top-k individuals based on fitness scores.
        
        Args:
            population: List of individuals
            fitness_scores: Corresponding fitness scores
            k: Number of top individuals to select
            
        Returns:
            List of top-k individuals
        """
        # Create pairs of (individual, fitness) and sort by fitness
        paired = list(zip(population, fitness_scores))
        paired.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness (descending)
        
        # Return top-k individuals
        return [individual for individual, _ in paired[:k]]
    
    def genetic_hyperparam_search(self, config_input: Union[str, Dict], 
                                 population_size: int = 50, 
                                 top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Main genetic algorithm hyperparameter search function.
        
        Args:
            config_input: Base configuration (dict or YAML file path) - currently unused
            population_size: Size of population per generation
            top_k: Number of best configurations to return
            
        Returns:
            List of top-k best hyperparameter configurations with fitness scores
        """
        self.logger.info(f"Starting GA hyperparameter search:")
        self.logger.info(f"  Generations: {self.generations}")
        self.logger.info(f"  Population size: {population_size}")
        self.logger.info(f"  Top-k: {top_k}")
        self.logger.info(f"  Search space: {len(self.search_space)} parameters")
        
        # Initialize population
        self.logger.info("Initializing random population...")
        population = [self.generate_individual() for _ in range(population_size)]
        
        # Track best individuals across generations
        all_time_best = []
        generation_stats = []
        
        # Evolution loop
        for generation in range(self.generations):
            self.logger.info(f"\n=== Generation {generation + 1}/{self.generations} ===")
            
            # Evaluate fitness for all individuals
            self.logger.info("Evaluating population fitness...")
            fitness_scores = []
            
            for i, individual in enumerate(population):
                self.logger.info(f"Evaluating individual {i + 1}/{population_size}")
                fitness = self.evaluate_fitness(individual) ###
                fitness_scores.append(fitness)
            
            # Track statistics
            best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            worst_fitness = min(fitness_scores)
            
            # Find best individual of this generation
            best_idx = fitness_scores.index(best_fitness)
            best_individual = population[best_idx].copy()
            best_individual['fitness'] = best_fitness
            best_individual['generation'] = generation + 1
            
            # Update all-time best
            all_time_best.append(best_individual)
            
            # Log generation statistics
            self.logger.info(f"Generation {generation + 1} Results:")
            self.logger.info(f"  Best fitness: {best_fitness:.4f}")
            self.logger.info(f"  Average fitness: {avg_fitness:.4f}")
            self.logger.info(f"  Worst fitness: {worst_fitness:.4f}")
            self.logger.info(f"  Best config: {best_individual}")
            
            # Store generation statistics
            generation_stats.append({
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'worst_fitness': worst_fitness,
                'best_individual': best_individual
            })
            
            # Create next generation (except for last generation)
            if generation < self.generations - 1:
                self.logger.info("Creating next generation...")
                new_population = []
                
                # Elitism: Keep best individuals based on user-specified percentage
                elite_count = max(1, int(population_size * self.elite_percentage))
                elite_individuals = self.select_top_k(population, fitness_scores, elite_count)
                new_population.extend(elite_individuals)
                
                self.logger.info(f"Preserving {elite_count} elite individuals ({self.elite_percentage:.1%})")
                
                # Generate offspring through crossover and mutation using survivor-based selection
                # Select survivors for breeding (top performers only)
                survivor_count = max(1, int(population_size * self.elite_percentage))
                survivors = self.select_top_k(population, fitness_scores, survivor_count)
                
                self.logger.info(f"Using top {survivor_count} survivors for breeding ({self.elite_percentage:.1%})")
                
                while len(new_population) < population_size:
                    # Selection: Choose parents randomly from survivors only
                    parent1 = random.choice(survivors)
                    parent2 = random.choice(survivors)
                    
                    # Crossover
                    if random.random() < self.crossover_rate:
                        offspring1, offspring2 = self.uniform_crossover(parent1, parent2)
                    else:
                        offspring1, offspring2 = parent1.copy(), parent2.copy()
                    
                    # Mutation
                    offspring1 = self.mutate(offspring1)
                    offspring2 = self.mutate(offspring2)
                    
                    # Add to new population
                    new_population.extend([offspring1, offspring2])
                
                # Trim to exact population size
                population = new_population[:population_size]
        
        # Select top-k best individuals from all generations
        self.logger.info(f"\nSelecting top-{top_k} individuals from all generations...")
        all_fitness = [ind['fitness'] for ind in all_time_best]
        top_individuals = self.select_top_k(all_time_best, all_fitness, top_k)
        
        # Save fitness cache
        self._save_cache()
        
        # Log cache statistics
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        self.logger.info(f"\n=== Cache Statistics ===")
        self.logger.info(f"  Cache hits: {self.cache_stats['hits']}")
        self.logger.info(f"  Cache misses: {self.cache_stats['misses']}")
        self.logger.info(f"  Hit rate: {hit_rate:.1f}%")
        self.logger.info(f"  Total evaluations: {self.cache_stats['total_evaluations']}")
        self.logger.info(f"  Cached configurations: {len(self.fitness_cache)}")
        
        # Save results
        self.save_results(top_individuals, generation_stats)
        
        # Log final results
        self.logger.info(f"\n=== Final Results ===")
        for i, individual in enumerate(top_individuals):
            self.logger.info(f"Rank {i + 1}: Fitness = {individual['fitness']:.4f}, "
                           f"Generation = {individual['generation']}")
        
        return top_individuals
    
    def save_results(self, top_individuals: List[Dict[str, Any]], 
                     generation_stats: List[Dict[str, Any]]):
        """
        Save GA results to JSON files.
        
        Args:
            top_individuals: Top-k best individuals
            generation_stats: Statistics for each generation
        """
        results_dir = "results/genetic_algorithm"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save top individuals
        top_results_file = f"{results_dir}/top_individuals_{timestamp}.json"
        with open(top_results_file, 'w') as f:
            json.dump(top_individuals, f, indent=2, default=str)
        
        # Save generation statistics
        stats_file = f"{results_dir}/generation_stats_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(generation_stats, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to:")
        self.logger.info(f"  Top individuals: {top_results_file}")
        self.logger.info(f"  Generation stats: {stats_file}")


# Convenience function for direct usage
def genetic_hyperparam_search(config_input: Union[str, Dict], 
                             generations: int = 10, 
                             population_size: int = 50, 
                             top_k: int = 5,
                             crossover_rate: float = 0.3,
                             mutation_rate: float = 0.2,
                             elite_percentage: float = 0.5) -> List[Dict[str, Any]]:
    """
    Genetic Algorithm hyperparameter search for NeuroGraph.
    
    Args:
        config_input: Base configuration (dict or YAML file path) - currently unused
        generations: Number of generations to evolve
        population_size: Size of population per generation  
        top_k: Number of best configurations to return
        crossover_rate: Probability of crossover between parents (0.0 to 1.0)
        mutation_rate: Probability of mutation per gene (0.0 to 1.0)
        elite_percentage: Percentage of population to preserve as elite (0.0 to 1.0)
        
    Returns:
        List of top-k best hyperparameter configurations with fitness scores
    """
    tuner = GeneticHyperparameterTuner(
        generations=generations,
        elite_percentage=elite_percentage,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate
    )
    return tuner.genetic_hyperparam_search(config_input, population_size, top_k)


if __name__ == "__main__":
    # Example usage
    print("NeuroGraph Genetic Algorithm Hyperparameter Tuner")
    print("=" * 50)
    
    # Run GA with small population for testing
    results = genetic_hyperparam_search(
        config_input={},  # Not used currently
        generations=3,
        population_size=10,
        top_k=3
    )
    
    print(f"\nTop {len(results)} configurations found:")
    for i, config in enumerate(results):
        print(f"\nRank {i + 1}:")
        print(f"  Fitness: {config['fitness']:.4f}")
        print(f"  Generation: {config['generation']}")
        print(f"  Parameters: {config}")
