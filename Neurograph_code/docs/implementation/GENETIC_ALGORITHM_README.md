# NeuroGraph Genetic Algorithm Hyperparameter Tuner

A comprehensive genetic algorithm implementation for optimizing NeuroGraph discrete neural network hyperparameters through evolutionary search with actual training runs.

## 🧬 Overview

This genetic algorithm tuner optimizes 10 key hyperparameters of the NeuroGraph discrete neural network by:
- Running actual 50-epoch training sessions for fitness evaluation
- Using validation accuracy (500 samples) as the fitness metric
- Evolving populations through tournament selection, uniform crossover, and mutation
- Tracking and saving the best configurations across generations

## 🎯 Optimized Hyperparameters

| Parameter | Search Space | Type | Description |
|-----------|--------------|------|-------------|
| `vector_dim` | [5, 8, 10] | Discrete | Dimension of phase-magnitude vectors |
| `phase_bins` | [16, 32, 64, 128] | Discrete | Number of phase discretization bins |
| `mag_bins` | [64, 128, 256, 512, 1024] | Discrete | Number of magnitude discretization bins |
| `cardinality` | [3, 4, 5, 6, 7, 8] | Discrete | Graph connectivity (connections per node) |
| `learning_rate` | [0.0001, 0.0005, 0.001, 0.005] | Discrete | Base learning rate for training |
| `decay_factor` | [0.9, 0.925, 0.95, 0.975] | Discrete | Activation decay factor |
| `orthogonality_threshold` | [0.05, 0.1, 0.15, 0.2] | Discrete | Class encoding separation threshold |
| `warmup_epochs` | [3, 5, 8, 10] | Discrete | Epochs before output node inclusion |
| `min_activation_strength` | [0.01, 0.05, 0.1, 0.2, 0.5] | Discrete | Minimum signal propagation threshold |
| `batch_size` | [3, 5, 8, 10] | Discrete | Training batch size |

## 🚀 Quick Start

### Basic Usage

```python
from genetic_hyperparameter_tuner import genetic_hyperparam_search

# Run genetic algorithm with default settings
best_configs = genetic_hyperparam_search(
    config_input={},        # Not used - configs generated from scratch
    generations=10,         # Number of generations
    population_size=50,     # Population size per generation
    top_k=5                # Number of best configs to return
)

# Display results
for i, config in enumerate(best_configs):
    print(f"Rank {i+1}: Fitness = {config['fitness']:.4f}")
    print(f"Parameters: {config}")
```

### Using the Example Script

```bash
# Run with interactive prompts
python run_genetic_tuning.py

# Or modify the script for custom parameters
```

### Testing the Implementation

```bash
# Run comprehensive tests
python test_genetic_tuner.py

# Test individual components without full training
```

## 📊 Algorithm Details

### Genetic Algorithm Components

#### 1. **Population Initialization**
- Random sampling from discrete parameter spaces
- Ensures valid parameter combinations
- Default population size: 50 individuals

#### 2. **Fitness Evaluation**
- **Training**: 50 epochs of actual NeuroGraph training
- **Validation**: Accuracy on 500 samples
- **Robustness**: Error handling for failed training runs
- **Cleanup**: Automatic temporary file management

#### 3. **Selection: Tournament Selection**
- Tournament size: 3 individuals
- Selects fittest individual from random tournament
- Maintains selection pressure while preserving diversity

#### 4. **Crossover: Uniform Crossover**
- 50% probability per gene to inherit from each parent
- Crossover rate: 80%
- Preserves building blocks from both parents

#### 5. **Mutation: Random Replacement**
- 10% probability per parameter to mutate
- Random selection from parameter's search space
- Maintains population diversity

#### 6. **Elitism**
- Top 10% of individuals preserved each generation
- Prevents loss of best solutions
- Accelerates convergence

### Performance Characteristics

#### Runtime Estimation
- **Per Individual**: ~6 minutes (50-epoch training)
- **Per Generation**: `population_size × 6` minutes
- **Total Runtime**: `generations × population_size × 6` minutes

#### Example Runtimes
| Configuration | Estimated Time |
|---------------|----------------|
| 5 gen × 20 pop | ~10 hours |
| 10 gen × 50 pop | ~50 hours |
| 3 gen × 10 pop | ~3 hours |

#### Resource Requirements
- **GPU**: Recommended (CUDA-compatible)
- **CPU Fallback**: Automatic detection
- **Memory**: ~2-4GB per training run
- **Storage**: Temporary files cleaned automatically

## 📁 File Structure

```
genetic_hyperparameter_tuner.py    # Main GA implementation
run_genetic_tuning.py              # Example usage script
test_genetic_tuner.py              # Test suite
GENETIC_ALGORITHM_README.md        # This documentation

# Generated during runs:
logs/genetic_algorithm/            # Training logs and GA progress
results/genetic_algorithm/         # Best configurations and statistics
```

## 🔧 Configuration Options

### GA Parameters

```python
class GeneticHyperparameterTuner:
    def __init__(self):
        # GA parameters (modifiable)
        self.tournament_size = 3        # Tournament selection size
        self.mutation_rate = 0.1        # Mutation probability per gene
        self.crossover_rate = 0.8       # Crossover probability
        
        # Fixed training parameters
        self.fixed_params = {
            'accumulation_steps': 8,    # Gradient accumulation
            'num_epochs': 50,           # Training epochs per evaluation
            'validation_samples': 500   # Samples for fitness evaluation
        }
```

### Customizing Search Spaces

To modify hyperparameter search spaces, edit the `search_space` dictionary:

```python
self.search_space = {
    'vector_dim': [5, 8, 10, 12],  # Add more options
    'learning_rate': [0.0001, 0.001, 0.01],  # Modify ranges
    # ... other parameters
}
```

## 📈 Results and Analysis

### Output Files

#### 1. **Top Individuals** (`top_individuals_YYYYMMDD_HHMMSS.json`)
```json
[
  {
    "vector_dim": 8,
    "phase_bins": 64,
    "mag_bins": 512,
    "cardinality": 5,
    "learning_rate": 0.001,
    "decay_factor": 0.925,
    "orthogonality_threshold": 0.1,
    "warmup_epochs": 5,
    "min_activation_strength": 0.05,
    "batch_size": 5,
    "fitness": 0.234,
    "generation": 8
  }
]
```

#### 2. **Generation Statistics** (`generation_stats_YYYYMMDD_HHMMSS.json`)
```json
[
  {
    "generation": 1,
    "best_fitness": 0.180,
    "avg_fitness": 0.145,
    "worst_fitness": 0.098,
    "best_individual": { ... }
  }
]
```

### Analyzing Results

```python
import json

# Load results
with open('results/genetic_algorithm/top_individuals_20250128_153000.json', 'r') as f:
    results = json.load(f)

# Analyze best configuration
best_config = results[0]
print(f"Best fitness: {best_config['fitness']:.4f}")
print(f"Best parameters: {best_config}")

# Use best config for production training
# (Create YAML config from best parameters)
```

## 🧪 Testing and Validation

### Test Suite Components

1. **Individual Generation Test**: Verifies random parameter generation
2. **Config Creation Test**: Validates NeuroGraph config generation
3. **Genetic Operators Test**: Tests crossover and mutation
4. **Selection Test**: Validates tournament and top-k selection
5. **Mini GA Test**: Full integration test with actual training

### Running Tests

```bash
# Run all tests (except mini GA)
python test_genetic_tuner.py

# For mini GA test, confirm when prompted
# (Takes ~12 minutes for 2 individuals)
```

## 🔍 Troubleshooting

### Common Issues

#### 1. **CUDA Out of Memory**
```python
# Reduce batch size or use CPU fallback
config['device'] = 'cpu'
```

#### 2. **Training Failures**
- Check NeuroGraph dependencies
- Verify MNIST dataset availability
- Monitor logs for specific errors

#### 3. **Slow Performance**
- Use smaller population sizes for testing
- Reduce number of generations
- Ensure GPU is being utilized

#### 4. **Import Errors**
```python
# Ensure NeuroGraph modules are in path
import sys
sys.path.append('/path/to/neurograph')
```

### Debugging Tips

1. **Enable Verbose Logging**:
   ```python
   import logging
   logging.getLogger().setLevel(logging.DEBUG)
   ```

2. **Monitor Resource Usage**:
   ```bash
   # GPU usage
   nvidia-smi -l 1
   
   # CPU/Memory usage
   htop
   ```

3. **Check Temporary Files**:
   - Temporary configs created in `/tmp/ga_eval_*`
   - Automatically cleaned up after evaluation

## 🚀 Advanced Usage

### Custom Fitness Functions

```python
class CustomGeneticTuner(GeneticHyperparameterTuner):
    def evaluate_fitness(self, individual):
        # Custom fitness evaluation
        # Could combine accuracy, training time, model size, etc.
        accuracy = super().evaluate_fitness(individual)
        
        # Example: Penalize large models
        model_size_penalty = individual['vector_dim'] * 0.01
        
        return accuracy - model_size_penalty
```

### Parallel Evaluation

```python
# Future enhancement: Parallel fitness evaluation
from concurrent.futures import ProcessPoolExecutor

def parallel_evaluate_population(self, population):
    with ProcessPoolExecutor(max_workers=4) as executor:
        fitness_scores = list(executor.map(self.evaluate_fitness, population))
    return fitness_scores
```

### Multi-Objective Optimization

```python
# Future enhancement: Multi-objective GA
# Optimize for accuracy AND training speed
def evaluate_multi_objective(self, individual):
    accuracy = self.evaluate_fitness(individual)
    training_time = self.measure_training_time(individual)
    
    return {
        'accuracy': accuracy,
        'speed': 1.0 / training_time,  # Higher is better
        'combined': accuracy * 0.8 + (1.0 / training_time) * 0.2
    }
```

## 📚 References

### Genetic Algorithm Theory
- Holland, J.H. (1992). "Adaptation in Natural and Artificial Systems"
- Goldberg, D.E. (1989). "Genetic Algorithms in Search, Optimization, and Machine Learning"

### NeuroGraph Architecture
- See NeuroGraph documentation for discrete neural network details
- Phase-magnitude signal processing principles
- Graph-based neural computation

### Hyperparameter Optimization
- Bergstra, J. & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization"
- Snoek, J. et al. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms"

## 🤝 Contributing

### Adding New Hyperparameters

1. **Update Search Space**:
   ```python
   self.search_space['new_param'] = [value1, value2, value3]
   ```

2. **Update Config Generation**:
   ```python
   config['new_section']['new_param'] = individual['new_param']
   ```

3. **Test Integration**:
   ```bash
   python test_genetic_tuner.py
   ```

### Improving GA Components

- **Selection**: Implement rank-based or roulette wheel selection
- **Crossover**: Add multi-point or arithmetic crossover
- **Mutation**: Implement adaptive mutation rates
- **Population**: Add diversity measures and niching

## 📄 License

This genetic algorithm implementation is part of the NeuroGraph project and follows the same licensing terms.

---

**NeuroGraph Genetic Algorithm Hyperparameter Tuner** - Evolving discrete neural networks through intelligent search.
