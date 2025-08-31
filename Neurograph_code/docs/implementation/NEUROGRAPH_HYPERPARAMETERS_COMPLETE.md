# NeuroGraph Complete Hyperparameter Reference

**Date**: January 28, 2025  
**Purpose**: Complete documentation of all hyperparameters used in the NeuroGraph discrete neural network system

## 🎯 Overview

This document provides a comprehensive reference of ALL hyperparameters used in the NeuroGraph system, categorized by whether they are optimized by the genetic algorithm or fixed in the implementation.

## 🧬 Genetic Algorithm Optimized Parameters (10 Parameters)

These hyperparameters are included in the GA search space and optimized through evolutionary search:

### 1. **Architecture Parameters**

| Parameter | Search Space | Type | Description | Impact |
|-----------|--------------|------|-------------|---------|
| `vector_dim` | [5, 8, 10] | Discrete | Dimension of phase-magnitude vectors | Higher dimensions allow more complex representations but increase computational cost |

### 2. **Resolution Parameters**

| Parameter | Search Space | Type | Description | Impact |
|-----------|--------------|------|-------------|---------|
| `phase_bins` | [16, 32, 64, 128] | Discrete | Number of phase discretization bins | Higher resolution improves signal precision but increases memory usage |
| `mag_bins` | [64, 128, 256, 512, 1024] | Discrete | Number of magnitude discretization bins | Higher resolution improves amplitude precision but increases memory usage |

### 3. **Graph Structure Parameters**

| Parameter | Search Space | Type | Description | Impact |
|-----------|--------------|------|-------------|---------|
| `cardinality` | [3, 4, 5, 6, 7, 8] | Discrete | Average number of connections per node | Higher cardinality increases information flow but may cause overfitting |

### 4. **Training Parameters**

| Parameter | Search Space | Type | Description | Impact |
|-----------|--------------|------|-------------|---------|
| `learning_rate` | [0.0001, 0.0005, 0.001, 0.005] | Discrete | Base learning rate for parameter updates | Higher rates speed training but may cause instability |
| `warmup_epochs` | [3, 5, 8, 10] | Discrete | Epochs before output node inclusion | Longer warmup allows better feature learning |
| `batch_size` | [3, 5, 8, 10] | Discrete | Number of samples per training batch | Larger batches provide more stable gradients |

### 5. **Forward Pass Parameters**

| Parameter | Search Space | Type | Description | Impact |
|-----------|--------------|------|-------------|---------|
| `decay_factor` | [0.9, 0.925, 0.95, 0.975] | Discrete | Activation decay rate per timestep | Higher values maintain activations longer |
| `min_activation_strength` | [0.01, 0.05, 0.1, 0.2, 0.5] | Discrete | Minimum threshold for signal propagation | Higher thresholds reduce noise but may block weak signals |

### 6. **Class Encoding Parameters**

| Parameter | Search Space | Type | Description | Impact |
|-----------|--------------|------|-------------|---------|
| `orthogonality_threshold` | [0.05, 0.1, 0.15, 0.2] | Discrete | Maximum allowed dot product between class encodings | Lower values enforce better class separation |

## 🔧 Fixed Hyperparameters (Not Optimized)

These hyperparameters are fixed in the implementation and not included in the GA search space:

### 1. **System Configuration**

| Parameter | Value | Type | Description | Rationale |
|-----------|-------|------|-------------|-----------|
| `mode` | 'modular' | String | NeuroGraph operation mode | Standard modular architecture |
| `device` | 'cuda' (with CPU fallback) | String | Computation device | GPU acceleration when available |
| `seed` | 42 | Integer | Random seed for reproducibility | Standard seed for consistent results |

### 2. **Architecture Configuration**

| Parameter | Value | Type | Description | Rationale |
|-----------|-------|------|-------------|-----------|
| `total_nodes` | 1000 | Integer | Total number of nodes in graph | Balanced complexity for MNIST |
| `input_nodes` | 200 | Integer | Number of input nodes | Sufficient for 784-dim MNIST projection |
| `output_nodes` | 10 | Integer | Number of output nodes | One per MNIST class |
| `intermediate_nodes` | 790 | Integer | Number of intermediate nodes | Calculated: total - input - output |

### 3. **Training Configuration**

| Parameter | Value | Type | Description | Rationale |
|-----------|-------|------|-------------|-----------|
| `num_epochs` | 50 | Integer | Training epochs per GA evaluation | Balance between accuracy and speed |
| `accumulation_steps` | 8 | Integer | Gradient accumulation steps | Effective batch size scaling |
| `lr_scaling` | 'sqrt' | String | Learning rate scaling method | √N scaling for gradient accumulation |
| `buffer_size` | 1500 | Integer | Gradient accumulator buffer size | Sufficient for node count |
| `validation_samples` | 500 | Integer | Samples for fitness evaluation | Balance between accuracy and speed |

### 4. **Forward Pass Configuration**

| Parameter | Value | Type | Description | Rationale |
|-----------|-------|------|-------------|-----------|
| `max_timesteps` | 50 | Integer | Maximum forward propagation steps | **Updated from 6 to 50 as requested** |
| `use_radiation` | True | Boolean | Enable radiation neighbor selection | Core NeuroGraph feature |
| `top_k_neighbors` | 4 | Integer | Number of radiation neighbors | Balanced connectivity |

### 5. **Input Processing Configuration**

| Parameter | Value | Type | Description | Rationale |
|-----------|-------|------|-------------|-----------|
| `adapter_type` | 'linear_projection' | String | Input adaptation method | Learnable linear transformation |
| `input_dim` | 784 | Integer | MNIST input dimension | 28×28 pixel images |
| `normalization` | 'layer_norm' | String | Input normalization method | Stable training |
| `dropout` | 0.1 | Float | Input dropout rate | Regularization |
| `learnable` | True | Boolean | Enable learnable input adapter | Adaptive input processing |

### 6. **Class Encoding Configuration**

| Parameter | Value | Type | Description | Rationale |
|-----------|-------|------|-------------|-----------|
| `encoding_type` | 'orthogonal' | String | Class encoding method | Orthogonal class separation |
| `num_classes` | 10 | Integer | Number of output classes | MNIST digit classes (0-9) |
| `encoding_dim` | `vector_dim` | Integer | Encoding dimension | Matches vector dimension |

### 7. **Loss Function Configuration**

| Parameter | Value | Type | Description | Rationale |
|-----------|-------|------|-------------|-----------|
| `loss_type` | 'categorical_crossentropy' | String | Loss function type | Standard classification loss |
| `temperature` | 1.0 | Float | Softmax temperature | No temperature scaling |
| `label_smoothing` | 0.0 | Float | Label smoothing factor | No label smoothing |

### 8. **Batch Evaluation Configuration**

| Parameter | Value | Type | Description | Rationale |
|-----------|-------|------|-------------|-----------|
| `batch_evaluation_enabled` | True | Boolean | Enable batch evaluation optimization | 5-10x speedup |
| `batch_evaluation_size` | 16 | Integer | Batch size for evaluation | Memory/speed balance |
| `streaming` | True | Boolean | Enable streaming evaluation | Memory efficiency |

### 9. **Debugging Configuration**

| Parameter | Value | Type | Description | Rationale |
|-----------|-------|------|-------------|-----------|
| `evaluation_samples` | 500 | Integer | Samples for standard evaluation | Same as validation_samples |
| `final_evaluation_samples` | 500 | Integer | Samples for final evaluation | Consistent evaluation |
| `log_level` | 'INFO' | String | Logging verbosity level | Informative but not verbose |

## 🧬 Genetic Algorithm Parameters

These parameters control the GA optimization process itself:

### User-Configurable GA Parameters

| Parameter | Default | Example | Type | Description | Impact |
|-----------|---------|---------|------|-------------|---------|
| `crossover_rate` | 0.8 | **0.3** | Float | Probability of crossover between parents | Higher rates increase exploration |
| `mutation_rate` | 0.1 | **0.2** | Float | Probability of mutation per gene | Higher rates increase diversity |
| `elite_percentage` | 0.1 | **0.5** | Float | Percentage of population preserved as elite | Higher values increase exploitation |

### Fixed GA Parameters

| Parameter | Value | Type | Description | Rationale |
|-----------|-------|------|-------------|-----------|
| `tournament_size` | 3 | Integer | Number of individuals in tournament selection | Balanced selection pressure |
| `generations` | 10 | Integer | Number of evolution generations (user-configurable) | Default for reasonable runtime |
| `population_size` | 50 | Integer | Population size per generation (user-configurable) | Balance between diversity and speed |
| `top_k` | 5 | Integer | Number of best configs to return (user-configurable) | Sufficient for analysis |

## 📊 Hyperparameter Impact Analysis

### High Impact Parameters (Major Performance Effect)
1. **`learning_rate`** - Directly affects convergence speed and stability
2. **`vector_dim`** - Controls model capacity and computational cost
3. **`phase_bins` × `mag_bins`** - Determines signal resolution and memory usage
4. **`cardinality`** - Controls information flow and model connectivity

### Medium Impact Parameters (Moderate Performance Effect)
5. **`decay_factor`** - Affects temporal dynamics of signal propagation
6. **`min_activation_strength`** - Controls signal-to-noise ratio
7. **`orthogonality_threshold`** - Affects class separation quality
8. **`warmup_epochs`** - Influences early training dynamics

### Lower Impact Parameters (Fine-tuning Effect)
9. **`batch_size`** - Affects gradient stability (limited by small values)
10. **GA Parameters** - Control optimization process efficiency

## 🎯 Optimization Strategy

### Search Space Design Rationale

1. **Discrete Choices**: All parameters use discrete search spaces for:
   - Computational efficiency (no continuous optimization)
   - Clear interpretability of results
   - Compatibility with NeuroGraph's discrete nature

2. **Balanced Ranges**: Search spaces cover:
   - Conservative values (safe, proven settings)
   - Moderate values (balanced performance/cost)
   - Aggressive values (high performance, high cost)

3. **Interdependent Parameters**: Some parameters interact:
   - `vector_dim` affects `encoding_dim` automatically
   - `phase_bins` × `mag_bins` determines total resolution
   - `learning_rate` scales with `accumulation_steps`

### Expected Optimization Patterns

1. **Early Generations**: Explore diverse parameter combinations
2. **Middle Generations**: Converge toward high-performing regions
3. **Late Generations**: Fine-tune best configurations

### Performance Expectations

- **Training Time**: ~6 minutes per individual (50 epochs)
- **Memory Usage**: 2-4GB per training run
- **Convergence**: Typically within 5-10 generations
- **Improvement**: 10-30% accuracy gain over random search

## 🔍 Usage Examples

### Basic Usage with Defaults
```python
from genetic_hyperparameter_tuner import genetic_hyperparam_search

results = genetic_hyperparam_search(
    config_input={},
    generations=10,
    population_size=50,
    top_k=5
)
```

### Custom GA Parameters (As Requested)
```python
results = genetic_hyperparam_search(
    config_input={},
    generations=10,
    population_size=50,
    top_k=5,
    crossover_rate=0.3,      # Example value
    mutation_rate=0.2,       # Example value  
    elite_percentage=0.5     # Example value (50%)
)
```

### Production Optimization
```python
results = genetic_hyperparam_search(
    config_input={},
    generations=20,          # More generations
    population_size=100,     # Larger population
    top_k=10,               # More results
    crossover_rate=0.7,     # Higher exploration
    mutation_rate=0.15,     # Moderate diversity
    elite_percentage=0.2    # Moderate elitism
)
```

## 📈 Performance Monitoring

### Key Metrics to Track
1. **Fitness Evolution**: Best/average/worst fitness per generation
2. **Parameter Convergence**: Which parameters stabilize first
3. **Diversity Metrics**: Population diversity over generations
4. **Resource Usage**: Training time and memory consumption

### Expected Results
- **Baseline Random**: ~15-20% accuracy
- **Optimized GA**: ~25-35% accuracy
- **Best Configurations**: >30% accuracy possible

---

**Note**: This documentation reflects the current NeuroGraph implementation with max_timesteps=50 and user-configurable GA parameters as requested. All hyperparameters are documented for complete transparency and reproducibility.
