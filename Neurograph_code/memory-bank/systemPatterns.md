# NeuroGraph System Patterns

## Architecture Overview
NeuroGraph implements a layered architecture with clear separation of concerns:

```
Input Layer (MNIST + PCA) → Graph Processing → Output Layer (Classification)
     ↓                           ↓                        ↓
Input Adapters            Core Engine              Output Adapters
     ↓                           ↓                        ↓
Phase-Mag Encoding       Hybrid Propagation       Cosine Similarity
```

## Core Design Patterns

### 1. Discrete Signal Processing Pattern
**Implementation**: PhaseCell + ExtendedLookupTableModule
```python
# Signal computation using discrete indices
phase_out = (ctx_phase_idx + self_phase_idx) % N
mag_out = (ctx_mag_idx + self_mag_idx) % M
signal = cos_lookup[phase_out] * exp_lookup[mag_out]
```
**Purpose**: Replace continuous activations with interpretable discrete representations

### 2. Hybrid Propagation Pattern
**Components**: 
- **Conduction**: Static DAG connections from graph topology
- **Radiation**: Dynamic top-K neighbors based on phase alignment

**Implementation**:
```python
# Static connections
static_targets = graph_df[graph_df["node_id"] == source]["input_connections"]

# Dynamic connections via phase alignment
dynamic_targets = get_radiation_neighbors(source, ctx_phase, node_store, top_k)

# Combined propagation
all_targets = static_targets + dynamic_targets
```

### 3. Temporal Activation Pattern
**Implementation**: ActivationTable with decay mechanism
- Signals decay over timesteps with configurable decay_factor (0.925)
- Minimum activation strength threshold filters weak signals
- Fixed timestep limit (6) prevents infinite propagation

### 4. Manual Gradient Pattern
**Implementation**: Custom backward pass without autograd
```python
# Forward pass returns gradients
_, _, signal, strength, grad_phase, grad_mag = phase_cell(...)

# Manual parameter updates
new_phase = (self_phase - lr * grad_phase) % phase_bins
new_mag = (self_mag - lr * grad_mag) % mag_bins
```

## Component Relationships

### Core Engine Components
1. **NodeStore**: Learnable phase-magnitude parameters per node
2. **PhaseCell**: Signal computation and gradient calculation
3. **PropagationEngine**: Orchestrates hybrid signal flow
4. **ActivationTable**: Tracks active signals with temporal decay
5. **ForwardEngine**: Multi-timestep propagation coordinator

### Data Flow Architecture
```
MNIST Image → PCA → Phase-Mag Encoding → Input Context
     ↓
Graph Propagation (T=6 timesteps)
     ↓
Output Activation → Signal Vectors → Cosine Similarity → Prediction
```

### Training Architecture
```
Forward Pass → Loss Computation → Manual Backward Pass → Parameter Update
     ↓              ↓                    ↓                    ↓
Signal Flow    MSE vs Target      Custom Gradients     Index Updates
```

## Key Architectural Decisions

### 1. Discrete vs Continuous
- **Decision**: Use discrete phase-magnitude indices instead of continuous values
- **Rationale**: Enables interpretable signal processing and biological plausibility
- **Implementation**: Lookup tables for cos/exp transformations

### 2. Static + Dynamic Connectivity
- **Decision**: Hybrid propagation combining fixed topology with dynamic routing
- **Rationale**: Balances structural inductive bias with adaptive information flow
- **Implementation**: DAG connections + phase-aligned neighbor selection

### 3. Manual Gradient Computation
- **Decision**: Implement custom backward pass without PyTorch autograd
- **Rationale**: Full control over learning dynamics and gradient computation
- **Implementation**: PhaseCell returns analytical gradients for discrete updates

### 4. Batch Processing Strategy
- **Decision**: Merge multiple samples into single forward pass
- **Rationale**: Efficient computation while maintaining individual loss calculation
- **Implementation**: Dictionary merging of input contexts

## Scalability Patterns

### Current Limitations
- **Node Count**: Limited to ~50 nodes due to manual gradient computation
- **Graph Size**: Static topology requires pre-generation
- **Memory**: Lookup tables scale with phase_bins × mag_bins
- **Computation**: Radiation neighbor selection is O(N²) brute-force

### Potential Optimizations
- **Sparse Representations**: Only store active node parameters
- **Efficient Neighbor Search**: Use approximate nearest neighbor algorithms
- **Parallel Processing**: Vectorize propagation across multiple nodes
- **Dynamic Graph Generation**: Runtime topology modification

## Configuration Patterns

### Hyperparameter Relationships
- **Graph Structure**: total_nodes = input_nodes + output_nodes + intermediate_nodes
- **Signal Resolution**: Higher phase_bins/mag_bins → finer signal granularity
- **Connectivity**: cardinality limits static connections per node
- **Propagation**: max_timesteps × decay_factor controls signal lifetime
- **Learning**: warmup_epochs determines output node inclusion strategy

### Modular Design
- **Adapters**: Swappable input/output interfaces for different datasets
- **Encodings**: Configurable target vector generation strategies
- **Propagation**: Toggle between pure static or hybrid connectivity
- **Loss Functions**: Modular loss computation for different objectives

## Hyperparameter Optimization Patterns

### Enhanced Genetic Algorithm Pattern
**Implementation**: Stratified sampling with multi-run evaluation
```python
# Stratified data sampling
training_samples = get_stratified_samples(run_id, samples_per_class=50)
test_samples = get_fixed_test_set()  # Same for all evaluations

# Multi-run fitness evaluation
fitness_scores = []
for run_id in range(num_runs):
    fitness = evaluate_single_run(individual, run_id)
    fitness_scores.append(fitness)
mean_fitness = np.mean(fitness_scores)
```

### Survivor-Based Selection Pattern
**Implementation**: Deterministic top-k selection replaces tournament selection
```python
# Elite selection based on fitness ranking
elite_count = int(population_size * elite_percentage)
survivors = select_top_k(population, fitness_scores, elite_count)

# Breeding from survivors only
while len(new_population) < population_size:
    parent1 = random.choice(survivors)
    parent2 = random.choice(survivors)
    offspring = crossover_and_mutate(parent1, parent2)
    new_population.append(offspring)
```

### Variance Reduction Pattern
**Components**:
- **Stratified Sampling**: Balanced class representation (50 samples per class)
- **Fixed Test Set**: Consistent evaluation across all candidates
- **Multiple Runs**: 5 independent training runs per candidate
- **Dynamic Epochs**: epochs = total_samples ÷ batch_size for consistent exposure

### Enhanced Caching Pattern
**Implementation**: Multi-run aware caching with validation
```python
# Cache key includes multi-run parameters
cache_key = hash({
    'hyperparams': sorted(individual.items()),
    'fixed_params': sorted(fixed_params.items()),
    'num_runs': num_runs,
    'stratified_config': stratified_config
})

# Cache validation ensures compatibility
if cached_fixed_params == current_fixed_params:
    return cached_fitness
else:
    invalidate_cache()
```

## Genetic Algorithm Architecture

### Data Management Layer
```
StratifiedDataManager → CustomDatasetAdapter → NeuroGraph Training
        ↓                      ↓                      ↓
Class-balanced sampling    Dataset interface    Actual training
```

### Evaluation Architecture
```
Individual → MultiRunEvaluator → [Run1, Run2, ..., Run5] → Mean Fitness
     ↓              ↓                      ↓                    ↓
Hyperparams    Stratified Data      Independent Training    Variance Reduction
```

### Selection Architecture
```
Population → Fitness Evaluation → Survivor Selection → Breeding → New Population
     ↓              ↓                    ↓               ↓           ↓
Random Init    Multi-run Eval      Top-k Selection   Crossover   Next Generation
```

## Key GA Design Decisions

### 1. Stratified vs Random Sampling
- **Decision**: Use stratified sampling with fixed class distribution
- **Rationale**: Eliminates class imbalance bias in fitness evaluation
- **Implementation**: 50 samples per class for both training and testing

### 2. Multi-Run vs Single-Run Evaluation
- **Decision**: 5 independent runs per candidate with different training data
- **Rationale**: Reduces variance from lucky/unlucky data sampling
- **Implementation**: Average fitness across successful runs

### 3. Survivor-Based vs Tournament Selection
- **Decision**: Deterministic top-k selection based on elite_percentage
- **Rationale**: More predictable selection pressure and gene preservation
- **Implementation**: Sort by fitness, select top performers for breeding

### 4. Enhanced vs Basic Caching
- **Decision**: Multi-run aware caching with parameter validation
- **Rationale**: Critical for performance with 5x longer evaluations
- **Implementation**: Include multi-run config in cache keys

## GA Performance Characteristics

### Evaluation Complexity
- **Time Complexity**: O(population_size × num_runs × training_time)
- **Space Complexity**: O(cache_size + concurrent_evaluations)
- **Cache Dependency**: 25-33% hit rate critical for acceptable performance

### Variance Reduction Metrics
- **Coefficient of Variation**: Typically 0.1-0.3 (10-30% relative std dev)
- **Fitness Range**: Reduced spread between min/max per candidate
- **Ranking Stability**: More consistent candidate ordering across GA runs

### Scalability Considerations
- **Population Size**: Linear scaling with evaluation time
- **Number of Runs**: Linear scaling with evaluation time
- **Cache Size**: Logarithmic impact on performance
- **Parallel Evaluation**: Future optimization opportunity
