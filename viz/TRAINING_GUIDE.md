# Neurograph Training Guide

This guide explains how to use the Neurograph training framework to train networks on various tasks and configurations.

## Quick Start

```bash
# Start the visualization server (optional, for UI)
cd viz && uvicorn server:app --port 8765

# List available configurations
python train.py --list

# Run a training experiment
python train.py --config xor_basic

# Run with custom epochs
python train.py --config xor_basic -e 200

# Compare configurations
python train.py --compare with_radiation without_radiation
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Framework                        │
├─────────────────────────────────────────────────────────────┤
│  train.py (CLI)                                             │
│    ↓                                                        │
│  training_framework.py (Core)                               │
│    ↓                                                        │
│  training_configs.yaml (Configurations)                     │
│    ↓                                                        │
│  manager.py (Network Engine)                                │
└─────────────────────────────────────────────────────────────┘
```

## Training Modes

### Direct Mode (Default)
Training runs directly in Python without a server. Fastest option.

```bash
python train.py --config xor_basic --direct
```

### HTTP Mode
Training communicates with a running server. Useful for live visualization.

```bash
# Terminal 1: Start server
uvicorn server:app --port 8765

# Terminal 2: Run training
python train.py --config xor_basic --http --url http://localhost:8765
```

## Configuration Files

### Structure

Configurations are defined in `training_configs.yaml`:

```yaml
experiment_name:
  description: "Human-readable description"
  network:
    num_input: 2
    num_input_connected: 4
    num_middle: 6
    num_output: 1
    vector_dim: 8
    beam_width: 10
    radiation_k: 3
    use_radiation: true
  training:
    epochs: 100
    learning_rate: 0.05
    early_stopping_patience: 15
  dataset:
    type: pattern
    patterns:
      - [0, 0]
      - [0, 1]
      - [1, 0]
      - [1, 1]
    targets: [0, 3.14159, 3.14159, 0]
```

### Network Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_input` | Number of input nodes | 2 |
| `num_input_connected` | Nodes connected to inputs | 4 |
| `num_middle` | Middle layer nodes (isolated) | 8 |
| `num_output` | Output nodes | 2 |
| `vector_dim` | Dimension of phase/magnitude vectors | 16 |
| `beam_width` | Max active nodes (None = unlimited) | None |
| `radiation_k` | Number of radiation neighbors | 3 |
| `use_radiation` | Enable radiation mechanism | true |
| `temporal_decay` | Energy decay per timestep | 0.4 |
| `conductance_efficiency` | Conductance energy efficiency | 0.9 |
| `radiation_efficiency` | Radiation energy efficiency | 0.95 |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `epochs` | Maximum training epochs | 100 |
| `learning_rate` | Gradient update scale | 0.05 |
| `early_stopping_patience` | Epochs without improvement before stopping | 10 |

### Dataset Types

#### Pattern Dataset
Fixed input-output mappings:

```yaml
dataset:
  type: pattern
  patterns:
    - [0, 0]
    - [0, 1]
    - [1, 0]
    - [1, 1]
  targets: [0, 3.14159, 3.14159, 0]  # Phase targets
  noise_level: 0.0  # Optional noise
```

#### Temporal Dataset
Sequence prediction:

```yaml
dataset:
  type: temporal
  sequence_length: 5
  prediction_horizon: 1
```

#### Sine Dataset
Function approximation:

```yaml
dataset:
  type: sine
```

#### Classification Dataset
Multi-class classification:

```yaml
dataset:
  type: classification
  num_classes: 3
```

## CLI Reference

### Basic Commands

```bash
# List all configurations
python train.py --list

# Run single config
python train.py --config <name>

# Run with custom epochs
python train.py --config <name> -e 200

# Run quietly (minimal output)
python train.py --config <name> -q
```

### Comparison

```bash
# Compare multiple configs
python train.py --compare config1 config2 config3

# Run all configs in a category
python train.py --category temporal

# Run all preset experiments
python train.py --presets
```

### Export Results

```bash
# Export to JSON
python train.py --config xor_basic --export results.json

# Export to CSV
python train.py --config xor_basic --export results.csv --format csv
```

### Custom Configurations

```bash
# Use custom YAML file
python train.py --custom my_experiments.yaml --config my_experiment
```

## Programmatic Usage

```python
from training_framework import (
    NeurographTrainer,
    ExperimentConfig,
    DatasetConfig,
    get_preset
)

# Create trainer
trainer = NeurographTrainer(use_direct=True)

# Use preset experiment
experiment = get_preset('xor')
result = trainer.train(experiment, epochs=100, verbose=True)

print(f"Final loss: {result.final_loss}")
print(f"Training time: {result.training_time:.2f}s")

# Custom experiment
custom_exp = ExperimentConfig(
    name='My Experiment',
    network={
        'num_input': 3,
        'num_input_connected': 6,
        'num_middle': 10,
        'num_output': 2,
        'vector_dim': 16
    },
    training={
        'epochs': 100,
        'learning_rate': 0.03
    },
    dataset=DatasetConfig(
        type='pattern',
        patterns=[[0,0,0], [1,0,0], [0,1,0], [1,1,0]],
        targets=[0, 1.57, 3.14, 4.71]
    )
)

result = trainer.train(custom_exp)
```

## Understanding Results

### Metrics

- **Loss**: Average phase error between output and target
- **Avg Gradient**: Mean gradient magnitude (learning signal)
- **Active Nodes**: Number of nodes with activation above threshold
- **Converged**: Whether early stopping triggered

### Interpreting Loss

| Loss Value | Interpretation |
|------------|----------------|
| 0.0 | Perfect match |
| 0.5 | Very good |
| 1.0 | Moderate error |
| 2.0 | Poor fit |
| 3.14 | Maximum error (π) |

### Tips for Training

1. **Start small**: Begin with `small_network` preset to verify setup
2. **Increase complexity**: Gradually add layers and dimensions
3. **Monitor gradients**: Zero gradients indicate learning stagnation
4. **Tune learning rate**: Too high = oscillation, too low = slow convergence
5. **Use beam width**: Limits computational cost for large networks

## Predefined Experiments

### Pattern Learning
- `xor_basic` - Classic XOR benchmark
- `xor_deep` - Deeper XOR network
- `and_gate` - AND gate pattern
- `or_gate` - OR gate pattern

### Function Approximation
- `sine_simple` - Basic sine approximation
- `sine_high_precision` - High-precision sine

### Temporal
- `temporal_short` - 3-step sequences
- `temporal_medium` - 5-step sequences
- `temporal_long` - 10-step sequences

### Classification
- `binary_classification` - 2 classes
- `multiclass_3` - 3 classes
- `multiclass_5` - 5 classes

### Architecture Comparison
- `small_network` - Minimal network
- `medium_network` - Balanced network
- `large_network` - Complex network

### Ablation Studies
- `with_radiation` - Radiation enabled
- `without_radiation` - Conductance only
- `narrow_beam` - Aggressive pruning
- `wide_beam` - Minimal pruning
- `high_decay` - Fast energy depletion
- `low_decay` - Persistent activations

## Troubleshooting

### Common Issues

**"Module not found" error**
```bash
pip install pyyaml requests numpy
```

**Training doesn't converge**
- Increase epochs
- Reduce learning rate
- Add more middle nodes
- Enable radiation

**Loss oscillates**
- Reduce learning rate
- Increase early_stopping_patience

**Very slow training**
- Reduce vector_dim
- Set beam_width
- Use direct mode (not HTTP)

### Performance Tips

1. Use direct mode for fastest training
2. Set appropriate beam_width for large networks
3. Monitor active_nodes - if too few, network may be underpowered
4. Export results for offline analysis

## Examples

### XOR Learning

```bash
python train.py --config xor_basic -e 100
```

Expected: Loss should decrease from ~3.0 to ~1.0 within 50 epochs.

### Comparing Radiation vs No-Radiation

```bash
python train.py --compare with_radiation without_radiation -e 50
```

Expected: Radiation-enabled should have lower final loss.

### Full Benchmark Suite

```bash
python train.py --presets --export benchmark_results.json
```

Runs all preset experiments and exports results.

## Next Steps

1. Explore the visualization UI at `http://localhost:8765`
2. Create custom configurations in `training_configs.yaml`
3. Analyze training dynamics with exported results
4. Compare network architectures for your task

