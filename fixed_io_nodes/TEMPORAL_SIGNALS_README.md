# Temporal Signal Classification Task

A simplified testbed for the GNN architecture using synthetic time-series data instead of MNIST.

## 🎯 Task Description

The model learns to classify temporal signals composed of different sine/cosine combinations. Each signal class has a unique frequency pattern that the GNN must learn to recognize.

### Dataset
- **Signal Types**: 5 different classes (configurable)
- **Signal Composition**: Combination of 2 sine/cosine waves with unique frequencies per class
- **Temporal Aspect**: Signals are fed as sliding windows (10 timesteps → predict class)
- **Noise**: Gaussian noise added for robustness
- **Size**: 200 samples per class (1000 total), creating ~40,000 temporal windows

## 🚀 Quick Start

### 1. Visualize the Data (Optional but Recommended)

```bash
cd /Volumes/T9/work/word-tree/fixed_io_nodes
python visualize_signals.py
```

This creates visualizations in `training_logs/`:
- `signal_visualization.png`: Sample signals from each class
- `temporal_windows.png`: How signals are windowed for temporal processing

### 2. Run Training

```bash
python main_temporal.py
```

### 3. Monitor Training

**TensorBoard:**
```bash
tensorboard --logdir=training_logs/temporal_tensorboard
```

**CSV Logs:**
```bash
tail -f training_logs/temporal_log.csv
```

## ⚙️ Configuration

Edit `configs/config_temporal.yaml`:

### Data Parameters
```yaml
data:
  num_classes: 5           # Number of signal types
  seq_length: 50           # Signal length
  input_window: 10         # Past timesteps as input
  noise_level: 0.1         # Noise strength
```

### Model Architecture
```yaml
graph:
  total_nodes: 100         # Graph size
  input_nodes: 10          # Must match input_window
  output_nodes: 5          # Must match num_classes
  
model:
  vector_dim: 1            # Per-node dimension
  iterations: 3            # Propagation steps
  gamma: 2.0              # Magnitude scaling
```

### Training
```yaml
training:
  lr: 0.01                # Learning rate
  batch_size: 32          # Gradient accumulation
  worker_count: 3         # Parallel workers
```

## 📊 Expected Behavior

- **Initial Loss**: ~1.6 (random guessing for 5 classes)
- **Training**: Loss should decrease gradually as the model learns signal patterns
- **Convergence**: Loss should drop below 0.5 if learning properly

## 🔍 Why This is Better for Testing

1. **Simpler Data**: Clear signal patterns vs. complex images
2. **Temporal Nature**: Tests the temporal decay and propagation dynamics
3. **Controllable**: Easy to adjust difficulty (noise, frequencies, classes)
4. **Fast Iteration**: Smaller data, faster training cycles
5. **Interpretable**: Can visualize what the model is learning

## 🐛 Debugging Tips

If loss doesn't decrease:
- Check signal visualization - classes should be visually distinct
- Reduce `noise_level` to make task easier
- Increase `iterations` for more propagation steps
- Try lower `learning_rate` (0.001)
- Ensure `input_nodes * vector_dim = input_window`

## 📁 Files

- `synthetic_data.py`: Dataset generation
- `main_temporal.py`: Training script
- `visualize_signals.py`: Data visualization
- `configs/config_temporal.yaml`: Configuration
- `TEMPORAL_SIGNALS_README.md`: This file

## 🔄 Return to MNIST

To switch back to MNIST:
```bash
python main.py  # Uses configs/config.yaml
```


