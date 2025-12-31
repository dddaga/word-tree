# Autoregressive Temporal Training

Sequential training mode where the GNN maintains temporal continuity across timesteps within a sequence.

## 🔄 How It Works

### Sliding Window Approach

```
Sequence: [v₀, v₁, v₂, v₃, v₄, v₅, v₆, v₇, v₈, v₉, ...]

Window Size = 5

┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: WARM-UP (No Gradients)                            │
│                                                               │
│  Window 1: [v₀, v₁, v₂, v₃, v₄] → Adapter → GNN → Predict │
│  Window 2: [v₁, v₂, v₃, v₄, v₅] → Adapter → GNN → Predict │
│  ...                                                          │
│  Window 5: [v₄, v₅, v₆, v₇, v₈] → Adapter → GNN → Predict │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: TRAINING (With Gradients)                          │
│                                                               │
│  Window 6: [v₅, v₆, v₇, v₈, v₉] → Adapter → GNN            │
│        ↓                                                      │
│     Predict Class → Compute Loss → Backward                  │
│        ↓                                                      │
│     Send Gradients to Accumulator                            │
│        ↓                                                      │
│     [Activations PERSIST - temporal continuity!]             │
│                                                               │
│  Window 7: [v₆, v₇, v₈, v₉, v₁₀] → Adapter → GNN           │
│        ↓ (builds on accumulated activations)                 │
│        ...                                                    │
└─────────────────────────────────────────────────────────────┘

After sequence ends: Reset activations, start next sequence
```

### Key Features

1. **Sliding Windows**: Input nodes represent the last N timesteps
2. **Window Representation**: Each input node receives one timestep from the window
3. **Warm-up Phase**: Initial windows prime the network without computing gradients
4. **Temporal Continuity**: ✨ Activations PERSIST across all windows in a sequence
5. **Journey Capture**: Network state accumulates throughout the entire sequence
6. **Input Adapter**: Transforms window vector to graph input space
7. **Single Worker**: Simplified architecture for validation

## 🚀 Quick Start

### Run Training

```bash
cd /Volumes/T9/work/word-tree/fixed_io_nodes
python main_autoregressive.py
```

### Expected Output

```
🎯 Using device: cpu
📊 Creating synthetic signal dataset...
✅ Dataset created: 25 sequences
✅ Autoregressive trainer initialized
📊 Logging to training_logs/autoregressive_log.csv

============================================================
🚀 Starting autoregressive training
============================================================
Warm-up steps: 10
Batch size: 16
Learning rate: 0.01
============================================================

📊 Processing sequence 1/10 (class 2)
  Warm-up phase: 10 steps
  Training phase: 40 steps
    t=10: loss=1.6234
    t=11: loss=1.5890
    ...
  ✅ Sequence 1 complete. Avg loss: 1.4523
```

## ⚙️ Configuration

Edit `configs/config_autoregressive.yaml`:

### Training Parameters

```yaml
training:
  window_size: 5           # Sliding window size (# of timesteps in each window)
  warmup_steps: 5          # Number of initial windows for warm-up phase
  lr: 0.01                 # Learning rate
  batch_size: 16           # Gradient accumulation
  num_sequences: 10        # Total sequences to train on
```

### Model Parameters

```yaml
model:
  iterations: 3            # GNN steps per timestep
  temporal_decay: 0.85     # Activation decay factor
  adapter_hidden_dims: [32, 16]  # Input adapter architecture
```

## 📊 Monitoring

### CSV Logs

```bash
tail -f training_logs/autoregressive_log.csv
```

Columns:
- `sequence_id`: Which sequence
- `timestep`: Position in sequence
- `phase`: "warmup" or "training"
- `loss`: Cross-entropy loss
- `target`: True class label

### Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_logs/autoregressive_log.csv')

# Filter training phase only
train_df = df[df['phase'] == 'training']

# Plot loss over time
plt.plot(train_df['loss'])
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Autoregressive Training Loss')
plt.show()
```

## 🎛️ Architecture Details

### Input Pipeline

```
Sliding Window [v_t, v_t+1, v_t+2, v_t+3, v_t+4]  (size=5)
  ↓
Input Adapter (MLP: 5 → 16 → 8 → 5)
  ↓
Reshape to (input_nodes=5, vector_dim=1)
  ↓
Scale to phase range [-π/2, π/2]
  ↓
GNN (3 iterations of propagation)
  ↓
Output Nodes (5 classes)
```

**Note**: Each input node corresponds to one timestep in the window:
- Node 0 ← v_t
- Node 1 ← v_t+1
- Node 2 ← v_t+2
- Node 3 ← v_t+3
- Node 4 ← v_t+4

### Gradient Flow

```
Output Logits
  ↓
Cross-Entropy Loss
  ↓
Backward through GNN
  ↓
Extract phase_grads, mag_grads
  ↓
Send to Gradient Accumulator
  ↓
Accumulate until batch_size reached
  ↓
Update weights in Qdrant
```

## 🐛 Debugging Tips

### Loss Not Decreasing?

1. **Check warm-up**: Increase `warmup_steps` to 15-20
2. **Reduce learning rate**: Try `lr: 0.001`
3. **Lower temporal decay**: Try `temporal_decay: 0.9` (stronger memory)
4. **Simplify signals**: Reduce `noise_level` to 0.01

### NaN/Inf Losses?

1. **Check input range**: Should be normalized
2. **Reduce learning rate**: Try `lr: 0.001`
3. **Check gamma**: Try lower value like `gamma: 1.0`

### Memory Issues?

1. **Reduce graph size**: Lower `total_nodes` to 50
2. **Reduce iterations**: Lower `iterations` to 2
3. **Smaller adapter**: Use `[16]` instead of `[32, 16]`

## 📈 Expected Behavior

- **Initial Loss**: ~1.6 (random for 5 classes)
- **After Warm-up**: Loss starts around 1.4-1.5
- **Training Progress**: Should decrease gradually to ~1.0-1.2
- **Convergence**: May take 50-100 sequences depending on complexity

## 🔄 Comparison with Other Modes

| Feature | Autoregressive | Temporal Windows | MNIST |
|---------|---------------|------------------|-------|
| Temporal Continuity | ✅ Yes | ❌ No | ❌ No |
| Sequential Input | ✅ One at a time | ❌ Batch window | ❌ Single image |
| Warm-up Phase | ✅ Yes | ❌ No | ❌ No |
| Activation Reset | Per sequence | Per sample | Per sample |
| Use Case | Time-series prediction | Pattern recognition | Image classification |

## 🎯 Next Steps

1. **Validate on simple data**: Run with current config
2. **Monitor convergence**: Check if loss decreases
3. **Tune hyperparameters**: Adjust warm-up, LR, decay
4. **Scale up**: Increase num_sequences gradually
5. **Real data**: Apply to actual time-series task

