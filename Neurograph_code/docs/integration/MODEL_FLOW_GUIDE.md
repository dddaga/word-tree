# NeuroGraph Model Flow Guide

## Overview
This document traces the complete execution flow of NeuroGraph from `main.py` through the entire system, showing how data flows through each component during training and evaluation.

## 🚀 Entry Point: `main.py`

### CLI Interface & Configuration
```python
main.py
├── Parse CLI arguments (mode, samples, config, etc.)
├── Auto-detect GPU availability (NVIDIA GeForce RTX 3050)
├── Load configuration from config/production.yaml
└── Initialize random seed (42)
```

**Key Parameters:**
- `--mode`: train/evaluate/benchmark
- `--eval-samples`: Number of evaluation samples
- `--config`: Configuration file path
- `--quick`: Quick evaluation mode

## 🔧 System Initialization Flow

### 1. Training Context Creation
```python
main.py → create_modular_train_context()
├── Load ModularConfig from YAML
├── Setup device (auto-detect CUDA)
├── Initialize core components
├── Setup input/output processing
├── Setup training components
└── Setup graph structure
```

### 2. Component Initialization Sequence
```python
ModularTrainContext.__init__()
├── 🔧 Core Components
│   ├── HighResolutionLookupTables (32×512 resolution)
│   ├── ModularPhaseCell (discrete signal computation)
│   └── VectorizedActivationTable (GPU tensor-based)
│
├── 🔧 Input Processing
│   ├── LinearInputAdapter (784→200 nodes, learnable projection)
│   └── MNIST Dataset Loading (60,000 samples)
│
├── 🔧 Output Processing
│   ├── OrthogonalClassEncoder (10 classes, 5D encoding)
│   └── ClassificationLoss (categorical cross-entropy)
│
├── 🔧 Training Components
│   ├── GradientAccumulator (8 steps, √8 scaling)
│   └── BatchController (gradient management)
│
└── 🔧 Graph Structure
    ├── Load/Generate static graph (1000 nodes, 4800 edges)
    ├── NodeStore (parameter storage)
    └── VectorizedForwardEngine (GPU-optimized)
```

## 📊 Data Flow Architecture

### System Specifications
- **Architecture**: 1000 nodes (200 input, 10 output, 790 intermediate)
- **Vector Dimension**: 5
- **Resolution**: 32 phase bins × 512 magnitude bins
- **Parameters**: 1,584,000 trainable
- **Device**: CUDA (RTX 3050 GPU)

## 🔄 Training Flow (mode=train)

### Single Sample Training Loop
```python
train_single_sample(sample_idx)
├── 1. Input Processing
│   ├── Get MNIST sample (784 pixels)
│   ├── LinearInputAdapter.get_input_context()
│   │   ├── Linear projection: 784 → 200×5 = 1000 values
│   │   ├── Phase-magnitude quantization
│   │   └── Return: {node_id: (phase_indices, mag_indices)}
│   └── Target label extraction
│
├── 2. Forward Pass
│   ├── Convert to string node IDs (n0, n1, ...)
│   ├── VectorizedForwardEngine.forward_pass_vectorized()
│   │   ├── Clear activation table
│   │   ├── Inject input context (batch GPU operation)
│   │   ├── Propagation loop (2-25 timesteps):
│   │   │   ├── Get active nodes (GPU tensors)
│   │   │   ├── Check output activation (vectorized)
│   │   │   ├── VectorizedPropagationEngine.propagate_vectorized()
│   │   │   │   ├── Static propagation (graph edges)
│   │   │   │   ├── Dynamic radiation (phase alignment)
│   │   │   │   └── Batch phase cell computation
│   │   │   ├── Clear previous activations
│   │   │   ├── Inject new activations (batch)
│   │   │   └── Decay and prune (vectorized)
│   │   └── Early termination on output activation
│   └── Extract output signals from final state
│
├── 3. Loss Computation
│   ├── Get class encodings (orthogonal 5D vectors)
│   ├── Compute logits via cosine similarity
│   ├── Categorical cross-entropy loss
│   └── Accuracy calculation
│
├── 4. Backward Pass (Discrete Gradient Approximation)
│   ├── Compute upstream gradients (loss derivatives)
│   ├── For each output node:
│   │   ├── Get current discrete parameters
│   │   ├── Compute continuous gradients (lookup tables)
│   │   └── Store node gradients
│   └── Vectorized intermediate node credit assignment
│
└── 5. Parameter Updates
    ├── Gradient accumulation (8 steps)
    ├── Apply continuous→discrete gradient conversion
    ├── Update NodeStore parameters (modular arithmetic)
    └── Learning rate: 0.0028 (√8 scaled from 0.001)
```

## 📈 Evaluation Flow (mode=evaluate)

### Batch Evaluation Process
```python
evaluate_accuracy(num_samples)
├── Set model to eval mode
├── Sample random indices from dataset
├── For each sample:
│   ├── Get input context (no gradients)
│   ├── Forward pass (same as training)
│   ├── Compute prediction (argmax of logits)
│   └── Compare with ground truth
├── Calculate accuracy percentage
└── Return final accuracy
```

## 🧠 Core Data Transformations

### 1. Input Transformation
```
MNIST Image (28×28=784 pixels)
    ↓ LinearInputAdapter
200 nodes × 5 dimensions = 1000 values
    ↓ Phase-Magnitude Quantization
Phase indices [0-31] + Magnitude indices [0-511]
    ↓ GPU Tensor Conversion
CUDA tensors for batch processing
```

### 2. Forward Propagation
```
Input Context: {node_id: (phase_idx, mag_idx)}
    ↓ VectorizedActivationTable
GPU tensors [max_nodes, vector_dim]
    ↓ Propagation Loop (2-25 timesteps)
Active nodes → Targets → New activations
    ↓ Early Termination
Output nodes activated → Extract signals
```

### 3. Signal Processing
```
Discrete Indices (phase, magnitude)
    ↓ HighResolutionLookupTables
Continuous signals (cos/sin values)
    ↓ Cosine Similarity
Logits for each class
    ↓ Softmax + Cross-entropy
Loss value + Gradients
```

## ⚡ GPU Acceleration Details

### Vectorized Operations
- **Activation Table**: GPU tensors instead of Python dictionaries
- **Batch Processing**: Multiple nodes processed simultaneously
- **Memory Pre-allocation**: Efficient tensor reuse
- **Vectorized Propagation**: Parallel neighbor computation

### Memory Usage
- **Total GPU Memory**: 8.76 MB allocated
- **Activation Table**: 0.10 MB
- **Forward Engine**: 8.66 MB (batch tensors + propagation)
- **Device**: NVIDIA GeForce RTX 3050 Laptop GPU

## 🔧 Key Components Deep Dive

### VectorizedForwardEngine
- **Purpose**: GPU-optimized forward propagation
- **Key Features**: Batch processing, early termination, memory efficiency
- **Performance**: 2-5x speedup over CPU version

### VectorizedActivationTable
- **Purpose**: GPU tensor-based activation management
- **Replaces**: Python dictionary operations
- **Performance**: 10x+ faster activation tracking

### ModularPhaseCell
- **Purpose**: Discrete signal computation
- **Method**: Direct phase-magnitude transfer with modular arithmetic
- **Resolution**: 32×512 bins for high precision

### LinearInputAdapter
- **Purpose**: Learnable input transformation
- **Method**: Neural network projection (NOT PCA)
- **Parameters**: 784×1000 = 784,000 learnable weights

## 🎯 Execution Examples

### Quick Evaluation
```bash
python main.py --mode evaluate --eval-samples 5 --quick
```
**Flow**: main.py → evaluate_model() → trainer.evaluate_accuracy(5) → 40% accuracy

### Training Session
```bash
python main.py --mode train --epochs 10
```
**Flow**: main.py → run_training() → trainer.train() → epoch loop → sample loop

### Benchmarking
```bash
python main.py --mode benchmark --samples 100
```
**Flow**: main.py → run_benchmark() → performance measurement → timing statistics

## 📊 Performance Characteristics

### Typical Execution Times (RTX 3050)
- **Initialization**: ~2-3 seconds
- **Forward Pass**: ~0.1-0.3 seconds per sample
- **Training Step**: ~0.2-0.5 seconds per sample
- **Evaluation**: ~0.05-0.1 seconds per sample

### Memory Efficiency
- **Base Memory**: ~6.1 MB system memory
- **GPU Memory**: 8.76 MB CUDA memory
- **Parameter Storage**: 1,584,000 discrete indices
- **Batch Tensors**: Pre-allocated for efficiency

## 🔄 Configuration System

### Production Configuration (config/production.yaml)
```yaml
architecture:
  total_nodes: 1000
  vector_dim: 5
  
resolution:
  phase_bins: 32
  mag_bins: 512
  
training:
  gradient_accumulation:
    enabled: true
    accumulation_steps: 8
```

### Dynamic Configuration Loading
- Auto-detection of optimal settings
- Device-specific optimizations
- Memory fraction management (80% GPU usage)

## 🚀 Optimization Features

### GPU Acceleration
- CUDA tensor operations throughout
- Vectorized batch processing
- Memory-efficient tensor pooling
- Optimized GPU memory layout

### Discrete Computation
- High-resolution phase-magnitude representation
- Modular arithmetic for parameter updates
- Continuous gradient approximation
- Efficient lookup table operations

This flow guide provides a complete picture of how NeuroGraph processes data from input to output, leveraging GPU acceleration and discrete computation for efficient neural network training and evaluation.
