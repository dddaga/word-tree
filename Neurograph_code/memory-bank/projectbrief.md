# NeuroGraph Project Brief

## Core Concept
NeuroGraph is a biologically-inspired, graph-based neural network prototype designed as an alternative to transformers. It implements discrete signal-based neural computation on a static Directed Acyclic Graph (DAG) using phase-magnitude vector representations and hybrid propagation mechanisms.

## Primary Innovation
The system replaces traditional continuous neural activations with discrete phase-magnitude index pairs that undergo lookup-table-based transformations, enabling:
- Discrete signal computation using cosine phase and exponential magnitude functions
- Hybrid propagation combining static graph topology with dynamic phase-based radiation
- Manual gradient computation without PyTorch autograd dependency

## Architecture Overview
- **Total Nodes**: 50 (configurable)
- **Input Nodes**: 5 (receive PCA-transformed MNIST data)
- **Output Nodes**: 10 (one per digit class)
- **Intermediate Nodes**: 35 (processing layer)
- **Graph Structure**: Static DAG with cardinality-limited connections
- **Vector Representation**: Phase-magnitude index pairs of dimension 5

## Core Components
1. **PhaseCell**: Discrete signal computation unit using lookup tables
2. **Propagation Engine**: Hybrid conduction (static) + radiation (dynamic) signal flow
3. **Node Store**: Learnable phase-magnitude parameter storage
4. **Activation Table**: Temporal signal decay and strength tracking
5. **Input/Output Adapters**: MNIST-to-graph and graph-to-prediction interfaces

## Learning Paradigm
- **Target Encoding**: Fixed phase-magnitude vectors per digit class (0-9)
- **Loss Function**: MSE between predicted and target signal vectors
- **Optimization**: Manual gradient-based updates using PhaseCell derivatives
- **Training Strategy**: Batch processing with warmup phase for output node inclusion

## Current Status
Fully implemented prototype capable of:
- MNIST digit classification through graph-based signal propagation
- Hybrid static-dynamic neighbor selection via phase alignment
- Manual backpropagation with discrete parameter updates
- Configurable hyperparameter experimentation
- Training convergence visualization and evaluation metrics

## Research Goals
Exploring discrete neural computation as an alternative to continuous activation paradigms, investigating biologically-plausible signal propagation mechanisms, and developing graph-based architectures that combine structural and dynamic connectivity patterns.
