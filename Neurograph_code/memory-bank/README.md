# NeuroGraph Memory Bank

This memory bank contains comprehensive documentation for the NeuroGraph project - a biologically-inspired, graph-based neural network prototype that implements discrete signal processing as an alternative to traditional continuous neural computation.

## 📁 Documentation Structure

### Core Documentation Files

#### 🎯 [projectbrief.md](./projectbrief.md)
**Foundation document** - Core concept, architecture overview, and research goals
- Primary innovation: discrete phase-magnitude neural computation
- Architecture: 50-node DAG with hybrid propagation
- Current status: fully functional MNIST classification prototype

#### 🌟 [productContext.md](./productContext.md)
**Purpose and vision** - Why NeuroGraph exists and what problems it solves
- Research questions: discrete vs continuous computation
- Target use cases: research platform, biological modeling
- Value proposition: transparency, configurability, interpretability

#### ⚡ [activeContext.md](./activeContext.md)
**Current state** - Active work, recent changes, and immediate next steps
- Current focus: comprehensive memory bank documentation
- Key insights: hybrid propagation, discrete signal processing
- Technical considerations: configuration mismatches, scalability limits

#### 🏗️ [systemPatterns.md](./systemPatterns.md)
**Architecture and design** - Core patterns, component relationships, and design decisions
- Discrete signal processing pattern with lookup tables
- Hybrid propagation: static DAG + dynamic radiation
- Manual gradient computation without autograd
- Temporal activation with decay mechanisms

#### 🔧 [techContext.md](./techContext.md)
**Implementation details** - Technology stack, development setup, and technical constraints
- Python/PyTorch stack with manual gradient computation
- Project structure and key implementation patterns
- Performance characteristics and scalability limits
- Future optimization opportunities

#### 📊 [progress.md](./progress.md)
**Development status** - What works, what's left to build, and roadmap
- Current status: fully functional prototype
- Immediate improvements: configuration consistency, performance optimization
- Long-term vision: production-ready discrete neural computation platform

## 🧠 Key Concepts

### Discrete Neural Computation
NeuroGraph replaces continuous neural activations with discrete phase-magnitude index pairs, processed through lookup tables containing pre-computed trigonometric and exponential functions.

### Hybrid Propagation
The system combines two propagation mechanisms:
- **Conduction**: Static connections from pre-defined DAG topology
- **Radiation**: Dynamic connections to top-K phase-aligned neighbors

### Manual Gradient Computation
Custom backward pass implementation that bypasses PyTorch autograd, providing full control over learning dynamics and parameter updates.

### Temporal Signal Processing
Signals decay over timesteps with configurable decay factors, creating natural attention-like mechanisms and preventing infinite propagation.

## 🚀 Quick Start

```bash
# Run training and evaluation
python main.py

# Configuration in config/default.yaml
# Results saved to logs/ directory
```

## 📈 Current Capabilities

- ✅ MNIST digit classification through graph-based signal propagation
- ✅ Configurable hyperparameter experimentation
- ✅ Training convergence visualization
- ✅ Hybrid static-dynamic connectivity patterns
- ✅ Discrete signal processing with interpretable representations

## 🔬 Research Applications

- **Discrete vs Continuous**: Investigating alternatives to continuous activation functions
- **Biological Plausibility**: Exploring spike-based neural computation models
- **Graph Neural Networks**: Novel approaches to structured data processing
- **Attention Alternatives**: Graph-based alternatives to transformer architectures

## 📋 Memory Bank Usage

This memory bank follows the hierarchical documentation structure:
1. **projectbrief.md** - Start here for project overview
2. **productContext.md** - Understand the research motivation
3. **systemPatterns.md** - Learn the architectural patterns
4. **techContext.md** - Explore implementation details
5. **activeContext.md** - Check current development status
6. **progress.md** - Review accomplishments and roadmap

Each file builds upon the previous ones, creating a comprehensive understanding of the NeuroGraph system from concept to implementation.

## 🔄 Maintenance

The memory bank is updated when:
- Significant architectural changes are made
- New features or capabilities are added
- Research insights or patterns are discovered
- Development priorities or roadmap changes
- Configuration or setup procedures are modified

Last updated: Current session - comprehensive initial documentation creation
