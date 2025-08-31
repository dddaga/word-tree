🧠 Project Name: NeuroGraph
A biologically inspired, graph-based neural network prototype designed as an alternative to transformers — built from scratch with full control over propagation, encoding, and learning logic.

🔍 Purpose
This project explores discrete signal-based neural computation on a static DAG graph, using:

Phase (angle) and magnitude (intensity) vectors

Hybrid signal propagation

Custom forward + backward logic

Manual loss updates without PyTorch autograd

🧱 Architecture Overview
Total Nodes: 35

5 input nodes

25 intermediate nodes

5 output nodes

Graph Structure: Directed Acyclic Graph (DAG)

Random cardinality-limited forward-only connections

Ensured no output→output or backward links

Connectivity:

Conduction: pre-defined edges from static DAG

Radiation: dynamic top-K nodes by cosine phase alignment

🔁 Data Flow
1. Input →
MNIST digit images (28×28) are flattened → PCA to 5×2=10 → split across 5 input nodes as:

(phase_idx[D], mag_idx[D])

2. Forward Propagation →
Fixed number of timesteps (T = 6)

Signal decays over time

Hybrid propagation to neighbors:

DAG connections (conduction)

Top-K phase-aligned nodes (radiation)

3. Output →
5 output nodes each emit:

Phase vector

Magnitude vector

🎯 Learning Logic
Target Vector Encoding
Each digit class (0–9) is assigned a fixed:

phase_idx[D]

mag_idx[D]

All output nodes must match the vector for the correct class

Loss Function
L2 loss (in index space) between output nodes and target class vector

Averaged across active output nodes

Backward Pass
Manual update using:

Gradients returned by PhaseCell forward

Optional Straight-Through Estimation (STE)

Only updates phase/mag indices in the node store

Includes a warm-up phase where all output nodes are included regardless of activity

✅ Key Modules
File	Purpose
main.py	Runs training + evaluation, saves loss plot
train/train_context.py	Core training loop, computes loss
modules/input_adapters.py	MNIST + PCA → graph input vectors
modules/class_encoding.py	Fixed per-class target vector (phase+mag)
modules/output_adapters.py	Cosine similarity prediction logic
core/forward_engine.py	Multi-step propagation logic
core/propagation.py	One-step propagation (conduction + radiation)
core/backward.py	Manual signal gradient-based updates
core/activation_table.py	Stores and decays active nodes
core/phase_cell.py	Main signal computation logic
core/node_store.py	Stores learnable phase/mag values
config/default.yaml	All hyperparameters

🧪 Evaluation Logic
At end of training, model is tested on N=100 unseen MNIST samples

For each sample:

Signal flows through the graph

Final output vectors are compared (cosine) to all class vectors

Prediction is the closest class

🔧 Current Config (default.yaml)
yaml
Copy
Edit
total_nodes: 35
num_input_nodes: 5
num_output_nodes: 5
vector_dim: 5
phase_bins: 8
mag_bins: 256
cardinality: 3
decay_factor: 0.95
min_activation_strength: 0.001
max_timesteps: 6
top_k_neighbors: 4
learning_rate: 0.01
warmup_epochs: 5
num_epochs: 50
use_radiation: true
graph_path: config/static_graph.pkl
log_path: logs/
📌 Current State
✅ Forward pass (hybrid propagation) implemented

✅ Manual backward pass using custom gradients

✅ Option 1 target strategy (shared vector per digit class)

✅ Trained and evaluated on MNIST with PCA input

✅ Cosine-based output prediction

✅ CLI evaluation accuracy logging and loss plotting