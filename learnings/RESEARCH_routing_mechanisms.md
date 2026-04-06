# Routing Mechanisms Research for SGNNET

Research date: 2026-04-09. Focus: dynamic routing compatible with O(N*K) sparsity, K_iter=8-12 iterative refinement, and S^{D-1} hypersphere geometry.

---

## 1. Sparse Attention for GNNs

**Exphormer** (ICML 2023): Sparse attention via expander graphs + virtual global nodes. O(N+E) complexity. Expander edges provide mathematically guaranteed spectral gap (no attention collapse). Compatible: expander edges are a fixed O(N*K) budget augmentation -- could replace SGNNET's small-world init with expander graph for better information propagation. Virtual nodes act as global memory without O(N^2).

**SGFormer** (NeurIPS 2023): Single-layer global attention with O(N) complexity via kernel trick. Decouples local message-passing from global attention. Key insight: simple linear global attention (no softmax) matches complex multi-head attention on graphs. SGNNET relevance: the "no softmax" finding aligns with SGNNET's gate-death problem -- softmax routing collapses under K_iter iteration.

**NodeFormer** (NeurIPS 2022): Kernelized Gumbel-Softmax for all-pair message passing at O(N) cost. Uses random feature maps to approximate softmax attention. OOM above 30K nodes. Less relevant given SGNNET's strict K<<N constraint.

**Experiment suggestion**: Replace small-world init with Ramanujan expander graph (deterministic, optimal spectral gap). No learned routing -- pure topology swap. Expected: better information flow at same K_hh budget.

---

## 2. Signal Conservation in Message Passing

**GRAND** (ICML 2021): Frames GNN as diffusion PDE discretization. Message passing = heat diffusion. Row-stochastic diffusion matrix enforces signal conservation (sum of incoming weights = 1 per node). Implicit discretization enables stable deep propagation. Key: diffusion naturally conserves total signal mass -- analogous to SGNNET's redistribution routing (sum w=1) that showed +3.98pp at N=1024.

**GRAND++** (2023): Generalized neural diffusion with source terms. Adds learnable source/sink to break pure diffusion's tendency toward equilibrium (over-smoothing). Source terms inject fresh signal at each step -- prevents the convergence that kills deep message passing.

**Graph Neural Reaction-Diffusion** (SIAM 2024): Couples diffusion with reaction terms (Allen-Cahn, Fisher-KPP). Reaction creates/destroys signal locally while diffusion spreads it. Maintains sparsity (operates on existing edges only).

**Beltrami Flow** (NeurIPS 2021): Joint evolution of features AND positional coordinates on a Riemannian manifold. Feature evolution = message passing; position evolution = graph rewiring. Naturally compatible with S^{D-1} geometry. Positions live on the manifold and evolve via gradient flow -- this IS dynamic connectivity from learned positions.

**Experiment suggestion**: Implement GRAND-style implicit diffusion as the message-passing kernel. Replace current explicit update h_{t+1} = f(h_t, neighbors) with implicit solve (I - dt*L)h_{t+1} = h_t. Implicit schemes are unconditionally stable at any K_iter depth. Add source term h_0 (initial representation) to prevent over-smoothing. This is a different mechanism than AH -- it's about HOW signals propagate, not which edges are suppressed.

---

## 3. Graph Rewiring / Dynamic Topology

**SDRF** (Topping et al. 2021): Stochastic Discrete Ricci Flow -- adds edges where curvature is most negative (bottlenecks), removes where most positive (redundant). Addresses over-squashing. ICLR 2025 evaluation showed theoretical motivation often fails on real data. Pre-processing step, not learned.

**DiffWire** (LoG 2022): Differentiable rewiring via Lovasz bound. CT-Layer rewires based on commute times; GAP-Layer based on spectral gap. Parameter-free. Compatible with sparse graphs. Key limitation: rewiring is a preprocessing step per layer, not per-step within K_iter.

**PR-MPNN** (2024): Probabilistically Rewired MPNNs. Differentiable k-subset sampling for edge selection. Learns which edges to add/remove during training. Maintains sparsity budget. Most promising for SGNNET: can learn to rewire while keeping O(N*K).

**Spectral Graph Pruning** (NeurIPS 2024): Prunes edges that hurt spectral gap while preserving connectivity. Addresses both over-squashing and over-smoothing simultaneously. Could be used as periodic rewiring during SGNNET training (every M epochs, re-prune the graph).

**Experiment suggestion**: Periodic spectral rewiring every E epochs (E=10-25). Compute Ollivier-Ricci curvature on current graph, add edges at bottlenecks, remove redundant edges, maintaining K_hh budget. This is NOT per-forward-pass dynamic routing (which has failed 9 times) -- it's slow structural adaptation during training. Avoids gate-death because no learned gates are involved.

---

## 4. Physics-Inspired Dynamics

**Hamiltonian GNN**: Hamiltonian dynamics preserve energy (symplectic integration). h_t evolves along energy-conserving trajectories. No dissipation = no over-smoothing by construction. Requires augmenting state with "momentum" variable p. Update: dh/dt = dH/dp, dp/dt = -dH/dh. The K_iter steps become leapfrog integration steps.

**Quantum Walk on Graphs**: Discrete-time quantum walk uses unitary evolution (coin + shift operators). Unitary = energy-preserving. Naturally sparse (walks on edges). Interference patterns enable non-local information flow without O(N^2). Wave-like spreading vs diffusive spreading: quantum walks spread as O(sqrt(t)) vs diffusion O(t), reaching further in fewer steps.

**Schrodinger Bridge on Graphs**: Optimal transport between distributions on graphs. Connects initial node features to target features via minimum-energy path. Natural sparsity preservation.

**Experiment suggestion**: Hamiltonian message passing. Augment each neuron with momentum p_i (same dim as h_i). K_iter steps become leapfrog: p_{t+1/2} = p_t - (dt/2)*dV/dh, h_{t+1} = h_t + dt*p_{t+1/2}, p_{t+1} = p_{t+1/2} - (dt/2)*dV/dh. V is the potential from neighbor interactions (existing W_pos mechanism). Energy conservation prevents signal death over K_iter=12 steps. Momentum carries information from early steps -- acts as implicit skip connection. Compatible with AH (AH modifies V, not the integration scheme).

---

## 5. Position-Based / Geometric Connectivity

**DGCNN / EdgeConv** (ACM TOG 2019): Dynamic graph CNN. Recomputes K-NN in feature space at each layer. Key: neighbors change every layer based on learned representations. O(N*K*log(N)) with KD-tree. "Semantic neighbors" emerge -- geometrically distant but functionally related nodes connect.

**Point Transformer v3** (2024): Serialization-based attention. Maps 3D points to 1D via space-filling curves, then applies windowed attention. O(N*W) where W=window size. Avoids KNN entirely.

**PointNet++** (2017): Hierarchical set abstraction. Farthest point sampling + ball query for local neighborhoods. Multi-scale grouping captures both fine and coarse structure.

**Beltrami Flow** (see Section 2): Jointly evolves features AND positions. The graph topology emerges from position proximity. This is the closest existing work to "position-only architecture" for SGNNET.

**Experiment suggestion**: DGCNN-style dynamic KNN at the GROUP level (not per-neuron). Compute group centroids S_g = mean(h_i) for each of n_groups=8 groups. Build KNN graph between groups (K_group=3-4) in feature space. Route inter-group messages along this dynamic group graph. Per-step cost: n_groups^2 * D = 8^2 * 64 = 4096 FLOPs (negligible). This addresses the step83 failure mode: S_g was used for softmax routing (collapsed), but here it's used only for topology (binary: connected or not). No learned gates, no softmax, no gate death.

---

## 6. MoE Routing for GNNs

**Expert Choice Routing** (Google, 2022): Inverts routing -- experts pick their top-k tokens instead of tokens picking experts. Perfect load balancing by construction. Each token gets variable number of experts. Avoids token dropping entirely.

**Soft MoE** (Puigcerver et al., ICLR 2024): Fully differentiable. Each expert receives a weighted combination of ALL tokens (soft assignment). No token dropping, no routing collapse, no expert death -- even with 128 experts. Weights are everywhere-positive. Cost: O(N*E*D) where E=experts.

**ReMoE** (ICLR 2025): ReLU routing replaces TopK+Softmax. Fully differentiable. Sparsity emerges naturally from ReLU zeros. L1 regularization for load balancing instead of auxiliary losses. Scales better than TopK as expert count grows. Key insight: ReLU naturally produces sparse routing without the discontinuity of TopK.

**Graph MoE** (NeurIPS 2023): Different GNN experts (GCN, GAT, GraphSAGE) routed per-node. Router predicts which expert architecture suits each node's local structure. Demonstrated on heterogeneous graphs.

**Experiment suggestion -- ReLU group routing**: Groups of neurons as "experts". Router: ReLU(W_route @ h_i) produces sparse, non-negative routing weights over groups. ReLU replaces softmax (which collapsed in step83). L1 penalty on routing weights for sparsity. Key difference from step83: (a) ReLU not softmax -- no normalization collapse, (b) L1 regularization not aux loss, (c) routing weights are non-negative by construction (no sign flip instability). This directly addresses the gate-death theorem: ReLU(x) for x>0 has gradient 1 (no exponential decay under iteration), and for x<=0 it's cleanly off (no near-zero leaking).

---

## 7. Over-Smoothing for Deep Message Passing

SGNNET's K_iter=8-12 is equivalent to 8-12 GNN layers. Most GNNs degrade past 4-6 layers.

**GCNII** (ICML 2020): Initial residual connection + identity mapping. h^(l) = ((1-a)*P*h^(l-1) + a*h^(0)) * ((1-b)*I + b*W^(l)). Two mechanisms: (a) residual to INITIAL features h^(0), not previous layer, (b) identity mapping preserves feature magnitude. Proven to prevent over-smoothing at 64 layers.

**JKNet** (ICML 2018): Jumping Knowledge -- aggregates representations from ALL layers (not just final). Uses max-pool, LSTM, or attention over {h^(0), h^(1), ..., h^(L)}. Different nodes can "select" their optimal depth.

**PairNorm** (ICLR 2020): Normalizes node features to maintain total pairwise distance. Prevents embeddings from collapsing to same point. Parameter-free, architecture-agnostic. SGNNET already L2-normalizes to S^{D-1} -- PairNorm-style thinking may be partially present.

**DropMessage** (AAAI 2023): Randomly drops messages (not edges or nodes) during propagation. More fine-grained than DropEdge. Unifies DropEdge, DropNode, and dropout under one framework. Directly applicable to SGNNET's message-passing loop.

**DGN / Directional GNN** (NeurIPS 2020): Uses eigenvector-derived directional flows. Messages carry directional information preventing isotropic smoothing. Requires Laplacian eigenvectors.

**Experiment suggestion**: GCNII-style initial residual in SGNNET's K_iter loop. At each routing step t: h_t = (1-a)*message_pass(h_{t-1}) + a*h_0, where h_0 is the initial projection. a=0.1-0.2 (small -- most signal from routing, but guaranteed h_0 injection). This is different from a skip connection (which connects h_{t-1} to h_t). Initial residual connects h_0 to EVERY step, preventing drift. AH still operates on the message_pass component. Cost: one lerp per step (negligible). GCNII proved this works at 64 layers -- SGNNET only needs 12.

---

## Recommended Experiments (Ranked)

### 1. GCNII-Style Initial Residual (Highest confidence)
- **Why**: Proven at 64 layers, zero new parameters, directly addresses K_iter depth. SGNNET's AH already handles neighbor selection -- this handles signal preservation orthogonally.
- **Config**: h_t = (1-a)*AH_message_pass(h_{t-1}) + a*h_0. Sweep a in {0.05, 0.1, 0.2, 0.3}. K_iter=12, N=4096, K_hh=4.
- **Why it won't gate-death**: No gates. It's a fixed interpolation coefficient (or learnable scalar, but start fixed).

### 2. ReLU Group Routing (Novel, directly addresses failure mode)
- **Why**: ReLU routing (ICLR 2025) solves exactly the softmax collapse that killed step83. Non-negative, sparse, fully differentiable, gradient=1 for active routes.
- **Config**: n_groups=8, K_group=3. Router: ReLU(W_route @ S_g) where S_g=group centroid. L1 penalty lambda=0.01. Inter-group messages weighted by ReLU output.
- **Risk**: Medium. ReLU dead neurons possible, but L1 prevents all-active (which is the opposite failure mode).

### 3. Implicit Diffusion Message Passing (Physics-grounded)
- **Why**: GRAND's implicit scheme is unconditionally stable at any depth. Current explicit updates accumulate error over K_iter steps. Implicit solve naturally conserves signal.
- **Config**: Replace h_{t+1} = f(h_t, N(h_t)) with implicit: h_{t+1} = h_t + dt*(A*h_{t+1} - h_{t+1} + h_0). Solve via 2-3 fixed-point iterations (cheap at K_hh=4). Source term h_0 prevents over-smoothing.
- **Risk**: Medium. Implicit solve adds compute. Start with 1 fixed-point iteration (= one extra matmul per step).

### 4. Hamiltonian Message Passing (Energy-conserving)
- **Why**: Leapfrog integration preserves energy exactly -- signal cannot decay over K_iter. Momentum provides implicit memory across steps (like a skip connection that carries velocity, not just position).
- **Config**: Augment state with p_i (momentum). Leapfrog integration with dt=0.1. Potential V from AH-weighted neighbor interactions. K_iter=12 steps.
- **Risk**: Higher. Doubles state size (h + p). Momentum might interfere with AH suppression dynamics. But energy conservation is a hard guarantee against over-smoothing.

### 5. Dynamic Group Topology via KNN (Simplest dynamic routing)
- **Why**: Binary topology (connected/not) avoids all gate-death issues. KNN on group centroids is O(n_groups^2) = O(64) -- negligible. Changes topology without learned routing weights.
- **Config**: n_groups=8, K_group=3. Every K_iter step (or every other step), recompute group KNN from centroids. Route inter-group messages along dynamic edges. Intra-group stays static small-world.
- **Risk**: Low. No new parameters. But effect may be small since n_groups=8 means only 8 centroids to route between.

### Honorable Mentions
- **Expander graph init**: Replace small-world with Ramanujan expander. Zero-cost experiment, potentially better spectral properties. Test as init-only change.
- **DropMessage**: Random message dropping during K_iter. Trivial to implement, proven anti-over-smoothing. Sweep drop rate {0.1, 0.2, 0.3}.
- **Periodic Ricci rewiring**: Every 10 epochs, recompute edge curvature, swap bottom-K edges. Slow adaptation without per-step gates.
