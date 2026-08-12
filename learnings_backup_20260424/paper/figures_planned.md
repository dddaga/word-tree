# Planned Figures and Tables

## Figure 1: Architecture diagram
SGNNET architecture: input scatter → random graph → K_iter routing with AH suppression → mean-pool → classify. Show the random connectivity, F.normalize on hypersphere, and how information flows.

## Figure 2: Parameter efficiency comparison
Bar chart: VGG16 FC (123.6M params, 93.5%) vs SGNNET (529K params, 97.86%). Include other baselines (MLP, pruned VGG, GCN).

## Figure 3: Pareto frontier — accuracy vs FLOPs
Scatter plot of all configurations tested. X-axis: FLOPs (log scale). Y-axis: accuracy. Highlight the Pareto frontier. Show VGG16 FC as reference point.

## Figure 4: K_iter ablation
Line plot: accuracy vs K_iter for multiple N values. Shows that routing depth is the primary capacity lever.

## Figure 5: N-scaling curve
Accuracy vs N for N={256..8192}. Shows diminishing returns and motivates efficiency track.

## Figure 6: Scale transfer compression
Paired bar chart: mechanism Δ at N=1024 vs N=4096 for W_proj, group topology, RigL, etc. Shows systematic compression.

## Figure 7: Load-bearing walls
Ablation heatmap: removing each component (F.normalize, AH, mean-pool, K_iter steps) and the resulting accuracy drop.

## Table 1: Main results
Accuracy, params, FLOPs for SGNNET vs all baselines on Imagenette (and CIFAR-10 when available).

## Table 2: Comprehensive ablation
Every architectural choice and its isolated effect. One row per ablation.

## Table 3: Confirmed hyperparameter defaults
Final recommended configuration with source experiments.
