"""
FFNN Conduction Sparsity Benchmark — UCI Wine Quality (Red)
SUB-EXPERIMENT: Depth Saturation — Fixed Nodes (240) + Fixed Sparsity (90%)

This experiment investigates the saturation effect of increasing the number of
hidden layers while keeping total neurons and sparsity completely fixed:

    total_neurons = 240  (constant)
    sparsity      = 0.90 (constant)
    num_layers    varies — each layer gets floor(total_neurons / num_layers) neurons

Node-level sparsity (weights + bias) is applied to ALL hidden layers:
floor(0.90 × layer_size) neurons are fully deactivated per layer.
The output layer is NOT masked.

The "saturation" question: beyond some depth, adding more (thinner) layers
with 90 % of neurons already dead yields diminishing or reversed returns.

Sweep:
  num_layers ∈ {2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 60, 80, 120}  (14 values)
  Each num_layers value → one training run

Outputs written to a timestamped sub-folder under:
    fixed_io_nodes/experiments/ffnn_sparsity_benchmark/runs_depth_saturation/

Run from repo root:
    python3 fixed_io_nodes/experiments/ffnn_sparsity_benchmark/run_experiment_depth_saturation.py
"""

import time
import tracemalloc
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ── paths ────────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_RUN_DIR = _HERE / "runs_depth_saturation" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
_RUN_DIR.mkdir(parents=True, exist_ok=True)

# ── hyper-parameters ─────────────────────────────────────────────────────────
EPOCHS         = 100
LR             = 1e-3
BATCH_SIZE     = 64
TOTAL_NEURONS  = 240
SPARSITY       = 0.90
# Layer counts to sweep — each gives layer_size = TOTAL_NEURONS // num_layers
NUM_LAYERS_LIST = [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 60, 80, 120]
SEED           = 42
DEVICE         = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

RED_WINE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
NUM_CLASSES = 11   # wine quality 0–10


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(RED_WINE_URL, sep=";")
    X  = df.drop("quality", axis=1).values.astype(np.float32)
    y  = df["quality"].values.astype(np.int64)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp
    )

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_val  = scaler.transform(X_val)
    X_te   = scaler.transform(X_te)

    print(f"Train: {X_tr.shape}  Val: {X_val.shape}  Test: {X_te.shape}")
    uniq, cnts = np.unique(y_tr, return_counts=True)
    print(f"Classes (quality 0–10): {dict(zip(uniq.tolist(), cnts.tolist()))}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ─────────────────────────────────────────────────────────────────────────────
# Node-level conduction mask (weights + bias from the same mask)
# ─────────────────────────────────────────────────────────────────────────────

def make_node_mask(
    out_features: int,
    in_features: int,
    sparsity: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Node-level mask: deactivate entire neurons (all incoming weights + bias).

    floor(sparsity × out_features) neurons are completely zeroed out.
    Returns (weight_mask, bias_mask).
    """
    if sparsity >= 1.0:
        return torch.zeros(out_features, in_features), torch.zeros(out_features)

    n_dead = int(sparsity * out_features)

    node_active = torch.ones(out_features)
    if n_dead > 0:
        dead_idxs = rng.choice(out_features, size=n_dead, replace=False)
        node_active[dead_idxs] = 0.0

    weight_mask = node_active.unsqueeze(1).expand(out_features, in_features).clone()
    bias_mask   = node_active

    return weight_mask, bias_mask


# ─────────────────────────────────────────────────────────────────────────────
# Model — Variable Depth with Fixed Total Neurons and Fixed Sparsity
# ─────────────────────────────────────────────────────────────────────────────

class SparseFFNNDepth(nn.Module):
    """Variable-depth FFNN with node-level conduction masks.

    Architecture:
        input → [hidden_layer × num_layers] → output
        layer_size = total_neurons // num_layers  (each hidden layer same width)

    Node-level sparsity is applied to ALL hidden layers; output is unmasked.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        layer_size: int,
        output_dim: int,
        sparsity: float,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layer_size = layer_size

        # Build hidden layers
        self.hidden = nn.ModuleList()
        prev_size = input_dim
        for _ in range(num_layers):
            self.hidden.append(nn.Linear(prev_size, layer_size))
            prev_size = layer_size

        self.output = nn.Linear(prev_size, output_dim)

        rng = np.random.default_rng(SEED)

        # Register masks for each hidden layer
        prev_size = input_dim
        for i, layer in enumerate(self.hidden):
            w_mask, b_mask = make_node_mask(layer_size, prev_size, sparsity, rng)
            self.register_buffer(f"w_mask_{i}", w_mask)
            self.register_buffer(f"b_mask_{i}", b_mask)
            prev_size = layer_size

        # Output layer: no sparsity
        w_mask_out = torch.ones(output_dim, prev_size)
        b_mask_out = torch.ones(output_dim)
        self.register_buffer("w_mask_out", w_mask_out)
        self.register_buffer("b_mask_out", b_mask_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.hidden):
            w_mask = getattr(self, f"w_mask_{i}")
            b_mask = getattr(self, f"b_mask_{i}")
            x = F.relu(F.linear(x, layer.weight * w_mask, layer.bias * b_mask))
        return F.linear(x, self.output.weight * self.w_mask_out,
                           self.output.bias   * self.b_mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_active_params(model: SparseFFNNDepth) -> int:
    """Active weight connections + active biases across all layers."""
    total = 0
    for i in range(model.num_layers):
        total += int(getattr(model, f"w_mask_{i}").sum().item())
        total += int(getattr(model, f"b_mask_{i}").sum().item())
    total += int(model.w_mask_out.sum().item())
    total += int(model.b_mask_out.sum().item())
    return total


def count_active_macs(model: SparseFFNNDepth) -> int:
    """MACs proportional to active (unmasked) weight connections only."""
    total = 0
    for i in range(model.num_layers):
        total += int(getattr(model, f"w_mask_{i}").sum().item())
    total += int(model.w_mask_out.sum().item())
    return total


def count_active_nodes_per_layer(model: SparseFFNNDepth) -> list[int]:
    """Number of active (unmasked) nodes in each hidden layer."""
    counts = []
    for i in range(model.num_layers):
        b_mask = getattr(model, f"b_mask_{i}")
        counts.append(int(b_mask.sum().item()))
    return counts


def _make_loaders(X_tr, y_tr, X_val, y_val, X_te, y_te):
    def to_ds(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
    return (
        DataLoader(to_ds(X_tr,  y_tr),  batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(to_ds(X_val, y_val), batch_size=256),
        DataLoader(to_ds(X_te,  y_te),  batch_size=256),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_and_eval(
    num_layers: int,
    X_tr, y_tr, X_val, y_val, X_te, y_te,
    store_curves: bool = True,
) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    layer_size = TOTAL_NEURONS // num_layers

    model     = SparseFFNNDepth(
        X_tr.shape[1], num_layers, layer_size, NUM_CLASSES, SPARSITY
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader, test_loader = _make_loaders(
        X_tr, y_tr, X_val, y_val, X_te, y_te
    )

    train_losses, val_losses = [], []

    tracemalloc.start()
    t_start = time.perf_counter()

    for _ in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        if store_curves:
            train_losses.append(ep_loss / len(X_tr))

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                v_loss += criterion(model(xb), yb).item() * len(xb)
        val_losses.append(v_loss / len(X_val))

    t_end = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    sample = torch.tensor(X_te[:1], dtype=torch.float32).to(DEVICE)
    inf_times = []
    with torch.no_grad():
        for _ in range(200):
            t0 = time.perf_counter()
            model(sample)
            inf_times.append(time.perf_counter() - t0)

    active_nodes = count_active_nodes_per_layer(model)

    result = {
        "num_layers":        num_layers,
        "layer_size":        layer_size,
        "sparsity":          SPARSITY,
        "total_neurons":     TOTAL_NEURONS,
        "active_nodes":      active_nodes,            # list, per layer
        "total_active_nodes": sum(active_nodes),
        "n_params":          count_params(model),
        "active_params":     count_active_params(model),
        "active_macs":       count_active_macs(model),
        "train_time_s":      round(t_end - t_start, 3),
        "peak_mem_kb":       round(peak_mem / 1024, 2),
        "accuracy":          round(accuracy_score(all_true, all_preds), 6),
        "f1":                round(f1_score(all_true, all_preds, average="weighted"), 6),
        "precision":         round(precision_score(all_true, all_preds, average="weighted", zero_division=0), 6),
        "recall":            round(recall_score(all_true, all_preds, average="weighted"), 6),
        "best_val_loss":     round(min(val_losses), 6),
        "convergence_epoch": int(np.argmin(val_losses)) + 1,
        "inference_time_us": round(np.median(inf_times) * 1e6, 3),
    }
    if store_curves:
        result["train_losses"] = train_losses
        result["val_losses"]   = val_losses

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_vs_depth(df: pd.DataFrame, out_dir: Path):
    """All key metrics plotted against number of layers (the primary sweep axis)."""
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    fig.suptitle(
        f"Depth Saturation  |  total_neurons={TOTAL_NEURONS}, sparsity={int(SPARSITY*100)}%"
        "\nMetrics vs Number of Hidden Layers",
        fontsize=12, fontweight="bold", y=1.01,
    )

    def _line(ax, y_col, title, ylabel, color):
        ax.plot(df["num_layers"], df[y_col], "o-", color=color, lw=2, ms=7)
        for x, y in zip(df["num_layers"], df[y_col]):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=6.5)
        ax.set_title(title); ax.set_xlabel("Number of Hidden Layers")
        ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)
        ax.set_xticks(df["num_layers"])

    _line(axes[0, 0], "accuracy",          "Test Accuracy",             "Accuracy",    "crimson")
    _line(axes[0, 1], "f1",                "Weighted F1",               "F1",          "mediumpurple")
    _line(axes[0, 2], "best_val_loss",      "Best Validation Loss",      "Cross-Entropy","indianred")
    _line(axes[1, 0], "active_params",      "Active Parameters",         "Count",       "steelblue")
    _line(axes[1, 1], "active_macs",        "Active MACs / Forward Pass","MACs",        "darkorange")
    _line(axes[1, 2], "train_time_s",       "Training Time",             "Seconds",     "seagreen")
    _line(axes[2, 0], "inference_time_us",  "Inference Latency (µs)",    "Microseconds","goldenrod")
    _line(axes[2, 1], "convergence_epoch",  "Convergence Epoch",         "Epoch",       "slateblue")
    _line(axes[2, 2], "total_active_nodes", "Total Active Nodes",        "Count",       "teal")

    plt.tight_layout()
    plt.savefig(out_dir / "metrics_vs_depth.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_active_nodes_per_layer(df: pd.DataFrame, out_dir: Path):
    """Bar chart showing active node count per layer for each depth configuration."""
    n_configs = len(df)
    ncols = 4
    nrows = int(np.ceil(n_configs / ncols))
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, n_configs))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten() if n_configs > 1 else [axes]

    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes_flat[idx]
        active = row["active_nodes"]
        layer_size = int(row["layer_size"])
        x = range(1, len(active) + 1)
        ax.bar(x, active, color=palette[idx], alpha=0.85, edgecolor="k", linewidth=0.5)
        ax.axhline(layer_size, color="gray", lw=1, ls="--", label=f"full ({layer_size})")
        ax.set_title(
            f"{int(row['num_layers'])} layers × {layer_size} nodes\n"
            f"acc={row['accuracy']:.4f}",
            fontsize=10,
        )
        ax.set_xlabel("Layer index"); ax.set_ylabel("Active nodes")
        ax.set_ylim(0, layer_size + 2)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    for idx in range(n_configs, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Active Nodes per Layer  |  total_neurons={TOTAL_NEURONS}, sparsity={int(SPARSITY*100)}%",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "active_nodes_per_layer.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_curves(results: list, out_dir: Path):
    """Training and validation loss curves for all depth configurations."""
    palette = plt.cm.plasma(np.linspace(0.05, 0.95, len(results)))
    epoch_range = range(1, EPOCHS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"Loss Curves  |  total_neurons={TOTAL_NEURONS}, sparsity={int(SPARSITY*100)}%"
        "  |  each line = one depth",
        fontsize=12, fontweight="bold",
    )

    for r, c in zip(results, palette):
        lbl = f"{r['num_layers']}L×{r['layer_size']}n"
        if "train_losses" in r:
            axes[0].plot(epoch_range, r["train_losses"], color=c, lw=1.4, label=lbl, alpha=0.9)
        axes[1].plot(epoch_range, r["val_losses"], color=c, lw=1.4, label=lbl, alpha=0.9)

    for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.13, 0.5),
               title="Depth config", fontsize=7, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_saturation_curve(df: pd.DataFrame, out_dir: Path):
    """Focused saturation plot: Accuracy and Active Nodes vs depth, dual-axis."""
    fig, ax1 = plt.subplots(figsize=(12, 5))

    color_acc   = "crimson"
    color_nodes = "steelblue"

    ln1 = ax1.plot(df["num_layers"], df["accuracy"], "o-", color=color_acc,
                   lw=2.5, ms=8, label="Test Accuracy")
    ax1.set_xlabel("Number of Hidden Layers", fontsize=13)
    ax1.set_ylabel("Test Accuracy", fontsize=12, color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_xticks(df["num_layers"])

    ax2 = ax1.twinx()
    ln2 = ax2.plot(df["num_layers"], df["total_active_nodes"], "s--", color=color_nodes,
                   lw=2, ms=7, label="Total Active Nodes")
    ax2.set_ylabel("Total Active Nodes", fontsize=12, color=color_nodes)
    ax2.tick_params(axis="y", labelcolor=color_nodes)

    # Annotate accuracy points
    for x, y in zip(df["num_layers"], df["accuracy"]):
        ax1.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=7, color=color_acc)

    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    fig.suptitle(
        f"Depth Saturation Curve  |  total_neurons={TOTAL_NEURONS}, sparsity={int(SPARSITY*100)}%\n"
        f"layer_size = {TOTAL_NEURONS} ÷ num_layers",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_dir / "saturation_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_tradeoffs(df: pd.DataFrame, out_dir: Path):
    """Scatter plots of Accuracy vs efficiency metrics, coloured by depth."""
    n = len(df)
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, n))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Efficiency vs Accuracy  |  total_neurons={TOTAL_NEURONS}, sparsity={int(SPARSITY*100)}%"
        "  |  colour = num_layers",
        fontsize=12, fontweight="bold",
    )

    pairs = [
        ("active_params",    "Active Parameters",     axes[0]),
        ("train_time_s",     "Training Time (s)",     axes[1]),
        ("inference_time_us","Inference Latency (µs)", axes[2]),
    ]
    for xcol, xlabel, ax in pairs:
        sc = ax.scatter(df[xcol], df["accuracy"], c=range(n), cmap="plasma",
                        s=80, zorder=3, edgecolors="k", linewidths=0.5)
        for _, row in df.iterrows():
            ax.annotate(f"{int(row['num_layers'])}L",
                        (row[xcol], row["accuracy"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)
        ax.set_xlabel(xlabel); ax.set_ylabel("Test Accuracy")
        ax.grid(True, alpha=0.3)

    plt.colorbar(sc, ax=axes[-1], label="Depth index (increasing layers →)")
    plt.tight_layout()
    plt.savefig(out_dir / "tradeoffs.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Findings
# ─────────────────────────────────────────────────────────────────────────────

def write_findings(df: pd.DataFrame, out_dir: Path):
    lines = []
    sep   = "=" * 72

    lines += [sep,
              "  FFNN Depth Saturation — Key Findings",
              f"  total_neurons={TOTAL_NEURONS}  |  sparsity={int(SPARSITY*100)}%",
              sep, ""]

    lines.append(f"  {'layers':>7}  {'l_size':>7}  {'act_nodes':>10}  "
                 f"{'act_params':>11}  {'acc':>8}  {'f1':>8}  {'train(s)':>9}  {'inf(µs)':>9}")
    lines.append("  " + "-" * 80)
    for _, row in df.iterrows():
        lines.append(
            f"  {int(row['num_layers']):>7}  {int(row['layer_size']):>7}  "
            f"{int(row['total_active_nodes']):>10}  {int(row['active_params']):>11,}  "
            f"{row['accuracy']:>8.4f}  {row['f1']:>8.4f}  "
            f"{row['train_time_s']:>9.2f}  {row['inference_time_us']:>9.2f}"
        )

    lines.append("")

    best = df.loc[df["accuracy"].idxmax()]
    lines.append(
        f"  Best accuracy : {best['num_layers']:.0f} layers "
        f"(size={int(best['layer_size'])})  →  {best['accuracy']:.4f}"
    )

    worst = df.loc[df["accuracy"].idxmin()]
    lines.append(
        f"  Worst accuracy: {worst['num_layers']:.0f} layers "
        f"(size={int(worst['layer_size'])})  →  {worst['accuracy']:.4f}"
    )

    # Saturation: find depth where accuracy drops below 95% of peak
    peak_acc = df["accuracy"].max()
    thresh   = peak_acc * 0.95
    safe     = df[df["accuracy"] >= thresh]["num_layers"].max()
    lines.append(f"\n  Peak accuracy   : {peak_acc:.4f}")
    lines.append(f"  95% threshold   : {thresh:.4f}")
    lines.append(f"  Saturates beyond: {int(safe)} layers (accuracy falls <95% of peak after this)")

    # Active node collapse info
    lines.append("\n  Active nodes per layer at selected depths:")
    for _, row in df.iterrows():
        nodes_str = ", ".join(str(n) for n in row["active_nodes"])
        lines.append(
            f"    {int(row['num_layers']):>4}L × {int(row['layer_size']):>3}n/layer : [{nodes_str}]"
        )

    lines += ["", sep]
    summary = "\n".join(lines)
    print("\n" + summary)
    (out_dir / "findings.txt").write_text(summary + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Device  : {DEVICE}")
    print(f"Outputs : {_RUN_DIR}\n")
    print(
        f"DEPTH SATURATION EXPERIMENT\n"
        f"  total_neurons = {TOTAL_NEURONS}  (fixed)\n"
        f"  sparsity      = {int(SPARSITY*100)}%        (fixed)\n"
        f"  layer configs : {[(n, TOTAL_NEURONS // n) for n in NUM_LAYERS_LIST]}\n"
    )

    X_tr, X_val, X_te, y_tr, y_val, y_te = load_data()

    print(
        f"\nSweeping num_layers ∈ {NUM_LAYERS_LIST}"
        f"\nTotal runs: {len(NUM_LAYERS_LIST)}  |  epochs={EPOCHS}\n"
    )
    print(
        f"{'layers':>7} {'l_size':>7} {'act_nodes':>10} {'act_params':>11} "
        f"{'train(s)':>9} {'inf(µs)':>9} {'acc':>8} {'f1':>8}"
    )
    print("-" * 75)

    results = []

    for num_layers in NUM_LAYERS_LIST:
        layer_size = TOTAL_NEURONS // num_layers
        r = train_and_eval(
            num_layers, X_tr, y_tr, X_val, y_val, X_te, y_te,
            store_curves=True,
        )
        results.append(r)
        print(
            f"{r['num_layers']:>7}"
            f"{r['layer_size']:>8}"
            f"{r['total_active_nodes']:>11}"
            f"{r['active_params']:>12,}"
            f"{r['train_time_s']:>9.2f}"
            f"{r['inference_time_us']:>9.2f}"
            f"{r['accuracy']:>8.4f}"
            f"{r['f1']:>8.4f}"
        )

    # ── save CSV ──────────────────────────────────────────────────────────────
    scalar_cols = [
        "num_layers", "layer_size", "sparsity", "total_neurons",
        "total_active_nodes", "n_params", "active_params", "active_macs",
        "train_time_s", "peak_mem_kb", "accuracy", "f1", "precision", "recall",
        "best_val_loss", "convergence_epoch", "inference_time_us",
    ]
    df = pd.DataFrame([{c: r[c] for c in scalar_cols} for r in results])
    # Store active_nodes list as string for CSV
    df["active_nodes_per_layer"] = [str(r["active_nodes"]) for r in results]
    df.to_csv(_RUN_DIR / "results.csv", index=False)

    # Attach list column to results for plotting
    for i, r in enumerate(results):
        pass  # already in r["active_nodes"]

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nSaving plots ...")

    # Attach active_nodes list back to df for plotting
    df["active_nodes"] = [r["active_nodes"] for r in results]

    plot_metrics_vs_depth(df, _RUN_DIR)
    plot_active_nodes_per_layer(df, _RUN_DIR)
    plot_loss_curves(results, _RUN_DIR)
    plot_saturation_curve(df, _RUN_DIR)
    plot_tradeoffs(df, _RUN_DIR)

    # ── findings ──────────────────────────────────────────────────────────────
    write_findings(df, _RUN_DIR)

    print(f"\nAll outputs saved to: {_RUN_DIR}")
    print(
        "  results.csv  |  findings.txt\n"
        "  metrics_vs_depth.png  |  saturation_curve.png\n"
        "  active_nodes_per_layer.png  |  loss_curves.png  |  tradeoffs.png"
    )


if __name__ == "__main__":
    main()
