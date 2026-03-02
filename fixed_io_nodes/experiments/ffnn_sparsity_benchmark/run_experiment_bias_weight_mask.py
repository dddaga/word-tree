"""
FFNN Conduction Sparsity Benchmark — UCI Wine Quality (Red)
SUB-EXPERIMENT: Masking both Weights AND Biases (node-level)

In the original experiment, only individual weight connections are masked while
biases always survive. Here, sparsity operates at the **node level**: for each
layer, floor(sparsity × out_features) neurons are completely deactivated — ALL
their incoming weights AND their bias are zeroed. The remaining neurons keep
full connectivity. This simulates truly removing nodes from the graph.

The bias mask is NOT a separate mask — it is derived directly from the weight
mask: bias[i] is zeroed iff the entire i-th row of the weight mask is zero
(i.e. the node is inactive).

Sweep:
  hidden_size  ∈ {100, 200, …, 1000}   (10 values)
  sparsity     ∈ {0.0, 0.1, …, 1.0}    (11 values)
  Total        = 110 training runs

Extra plots (not in baseline):
  - 3D scatter: Accuracy vs Active Parameters vs Training & Inference Time
  - Per-hidden-size lines: Accuracy, Params, Train/Inference Time vs Sparsity

Outputs written to a timestamped sub-folder under:
    fixed_io_nodes/experiments/ffnn_sparsity_benchmark/runs_bias_weight_mask/

Run from repo root:
    python3 fixed_io_nodes/experiments/ffnn_sparsity_benchmark/run_experiment_bias_weight_mask.py
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
_RUN_DIR = _HERE / "runs_bias_weight_mask" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
_RUN_DIR.mkdir(parents=True, exist_ok=True)

# ── hyper-parameters ─────────────────────────────────────────────────────────
EPOCHS          = 100
LR              = 1e-3
BATCH_SIZE      = 64
HIDDEN_SIZES    = list(range(100, 10100, 100))
SPARSITY_LEVELS = [round(i / 10, 1) for i in range(0, 11)]   # 0.0 … 1.0
SEED            = 42
# DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE          = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SAMPLE_HIDDEN   = 500

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
    The remaining neurons keep ALL their incoming connections.

    Returns (weight_mask, bias_mask) where bias_mask is derived from weight_mask:
      bias_mask[i] = 1.0  iff  weight_mask[i, :].any()   (node is active)
      bias_mask[i] = 0.0  iff  weight_mask[i, :] is all zeros (node is dead)
    """
    if sparsity >= 1.0:
        return torch.zeros(out_features, in_features), torch.zeros(out_features)

    n_dead = int(sparsity * out_features)

    # Choose which nodes to deactivate
    node_active = torch.ones(out_features)
    if n_dead > 0:
        dead_idxs = rng.choice(out_features, size=n_dead, replace=False)
        node_active[dead_idxs] = 0.0

    # Weight mask: active nodes get a full row of 1s, dead nodes get all 0s
    weight_mask = node_active.unsqueeze(1).expand(out_features, in_features).clone()

    # Bias mask: directly derived from the weight mask (same node-level decision)
    bias_mask = node_active

    return weight_mask, bias_mask


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class SparseFFNN(nn.Module):
    """Single hidden layer FFNN with node-level conduction masks.

    Both weight rows AND bias entries are zeroed for deactivated nodes.
    Active nodes retain full connectivity. The mask is fixed at init.
    """

    def __init__(
        self, input_dim: int, hidden_size: int, output_dim: int, sparsity: float
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

        rng = np.random.default_rng(SEED)

        w_mask1, b_mask1 = make_node_mask(hidden_size, input_dim,   sparsity, rng)
        w_mask2, b_mask2 = make_node_mask(output_dim,  hidden_size, sparsity, rng)

        self.register_buffer("w_mask1", w_mask1)
        self.register_buffer("b_mask1", b_mask1)
        self.register_buffer("w_mask2", w_mask2)
        self.register_buffer("b_mask2", b_mask2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.linear(
            x,
            self.fc1.weight * self.w_mask1,
            self.fc1.bias * self.b_mask1,
        ))
        return F.linear(
            x,
            self.fc2.weight * self.w_mask2,
            self.fc2.bias * self.b_mask2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_active_params(model: SparseFFNN) -> int:
    """Active weight connections + active biases."""
    w = int(model.w_mask1.sum().item()) + int(model.w_mask2.sum().item())
    b = int(model.b_mask1.sum().item()) + int(model.b_mask2.sum().item())
    return w + b


def count_active_macs(model: SparseFFNN) -> int:
    """MACs proportional to active (unmasked) weight connections only."""
    return int(model.w_mask1.sum().item()) + int(model.w_mask2.sum().item())


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
    hidden_size: int,
    sparsity: float,
    X_tr, y_tr, X_val, y_val, X_te, y_te,
    store_curves: bool = False,
) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model     = SparseFFNN(X_tr.shape[1], hidden_size, NUM_CLASSES, sparsity).to(DEVICE)
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

    result = {
        "hidden_size":       hidden_size,
        "sparsity":          sparsity,
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

def plot_heatmap(pivot: pd.DataFrame, metric: str, title: str, out_dir: Path, fname: str):
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(s*100)}%" for s in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int))
    ax.set_xlabel("Sparsity"); ax.set_ylabel("Hidden Size")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label=metric)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color="black")

    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()


def plot_accuracy_vs_sparsity(df: pd.DataFrame, out_dir: Path):
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(HIDDEN_SIZES)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Test Accuracy vs Sparsity (node-level mask)  |  each line = one hidden width",
        fontsize=12, fontweight="bold",
    )

    for h, c in zip(HIDDEN_SIZES, palette):
        sub = df[df["hidden_size"] == h].sort_values("sparsity")
        axes[0].plot(sub["sparsity"] * 100, sub["accuracy"],
                     "o-", color=c, lw=1.6, ms=4, label=f"h={h}")
        axes[1].plot(sub["sparsity"] * 100, sub["f1"],
                     "o-", color=c, lw=1.6, ms=4, label=f"h={h}")

    for ax, ylabel in zip(axes, ["Test Accuracy", "Weighted F1"]):
        ax.set_xlabel("Sparsity (%)"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3); ax.set_xlim(-2, 102)

    axes[0].set_title("Accuracy vs Sparsity")
    axes[1].set_title("F1 vs Sparsity")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.12, 0.5),
               title="Hidden Size", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_sparsity.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_accuracy_vs_hidden(df: pd.DataFrame, out_dir: Path):
    palette = plt.cm.plasma(np.linspace(0.05, 0.95, len(SPARSITY_LEVELS)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Test Accuracy vs Hidden Width (node-level mask)  |  each line = one sparsity level",
        fontsize=12, fontweight="bold",
    )

    for s, c in zip(SPARSITY_LEVELS, palette):
        sub = df[df["sparsity"] == s].sort_values("hidden_size")
        lbl = f"{int(s*100)}%"
        axes[0].plot(sub["hidden_size"], sub["accuracy"],
                     "o-", color=c, lw=1.6, ms=4, label=lbl)
        axes[1].plot(sub["hidden_size"], sub["f1"],
                     "o-", color=c, lw=1.6, ms=4, label=lbl)

    for ax, ylabel in zip(axes, ["Test Accuracy", "Weighted F1"]):
        ax.set_xlabel("Hidden Neurons"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Accuracy vs Hidden Width")
    axes[1].set_title("F1 vs Hidden Width")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.1, 0.5),
               title="Sparsity", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_hidden.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_dashboard(df: pd.DataFrame, out_dir: Path):
    """Metrics averaged over hidden sizes, plotted against sparsity."""
    grouped = df.groupby("sparsity").mean(numeric_only=True).reset_index()
    sp_pct  = grouped["sparsity"] * 100

    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    fig.suptitle(
        "FFNN Node-Level Mask Sparsity Benchmark  |  metrics averaged over hidden widths",
        fontsize=12, fontweight="bold", y=1.01,
    )

    def _line(ax, y_col, title, ylabel, color):
        ax.plot(sp_pct, grouped[y_col], "o-", color=color, lw=2, ms=6)
        ax.set_title(title); ax.set_xlabel("Sparsity (%)"); ax.set_ylabel(ylabel)
        ax.set_xlim(-2, 102); ax.grid(True, alpha=0.3)

    _line(axes[0, 0], "accuracy",         "Test Accuracy",             "Accuracy",    "crimson")
    _line(axes[0, 1], "f1",               "Weighted F1",               "F1",          "mediumpurple")
    _line(axes[0, 2], "best_val_loss",     "Best Validation Loss",      "Cross-Entropy","indianred")
    _line(axes[1, 0], "active_params",     "Active Parameters",         "Count",       "steelblue")
    _line(axes[1, 1], "active_macs",       "Active MACs / Forward Pass","MACs",        "darkorange")
    _line(axes[1, 2], "train_time_s",      "Training Time",             "Seconds",     "seagreen")
    _line(axes[2, 0], "inference_time_us", "Inference Latency (µs)",    "Microseconds","goldenrod")
    _line(axes[2, 1], "convergence_epoch", "Convergence Epoch",         "Epoch",       "slateblue")
    _line(axes[2, 2], "peak_mem_kb",       "Peak Memory (KB)",          "KB",          "teal")

    plt.tight_layout()
    plt.savefig(out_dir / "metrics_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_tradeoff(df: pd.DataFrame, out_dir: Path):
    sp_pct     = df["sparsity"] * 100
    scatter_kw = dict(c=sp_pct, cmap="RdYlGn_r", s=60, alpha=0.8, zorder=3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Complexity vs Accuracy Trade-off (node-level mask)", fontsize=12, fontweight="bold")

    ax = axes[0]
    sc = ax.scatter(df["active_params"], df["accuracy"], **scatter_kw)
    ax.set_xlabel("Active Parameter Count"); ax.set_ylabel("Test Accuracy")
    ax.set_title("Active Params vs Accuracy")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(df["active_macs"], df["accuracy"], **scatter_kw)
    ax.set_xlabel("Active MACs"); ax.set_ylabel("Test Accuracy")
    ax.set_title("Active MACs vs Accuracy")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, alpha=0.3)

    plt.colorbar(sc, ax=axes, label="Sparsity (%)", shrink=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / "tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_sample_loss_curves(sample_curves: list, out_dir: Path):
    """Loss curves for hidden_size=SAMPLE_HIDDEN across all sparsity levels."""
    epoch_range = range(1, EPOCHS + 1)
    palette     = plt.cm.RdYlGn_r(np.linspace(0.05, 0.95, len(sample_curves)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"Loss Curves (node-level mask)  |  hidden_size={SAMPLE_HIDDEN}, varying sparsity",
        fontsize=12, fontweight="bold",
    )

    for r, c in zip(sample_curves, palette):
        lbl = f"{int(r['sparsity']*100)}%"
        axes[0].plot(epoch_range, r["train_losses"], color=c, lw=1.4, label=lbl, alpha=0.9)
        axes[1].plot(epoch_range, r["val_losses"],   color=c, lw=1.4, label=lbl, alpha=0.9)

    for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.1, 0.5),
               title="Sparsity", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "sample_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Accuracy vs Active Parameters (sparsity) vs Training & Inference Time
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_vs_params_vs_time(df: pd.DataFrame, out_dir: Path):
    """3D scatter + 2D bubble plots showing how accuracy varies jointly with
    active parameters (driven by sparsity) and training/inference time."""

    sp_pct = df["sparsity"] * 100

    fig = plt.figure(figsize=(14, 10))

    # ── 3D: Accuracy vs Active Params vs Training Time ────────────────────
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    sc = ax.scatter(
        df["active_params"], df["train_time_s"], df["accuracy"],
        c=sp_pct, cmap="RdYlGn_r", s=50, alpha=0.85, edgecolors="k", linewidths=0.3,
    )
    ax.set_xlabel("Active Params", fontsize=9, labelpad=8)
    ax.set_ylabel("Train Time (s)", fontsize=9, labelpad=8)
    ax.set_zlabel("Accuracy", fontsize=9, labelpad=8)
    ax.set_title("Accuracy vs Active Params vs Train Time", fontsize=10, fontweight="bold")
    fig.colorbar(sc, ax=ax, shrink=0.55, label="Sparsity (%)", pad=0.12)

    # ── 3D: Accuracy vs Active Params vs Inference Time ───────────────────
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    sc2 = ax2.scatter(
        df["active_params"], df["inference_time_us"], df["accuracy"],
        c=sp_pct, cmap="RdYlGn_r", s=50, alpha=0.85, edgecolors="k", linewidths=0.3,
    )
    ax2.set_xlabel("Active Params", fontsize=9, labelpad=8)
    ax2.set_ylabel("Inference (µs)", fontsize=9, labelpad=8)
    ax2.set_zlabel("Accuracy", fontsize=9, labelpad=8)
    ax2.set_title("Accuracy vs Active Params vs Inference Time", fontsize=10, fontweight="bold")
    fig.colorbar(sc2, ax=ax2, shrink=0.55, label="Sparsity (%)", pad=0.12)

    # ── 2D: Accuracy vs Active Params, bubble size = train time ───────────
    ax3 = fig.add_subplot(2, 2, 3)
    t_range = df["train_time_s"].max() - df["train_time_s"].min() + 1e-9
    sizes = (df["train_time_s"] - df["train_time_s"].min()) / t_range * 200 + 20
    sc3 = ax3.scatter(
        df["active_params"], df["accuracy"],
        c=sp_pct, cmap="RdYlGn_r", s=sizes, alpha=0.7, edgecolors="k", linewidths=0.3,
    )
    ax3.set_xlabel("Active Parameters")
    ax3.set_ylabel("Test Accuracy")
    ax3.set_title("Accuracy vs Params  (bubble = train time)", fontsize=10, fontweight="bold")
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax3.grid(True, alpha=0.3)
    fig.colorbar(sc3, ax=ax3, label="Sparsity (%)", shrink=0.8)

    # ── 2D: Accuracy vs Active Params, bubble size = inference time ───────
    ax4 = fig.add_subplot(2, 2, 4)
    i_range = df["inference_time_us"].max() - df["inference_time_us"].min() + 1e-9
    sizes_inf = (df["inference_time_us"] - df["inference_time_us"].min()) / i_range * 200 + 20
    sc4 = ax4.scatter(
        df["active_params"], df["accuracy"],
        c=sp_pct, cmap="RdYlGn_r", s=sizes_inf, alpha=0.7, edgecolors="k", linewidths=0.3,
    )
    ax4.set_xlabel("Active Parameters")
    ax4.set_ylabel("Test Accuracy")
    ax4.set_title("Accuracy vs Params  (bubble = inference time)", fontsize=10, fontweight="bold")
    ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax4.grid(True, alpha=0.3)
    fig.colorbar(sc4, ax=ax4, label="Sparsity (%)", shrink=0.8)

    fig.suptitle(
        "Accuracy vs Parameters (Sparsity) vs Training & Inference Time\n"
        "(node-level weight+bias mask)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_params_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Per-hidden-size line plot: Accuracy & Time vs Sparsity ────────────
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(HIDDEN_SIZES)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Accuracy & Time vs Sparsity per Hidden Size  (node-level mask)",
        fontsize=13, fontweight="bold",
    )

    for h, c in zip(HIDDEN_SIZES, palette):
        sub = df[df["hidden_size"] == h].sort_values("sparsity")
        sp  = sub["sparsity"] * 100
        axes[0, 0].plot(sp, sub["accuracy"],         "o-", color=c, lw=1.4, ms=3, label=f"h={h}")
        axes[0, 1].plot(sp, sub["active_params"],     "o-", color=c, lw=1.4, ms=3, label=f"h={h}")
        axes[1, 0].plot(sp, sub["train_time_s"],      "o-", color=c, lw=1.4, ms=3, label=f"h={h}")
        axes[1, 1].plot(sp, sub["inference_time_us"], "o-", color=c, lw=1.4, ms=3, label=f"h={h}")

    titles  = ["Test Accuracy", "Active Parameters", "Training Time (s)", "Inference Latency (µs)"]
    ylabels = ["Accuracy", "Count", "Seconds", "Microseconds"]
    for ax, t, yl in zip(axes.flat, titles, ylabels):
        ax.set_title(t); ax.set_xlabel("Sparsity (%)"); ax.set_ylabel(yl)
        ax.set_xlim(-2, 102); ax.grid(True, alpha=0.3)

    axes[0, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.1, 0.5),
               title="Hidden Size", fontsize=7, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_params_time_lines.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Findings
# ─────────────────────────────────────────────────────────────────────────────

def write_findings(df: pd.DataFrame, out_dir: Path):
    lines = []
    sep   = "=" * 68

    lines += [sep, "  FFNN Node-Level Mask Sparsity Benchmark — Key Findings", sep, ""]

    lines.append("  NOTE: This experiment masks BOTH weights AND biases at the node")
    lines.append("  level. Deactivated nodes have ALL incoming weights + bias zeroed,")
    lines.append("  unlike the baseline which only masks individual weight connections")
    lines.append("  while biases always survive.\n")

    # Best run overall
    best = df.loc[df["accuracy"].idxmax()]
    lines.append(
        f"  Global best accuracy : h={int(best.hidden_size):4d}, "
        f"sparsity={int(best.sparsity*100):3d}%  → {best.accuracy:.4f}"
    )

    # Dense baseline (sparsity=0) per hidden size
    dense = df[df["sparsity"] == 0.0].set_index("hidden_size")["accuracy"]
    lines.append(f"\n  Dense (0 % sparsity) accuracy range : "
                 f"{dense.min():.4f} – {dense.max():.4f}")

    # Resilience table
    lines.append("\n  Resilience: highest sparsity ≤ 95% of dense accuracy")
    lines.append(f"  {'h':>6}  {'dense acc':>10}  {'95% threshold':>14}  {'max safe sparsity':>18}")
    lines.append("  " + "-" * 54)
    for h in HIDDEN_SIZES:
        sub    = df[df["hidden_size"] == h].sort_values("sparsity")
        d_acc  = float(sub[sub["sparsity"] == 0.0]["accuracy"].iloc[0])
        thresh = d_acc * 0.95
        safe   = sub[sub["accuracy"] >= thresh]["sparsity"].max()
        lines.append(
            f"  {h:>6}  {d_acc:>10.4f}  {thresh:>14.4f}  {int(safe*100):>17}%"
        )

    # Efficiency comparisons
    for s_label, s_val in [("0%", 0.0), ("50%", 0.5)]:
        sub = df[df["sparsity"] == s_val]
        lines.append(
            f"\n  Avg active params at sparsity={s_label:>4s} : "
            f"{sub['active_params'].mean():,.0f}"
        )
        lines.append(
            f"  Avg accuracy      at sparsity={s_label:>4s} : "
            f"{sub['accuracy'].mean():.4f}"
        )
        lines.append(
            f"  Avg train time    at sparsity={s_label:>4s} : "
            f"{sub['train_time_s'].mean():.2f}s"
        )
        lines.append(
            f"  Avg inference     at sparsity={s_label:>4s} : "
            f"{sub['inference_time_us'].mean():.2f}µs"
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
    print("SUB-EXPERIMENT: Node-level masking (weights + bias masked together)\n")

    X_tr, X_val, X_te, y_tr, y_val, y_te = load_data()

    total_runs = len(HIDDEN_SIZES) * len(SPARSITY_LEVELS)
    print(
        f"\nSweeping hidden_size ∈ {HIDDEN_SIZES}"
        f"\n        sparsity    ∈ {[f'{int(s*100)}%' for s in SPARSITY_LEVELS]}"
        f"\nTotal runs: {total_runs}  |  epochs={EPOCHS}\n"
    )
    print(
        f"{'hidden':>8} {'sparsity':>9} {'active_p':>10} "
        f"{'train(s)':>9} {'infer(µs)':>10} {'acc':>8} {'f1':>8}"
    )
    print("-" * 68)

    results       = []
    sample_curves = []

    for h in HIDDEN_SIZES:
        for s in SPARSITY_LEVELS:
            is_sample = (h == SAMPLE_HIDDEN)
            r = train_and_eval(
                h, s, X_tr, y_tr, X_val, y_val, X_te, y_te,
                store_curves=is_sample,
            )
            results.append(r)
            if is_sample:
                sample_curves.append(r)

            print(
                f"{r['hidden_size']:>8}"
                f"{int(r['sparsity']*100):>8}%"
                f"{r['active_params']:>10,}"
                f"{r['train_time_s']:>9.2f}"
                f"{r['inference_time_us']:>10.2f}"
                f"{r['accuracy']:>8.4f}"
                f"{r['f1']:>8.4f}"
            )

    # ── save CSV ──────────────────────────────────────────────────────────────
    scalar_cols = [
        "hidden_size", "sparsity", "n_params", "active_params", "active_macs",
        "train_time_s", "peak_mem_kb", "accuracy", "f1", "precision", "recall",
        "best_val_loss", "convergence_epoch", "inference_time_us",
    ]
    df = pd.DataFrame([{c: r[c] for c in scalar_cols} for r in results])
    df.to_csv(_RUN_DIR / "results.csv", index=False)

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nSaving plots ...")

    pivot_acc = df.pivot(index="hidden_size", columns="sparsity", values="accuracy")
    pivot_f1  = df.pivot(index="hidden_size", columns="sparsity", values="f1")

    plot_heatmap(pivot_acc, "Accuracy",
                 "Test Accuracy (node-level mask)  |  rows=hidden_size, cols=sparsity",
                 out_dir=_RUN_DIR, fname="heatmap_accuracy.png")
    plot_heatmap(pivot_f1,  "Weighted F1",
                 "Weighted F1 (node-level mask)    |  rows=hidden_size, cols=sparsity",
                 out_dir=_RUN_DIR, fname="heatmap_f1.png")

    plot_accuracy_vs_sparsity(df, _RUN_DIR)
    plot_accuracy_vs_hidden(df, _RUN_DIR)
    plot_metrics_dashboard(df, _RUN_DIR)
    plot_tradeoff(df, _RUN_DIR)
    plot_sample_loss_curves(sample_curves, _RUN_DIR)
    plot_accuracy_vs_params_vs_time(df, _RUN_DIR)

    # ── findings ──────────────────────────────────────────────────────────────
    write_findings(df, _RUN_DIR)

    print(f"\nAll outputs saved to: {_RUN_DIR}")
    print(
        "  results.csv  |  findings.txt\n"
        "  heatmap_accuracy.png  |  heatmap_f1.png\n"
        "  accuracy_vs_sparsity.png  |  accuracy_vs_hidden.png\n"
        "  metrics_dashboard.png  |  tradeoff.png  |  sample_loss_curves.png\n"
        "  accuracy_vs_params_vs_time.png  |  accuracy_params_time_lines.png"
    )


if __name__ == "__main__":
    main()
