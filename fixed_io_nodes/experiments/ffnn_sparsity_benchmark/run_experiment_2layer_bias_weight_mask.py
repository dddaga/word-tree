"""
FFNN Conduction Sparsity Benchmark — UCI Wine Quality (Red)
SUB-EXPERIMENT: Two Hidden Layers with Node-Level Masking (Weights + Biases)

This experiment extends the single hidden layer node-level masking experiment
to a TWO hidden layer architecture. The total neuron budget is split equally
between the two layers:
    total_neurons = 100, 120, 140, …, 240
    layer1 = total_neurons // 2
    layer2 = total_neurons // 2

Node-level sparsity is applied to BOTH hidden layers independently:
for each layer, floor(sparsity × layer_size) neurons are completely
deactivated — ALL their incoming weights AND their bias are zeroed.
The output layer is NOT masked (all class logits remain active).

Sweep:
  total_neurons  ∈ {100, 120, 140, …, 240}   (8 values)
  sparsity       ∈ {0.0, 0.1, …, 1.0}        (11 values)
  Total          = 88 training runs

Outputs written to a timestamped sub-folder under:
    fixed_io_nodes/experiments/ffnn_sparsity_benchmark/runs_2layer_bias_weight_mask/

Run from repo root:
    python3 fixed_io_nodes/experiments/ffnn_sparsity_benchmark/run_experiment_2layer_bias_weight_mask.py
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
_RUN_DIR = _HERE / "runs_2layer_bias_weight_mask" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
_RUN_DIR.mkdir(parents=True, exist_ok=True)

# ── hyper-parameters ─────────────────────────────────────────────────────────
EPOCHS          = 100
LR              = 1e-3
BATCH_SIZE      = 64
TOTAL_NEURONS   = list(range(100, 241, 20))    # 100, 120, …, 240
SPARSITY_LEVELS = [round(i / 10, 1) for i in range(0, 11)]   # 0.0 … 1.0
SEED            = 42
DEVICE          = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SAMPLE_TOTAL    = 200   # total neurons for sample loss curves

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
    bias_mask = node_active

    return weight_mask, bias_mask


# ─────────────────────────────────────────────────────────────────────────────
# Model — Two Hidden Layers
# ─────────────────────────────────────────────────────────────────────────────

class SparseFFNN2Layer(nn.Module):
    """Two hidden layer FFNN with node-level conduction masks.

    Both weight rows AND bias entries are zeroed for deactivated nodes
    in BOTH hidden layers. The output layer is NOT masked.
    """

    def __init__(
        self, input_dim: int, h1_size: int, h2_size: int,
        output_dim: int, sparsity: float,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, output_dim)

        rng = np.random.default_rng(SEED)

        # Sparsity masks for both hidden layers
        w_mask1, b_mask1 = make_node_mask(h1_size, input_dim, sparsity, rng)
        w_mask2, b_mask2 = make_node_mask(h2_size, h1_size,   sparsity, rng)

        # Output layer: NO sparsity mask
        w_mask3 = torch.ones(output_dim, h2_size)
        b_mask3 = torch.ones(output_dim)

        self.register_buffer("w_mask1", w_mask1)
        self.register_buffer("b_mask1", b_mask1)
        self.register_buffer("w_mask2", w_mask2)
        self.register_buffer("b_mask2", b_mask2)
        self.register_buffer("w_mask3", w_mask3)
        self.register_buffer("b_mask3", b_mask3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.linear(
            x,
            self.fc1.weight * self.w_mask1,
            self.fc1.bias * self.b_mask1,
        ))
        x = F.relu(F.linear(
            x,
            self.fc2.weight * self.w_mask2,
            self.fc2.bias * self.b_mask2,
        ))
        return F.linear(
            x,
            self.fc3.weight * self.w_mask3,
            self.fc3.bias * self.b_mask3,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_active_params(model: SparseFFNN2Layer) -> int:
    """Active weight connections + active biases."""
    w = (int(model.w_mask1.sum().item())
         + int(model.w_mask2.sum().item())
         + int(model.w_mask3.sum().item()))
    b = (int(model.b_mask1.sum().item())
         + int(model.b_mask2.sum().item())
         + int(model.b_mask3.sum().item()))
    return w + b


def count_active_macs(model: SparseFFNN2Layer) -> int:
    """MACs proportional to active (unmasked) weight connections only."""
    return (int(model.w_mask1.sum().item())
            + int(model.w_mask2.sum().item())
            + int(model.w_mask3.sum().item()))


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
    total_neurons: int,
    sparsity: float,
    X_tr, y_tr, X_val, y_val, X_te, y_te,
    store_curves: bool = False,
) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    h1 = total_neurons // 2
    h2 = total_neurons // 2

    model     = SparseFFNN2Layer(X_tr.shape[1], h1, h2, NUM_CLASSES, sparsity).to(DEVICE)
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
        "total_neurons":     total_neurons,
        "h1_size":           h1,
        "h2_size":           h2,
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
    ax.set_xlabel("Sparsity"); ax.set_ylabel("Total Neurons (2 layers)")
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
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(TOTAL_NEURONS)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Test Accuracy vs Sparsity (2-layer node-level mask)  |  each line = one total width",
        fontsize=12, fontweight="bold",
    )

    for tn, c in zip(TOTAL_NEURONS, palette):
        sub = df[df["total_neurons"] == tn].sort_values("sparsity")
        axes[0].plot(sub["sparsity"] * 100, sub["accuracy"],
                     "o-", color=c, lw=1.6, ms=4, label=f"n={tn}")
        axes[1].plot(sub["sparsity"] * 100, sub["f1"],
                     "o-", color=c, lw=1.6, ms=4, label=f"n={tn}")

    for ax, ylabel in zip(axes, ["Test Accuracy", "Weighted F1"]):
        ax.set_xlabel("Sparsity (%)"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3); ax.set_xlim(-2, 102)

    axes[0].set_title("Accuracy vs Sparsity")
    axes[1].set_title("F1 vs Sparsity")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.12, 0.5),
               title="Total Neurons", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_sparsity.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_accuracy_vs_neurons(df: pd.DataFrame, out_dir: Path):
    palette = plt.cm.plasma(np.linspace(0.05, 0.95, len(SPARSITY_LEVELS)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Test Accuracy vs Total Neurons (2-layer node-level mask)  |  each line = one sparsity level",
        fontsize=12, fontweight="bold",
    )

    for s, c in zip(SPARSITY_LEVELS, palette):
        sub = df[df["sparsity"] == s].sort_values("total_neurons")
        lbl = f"{int(s*100)}%"
        axes[0].plot(sub["total_neurons"], sub["accuracy"],
                     "o-", color=c, lw=1.6, ms=4, label=lbl)
        axes[1].plot(sub["total_neurons"], sub["f1"],
                     "o-", color=c, lw=1.6, ms=4, label=lbl)

    for ax, ylabel in zip(axes, ["Test Accuracy", "Weighted F1"]):
        ax.set_xlabel("Total Hidden Neurons"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Accuracy vs Total Neurons")
    axes[1].set_title("F1 vs Total Neurons")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.1, 0.5),
               title="Sparsity", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_neurons.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_dashboard(df: pd.DataFrame, out_dir: Path):
    """Metrics averaged over total neuron counts, plotted against sparsity."""
    grouped = df.groupby("sparsity").mean(numeric_only=True).reset_index()
    sp_pct  = grouped["sparsity"] * 100

    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    fig.suptitle(
        "FFNN 2-Layer Node-Level Mask Sparsity Benchmark  |  metrics averaged over neuron counts",
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
    fig.suptitle("Complexity vs Accuracy Trade-off (2-layer node-level mask)", fontsize=12, fontweight="bold")

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
    """Loss curves for total_neurons=SAMPLE_TOTAL across all sparsity levels."""
    epoch_range = range(1, EPOCHS + 1)
    palette     = plt.cm.RdYlGn_r(np.linspace(0.05, 0.95, len(sample_curves)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"Loss Curves (2-layer node-level mask)  |  total_neurons={SAMPLE_TOTAL} ({SAMPLE_TOTAL//2}+{SAMPLE_TOTAL//2}), varying sparsity",
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
        "(2-layer node-level weight+bias mask)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_params_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Per-total-neurons line plot: Accuracy & Time vs Sparsity ──────────
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(TOTAL_NEURONS)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Accuracy & Time vs Sparsity per Total Neurons  (2-layer node-level mask)",
        fontsize=13, fontweight="bold",
    )

    for tn, c in zip(TOTAL_NEURONS, palette):
        sub = df[df["total_neurons"] == tn].sort_values("sparsity")
        sp  = sub["sparsity"] * 100
        axes[0, 0].plot(sp, sub["accuracy"],         "o-", color=c, lw=1.4, ms=3, label=f"n={tn}")
        axes[0, 1].plot(sp, sub["active_params"],     "o-", color=c, lw=1.4, ms=3, label=f"n={tn}")
        axes[1, 0].plot(sp, sub["train_time_s"],      "o-", color=c, lw=1.4, ms=3, label=f"n={tn}")
        axes[1, 1].plot(sp, sub["inference_time_us"], "o-", color=c, lw=1.4, ms=3, label=f"n={tn}")

    titles  = ["Test Accuracy", "Active Parameters", "Training Time (s)", "Inference Latency (µs)"]
    ylabels = ["Accuracy", "Count", "Seconds", "Microseconds"]
    for ax, t, yl in zip(axes.flat, titles, ylabels):
        ax.set_title(t); ax.set_xlabel("Sparsity (%)"); ax.set_ylabel(yl)
        ax.set_xlim(-2, 102); ax.grid(True, alpha=0.3)

    axes[0, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.1, 0.5),
               title="Total Neurons", fontsize=7, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_params_time_lines.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Per-sparsity subplots & variance plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_loss_per_sparsity(df: pd.DataFrame, out_dir: Path):
    """Separate subplot for each sparsity level: Accuracy and Val Loss vs Total Neurons."""
    sparsity_levels = sorted(df["sparsity"].unique())
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(sparsity_levels) - 1)) for i in range(len(sparsity_levels))]

    n_sp = len(sparsity_levels)
    ncols = 4
    nrows = int(np.ceil(n_sp / ncols))

    # ── Accuracy vs Total Neurons per sparsity ──
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, (sp, color) in enumerate(zip(sparsity_levels, colors)):
        ax = axes_flat[idx]
        sub = df[df["sparsity"] == sp].sort_values("total_neurons")
        ax.plot(sub["total_neurons"], sub["accuracy"], color=color, linewidth=1.2, alpha=0.85)
        ax.set_title(f"Sparsity = {sp:.1f}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        if idx % ncols == 0:
            ax.set_ylabel("Accuracy", fontsize=10)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Total Neurons", fontsize=10)

    for idx in range(n_sp, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Accuracy vs Total Neurons (separate per Sparsity level)  [2-layer]", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_neurons_per_sparsity.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── Val Loss vs Total Neurons per sparsity ──
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True)
    axes2_flat = axes2.flatten()

    for idx, (sp, color) in enumerate(zip(sparsity_levels, colors)):
        ax2 = axes2_flat[idx]
        sub = df[df["sparsity"] == sp].sort_values("total_neurons")
        ax2.plot(sub["total_neurons"], sub["best_val_loss"], color=color, linewidth=1.2, alpha=0.85)
        ax2.set_title(f"Sparsity = {sp:.1f}", fontsize=11)
        ax2.grid(True, alpha=0.3)
        if idx % ncols == 0:
            ax2.set_ylabel("Best Val Loss", fontsize=10)
        if idx >= (nrows - 1) * ncols:
            ax2.set_xlabel("Total Neurons", fontsize=10)

    for idx in range(n_sp, len(axes2_flat)):
        axes2_flat[idx].set_visible(False)

    fig2.suptitle("Validation Loss vs Total Neurons (separate per Sparsity level)  [2-layer]", fontsize=15, y=1.01)
    fig2.tight_layout()
    fig2.savefig(out_dir / "loss_vs_neurons_per_sparsity.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_variance_plots(df: pd.DataFrame, out_dir: Path):
    """Accuracy and Loss variance across sparsity levels for each total neuron count."""
    # ── Accuracy variance ──
    variance_df = df.groupby("total_neurons")["accuracy"].var().reset_index()
    variance_df.columns = ["total_neurons", "accuracy_variance"]

    fig3, ax3 = plt.subplots(figsize=(14, 6))
    ax3.bar(variance_df["total_neurons"], variance_df["accuracy_variance"],
            width=15, color="steelblue", alpha=0.8)
    ax3.set_xlabel("Total Neurons", fontsize=13)
    ax3.set_ylabel("Variance of Accuracy (across sparsity levels)", fontsize=13)
    ax3.set_title("Accuracy Variance across Sparsity levels vs Total Neurons  [2-layer]", fontsize=15)
    ax3.grid(True, alpha=0.3, axis="y")
    fig3.tight_layout()
    fig3.savefig(out_dir / "accuracy_variance_vs_neurons.png", dpi=200)
    plt.close()

    # ── Loss variance ──
    loss_var_df = df.groupby("total_neurons")["best_val_loss"].var().reset_index()
    loss_var_df.columns = ["total_neurons", "loss_variance"]

    fig4, ax4 = plt.subplots(figsize=(14, 6))
    ax4.bar(loss_var_df["total_neurons"], loss_var_df["loss_variance"],
            width=15, color="indianred", alpha=0.8)
    ax4.set_xlabel("Total Neurons", fontsize=13)
    ax4.set_ylabel("Variance of Val Loss (across sparsity levels)", fontsize=13)
    ax4.set_title("Validation Loss Variance across Sparsity levels vs Total Neurons  [2-layer]", fontsize=15)
    ax4.grid(True, alpha=0.3, axis="y")
    fig4.tight_layout()
    fig4.savefig(out_dir / "loss_variance_vs_neurons.png", dpi=200)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Findings
# ─────────────────────────────────────────────────────────────────────────────

def write_findings(df: pd.DataFrame, out_dir: Path):
    lines = []
    sep   = "=" * 72

    lines += [sep, "  FFNN 2-Layer Node-Level Mask Sparsity Benchmark — Key Findings", sep, ""]

    lines.append("  Architecture: 2 hidden layers, each with total_neurons // 2 neurons.")
    lines.append("  Node-level masking applied to BOTH hidden layers (weights + bias).")
    lines.append("  Output layer is NOT masked.\n")

    # Best run overall
    best = df.loc[df["accuracy"].idxmax()]
    lines.append(
        f"  Global best accuracy : total={int(best.total_neurons):4d} "
        f"({int(best.h1_size)}+{int(best.h2_size)}), "
        f"sparsity={int(best.sparsity*100):3d}%  → {best.accuracy:.4f}"
    )

    # Dense baseline (sparsity=0) per neuron count
    dense = df[df["sparsity"] == 0.0].set_index("total_neurons")["accuracy"]
    lines.append(f"\n  Dense (0 % sparsity) accuracy range : "
                 f"{dense.min():.4f} – {dense.max():.4f}")

    # Resilience table
    lines.append("\n  Resilience: highest sparsity ≤ 95% of dense accuracy")
    lines.append(f"  {'total':>7}  {'(h1+h2)':>9}  {'dense acc':>10}  {'95% thresh':>11}  {'max safe sparsity':>18}")
    lines.append("  " + "-" * 60)
    for tn in TOTAL_NEURONS:
        sub    = df[df["total_neurons"] == tn].sort_values("sparsity")
        d_acc  = float(sub[sub["sparsity"] == 0.0]["accuracy"].iloc[0])
        thresh = d_acc * 0.95
        safe   = sub[sub["accuracy"] >= thresh]["sparsity"].max()
        h = tn // 2
        lines.append(
            f"  {tn:>7}  {f'{h}+{h}':>9}  {d_acc:>10.4f}  {thresh:>11.4f}  {int(safe*100):>17}%"
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
    print("SUB-EXPERIMENT: 2-layer node-level masking (weights + bias masked together)\n")

    X_tr, X_val, X_te, y_tr, y_val, y_te = load_data()

    total_runs = len(TOTAL_NEURONS) * len(SPARSITY_LEVELS)
    print(
        f"\nSweeping total_neurons ∈ {TOTAL_NEURONS}"
        f"\n        sparsity       ∈ {[f'{int(s*100)}%' for s in SPARSITY_LEVELS]}"
        f"\nTotal runs: {total_runs}  |  epochs={EPOCHS}\n"
    )
    print(
        f"{'total':>8} {'(h1+h2)':>9} {'sparsity':>9} {'active_p':>10} "
        f"{'train(s)':>9} {'infer(µs)':>10} {'acc':>8} {'f1':>8}"
    )
    print("-" * 78)

    results       = []
    sample_curves = []

    for tn in TOTAL_NEURONS:
        for s in SPARSITY_LEVELS:
            is_sample = (tn == SAMPLE_TOTAL)
            r = train_and_eval(
                tn, s, X_tr, y_tr, X_val, y_val, X_te, y_te,
                store_curves=is_sample,
            )
            results.append(r)
            if is_sample:
                sample_curves.append(r)

            h = tn // 2
            print(
                f"{r['total_neurons']:>8}"
                f"  {f'{h}+{h}':>7}"
                f"{int(r['sparsity']*100):>8}%"
                f"{r['active_params']:>10,}"
                f"{r['train_time_s']:>9.2f}"
                f"{r['inference_time_us']:>10.2f}"
                f"{r['accuracy']:>8.4f}"
                f"{r['f1']:>8.4f}"
            )

    # ── save CSV ──────────────────────────────────────────────────────────────
    scalar_cols = [
        "total_neurons", "h1_size", "h2_size", "sparsity",
        "n_params", "active_params", "active_macs",
        "train_time_s", "peak_mem_kb", "accuracy", "f1", "precision", "recall",
        "best_val_loss", "convergence_epoch", "inference_time_us",
    ]
    df = pd.DataFrame([{c: r[c] for c in scalar_cols} for r in results])
    df.to_csv(_RUN_DIR / "results.csv", index=False)

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nSaving plots ...")

    pivot_acc = df.pivot(index="total_neurons", columns="sparsity", values="accuracy")
    pivot_f1  = df.pivot(index="total_neurons", columns="sparsity", values="f1")

    plot_heatmap(pivot_acc, "Accuracy",
                 "Test Accuracy (2-layer node-level mask)  |  rows=total_neurons, cols=sparsity",
                 out_dir=_RUN_DIR, fname="heatmap_accuracy.png")
    plot_heatmap(pivot_f1,  "Weighted F1",
                 "Weighted F1 (2-layer node-level mask)    |  rows=total_neurons, cols=sparsity",
                 out_dir=_RUN_DIR, fname="heatmap_f1.png")

    plot_accuracy_vs_sparsity(df, _RUN_DIR)
    plot_accuracy_vs_neurons(df, _RUN_DIR)
    plot_metrics_dashboard(df, _RUN_DIR)
    plot_tradeoff(df, _RUN_DIR)
    plot_sample_loss_curves(sample_curves, _RUN_DIR)
    plot_accuracy_vs_params_vs_time(df, _RUN_DIR)
    plot_accuracy_loss_per_sparsity(df, _RUN_DIR)
    plot_variance_plots(df, _RUN_DIR)

    # ── findings ──────────────────────────────────────────────────────────────
    write_findings(df, _RUN_DIR)

    print(f"\nAll outputs saved to: {_RUN_DIR}")
    print(
        "  results.csv  |  findings.txt\n"
        "  heatmap_accuracy.png  |  heatmap_f1.png\n"
        "  accuracy_vs_sparsity.png  |  accuracy_vs_neurons.png\n"
        "  metrics_dashboard.png  |  tradeoff.png  |  sample_loss_curves.png\n"
        "  accuracy_vs_params_vs_time.png  |  accuracy_params_time_lines.png\n"
        "  accuracy_vs_neurons_per_sparsity.png  |  loss_vs_neurons_per_sparsity.png\n"
        "  accuracy_variance_vs_neurons.png  |  loss_variance_vs_neurons.png"
    )


if __name__ == "__main__":
    main()
