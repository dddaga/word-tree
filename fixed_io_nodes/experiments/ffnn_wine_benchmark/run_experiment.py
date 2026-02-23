"""
FFNN Hidden-Width Benchmark — UCI Wine Quality (Red)

Multi-class classification: quality labels 0–10 (11 classes). Red wine CSV has 3–8.
Trains a single-hidden-layer feedforward network for hidden_size in
[100, 200, ..., 1000]; records space/time complexity, training time,
inference latency, peak memory, accuracy, F1/precision/recall, best
validation loss, convergence epoch.

All outputs (plots + CSV + findings summary) are written to a
timestamped sub-folder under:
    fixed_io_nodes/experiments/ffnn_wine_benchmark/runs/

Run from repo root:
    python3 fixed_io_nodes/experiments/ffnn_wine_benchmark/run_experiment.py
"""

import time
import tracemalloc
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
_HERE = Path(__file__).resolve().parent
_RUN_DIR = _HERE / "runs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
_RUN_DIR.mkdir(parents=True, exist_ok=True)

# ── hyper-parameters ─────────────────────────────────────────────────────────
EPOCHS       = 100
LR           = 1e-3
BATCH_SIZE   = 64
HIDDEN_SIZES = list(range(100, 1100, 100))
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    y  = df["quality"].values.astype(np.int64)   # 0–10 scale (red wine typically 3–8)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp
    )

    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    print(f"Train: {X_tr.shape}  Val: {X_val.shape}  Test: {X_te.shape}")
    uniq, cnts = np.unique(y_tr, return_counts=True)
    print(f"Classes (quality 0–10): {dict(zip(uniq.tolist(), cnts.tolist()))}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class FeedForwardNet(nn.Module):
    """Single hidden layer: Linear → ReLU → Linear."""

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_macs(input_dim: int, hidden_size: int, output_dim: int) -> int:
    """MACs for one forward pass (weights + biases for both layers)."""
    return (input_dim * hidden_size + hidden_size) + (hidden_size * output_dim + output_dim)


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

def train_and_eval(hidden_size: int, X_tr, y_tr, X_val, y_val, X_te, y_te) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model     = FeedForwardNet(X_tr.shape[1], hidden_size, NUM_CLASSES).to(DEVICE)
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

    # test metrics
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    # inference latency: median of 200 single-sample runs
    sample = torch.tensor(X_te[:1], dtype=torch.float32).to(DEVICE)
    inf_times = []
    with torch.no_grad():
        for _ in range(200):
            t0 = time.perf_counter()
            model(sample)
            inf_times.append(time.perf_counter() - t0)

    return {
        "hidden_size":       hidden_size,
        "n_params":          count_params(model),
        "macs":              estimate_macs(X_tr.shape[1], hidden_size, NUM_CLASSES),
        "train_time_s":      round(t_end - t_start, 3),
        "peak_mem_kb":       round(peak_mem / 1024, 2),
        "accuracy":          round(accuracy_score(all_true, all_preds), 6),
        "f1":                round(f1_score(all_true, all_preds, average="weighted"), 6),
        "precision":         round(precision_score(all_true, all_preds, average="weighted", zero_division=0), 6),
        "recall":            round(recall_score(all_true, all_preds, average="weighted"), 6),
        "best_val_loss":     round(min(val_losses), 6),
        "convergence_epoch": int(np.argmin(val_losses)) + 1,
        "inference_time_us": round(np.median(inf_times) * 1e6, 3),
        "train_losses":      train_losses,
        "val_losses":        val_losses,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_int_axis(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))


def plot_dashboard(df_res: pd.DataFrame, out_dir: Path):
    h_vals = df_res.index.tolist()
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    fig.suptitle(
        "FFNN Hidden-Width Benchmark  |  UCI Wine Quality (Red)  |  Multi-class (Quality 0–10)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    ax = axes[0, 0]
    ax.plot(h_vals, df_res["n_params"], "o-", color="steelblue", lw=2, ms=6)
    ax.set_title("Space Complexity — # Parameters")
    ax.set_xlabel("Hidden Neurons"); ax.set_ylabel("Parameters")
    _fmt_int_axis(ax); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(h_vals, df_res["macs"], "o-", color="darkorange", lw=2, ms=6)
    ax.set_title("Time Complexity — MACs / Forward Pass")
    ax.set_xlabel("Hidden Neurons"); ax.set_ylabel("MACs")
    _fmt_int_axis(ax); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(h_vals, df_res["train_time_s"], "o-", color="seagreen", lw=2, ms=6)
    ax.set_title("Training Time (all epochs)")
    ax.set_xlabel("Hidden Neurons"); ax.set_ylabel("Seconds")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(h_vals, df_res["accuracy"], "o-", color="crimson", lw=2, ms=6)
    ax.set_title("Test Accuracy")
    ax.set_xlabel("Hidden Neurons"); ax.set_ylabel("Accuracy")
    margin = (df_res["accuracy"].max() - df_res["accuracy"].min()) * 0.5 + 0.002
    ax.set_ylim([df_res["accuracy"].min() - margin, df_res["accuracy"].max() + margin])
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(h_vals, df_res["f1"],        "o-", label="F1",        color="mediumpurple", lw=2, ms=5)
    ax.plot(h_vals, df_res["precision"], "s-", label="Precision", color="coral",        lw=2, ms=5)
    ax.plot(h_vals, df_res["recall"],    "^-", label="Recall",    color="dodgerblue",   lw=2, ms=5)
    ax.set_title("F1 / Precision / Recall")
    ax.set_xlabel("Hidden Neurons"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(h_vals, df_res["inference_time_us"], "o-", color="goldenrod", lw=2, ms=6)
    ax.set_title("Inference Latency (single sample, median)")
    ax.set_xlabel("Hidden Neurons"); ax.set_ylabel("Microseconds (µs)")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(h_vals, df_res["peak_mem_kb"], "o-", color="teal", lw=2, ms=6)
    ax.set_title("Peak Memory Usage (tracemalloc)")
    ax.set_xlabel("Hidden Neurons"); ax.set_ylabel("KB")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(h_vals, df_res["best_val_loss"], "o-", color="indianred", lw=2, ms=6)
    ax.set_title("Best Validation Loss")
    ax.set_xlabel("Hidden Neurons"); ax.set_ylabel("Cross-Entropy")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    bars = ax.bar(h_vals, df_res["convergence_epoch"], width=70, color="slateblue", alpha=0.85)
    ax.bar_label(bars, fmt="%d", fontsize=7, padding=2)
    ax.set_title("Convergence Epoch (best val loss)")
    ax.set_xlabel("Hidden Neurons"); ax.set_ylabel("Epoch")
    ax.set_ylim(0, EPOCHS + 10); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_curves(results: list, out_dir: Path):
    epoch_range = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharey=False)
    fig.suptitle("Training vs Validation Loss — per Hidden Width", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for ax, r in zip(axes, results):
        ax.plot(epoch_range, r["train_losses"], color="steelblue", lw=1.5, label="Train")
        ax.plot(epoch_range, r["val_losses"],   color="tomato",    lw=1.5, ls="--", label="Val")
        ax.axvline(r["convergence_epoch"], color="gray", lw=1, ls=":", alpha=0.8)
        ax.set_title(f"h = {r['hidden_size']}", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=8); ax.set_ylabel("Loss", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.25)

    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_overlaid_loss(results: list, out_dir: Path):
    epoch_range = range(1, EPOCHS + 1)
    palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(results)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Overlaid Loss Curves Across Hidden Widths", fontsize=12, fontweight="bold")

    for ax, loss_key, title in zip(
        axes, ["train_losses", "val_losses"], ["Training Loss", "Validation Loss"]
    ):
        for r, c in zip(results, palette):
            ax.plot(epoch_range, r[loss_key], color=c, lw=1.4, alpha=0.85, label=f"h={r['hidden_size']}")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.12, 0.5),
               title="Hidden Size", fontsize=9, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "overlaid_loss.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_tradeoff(df_res: pd.DataFrame, out_dir: Path):
    h_vals = df_res.index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Complexity vs. Quality Trade-off", fontsize=12, fontweight="bold")
    scatter_kw = dict(s=100, zorder=3, c=h_vals, cmap="viridis")

    ax = axes[0]
    sc = ax.scatter(df_res["n_params"], df_res["accuracy"], **scatter_kw)
    for x, y_, lbl in zip(df_res["n_params"], df_res["accuracy"], h_vals):
        ax.annotate(str(lbl), (x, y_), textcoords="offset points", xytext=(5, 3), fontsize=7, color="dimgray")
    ax.set_xlabel("Parameter Count"); ax.set_ylabel("Test Accuracy")
    ax.set_title("Space Complexity vs Accuracy")
    _fmt_int_axis(ax); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(df_res["train_time_s"], df_res["accuracy"], **scatter_kw)
    for x, y_, lbl in zip(df_res["train_time_s"], df_res["accuracy"], h_vals):
        ax.annotate(str(lbl), (x, y_), textcoords="offset points", xytext=(5, 3), fontsize=7, color="dimgray")
    ax.set_xlabel("Training Time (s)"); ax.set_ylabel("Test Accuracy")
    ax.set_title("Training Time vs Accuracy"); ax.grid(True, alpha=0.3)

    plt.colorbar(sc, ax=axes, label="Hidden Size", shrink=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / "tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_efficiency(df_res: pd.DataFrame, out_dir: Path):
    h_vals  = df_res.index.tolist()
    eff     = df_res["accuracy"] / np.log10(df_res["n_params"])

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(h_vals, eff, width=70, color="mediumseagreen", alpha=0.85)
    ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=2)
    ax.set_title("Param-Efficiency Score  (Accuracy / log₁₀(# Params))", fontsize=11)
    ax.set_xlabel("Hidden Neurons"); ax.set_ylabel("Efficiency")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "efficiency.png", dpi=150, bbox_inches="tight")
    plt.close()

    return eff


# ─────────────────────────────────────────────────────────────────────────────
# Findings
# ─────────────────────────────────────────────────────────────────────────────

def write_findings(df_res: pd.DataFrame, eff: pd.Series, out_dir: Path):
    lines = []
    lines.append("=" * 52)
    lines.append("  FFNN Wine Benchmark — Key Findings")
    lines.append("=" * 52)

    best_acc = df_res["accuracy"].idxmax()
    best_f1  = df_res["f1"].idxmax()
    fastest  = df_res["train_time_s"].idxmin()
    smallest = df_res["n_params"].idxmin()
    best_eff = eff.idxmax()

    lines.append(f"  Best accuracy    : h={best_acc:4d}  ({df_res.loc[best_acc, 'accuracy']:.4f})")
    lines.append(f"  Best F1          : h={best_f1:4d}  ({df_res.loc[best_f1, 'f1']:.4f})")
    lines.append(f"  Fastest training : h={fastest:4d}  ({df_res.loc[fastest, 'train_time_s']:.2f}s)")
    lines.append(f"  Smallest model   : h={smallest:4d}  ({df_res.loc[smallest, 'n_params']:,} params)")
    lines.append(f"  Most efficient   : h={best_eff:4d}  (score={eff.loc[best_eff]:.4f})")

    lines.append("")
    lines.append("  Scaling factors  (h=100  →  h=1000)")
    lines.append(f"    Parameters  : x{df_res.loc[1000, 'n_params'] / df_res.loc[100, 'n_params']:.1f}")
    lines.append(f"    MACs        : x{df_res.loc[1000, 'macs']     / df_res.loc[100, 'macs']:.1f}")
    lines.append(f"    Train time  : x{df_res.loc[1000, 'train_time_s'] / df_res.loc[100, 'train_time_s']:.1f}")
    lines.append(f"    Infer. time : x{df_res.loc[1000, 'inference_time_us'] / df_res.loc[100, 'inference_time_us']:.1f}")
    lines.append(f"    Delta acc   : {df_res.loc[1000, 'accuracy'] - df_res.loc[100, 'accuracy']:+.4f}")
    lines.append("=" * 52)

    summary = "\n".join(lines)
    print("\n" + summary)
    (out_dir / "findings.txt").write_text(summary + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Device  : {DEVICE}")
    print(f"Outputs : {_RUN_DIR}\n")

    X_tr, X_val, X_te, y_tr, y_val, y_te = load_data()

    print(f"\nBenchmarking hidden_size ∈ {HIDDEN_SIZES}  |  epochs={EPOCHS}\n")
    print(f"{'hidden':>8} {'params':>10} {'MACs':>10} {'train(s)':>9} {'infer(µs)':>10} {'acc':>8} {'f1':>8}")
    print("-" * 68)

    results = []
    for h in HIDDEN_SIZES:
        r = train_and_eval(h, X_tr, y_tr, X_val, y_val, X_te, y_te)
        results.append(r)
        print(
            f"{r['hidden_size']:>8}"
            f"{r['n_params']:>10,}"
            f"{r['macs']:>10,}"
            f"{r['train_time_s']:>9.2f}"
            f"{r['inference_time_us']:>10.2f}"
            f"{r['accuracy']:>8.4f}"
            f"{r['f1']:>8.4f}"
        )

    # ── save CSV ──────────────────────────────────────────────────────────────
    scalar_cols = [
        "hidden_size", "n_params", "macs", "train_time_s", "peak_mem_kb",
        "accuracy", "f1", "precision", "recall",
        "best_val_loss", "convergence_epoch", "inference_time_us",
    ]
    df_res = pd.DataFrame([{c: r[c] for c in scalar_cols} for r in results])
    df_res = df_res.set_index("hidden_size")
    df_res.to_csv(_RUN_DIR / "results.csv")

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nSaving plots ...")
    plot_dashboard(df_res, _RUN_DIR)
    plot_loss_curves(results, _RUN_DIR)
    plot_overlaid_loss(results, _RUN_DIR)
    plot_tradeoff(df_res, _RUN_DIR)
    eff = plot_efficiency(df_res, _RUN_DIR)

    # ── findings ──────────────────────────────────────────────────────────────
    write_findings(df_res, eff, _RUN_DIR)

    print(f"\nAll outputs saved to: {_RUN_DIR}")
    print("  results.csv  |  findings.txt  |  benchmark_dashboard.png")
    print("  loss_curves.png  |  overlaid_loss.png  |  tradeoff.png  |  efficiency.png")


if __name__ == "__main__":
    main()
