"""
2D FFNN Conduction Sparsity Benchmark — UCI Wine (Red).

Same as run_ffnn_2d_experiment (features + PE, two towers) but with fixed conduction
masks on all four weight matrices. Sweep: hidden_size × sparsity (110 runs).
Outputs: runs/run_2d_sparsity_<timestamp>/
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from run_experiment import load_data, NUM_CLASSES
from ffnn_2d import (
    sinusoidal_positional_encoding,
    to_2d_input,
    SparseFeedForwardNet2D,
    count_active_params_2d,
    count_active_macs_2d,
)

_HERE = Path(__file__).resolve().parent
_RUN_DIR = _HERE / "runs" / f"run_2d_sparsity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
_RUN_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 64
HIDDEN_SIZES = list(range(100, 1100, 100))
SPARSITY_LEVELS = [round(i / 10, 1) for i in range(0, 11)]
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_HIDDEN = 500


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _make_loaders_2d(X_tr_2d, y_tr, X_val_2d, y_val, X_te_2d, y_te):
    def to_ds(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
    return (
        DataLoader(to_ds(X_tr_2d, y_tr), batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(to_ds(X_val_2d, y_val), batch_size=256),
        DataLoader(to_ds(X_te_2d, y_te), batch_size=256),
    )


def train_and_eval_2d(
    hidden_size: int,
    sparsity: float,
    feature_dim: int,
    X_tr_2d, y_tr, X_val_2d, y_val, X_te_2d, y_te,
    store_curves: bool = False,
) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = SparseFeedForwardNet2D(feature_dim, hidden_size, NUM_CLASSES, sparsity, seed=SEED).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader, test_loader = _make_loaders_2d(
        X_tr_2d, y_tr, X_val_2d, y_val, X_te_2d, y_te
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
            train_losses.append(ep_loss / len(X_tr_2d))

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                v_loss += criterion(model(xb), yb).item() * len(xb)
        val_losses.append(v_loss / len(X_val_2d))

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

    sample = torch.tensor(X_te_2d[:1], dtype=torch.float32).to(DEVICE)
    inf_times = []
    with torch.no_grad():
        for _ in range(200):
            t0 = time.perf_counter()
            model(sample)
            inf_times.append(time.perf_counter() - t0)

    result = {
        "hidden_size": hidden_size,
        "sparsity": sparsity,
        "n_params": count_params(model),
        "active_params": count_active_params_2d(model),
        "active_macs": count_active_macs_2d(model),
        "train_time_s": round(t_end - t_start, 3),
        "peak_mem_kb": round(peak_mem / 1024, 2),
        "accuracy": round(accuracy_score(all_true, all_preds), 6),
        "f1": round(f1_score(all_true, all_preds, average="weighted"), 6),
        "precision": round(precision_score(all_true, all_preds, average="weighted", zero_division=0), 6),
        "recall": round(recall_score(all_true, all_preds, average="weighted"), 6),
        "best_val_loss": round(min(val_losses), 6),
        "convergence_epoch": int(np.argmin(val_losses)) + 1,
        "inference_time_us": round(np.median(inf_times) * 1e6, 3),
    }
    if store_curves:
        result["train_losses"] = train_losses
        result["val_losses"] = val_losses
    return result


# ── Plots ─────────────────────────────────────────────────────────────────────

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
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color="black")
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()


def plot_accuracy_vs_sparsity(df: pd.DataFrame, out_dir: Path):
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(HIDDEN_SIZES)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("2D FFNN — Test Accuracy vs Sparsity  |  each line = one hidden width", fontsize=12, fontweight="bold")
    for h, c in zip(HIDDEN_SIZES, palette):
        sub = df[df["hidden_size"] == h].sort_values("sparsity")
        axes[0].plot(sub["sparsity"] * 100, sub["accuracy"], "o-", color=c, lw=1.6, ms=4, label=f"h={h}")
        axes[1].plot(sub["sparsity"] * 100, sub["f1"], "o-", color=c, lw=1.6, ms=4, label=f"h={h}")
    for ax, ylabel in zip(axes, ["Test Accuracy", "Weighted F1"]):
        ax.set_xlabel("Sparsity (%)"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3); ax.set_xlim(-2, 102)
    axes[0].set_title("Accuracy vs Sparsity"); axes[1].set_title("F1 vs Sparsity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.12, 0.5), title="Hidden Size", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_sparsity.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_accuracy_vs_hidden(df: pd.DataFrame, out_dir: Path):
    palette = plt.cm.plasma(np.linspace(0.05, 0.95, len(SPARSITY_LEVELS)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("2D FFNN — Test Accuracy vs Hidden Width  |  each line = one sparsity level", fontsize=12, fontweight="bold")
    for s, c in zip(SPARSITY_LEVELS, palette):
        sub = df[df["sparsity"] == s].sort_values("hidden_size")
        lbl = f"{int(s*100)}%"
        axes[0].plot(sub["hidden_size"], sub["accuracy"], "o-", color=c, lw=1.6, ms=4, label=lbl)
        axes[1].plot(sub["hidden_size"], sub["f1"], "o-", color=c, lw=1.6, ms=4, label=lbl)
    for ax, ylabel in zip(axes, ["Test Accuracy", "Weighted F1"]):
        ax.set_xlabel("Hidden Neurons"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.1, 0.5), title="Sparsity", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_hidden.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_dashboard(df: pd.DataFrame, out_dir: Path):
    grouped = df.groupby("sparsity").mean(numeric_only=True).reset_index()
    sp_pct = grouped["sparsity"] * 100
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    fig.suptitle("2D FFNN Conduction Sparsity  |  metrics averaged over hidden widths", fontsize=12, fontweight="bold", y=1.01)

    def _line(ax, y_col, title, ylabel, color):
        ax.plot(sp_pct, grouped[y_col], "o-", color=color, lw=2, ms=6)
        ax.set_title(title); ax.set_xlabel("Sparsity (%)"); ax.set_ylabel(ylabel)
        ax.set_xlim(-2, 102); ax.grid(True, alpha=0.3)

    _line(axes[0, 0], "accuracy", "Test Accuracy", "Accuracy", "crimson")
    _line(axes[0, 1], "f1", "Weighted F1", "F1", "mediumpurple")
    _line(axes[0, 2], "best_val_loss", "Best Validation Loss", "Cross-Entropy", "indianred")
    _line(axes[1, 0], "active_params", "Active Parameters", "Count", "steelblue")
    _line(axes[1, 1], "active_macs", "Active MACs / Forward Pass", "MACs", "darkorange")
    _line(axes[1, 2], "train_time_s", "Training Time", "Seconds", "seagreen")
    _line(axes[2, 0], "inference_time_us", "Inference Latency (µs)", "Microseconds", "goldenrod")
    _line(axes[2, 1], "convergence_epoch", "Convergence Epoch", "Epoch", "slateblue")
    _line(axes[2, 2], "peak_mem_kb", "Peak Memory (KB)", "KB", "teal")
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_tradeoff(df: pd.DataFrame, out_dir: Path):
    sp_pct = df["sparsity"] * 100
    scatter_kw = dict(c=sp_pct, cmap="RdYlGn_r", s=60, alpha=0.8, zorder=3)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("2D FFNN — Complexity vs Accuracy Trade-off", fontsize=12, fontweight="bold")
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
    epoch_range = range(1, EPOCHS + 1)
    palette = plt.cm.RdYlGn_r(np.linspace(0.05, 0.95, len(sample_curves)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"2D FFNN Loss Curves  |  hidden_size={SAMPLE_HIDDEN}, varying sparsity", fontsize=12, fontweight="bold")
    for r, c in zip(sample_curves, palette):
        lbl = f"{int(r['sparsity']*100)}%"
        axes[0].plot(epoch_range, r["train_losses"], color=c, lw=1.4, label=lbl, alpha=0.9)
        axes[1].plot(epoch_range, r["val_losses"], color=c, lw=1.4, label=lbl, alpha=0.9)
    for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.1, 0.5), title="Sparsity", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "sample_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def write_findings(df: pd.DataFrame, out_dir: Path):
    lines = []
    sep = "=" * 60
    lines += [sep, "  2D FFNN Conduction Sparsity Benchmark — Key Findings", sep, ""]
    best = df.loc[df["accuracy"].idxmax()]
    lines.append(f"  Global best accuracy : h={int(best.hidden_size):4d}, sparsity={int(best.sparsity*100):3d}%  → {best.accuracy:.4f}")
    dense = df[df["sparsity"] == 0.0].set_index("hidden_size")["accuracy"]
    lines.append(f"\n  Dense (0% sparsity) accuracy range : {dense.min():.4f} – {dense.max():.4f}")
    lines.append("\n  Resilience: highest sparsity still ≥95% of dense accuracy")
    lines.append(f"  {'h':>6}  {'dense acc':>10}  {'95% threshold':>14}  {'max safe sparsity':>18}")
    lines.append("  " + "-" * 54)
    for h in HIDDEN_SIZES:
        sub = df[df["hidden_size"] == h].sort_values("sparsity")
        d_acc = float(sub[sub["sparsity"] == 0.0]["accuracy"].iloc[0])
        thresh = d_acc * 0.95
        safe = sub[sub["accuracy"] >= thresh]["sparsity"].max()
        lines.append(f"  {h:>6}  {d_acc:>10.4f}  {thresh:>14.4f}  {int(safe*100):>17}%")
    for s_label, s_val in [("0%", 0.0), ("50%", 0.5)]:
        sub = df[df["sparsity"] == s_val]
        lines.append(f"\n  Avg active params at sparsity={s_label:>4s} : {sub['active_params'].mean():,.0f}")
        lines.append(f"  Avg accuracy      at sparsity={s_label:>4s} : {sub['accuracy'].mean():.4f}")
    lines += ["", sep]
    summary = "\n".join(lines)
    print("\n" + summary)
    (out_dir / "findings.txt").write_text(summary + "\n")


def main():
    print(f"Device  : {DEVICE}")
    print(f"Outputs : {_RUN_DIR}\n")

    X_tr, X_val, X_te, y_tr, y_val, y_te = load_data()
    feature_dim = X_tr.shape[1]
    pe = sinusoidal_positional_encoding(feature_dim)
    X_tr_2d = to_2d_input(X_tr, pe)
    X_val_2d = to_2d_input(X_val, pe)
    X_te_2d = to_2d_input(X_te, pe)
    print(f"2D input shape: train {X_tr_2d.shape}\n")

    total_runs = len(HIDDEN_SIZES) * len(SPARSITY_LEVELS)
    print(f"Sweeping hidden_size ∈ {HIDDEN_SIZES}")
    print(f"        sparsity    ∈ {[f'{int(s*100)}%' for s in SPARSITY_LEVELS]}")
    print(f"Total runs: {total_runs}  |  epochs={EPOCHS}\n")
    print(f"{'hidden':>8} {'sparsity':>9} {'active_p':>10} {'train(s)':>9} {'infer(µs)':>10} {'acc':>8} {'f1':>8}")
    print("-" * 68)

    results = []
    sample_curves = []
    for h in HIDDEN_SIZES:
        for s in SPARSITY_LEVELS:
            is_sample = h == SAMPLE_HIDDEN
            r = train_and_eval_2d(
                h, s, feature_dim,
                X_tr_2d, y_tr, X_val_2d, y_val, X_te_2d, y_te,
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

    scalar_cols = [
        "hidden_size", "sparsity", "n_params", "active_params", "active_macs",
        "train_time_s", "peak_mem_kb", "accuracy", "f1", "precision", "recall",
        "best_val_loss", "convergence_epoch", "inference_time_us",
    ]
    df = pd.DataFrame([{c: r[c] for c in scalar_cols} for r in results])
    df.to_csv(_RUN_DIR / "results.csv", index=False)

    print("\nSaving plots ...")
    pivot_acc = df.pivot(index="hidden_size", columns="sparsity", values="accuracy")
    pivot_f1 = df.pivot(index="hidden_size", columns="sparsity", values="f1")
    plot_heatmap(pivot_acc, "Accuracy", "2D FFNN Test Accuracy  |  rows=hidden_size, cols=sparsity", _RUN_DIR, "heatmap_accuracy.png")
    plot_heatmap(pivot_f1, "Weighted F1", "2D FFNN Weighted F1  |  rows=hidden_size, cols=sparsity", _RUN_DIR, "heatmap_f1.png")
    plot_accuracy_vs_sparsity(df, _RUN_DIR)
    plot_accuracy_vs_hidden(df, _RUN_DIR)
    plot_metrics_dashboard(df, _RUN_DIR)
    plot_tradeoff(df, _RUN_DIR)
    plot_sample_loss_curves(sample_curves, _RUN_DIR)
    write_findings(df, _RUN_DIR)

    print(f"\nAll outputs saved to: {_RUN_DIR}")
    print("  results.csv  |  findings.txt  |  heatmap_*.png  |  accuracy_vs_*.png  |  metrics_dashboard.png  |  tradeoff.png  |  sample_loss_curves.png")


if __name__ == "__main__":
    main()
