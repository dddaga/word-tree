"""
2D FFNN sub-experiment: same as ffnn_wine_benchmark but input is (features, positional_encoding).
Two parallel towers (feature + PE), summed at output — two components per node for future radiation-resonance.
"""

import time
import tracemalloc
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from run_experiment import (
    load_data,
    NUM_CLASSES,
    EPOCHS,
    LR,
    BATCH_SIZE,
    HIDDEN_SIZES,
    SEED,
    DEVICE,
    count_params,
    plot_dashboard,
    plot_loss_curves,
    plot_overlaid_loss,
    plot_tradeoff,
    plot_efficiency,
    write_findings,
)
from ffnn_2d import (
    sinusoidal_positional_encoding,
    to_2d_input,
    FeedForwardNet2D,
    estimate_macs_2d,
)

_HERE = Path(__file__).resolve().parent
_RUN_DIR = _HERE / "runs" / f"run_2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
_RUN_DIR.mkdir(parents=True, exist_ok=True)


def _make_loaders_2d(X_tr_2d, y_tr, X_val_2d, y_val, X_te_2d, y_te):
    def to_ds(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
    return (
        DataLoader(to_ds(X_tr_2d,  y_tr),  batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(to_ds(X_val_2d, y_val), batch_size=256),
        DataLoader(to_ds(X_te_2d,  y_te), batch_size=256),
    )


def train_and_eval_2d(hidden_size: int, feature_dim: int,
                      X_tr_2d, y_tr, X_val_2d, y_val, X_te_2d, y_te) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = FeedForwardNet2D(feature_dim, hidden_size, NUM_CLASSES).to(DEVICE)
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

    return {
        "hidden_size":       hidden_size,
        "n_params":          count_params(model),
        "macs":              estimate_macs_2d(feature_dim, hidden_size, NUM_CLASSES),
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


def main():
    print(f"Device  : {DEVICE}")
    print(f"Outputs : {_RUN_DIR}\n")

    X_tr, X_val, X_te, y_tr, y_val, y_te = load_data()
    feature_dim = X_tr.shape[1]
    pe = sinusoidal_positional_encoding(feature_dim)
    X_tr_2d = to_2d_input(X_tr, pe)
    X_val_2d = to_2d_input(X_val, pe)
    X_te_2d = to_2d_input(X_te, pe)
    print(f"2D input shape: train {X_tr_2d.shape} (batch, 2, features)\n")

    print(f"Benchmarking 2D FFNN  hidden_size ∈ {HIDDEN_SIZES}  |  epochs={EPOCHS}\n")
    print(f"{'hidden':>8} {'params':>10} {'MACs':>10} {'train(s)':>9} {'infer(µs)':>10} {'acc':>8} {'f1':>8}")
    print("-" * 68)

    results = []
    for h in HIDDEN_SIZES:
        r = train_and_eval_2d(h, feature_dim, X_tr_2d, y_tr, X_val_2d, y_val, X_te_2d, y_te)
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

    scalar_cols = [
        "hidden_size", "n_params", "macs", "train_time_s", "peak_mem_kb",
        "accuracy", "f1", "precision", "recall",
        "best_val_loss", "convergence_epoch", "inference_time_us",
    ]
    df_res = pd.DataFrame([{c: r[c] for c in scalar_cols} for r in results])
    df_res = df_res.set_index("hidden_size")
    df_res.to_csv(_RUN_DIR / "results.csv")

    print("\nSaving plots ...")
    plot_dashboard(df_res, _RUN_DIR)
    plot_loss_curves(results, _RUN_DIR)
    plot_overlaid_loss(results, _RUN_DIR)
    plot_tradeoff(df_res, _RUN_DIR)
    eff = plot_efficiency(df_res, _RUN_DIR)
    write_findings(df_res, eff, _RUN_DIR)

    print(f"\nAll outputs saved to: {_RUN_DIR}")
    print("  results.csv  |  findings.txt  |  benchmark_dashboard.png  |  ...")


if __name__ == "__main__":
    main()
