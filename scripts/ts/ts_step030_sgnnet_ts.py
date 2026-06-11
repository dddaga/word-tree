"""ts_step030: SGNNET-TS T0 (20ep, directional loss).
Fixes: MSE warmup replaced by directional_loss from ep1 → avoids mean-predictor collapse.
Baselines: Linear (OLS on flattened window), MLP_small (2-layer).
Run: python scripts/ts/ts_step030_sgnnet_ts.py --device cpu --epochs 20
"""
# See ts_common.py (data/losses), ts_model_sgnnet.py (model)
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ts"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ts_common import (
    load_universe, StockDataset, eval_epoch, train_epoch, count_params,
    directional_loss, log_wealth_loss, FEATURE_COLS, T_LOOKBACK, TC,
)
from ts_model_sgnnet import SGNNETRecurrent

RESULTS_DIR = ROOT / "results" / "ts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default=str(ROOT / "data/ts/raw"))
    p.add_argument("--D", type=int, default=16)
    p.add_argument("--K_wiring", type=int, default=4)
    p.add_argument("--K_iter", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


class LinearBaseline(nn.Module):
    """Flatten [B, T, N, F] → Linear → [B, N]."""
    def __init__(self, T, N, F):
        super().__init__()
        self.T = T; self.N = N
        self.fc = nn.Linear(T * N * F, N)

    def forward(self, x):
        return torch.tanh(self.fc(x.reshape(x.size(0), -1))) * 0.05


class MLPBaseline(nn.Module):
    """2-layer MLP: [B, T*N*F] → hidden → [B, N]."""
    def __init__(self, T, N, F, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(T * N * F, hidden), nn.ReLU(),
            nn.Linear(hidden, N)
        )

    def forward(self, x):
        return torch.tanh(self.net(x.reshape(x.size(0), -1))) * 0.05


def train_config(label, model, train_ds, val_ds, args, device):
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # FIX: directional_loss from epoch 1; no MSE warmup (avoids mean-predictor collapse)
    best_sharpe = -1e9; best_dir_acc = 0.0; t0 = time.time()
    for ep in range(1, args.epochs + 1):
        # Combine directional + log-wealth: directional provides strong direction signal
        def loss_fn(pred, actual):
            return directional_loss(pred, actual) + 0.5 * log_wealth_loss(pred, actual)
        tr_loss = train_epoch(model, loader, opt, loss_fn, device)
        _, dir_acc, sharpe, mae = eval_epoch(model, val_loader, device)
        if sharpe > best_sharpe: best_sharpe = sharpe; best_dir_acc = dir_acc
        if ep % 5 == 0 or ep == 1:
            print(f"  [{label}] ep{ep:3d} loss={tr_loss:.4f} dir_acc={dir_acc:.4f} sharpe={sharpe:.4f}")

    elapsed = time.time() - t0
    n_p = count_params(model)
    print(f"  [{label}] DONE: best_sharpe={best_sharpe:.4f} best_dir_acc={best_dir_acc:.4f} params={n_p:,} [{elapsed:.0f}s]")
    return {"label": label, "n_params": n_p, "best_sharpe": round(best_sharpe, 4),
            "best_dir_acc": round(best_dir_acc, 4), "elapsed_s": round(elapsed, 1)}


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    print(f"step030 SGNNET-TS T0 (ep={args.epochs}, directional loss, device={device})")
    print("Loading universe...")
    universe = load_universe(data_dir)
    SPLITS = [("2015-01-01", "2021-12-31"), ("2022-01-01", "2023-12-31")]
    universe = {t: df for t, df in universe.items()
                if all(len(df.loc[s:e]) >= T_LOOKBACK + 50 for s, e in SPLITS)}
    print(f"Universe: {len(universe)} tickers after cross-split filter")

    train_ds = StockDataset(universe, "2015-01-01", "2021-12-31")
    val_ds = StockDataset(universe, "2022-01-01", "2023-12-31")
    if train_ds.N == 0: print("ERROR: no valid tickers"); return
    N, F = train_ds.N, len(FEATURE_COLS)
    print(f"Train={len(train_ds)} Val={len(val_ds)} N={N} F={F}")

    configs = {
        "Linear": LinearBaseline(T_LOOKBACK, N, F).to(device),
        "MLP_256": MLPBaseline(T_LOOKBACK, N, F, 256).to(device),
        "SGNNET_K3": SGNNETRecurrent(N, F, args.D, args.K_wiring, args.K_iter, args.dropout).to(device),
        "SGNNET_K0": SGNNETRecurrent(N, F, args.D, args.K_wiring, 0, args.dropout).to(device),
    }

    results = {}
    for label, model in configs.items():
        results[label] = train_config(label, model, train_ds, val_ds, args, device)

    print("\n=== SUMMARY ===")
    print(f"{'Config':<15} {'dir_acc':>9} {'sharpe':>9} {'params':>10}")
    for label, r in results.items():
        print(f"{label:<15} {r['best_dir_acc']:>9.4f} {r['best_sharpe']:>9.4f} {r['n_params']:>10,}")

    out = RESULTS_DIR / f"ts_step030_sgnnet_ts_seed{args.seed}.json"
    out.write_text(json.dumps({"step": "ts_step030", "epochs": args.epochs,
                               "loss": "directional+log_wealth", "results": results}, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
