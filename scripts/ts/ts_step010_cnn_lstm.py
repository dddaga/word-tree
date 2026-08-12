"""
ts_step010_cnn_lstm.py — CNN + LSTM baseline for next-day return prediction.

Architecture: [B, F, T, N] → 3×(Conv2d→BN→ReLU) → permute → [B, T', 64*N'] → LSTM(128,2) → Linear → [B, N_stocks]
Loss: MSE warmup (20 epochs) → log-wealth loss (fine-tune)
Walk-forward: train 2015-2021, val 2022-2023
Data: data/ts/raw/*.parquet, sliding window stride=1
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "data" / "ts" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", type=str, default="data/ts/raw")
    p.add_argument("--results_dir", type=str, default="results/ts")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading + feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "ret_1d",
    "ma_ratio_5", "ma_ratio_21", "ma_ratio_55", "ma_ratio_89",
    "ma_cross_5_21", "ma_cross_13_55", "ma_cross_21_89",
    "vol_20", "vol_60",
    "spread_z", "resid_z", "delta_corr", "lag1_ret",
    "high_low_ratio", "volume_z",
]
assert len(FEATURE_COLS) == 16, "F must equal 16"

T_LOOKBACK = 60
TC = 0.001  # transaction cost


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """All features use shift(1) before any rolling window — no lookahead."""
    out = pd.DataFrame(index=df.index)

    close = df["close"].shift(1)  # shift first — all features derived from this
    volume = df["volume"].shift(1)
    high = df["high"].shift(1)
    low = df["low"].shift(1)

    # ret_1d: unshifted close ratio (uses shift(1) close as numerator denominator)
    out["ret_1d"] = close / close.shift(1) - 1

    # MA ratios
    for w in [5, 21, 55, 89]:
        out[f"ma_ratio_{w}"] = close / close.rolling(w).mean()

    # MA crosses
    for f, s in [(5, 21), (13, 55), (21, 89)]:
        out[f"ma_cross_{f}_{s}"] = close.rolling(f).mean() / close.rolling(s).mean() - 1

    # Volatility
    ret = close / close.shift(1) - 1
    out["vol_20"] = ret.rolling(20).std()
    out["vol_60"] = ret.rolling(60).std()

    # Spread z-score (self, 60d rolling zscore of price)
    mu60 = close.rolling(60).mean()
    std60 = close.rolling(60).std()
    out["spread_z"] = (close - mu60) / (std60 + 1e-8)

    # Residual z-score (demeaned by 60d mean return)
    out["resid_z"] = (ret - ret.rolling(60).mean()) / (ret.rolling(60).std() + 1e-8)

    # Delta corr (approximated: vol_20/vol_60 ratio as proxy)
    out["delta_corr"] = out["vol_20"] / (out["vol_60"] + 1e-8) - 1.0

    # Lag1 return
    out["lag1_ret"] = ret.shift(1)

    # Extra features to reach F=16
    out["high_low_ratio"] = (high - low) / (close + 1e-8)
    vol_mu = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    out["volume_z"] = (volume - vol_mu) / (vol_std + 1e-8)

    return out


def _compute_target(df: pd.DataFrame) -> pd.Series:
    """next-day % return: close.pct_change().shift(-1)"""
    return df["close"].pct_change().shift(-1)


def load_universe(data_dir: Path):
    """Load all parquet files, compute features and targets, return dict keyed by ticker."""
    parquets = sorted(f for f in data_dir.glob("*.parquet") if not f.stem.startswith("."))
    if not parquets:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    universe = {}
    for pq in parquets:
        ticker = pq.stem
        df = pd.read_parquet(pq)
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_index()

        feats = _compute_features(df)
        target = _compute_target(df)

        combined = feats.copy()
        combined["target"] = target
        combined = combined.dropna()

        universe[ticker] = combined

    return universe


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StockDataset(Dataset):
    """
    Returns windows of shape [T_lookback, N_stocks, F_features] and target [N_stocks].
    Stocks ordered by sorted ticker name (replace with stock_order.json if available).
    """

    def __init__(self, universe: dict, start_date: str, end_date: str, T: int = T_LOOKBACK):
        self.T = T

        # Filter to tickers that have ≥ T+50 rows in [start_date, end_date]
        filtered = {}
        for t, df in universe.items():
            sub = df.loc[start_date:end_date]
            if len(sub) >= T + 50:
                filtered[t] = sub
        if not filtered:
            self.D = 0
            self.N = 0
            return

        tickers = sorted(filtered.keys())
        self.N = len(tickers)
        frames = [filtered[t] for t in tickers]

        # Common dates across filtered tickers only
        common_idx = frames[0].index
        for f in frames[1:]:
            common_idx = common_idx.intersection(f.index)
        common_idx = common_idx.sort_values()
        print(f"  [{start_date}:{end_date}] {len(tickers)} tickers, {len(common_idx)} common dates")

        # Build arrays: [D, N, F+1]
        feat_cols = FEATURE_COLS
        feat_arr = np.stack(
            [f.loc[common_idx, feat_cols].values for f in frames], axis=1
        ).astype(np.float32)  # [D, N, F]
        tgt_arr = np.stack(
            [f.loc[common_idx, "target"].values for f in frames], axis=1
        ).astype(np.float32)  # [D, N]

        # Replace nan/inf
        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
        tgt_arr = np.nan_to_num(tgt_arr, nan=0.0, posinf=0.0, neginf=0.0)

        self.feat_arr = feat_arr
        self.tgt_arr = tgt_arr
        self.D = len(common_idx)

    def __len__(self):
        return max(0, self.D - self.T - 1)

    def __getitem__(self, idx):
        # x: [T, N, F], y: [N]
        x = self.feat_arr[idx: idx + self.T]       # [T, N, F]
        y = self.tgt_arr[idx + self.T]              # [N]  — next day after window
        return torch.tensor(x), torch.tensor(y)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 16):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, F, T, N]
        out = self.cnn(x)  # [B, 64, T, N]
        return out


class CNN_LSTM(nn.Module):
    def __init__(self, N_stocks: int, F: int = 16, hidden: int = 128, lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.N = N_stocks
        self.encoder = CNNEncoder(in_channels=F)
        lstm_input = 64 * N_stocks
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, N_stocks)

    def forward(self, x):
        # x: [B, T, N, F]
        B, T, N, F = x.shape
        # Permute for CNN: [B, F, T, N]
        x = x.permute(0, 3, 1, 2)
        cnn_out = self.encoder(x)          # [B, 64, T', N']  (same T,N with padding=1)
        _, C, T2, N2 = cnn_out.shape
        # Permute to sequence: [B, T', 64*N']
        seq = cnn_out.permute(0, 2, 1, 3).contiguous().reshape(B, T2, C * N2)
        lstm_out, _ = self.lstm(seq)       # [B, T', hidden]
        last = lstm_out[:, -1, :]          # [B, hidden]
        return self.head(last)             # [B, N_stocks]


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mse_loss(pred, actual):
    return nn.functional.mse_loss(pred, actual)


def log_wealth_loss(pred, actual, tc: float = TC):
    position = torch.tanh(pred)                             # [B, N]
    daily_ret = position * actual - tc                      # [B, N]
    log_w = torch.log1p(daily_ret.clamp(min=-0.999))       # [B, N]
    return -log_w.mean()


def combined_loss(pred, actual, tc: float = TC):
    return log_wealth_loss(pred, actual, tc) + 0.1 * mse_loss(pred, actual)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe = mean(r)/std(r) * sqrt(252)."""
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns)
    if sigma < 1e-8:
        return 0.0
    return float(mu / sigma * math.sqrt(252))


def compute_metrics(preds: np.ndarray, actuals: np.ndarray):
    """
    preds, actuals: [n_samples, N_stocks]
    Returns: dir_acc, sharpe, mae
    """
    # Directional accuracy
    pred_dir = np.sign(preds)
    act_dir = np.sign(actuals)
    dir_acc = float((pred_dir == act_dir).mean())

    # Portfolio returns: equal-weight tanh positions
    positions = np.tanh(preds)
    port_ret = (positions * actuals - TC).mean(axis=1)  # [n_samples]
    sharpe = compute_sharpe(port_ret)

    # MAE on returns
    mae = float(np.abs(preds - actuals).mean())

    return dir_acc, sharpe, mae


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, loss_fn, device, clip_norm=1.0):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_actuals = [], []
    total_mse = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        mse = nn.functional.mse_loss(pred, y).item()
        total_mse += mse * x.size(0)
        n += x.size(0)
        all_preds.append(pred.cpu().numpy())
        all_actuals.append(y.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    dir_acc, sharpe, mae = compute_metrics(preds, actuals)
    return total_mse / max(n, 1), dir_acc, sharpe, mae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir} ...")
    universe = load_universe(data_dir)
    print(f"Tickers loaded: {len(universe)}")

    # Keep only tickers with enough rows in BOTH train and val periods so N is consistent
    SPLITS = [("2015-01-01", "2021-12-31"), ("2022-01-01", "2023-12-31")]
    valid_tickers = {
        t for t, df in universe.items()
        if all(len(df.loc[s:e]) >= T_LOOKBACK + 50 for s, e in SPLITS)
    }
    universe = {t: df for t, df in universe.items() if t in valid_tickers}
    print(f"Tickers after cross-split filter: {len(universe)}")

    train_ds = StockDataset(universe, "2015-01-01", "2021-12-31")
    val_ds = StockDataset(universe, "2022-01-01", "2023-12-31")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = CNN_LSTM(N_stocks=train_ds.N).to(device)
    n_params = count_params(model)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    header = f"{'epoch':>6} | {'train_mse':>10} | {'val_mae':>8} | {'val_dir_acc':>11} | {'val_sharpe':>10}"
    print(header)
    print("-" * len(header))

    best_sharpe = -1e9
    best_state = None

    for epoch in range(1, args.epochs + 1):
        # Loss selection: MSE warmup, then log-wealth
        if epoch <= args.warmup_epochs:
            loss_fn = mse_loss
        else:
            loss_fn = combined_loss

        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_mse, val_dir_acc, val_sharpe, val_mae = eval_epoch(model, val_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_mse": round(train_loss, 6),
            "val_mse": round(val_mse, 6),
            "val_mae": round(val_mae, 6),
            "val_dir_acc": round(val_dir_acc, 4),
            "val_sharpe": round(val_sharpe, 4),
            "elapsed_s": round(time.time() - t0, 2),
        }
        history.append(row)

        print(f"{epoch:>6} | {train_loss:>10.6f} | {val_mae:>8.5f} | {val_dir_acc:>11.4f} | {val_sharpe:>10.4f}")

        if val_sharpe > best_sharpe:
            best_sharpe = val_sharpe
            torch.save(model.state_dict(), CKPT_DIR / "ts_step010_best.pt")

    # Restore best checkpoint
    ckpt_path = CKPT_DIR / "ts_step010_best.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded best checkpoint from {ckpt_path}")

    # Final evaluation
    val_mse, val_dir_acc, val_sharpe, val_mae = eval_epoch(model, val_loader, device)

    results = {
        "step": "ts_step010_cnn_lstm",
        "model": "CNN_LSTM",
        "n_params": n_params,
        "n_stocks": train_ds.N,
        "T_lookback": T_LOOKBACK,
        "device": args.device,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "best_val_sharpe": round(best_sharpe, 4),
        "final_val_dir_acc": round(val_dir_acc, 4),
        "final_val_mae": round(val_mae, 6),
        "final_val_mse": round(val_mse, 6),
        "history": history,
    }

    out_path = results_dir / "ts_step010_cnn_lstm.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Best val sharpe: {best_sharpe:.4f} | Final dir_acc: {val_dir_acc:.4f} | MAE: {val_mae:.6f} | params: {n_params:,}")


if __name__ == "__main__":
    main()
