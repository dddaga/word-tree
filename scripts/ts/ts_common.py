"""Shared data loading, metrics, losses, and training utilities for TS experiments."""
# See ts_model_sgnnet.py (model), ts_step030_sgnnet_ts.py (main)
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

FEATURE_COLS = [
    "ret_1d",
    "ma_ratio_5", "ma_ratio_21", "ma_ratio_55", "ma_ratio_89",
    "ma_cross_5_21", "ma_cross_13_55", "ma_cross_21_89",
    "vol_20", "vol_60",
    "spread_z", "resid_z", "delta_corr", "lag1_ret",
    "high_low_ratio", "volume_z",
]
assert len(FEATURE_COLS) == 16

T_LOOKBACK = 60
TC = 0.001


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """All features use shift(1) before rolling — no lookahead."""
    out = pd.DataFrame(index=df.index)
    close = df["close"].shift(1)
    volume = df["volume"].shift(1)
    high = df["high"].shift(1)
    low = df["low"].shift(1)
    ret = close / close.shift(1) - 1

    out["ret_1d"] = ret
    for w in [5, 21, 55, 89]:
        out[f"ma_ratio_{w}"] = close / close.rolling(w).mean()
    for f, s in [(5, 21), (13, 55), (21, 89)]:
        out[f"ma_cross_{f}_{s}"] = close.rolling(f).mean() / close.rolling(s).mean() - 1

    out["vol_20"] = ret.rolling(20).std()
    out["vol_60"] = ret.rolling(60).std()

    mu60 = close.rolling(60).mean()
    std60 = close.rolling(60).std()
    out["spread_z"] = (close - mu60) / (std60 + 1e-8)
    out["resid_z"] = (ret - ret.rolling(60).mean()) / (ret.rolling(60).std() + 1e-8)
    out["delta_corr"] = out["vol_20"] / (out["vol_60"] + 1e-8) - 1.0
    out["lag1_ret"] = ret.shift(1)
    out["high_low_ratio"] = (high - low) / (close + 1e-8)
    vol_mu = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    out["volume_z"] = (volume - vol_mu) / (vol_std + 1e-8)
    return out


def _compute_target(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().shift(-1)


def load_universe(data_dir: Path) -> dict:
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


class StockDataset(Dataset):
    """Windows of shape [T_lookback, N_stocks, F_features] → target [N_stocks]."""

    def __init__(self, universe: dict, start_date: str, end_date: str, T: int = T_LOOKBACK):
        self.T = T
        filtered = {t: df.loc[start_date:end_date] for t, df in universe.items()
                    if len(df.loc[start_date:end_date]) >= T + 50}
        if not filtered:
            self.D = self.N = 0; return

        tickers = sorted(filtered.keys())
        self.N = len(tickers)
        frames = [filtered[t] for t in tickers]
        common_idx = frames[0].index
        for f in frames[1:]:
            common_idx = common_idx.intersection(f.index)
        common_idx = common_idx.sort_values()
        print(f"  [{start_date}:{end_date}] {len(tickers)} tickers, {len(common_idx)} common dates")

        feat_arr = np.stack([f.loc[common_idx, FEATURE_COLS].values
                             for f in frames], axis=1).astype(np.float32)
        tgt_arr = np.stack([f.loc[common_idx, "target"].values
                            for f in frames], axis=1).astype(np.float32)

        self.feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
        self.tgt_arr = np.nan_to_num(tgt_arr, nan=0.0, posinf=0.0, neginf=0.0)
        self.D = len(common_idx)

    def __len__(self): return max(0, self.D - self.T - 1)

    def __getitem__(self, idx):
        return (torch.tensor(self.feat_arr[idx: idx + self.T]),
                torch.tensor(self.tgt_arr[idx + self.T]))


def compute_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2: return 0.0
    mu, sigma = np.mean(returns), np.std(returns)
    return float(mu / sigma * math.sqrt(252)) if sigma >= 1e-8 else 0.0


def compute_metrics(preds: np.ndarray, actuals: np.ndarray):
    dir_acc = float((np.sign(preds) == np.sign(actuals)).mean())
    port_ret = (np.tanh(preds) * actuals - TC).mean(axis=1)
    sharpe = compute_sharpe(port_ret)
    mae = float(np.abs(preds - actuals).mean())
    return dir_acc, sharpe, mae


# --- Losses ---

def log_wealth_loss(pred, actual, tc: float = TC):
    position = torch.tanh(pred)
    daily_ret = position * actual - tc
    return -torch.log1p(daily_ret.clamp(min=-0.999)).mean()


def directional_loss(pred, actual):
    """BCE loss on direction (sign) prediction — fixes MSE mean-predictor collapse."""
    target_dir = (actual > 0).float()
    prob_up = torch.sigmoid(pred * 50.0)
    return F.binary_cross_entropy(prob_up, target_dir)


def mse_loss(pred, actual):
    return F.mse_loss(pred, actual)


# --- Training utilities ---

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, loss_fn, device, clip_norm=1.0):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total += loss.item() * x.size(0); n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_actuals, total_mse, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_mse += F.mse_loss(pred, y).item() * x.size(0); n += x.size(0)
        all_preds.append(pred.cpu().numpy())
        all_actuals.append(y.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    dir_acc, sharpe, mae = compute_metrics(preds, actuals)
    return total_mse / max(n, 1), dir_acc, sharpe, mae
