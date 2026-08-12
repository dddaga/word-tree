"""ts_step003 — Moving average filter baseline.

Question: which time horizon is most informative per stock?

Method:
  - MA windows: Fibonacci + standard [2,3,5,8,13,21,34,55,89,144,200]
  - Features: price/MA (relative strength) + fast/slow crossovers
  - Scorer: LinearRegression + 1-layer MLP (2 scorers to compare)
  - Target: next-day return (regression) + direction (classification)
  - Walk-forward: 2015-2021 train / 2022-2023 val / 2024+ test

Output:
  - data/ts/processed/ma_features.parquet   (all features, all tickers)
  - results/ts/ts_step003_ma_baseline.json  (per-ticker feature importance)

Usage:
    python scripts/ts/ts_step003_ma_baseline.py
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "ts" / "raw"
PROC_DIR = ROOT / "data" / "ts" / "processed"
RESULTS_DIR = ROOT / "results" / "ts"
PROC_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MA_WINDOWS = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 200]
MA_CROSS_PAIRS = [(5, 21), (13, 55), (21, 89)]

TRAIN_END   = "2021-12-31"
VAL_START   = "2022-01-01"
VAL_END     = "2023-12-31"
TEST_START  = "2024-01-01"


def compute_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Build MA features for a single ticker. NO lookahead (all shift(1))."""
    close = df["close"].copy()
    ret = close.pct_change()

    feats = pd.DataFrame(index=df.index)
    feats["ticker"] = ticker
    feats["close"] = close
    feats["ret_1d"] = ret

    # MA ratios (use shift(1) to avoid lookahead on close itself)
    for w in MA_WINDOWS:
        ma = close.shift(1).rolling(w).mean()
        feats[f"ma_ratio_{w}"] = close.shift(1) / ma - 1

    # MA crossovers
    for f, s in MA_CROSS_PAIRS:
        ma_f = close.shift(1).rolling(f).mean()
        ma_s = close.shift(1).rolling(s).mean()
        feats[f"ma_cross_{f}_{s}"] = ma_f / ma_s - 1

    # Volatility
    feats["vol_20"] = ret.shift(1).rolling(20).std()
    feats["vol_60"] = ret.shift(1).rolling(60).std()

    # Target: next-day return (shift -1 = tomorrow's return, known at close of t+1)
    feats["target_ret"] = ret.shift(-1)
    feats["target_dir"] = np.sign(feats["target_ret"])

    return feats.dropna()


class OnelayerMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def evaluate_ticker(ticker: str, df_feats: pd.DataFrame) -> dict:
    feature_cols = [c for c in df_feats.columns
                    if c.startswith("ma_ratio_") or c.startswith("ma_cross_")
                    or c in ("vol_20", "vol_60")]

    train = df_feats[df_feats.index <= TRAIN_END]
    val   = df_feats[(df_feats.index >= VAL_START) & (df_feats.index <= VAL_END)]

    if len(train) < 100 or len(val) < 20:
        return {"ticker": ticker, "skip": "insufficient data"}

    X_tr = train[feature_cols].values
    y_tr = train["target_ret"].values
    X_val = val[feature_cols].values
    y_val = val["target_ret"].values

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # Linear regression
    lr = Ridge(alpha=1.0).fit(X_tr_s, y_tr)
    pred_lr = lr.predict(X_val_s)
    mae_lr = mean_absolute_error(y_val, pred_lr)
    dir_lr = accuracy_score(np.sign(y_val), np.sign(pred_lr))

    # Feature importance (absolute Ridge coefficients)
    importance = dict(zip(feature_cols, np.abs(lr.coef_)))
    top_ma = sorted(
        [(k, v) for k, v in importance.items() if k.startswith("ma_ratio_")],
        key=lambda x: -x[1]
    )[:5]

    # 1-layer MLP
    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)

    mlp = OnelayerMLP(len(feature_cols))
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for _ in range(200):
        opt.zero_grad()
        loss = nn.MSELoss()(mlp(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()

    mlp.eval()
    with torch.no_grad():
        pred_mlp = mlp(X_val_t).numpy()
    mae_mlp = mean_absolute_error(y_val, pred_mlp)
    dir_mlp = accuracy_score(np.sign(y_val), np.sign(pred_mlp))

    return {
        "ticker": ticker,
        "val_rows": len(val),
        "lr_mae": round(float(mae_lr), 6),
        "lr_dir_acc": round(float(dir_lr), 4),
        "mlp_mae": round(float(mae_mlp), 6),
        "mlp_dir_acc": round(float(dir_mlp), 4),
        "top5_ma_windows": [(k.replace("ma_ratio_", ""), round(v, 4)) for k, v in top_ma],
    }


def main():
    parquet_files = sorted(f for f in RAW_DIR.glob("*.parquet") if not f.stem.startswith("."))
    if not parquet_files:
        print("No data found in data/ts/raw/. Run ts_step001 first.")
        return

    print(f"ts_step003 | tickers={len(parquet_files)} | MA windows={MA_WINDOWS}\n")

    all_feats = []
    results = []

    for fpath in parquet_files:
        ticker = fpath.stem
        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        feats = compute_features(df, ticker)
        if len(feats) < 200:
            print(f"  {ticker}: skipped ({len(feats)} rows after dropna)")
            continue

        all_feats.append(feats)
        res = evaluate_ticker(ticker, feats)
        results.append(res)
        if "lr_dir_acc" in res:
            status = f"lr_dir={res['lr_dir_acc']:.3f}  mlp_dir={res['mlp_dir_acc']:.3f}"
            top = res.get("top5_ma_windows", [])
            top_str = ", ".join(f"{w}d" for w, _ in top[:3]) if top else "N/A"
            print(f"  {ticker}: {status}  | top_MA: {top_str}")
        else:
            print(f"  {ticker}: skipped")

    # Save features
    all_df = pd.concat(all_feats)
    all_df.to_parquet(PROC_DIR / "ma_features.parquet")
    print(f"\nFeatures saved: {PROC_DIR / 'ma_features.parquet'}")

    # Save results
    out_path = RESULTS_DIR / "ts_step003_ma_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_path}")

    # Summary
    valid = [r for r in results if "lr_dir_acc" in r]
    if valid:
        avg_lr  = np.mean([r["lr_dir_acc"] for r in valid])
        avg_mlp = np.mean([r["mlp_dir_acc"] for r in valid])
        # Most common informative windows
        all_tops = [w for r in valid for w, _ in r.get("top5_ma_windows", [])]
        from collections import Counter
        window_freq = Counter(all_tops).most_common(5)
        print(f"\n=== Summary ===")
        print(f"Mean directional accuracy — LinearReg: {avg_lr:.3f}  MLP: {avg_mlp:.3f}")
        print(f"Most informative MA windows (across all tickers): {window_freq}")
        print(f"Baseline: 0.500 (random). Target: >0.52 to be useful.")


if __name__ == "__main__":
    main()
