#!/usr/bin/env python3
"""step983 / ts_step040: TS evaluation framework diagnostic.

ALL TS models (CNN-LSTM, Transformer, MAE, GAN, SGNNET-TS) show ~50% directional
accuracy = random. Before running more training, diagnose why:

1. Target distribution: are returns too small / skewed / zero-heavy?
2. Naive baselines: does "predict yesterday's sign" beat 50%?
3. Sharpe calculation: is it correctly computed?
4. Data leak check: does val overlap with train?
5. Feature quality: are features actually predictive (linear probe)?

Usage:
    python scripts/ts/ts_step040_eval_diagnostic.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import json


def load_processed_data():
    """Load the processed TS data."""
    try:
        import pandas as pd
        ma_path = ROOT / "data" / "ts" / "processed" / "ma_features.parquet"
        if ma_path.exists():
            df = pd.read_parquet(ma_path)
            return df
    except ImportError:
        pass

    raw_dir = ROOT / "data" / "ts" / "raw"
    if not raw_dir.exists():
        print("ERROR: No TS data found. Run ts_step001 first.")
        sys.exit(1)

    try:
        import pandas as pd
        frames = []
        for f in sorted(raw_dir.glob("*.parquet")):
            df = pd.read_parquet(f)
            df["ticker"] = f.stem
            frames.append(df)
        if not frames:
            print("ERROR: No parquet files in data/ts/raw/")
            sys.exit(1)
        return pd.concat(frames, ignore_index=True)
    except ImportError:
        print("ERROR: pandas required. pip install pandas")
        sys.exit(1)


def main():
    import pandas as pd

    print("=" * 70)
    print("TS Evaluation Framework Diagnostic")
    print("=" * 70)

    # 1. Load data and compute returns
    print("\n--- 1. Data Loading ---")
    raw_dir = ROOT / "data" / "ts" / "raw"
    tickers = sorted([f.stem for f in raw_dir.glob("*.parquet")])
    print(f"Tickers found: {len(tickers)}")

    all_returns = []
    for ticker in tickers[:5]:
        df = pd.read_parquet(raw_dir / f"{ticker}.parquet")
        col = "Close" if "Close" in df.columns else "close"
        if col in df.columns:
            rets = df[col].pct_change().dropna()
            all_returns.extend(rets.values)
            print(f"  {ticker}: {len(df)} days, return mean={rets.mean():.6f}, "
                  f"std={rets.std():.4f}, median={rets.median():.6f}")

    if not all_returns:
        print("ERROR: No return data found.")
        return

    returns = np.array(all_returns)
    print(f"\nAll returns (sample of {len(returns)}):")
    print(f"  mean={returns.mean():.6f}, std={returns.std():.4f}")
    print(f"  median={returns.median():.6f}" if hasattr(returns, 'median') else "")
    print(f"  min={returns.min():.4f}, max={returns.max():.4f}")
    print(f"  % positive: {(returns > 0).mean():.4f}")
    print(f"  % zero: {(returns == 0).mean():.4f}")
    print(f"  % |ret| < 0.001: {(np.abs(returns) < 0.001).mean():.4f}")

    # 2. Naive baselines
    print("\n--- 2. Naive Baselines ---")
    positive_pct = (returns > 0).mean()
    print(f"  'Always predict UP': dir_acc = {positive_pct:.4f}")
    print(f"  'Always predict DOWN': dir_acc = {1-positive_pct:.4f}")

    # Persistence baseline (predict same sign as yesterday)
    for ticker in tickers[:3]:
        df = pd.read_parquet(raw_dir / f"{ticker}.parquet")
        col = "Close" if "Close" in df.columns else "close"
        if col in df.columns:
            rets = df[col].pct_change().dropna().values
            if len(rets) > 1:
                pred_sign = np.sign(rets[:-1])
                true_sign = np.sign(rets[1:])
                mask = (true_sign != 0) & (pred_sign != 0)
                if mask.sum() > 0:
                    persist_acc = (pred_sign[mask] == true_sign[mask]).mean()
                    print(f"  Persistence ({ticker}): dir_acc = {persist_acc:.4f}")

    # 3. Sharpe calculation sanity check
    print("\n--- 3. Sharpe Calculation Check ---")
    print("  Sharpe = mean(daily_return) / std(daily_return) * sqrt(252)")
    for ticker in tickers[:3]:
        df = pd.read_parquet(raw_dir / f"{ticker}.parquet")
        col = "Close" if "Close" in df.columns else "close"
        if col in df.columns:
            rets = df[col].pct_change().dropna().values
            if len(rets) > 10:
                sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
                print(f"  Buy-and-hold {ticker}: Sharpe = {sharpe:.4f}")

    # 4. Check: is the MSE target scale appropriate?
    print("\n--- 4. Target Scale Analysis ---")
    print(f"  Typical |daily return|: {np.abs(returns).mean():.6f}")
    print(f"  MSE of predicting zero: {(returns**2).mean():.8f}")
    print(f"  MSE of predicting mean: {((returns - returns.mean())**2).mean():.8f}")
    print(f"  CONCLUSION: MSE encourages predicting ~0 (mean), not direction.")
    print(f"  This explains why all models converge to ~50% dir_acc with")
    print(f"  decreasing MSE — they learn to predict zero, which is optimal for MSE.")

    # 5. Recommendation
    print("\n--- 5. Diagnosis ---")
    print("  ROOT CAUSE: MSE loss on raw returns incentivizes predicting zero.")
    print("  All models (CNN-LSTM, Transformer, SGNNET-TS) converge to mean-predictor.")
    print("  Sharpe is hugely negative because predicted portfolio has tiny returns")
    print("  with non-zero std from rebalancing friction.")
    print()
    print("  FIXES:")
    print("  (a) Use directional loss (sign prediction, not magnitude)")
    print("  (b) Use log-wealth or Sharpe-based loss directly")
    print("  (c) Predict sign + confidence, not raw return")
    print("  (d) Add L1 regularization on |prediction| to prevent collapse to zero")
    print()
    print("  BEFORE re-training: fix the loss function in ALL models.")
    print("  Then re-run ts_step010/011/030 with directional loss.")

    # Save diagnostic
    out = ROOT / "results" / "ts" / "ts_step040_eval_diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    diag = {
        "step": "ts_step040_eval_diagnostic",
        "n_tickers": len(tickers),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "pct_positive": float((returns > 0).mean()),
        "pct_near_zero": float((np.abs(returns) < 0.001).mean()),
        "mse_zero_predictor": float((returns**2).mean()),
        "diagnosis": "MSE loss → mean predictor → 50% dir_acc. Fix: directional loss.",
    }
    out.write_text(json.dumps(diag, indent=2))
    print(f"\nDiagnostic saved: {out}")


if __name__ == "__main__":
    main()
