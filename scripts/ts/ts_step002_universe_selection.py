"""ts_step002 — Universe selection + correlation-ordered stock list.

Steps:
  1. Load all downloaded tickers from data/ts/raw/
  2. Compute annualised volatility → take top-N by volatility
  3. Compute 60d rolling correlation matrix → hierarchical cluster with optimal leaf order
  4. Output: data/ts/stock_order.json + correlation heatmap

Usage:
    python scripts/ts/ts_step002_universe_selection.py [--top-n 30]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list, dendrogram
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "ts" / "raw"
PROC_DIR = ROOT / "data" / "ts" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

CORR_WINDOW = 60   # days for rolling correlation
MIN_HISTORY = 500  # minimum trading days to include


def load_returns(min_history: int = MIN_HISTORY) -> pd.DataFrame:
    """Load all tickers, return daily returns DataFrame."""
    dfs = {}
    for fpath in sorted(f for f in RAW_DIR.glob("*.parquet") if not f.stem.startswith(".")):
        ticker = fpath.stem
        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if len(df) < min_history:
            print(f"  Skip {ticker}: only {len(df)} rows")
            continue
        dfs[ticker] = df["close"].pct_change()
    return pd.DataFrame(dfs).dropna(how="all")


def compute_volatility(returns: pd.DataFrame) -> pd.Series:
    """Annualised volatility (252-day std)."""
    return returns.std() * np.sqrt(252)


def correlation_cluster_order(returns: pd.DataFrame) -> list[str]:
    """Return tickers ordered by hierarchical clustering on (1 - |corr|) distance.

    Uses optimal_leaf_ordering (scipy) to minimise sum of adjacent distances —
    puts correlated stocks close together on the synthetic spatial axis.
    """
    corr = returns.corr().fillna(0)
    dist = 1 - np.abs(corr.values)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2  # ensure symmetry

    condensed = squareform(dist)
    Z = linkage(condensed, method="ward")
    Z_opt = optimal_leaf_ordering(Z, condensed)
    order_idx = leaves_list(Z_opt)
    return [corr.columns[i] for i in order_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=40,
                        help="Take top-N stocks by volatility (default: 40)")
    parser.add_argument("--min-history", type=int, default=MIN_HISTORY)
    args = parser.parse_args()

    print(f"ts_step002 | top_n={args.top_n} | min_history={args.min_history}")

    returns = load_returns(min_history=args.min_history)
    print(f"Loaded {returns.shape[1]} tickers, {len(returns)} dates "
          f"({returns.index.min().date()} → {returns.index.max().date()})")

    vol = compute_volatility(returns)
    top_tickers = vol.nlargest(args.top_n).index.tolist()
    print(f"\nTop-{args.top_n} by annualised volatility:")
    for t, v in vol[top_tickers].items():
        print(f"  {t:20s}  {v:.3f}")

    ret_subset = returns[top_tickers].dropna(how="all")
    ordered = correlation_cluster_order(ret_subset)

    out = {
        "ordered_tickers": ordered,
        "volatility": {t: round(float(vol[t]), 4) for t in ordered},
        "n_stocks": len(ordered),
        "corr_window": CORR_WINDOW,
        "date_range": [str(returns.index.min().date()), str(returns.index.max().date())],
        "note": "Synthetic spatial axis — hierarchical Ward clustering on (1-|corr|) distance + optimal leaf ordering",
    }
    out_path = PROC_DIR / "stock_order.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nstock_order.json saved: {out_path}")
    print(f"Final universe ({len(ordered)} tickers, correlation-ordered):")
    print("  " + ", ".join(ordered))


if __name__ == "__main__":
    main()
