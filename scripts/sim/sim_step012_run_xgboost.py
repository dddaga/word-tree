"""sim_step012 — XGBoost signal model backtested on 2022-2023 val period.

Trains a per-stock XGBoost binary classifier on MA features (same as ts_step003)
on the 2015-2021 train period. Uses predicted up-probability as signal.

Signal:  prob = xgb.predict_proba(features)[:,1]
         target_weight = tanh((prob - 0.5) * 4) * 0.15
         Only go long if prob > 0.52.

Cross-split filter: tickers with ≥ T+50 rows in BOTH train AND val periods (T=60).

Usage:
    python scripts/sim/sim_step012_run_xgboost.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "sim"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# XGBoost with sklearn GradientBoostingClassifier fallback
try:
    from xgboost import XGBClassifier
    _XGB_BACKEND = "xgboost"
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier  # type: ignore
    _XGB_BACKEND = "sklearn_gbc"
    warnings.warn("xgboost not found — using sklearn GradientBoostingClassifier as fallback.")

from scripts.sim.sim_step001_engine import (  # noqa: E402
    Backtester,
    Portfolio,
    Strategy,
    STARTING_CAPITAL,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_START = "2015-01-01"
TRAIN_END   = "2021-12-31"
VAL_START   = "2022-01-01"
VAL_END     = "2023-12-31"

T = 60          # lookback length for cross-split filter
MIN_ROWS = T + 50  # minimum rows per period

MA_WINDOWS      = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 200]
MA_CROSS_PAIRS  = [(5, 21), (13, 55), (21, 89)]

XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)

SIGNAL_SCALE   = 4.0   # tanh stretch
MAX_WEIGHT     = 0.15  # per-stock cap
PROB_THRESHOLD = 0.52  # minimum probability to go long


# ---------------------------------------------------------------------------
# Feature computation (copied from ts_step003 — no import to avoid path issues)
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Build MA features for a single ticker. NO lookahead (all shift(1))."""
    close = df["close"].copy()
    ret   = close.pct_change()

    feats = pd.DataFrame(index=df.index)
    feats["ticker"] = ticker
    feats["close"]  = close
    feats["ret_1d"] = ret

    # MA ratios
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

    # Target: next-day direction (1 if up, 0 if down/flat)
    next_ret = ret.shift(-1)
    feats["target_dir"] = (next_ret > 0).astype(int)

    return feats.dropna()


FEATURE_COLS = (
    [f"ma_ratio_{w}" for w in MA_WINDOWS]
    + [f"ma_cross_{f}_{s}" for f, s in MA_CROSS_PAIRS]
    + ["vol_20", "vol_60"]
)


# ---------------------------------------------------------------------------
# Per-stock model training
# ---------------------------------------------------------------------------

def train_models(universe: dict[str, pd.DataFrame]) -> dict[str, object]:
    """
    Train one XGBoost classifier per ticker on the train period.
    Returns {ticker: fitted_model} plus a scaler dict.
    """
    models: dict[str, object]  = {}
    scalers: dict[str, object] = {}
    skipped = []

    for ticker, df in universe.items():
        feats = compute_features(df, ticker)
        train = feats[feats.index <= TRAIN_END]

        if len(train) < 100:
            skipped.append(ticker)
            continue

        X = train[FEATURE_COLS].values
        y = train["target_dir"].values

        scaler = StandardScaler().fit(X)
        X_s    = scaler.transform(X)

        if _XGB_BACKEND == "xgboost":
            clf = XGBClassifier(**XGB_PARAMS)
        else:
            # sklearn GBC doesn't accept all xgb params
            clf = XGBClassifier(
                n_estimators=XGB_PARAMS["n_estimators"],
                max_depth=XGB_PARAMS["max_depth"],
                learning_rate=XGB_PARAMS["learning_rate"],
                subsample=XGB_PARAMS["subsample"],
                random_state=XGB_PARAMS["random_state"],
            )
        clf.fit(X_s, y)

        models[ticker]  = clf
        scalers[ticker] = scaler

    if skipped:
        print(f"  Skipped (insufficient train rows): {skipped}")

    return models, scalers


# ---------------------------------------------------------------------------
# XGBoost Strategy
# ---------------------------------------------------------------------------

class XGBoostStrategy(Strategy):
    """
    Per-stock XGBoost binary classifier signal.
    Features are computed on-the-fly using a rolling price buffer.
    """

    def __init__(
        self,
        models:  dict[str, object],
        scalers: dict[str, object],
        lookback: int = 210,  # max MA window + buffer
    ):
        self.models   = models
        self.scalers  = scalers
        self.lookback = lookback

    def reset(self, tickers: list[str]) -> None:
        self.tickers = tickers
        # Per-ticker rolling close history
        self._history: dict[str, list[float]] = {t: [] for t in tickers}

    def _compute_single_feature(self, prices_arr: np.ndarray) -> Optional[np.ndarray]:
        """
        Given a 1-D array of recent close prices (oldest first),
        compute a single feature row using MA ratios + crossovers + vol.
        Returns None if insufficient data.
        """
        n = len(prices_arr)
        max_window = max(MA_WINDOWS)
        if n < max_window + 1:
            return None

        close = pd.Series(prices_arr)
        ret   = close.pct_change()

        row = []
        # MA ratios — shift(1) means we use close[-2] as "current", close[-1] shifted away
        # In online mode: the last element is today's price (already observed),
        # so shift(1) translates to prices_arr[-2] for the "current" close.
        cur = close.iloc[-2]   # shift(1) of the last bar

        for w in MA_WINDOWS:
            if n - 1 < w:
                return None
            ma = close.iloc[-(w+1):-1].mean()
            row.append(cur / ma - 1 if ma != 0 else 0.0)

        for f, s in MA_CROSS_PAIRS:
            if n - 1 < s:
                return None
            ma_f = close.iloc[-(f+1):-1].mean()
            ma_s = close.iloc[-(s+1):-1].mean()
            row.append(ma_f / ma_s - 1 if ma_s != 0 else 0.0)

        # vol_20, vol_60 — shift(1) means we exclude the last return
        shifted_ret = ret.iloc[:-1]  # drop latest (lookahead guard)
        if len(shifted_ret) < 60:
            return None
        row.append(float(shifted_ret.iloc[-20:].std()))
        row.append(float(shifted_ret.iloc[-60:].std()))

        return np.array(row, dtype=np.float32)

    def on_bar(
        self,
        date: pd.Timestamp,
        prices: dict[str, float],
        volumes: dict[str, float],
        features: Optional[dict[str, np.ndarray]],
        portfolio: Portfolio,
    ) -> dict[str, float]:
        # Update history
        for t, p in prices.items():
            if t in self._history:
                self._history[t].append(p)
                # Trim to lookback
                if len(self._history[t]) > self.lookback:
                    self._history[t] = self._history[t][-self.lookback:]

        weights: dict[str, float] = {}

        for t in prices:
            if t not in self.models:
                continue

            hist = self._history.get(t, [])
            feat_row = self._compute_single_feature(np.array(hist, dtype=np.float32))
            if feat_row is None:
                continue

            scaler = self.scalers[t]
            feat_scaled = scaler.transform(feat_row.reshape(1, -1))

            prob = float(self.models[t].predict_proba(feat_scaled)[0, 1])
            signal = prob - 0.5

            if prob > PROB_THRESHOLD:
                weight = float(np.tanh(signal * SIGNAL_SCALE) * MAX_WEIGHT)
            else:
                weight = 0.0

            weights[t] = weight

        return weights


# ---------------------------------------------------------------------------
# Data loading + cross-split filter
# ---------------------------------------------------------------------------

def load_universe(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all parquet files; apply cross-split filter (T+50 rows in each period)."""
    splits = [(TRAIN_START, TRAIN_END), (VAL_START, VAL_END)]
    universe: dict[str, pd.DataFrame] = {}

    for fpath in sorted(raw_dir.glob("*.parquet")):
        if fpath.name.startswith("."):
            continue
        ticker = fpath.stem
        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        if all(len(df.loc[s:e]) >= MIN_ROWS for s, e in splits):
            universe[ticker] = df

    return universe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    raw_dir = ROOT / "data" / "ts" / "raw"

    print(f"Backend: {_XGB_BACKEND}")
    print(f"Loading universe from {raw_dir} ...")
    universe = load_universe(raw_dir)
    print(f"Tickers after cross-split filter: {len(universe)}")

    # Build val-period price + volume dataframes
    dfs_close: dict[str, pd.Series] = {}
    dfs_vol:   dict[str, pd.Series] = {}
    for ticker, df in universe.items():
        mask = (df.index >= VAL_START) & (df.index <= VAL_END)
        sub  = df.loc[mask]
        if len(sub) > 0:
            dfs_close[ticker] = sub["close"]
            dfs_vol[ticker]   = sub["volume"]

    prices  = pd.DataFrame(dfs_close).sort_index()
    volumes = pd.DataFrame(dfs_vol).sort_index()
    print(f"Val period: {VAL_START} → {VAL_END}  |  {len(prices)} trading days  |  {len(prices.columns)} tickers")

    # Train models
    print(f"\nTraining XGBoost per stock ({len(universe)} tickers)...")
    models, scalers = train_models(universe)
    print(f"Models trained: {len(models)}")

    # Build strategy + backtest
    strategy = XGBoostStrategy(models, scalers)
    bt = Backtester(strategy, prices, volumes, starting_capital=STARTING_CAPITAL)

    print("\nRunning backtest...")
    result = bt.run()

    # Tearsheet
    ts = result.tearsheet()

    # Save
    out = {
        "strategy": "XGBoostStrategy",
        "backend": _XGB_BACKEND,
        "xgb_params": {k: v for k, v in XGB_PARAMS.items() if k != "verbosity"},
        "signal": {
            "prob_threshold": PROB_THRESHOLD,
            "signal_scale":   SIGNAL_SCALE,
            "max_weight":     MAX_WEIGHT,
        },
        "universe_size":  len(universe),
        "models_trained": len(models),
        "val_period":    f"{VAL_START} to {VAL_END}",
        "tearsheet":     ts,
    }

    out_path = RESULTS_DIR / "sim_step012_xgboost.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
