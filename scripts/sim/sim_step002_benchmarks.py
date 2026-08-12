"""sim_step002 — Benchmark strategies for the backtesting engine.

Evaluates 4 strategies on the 2022-2023 validation period against the Nifty50 index:
  1. NiftyIndexStrategy     — equal-weight buy-and-hold on day 1 (index proxy).
  2. BuyAndHoldStrategy     — equal-weight, rebalanced monthly.
  3. MACrossStrategy        — 21d/55d golden-cross per stock, equal-weight among longs.
  4. RandomSignalStrategy   — 100-seed Monte Carlo null distribution (median NAV).

Usage:
    python scripts/sim/sim_step002_benchmarks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "sim"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

from scripts.sim.sim_step001_engine import (  # noqa: E402
    Backtester,
    Portfolio,
    Strategy,
    STARTING_CAPITAL,
)

VAL_START = "2022-01-01"
VAL_END   = "2023-12-31"
MIN_ROWS  = 200


# ---------------------------------------------------------------------------
# Strategy 1: Nifty Index Proxy — equal-weight all tickers on day 1, no rebal
# ---------------------------------------------------------------------------

class NiftyIndexStrategy(Strategy):
    """Equal-weight all tickers on the first bar; never rebalances."""

    def reset(self, tickers: list[str]) -> None:
        self.tickers = tickers
        self._invested = False

    def on_bar(
        self,
        date: pd.Timestamp,
        prices: dict[str, float],
        volumes: dict[str, float],
        features: Optional[dict[str, np.ndarray]],
        portfolio: Portfolio,
    ) -> dict[str, float]:
        if self._invested:
            return {}
        self._invested = True
        w = 0.95 / max(len(prices), 1)
        return {t: w for t in prices}


# ---------------------------------------------------------------------------
# Strategy 2: Buy-and-Hold — equal-weight, rebalance on first day of each month
# ---------------------------------------------------------------------------

class BuyAndHoldStrategy(Strategy):
    """Equal-weight all tickers, rebalanced on the first trading day of each month."""

    def reset(self, tickers: list[str]) -> None:
        self.tickers = tickers
        self._last_month: Optional[int] = None

    def on_bar(
        self,
        date: pd.Timestamp,
        prices: dict[str, float],
        volumes: dict[str, float],
        features: Optional[dict[str, np.ndarray]],
        portfolio: Portfolio,
    ) -> dict[str, float]:
        if self._last_month == date.month:
            return {}
        self._last_month = date.month
        w = 0.95 / max(len(prices), 1)
        return {t: w for t in prices}


# ---------------------------------------------------------------------------
# Strategy 3: MA Cross — 21d/55d golden cross, equal-weight among longs
# ---------------------------------------------------------------------------

class MACrossStrategy(Strategy):
    """
    Per-stock 21d/55d simple moving average crossover.
    Long any stock where fast MA > slow MA; equal-weight among longs.
    Uses only past prices — no lookahead.
    """

    FAST = 21
    SLOW = 55

    def reset(self, tickers: list[str]) -> None:
        self.tickers = tickers
        self._history: dict[str, list[float]] = {t: [] for t in tickers}

    def on_bar(
        self,
        date: pd.Timestamp,
        prices: dict[str, float],
        volumes: dict[str, float],
        features: Optional[dict[str, np.ndarray]],
        portfolio: Portfolio,
    ) -> dict[str, float]:
        for t, p in prices.items():
            if t in self._history:
                self._history[t].append(p)

        longs = []
        for t in prices:
            hist = self._history.get(t, [])
            if len(hist) < self.SLOW:
                continue
            fast_ma = np.mean(hist[-self.FAST:])
            slow_ma = np.mean(hist[-self.SLOW:])
            if fast_ma > slow_ma:
                longs.append(t)

        if not longs:
            return {t: 0.0 for t in prices}

        w = 0.95 / len(longs)
        weights = {t: 0.0 for t in prices}
        for t in longs:
            weights[t] = w
        return weights


# ---------------------------------------------------------------------------
# Strategy 4: Random Signal — Monte Carlo null distribution
# ---------------------------------------------------------------------------

class RandomSignalStrategy(Strategy):
    """
    Random ±1 direction per stock per day across n_seeds seeds.
    Returns the median NAV curve after running all seeds.
    Used as a null-distribution baseline.
    """

    def __init__(self, n_seeds: int = 100):
        self.n_seeds = n_seeds
        self._seed: int = 0

    def set_seed(self, seed: int) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, tickers: list[str]) -> None:
        self.tickers = tickers
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(self._seed)

    def on_bar(
        self,
        date: pd.Timestamp,
        prices: dict[str, float],
        volumes: dict[str, float],
        features: Optional[dict[str, np.ndarray]],
        portfolio: Portfolio,
    ) -> dict[str, float]:
        tickers = list(prices.keys())
        directions = self._rng.choice([-1, 1], size=len(tickers))
        longs = [t for t, d in zip(tickers, directions) if d > 0]
        if not longs:
            return {t: 0.0 for t in tickers}
        w = 0.95 / len(longs)
        weights = {t: 0.0 for t in tickers}
        for t in longs:
            weights[t] = w
        return weights


def run_random_strategy_median(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    n_seeds: int = 100,
) -> "BacktestResult":
    """Run RandomSignalStrategy n_seeds times; return result with median NAV."""
    from scripts.sim.sim_step001_engine import BacktestResult, DayResult

    all_navs: list[pd.Series] = []
    for seed in range(n_seeds):
        strat = RandomSignalStrategy(n_seeds=n_seeds)
        strat.set_seed(seed)
        bt = Backtester(strat, prices, volumes, starting_capital=STARTING_CAPITAL)
        res = bt.run()
        all_navs.append(res.nav_series)

    nav_matrix = pd.concat(all_navs, axis=1)
    median_nav = nav_matrix.median(axis=1)

    # Build a synthetic BacktestResult from the median NAV
    ret_series = median_nav.pct_change().fillna(0)
    days = [
        DayResult(
            date=date,
            nav=nav,
            daily_ret=ret,
            cash=0.0,
            n_positions=0,
            exec_log={},
        )
        for date, nav, ret in zip(median_nav.index, median_nav.values, ret_series.values)
    ]
    return BacktestResult(days, list(prices.columns), STARTING_CAPITAL)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import yfinance as yf

    raw_dir = ROOT / "data" / "ts" / "raw"
    parquet_files = [
        f for f in raw_dir.glob("*.parquet")
        if not f.name.startswith(".")
    ]

    dfs_close: dict[str, pd.Series] = {}
    dfs_vol:   dict[str, pd.Series] = {}

    for fpath in parquet_files:
        ticker = fpath.stem
        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index)
        mask = (df.index >= VAL_START) & (df.index <= VAL_END)
        sub = df.loc[mask]
        if len(sub) >= MIN_ROWS:
            dfs_close[ticker] = sub["close"]
            dfs_vol[ticker]   = sub["volume"]

    prices  = pd.DataFrame(dfs_close).sort_index()
    volumes = pd.DataFrame(dfs_vol).sort_index()

    print(f"Universe: {len(prices.columns)} tickers, {len(prices)} trading days")

    # Download Nifty50 index benchmark
    print("Downloading ^NSEI...")
    nifty_raw = yf.download("^NSEI", start=VAL_START, end=VAL_END, auto_adjust=True, progress=False)
    nifty_close = nifty_raw["Close"].squeeze().dropna()
    nifty_nav = (nifty_close / nifty_close.iloc[0]) * STARTING_CAPITAL
    nifty_nav = nifty_nav.reindex(prices.index).ffill()

    all_results: dict[str, dict] = {}

    # Strategy 1: Nifty Index Proxy
    print("\n--- NiftyIndexStrategy ---")
    bt = Backtester(NiftyIndexStrategy(), prices, volumes, starting_capital=STARTING_CAPITAL)
    res = bt.run()
    ts = res.tearsheet(benchmark_nav=nifty_nav)
    all_results["NiftyIndexStrategy"] = ts

    # Strategy 2: Buy and Hold
    print("\n--- BuyAndHoldStrategy ---")
    bt = Backtester(BuyAndHoldStrategy(), prices, volumes, starting_capital=STARTING_CAPITAL)
    res = bt.run()
    ts = res.tearsheet(benchmark_nav=nifty_nav)
    all_results["BuyAndHoldStrategy"] = ts

    # Strategy 3: MA Cross
    print("\n--- MACrossStrategy ---")
    bt = Backtester(MACrossStrategy(), prices, volumes, starting_capital=STARTING_CAPITAL)
    res = bt.run()
    ts = res.tearsheet(benchmark_nav=nifty_nav)
    all_results["MACrossStrategy"] = ts

    # Strategy 4: Random Signal (100 seeds, median)
    print("\n--- RandomSignalStrategy (100 seeds, median NAV) ---")
    rand_res = run_random_strategy_median(prices, volumes, n_seeds=100)
    ts = rand_res.tearsheet(benchmark_nav=nifty_nav)
    all_results["RandomSignalStrategy"] = ts

    # Save combined results
    out_path = RESULTS_DIR / "sim_step002_benchmarks.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
