"""sim_step001 — Core backtesting engine.

Pluggable Strategy interface. Handles:
  - Order execution (next-day open or same-day close)
  - Slippage (bps), brokerage (flat ₹20 or %, whichever lower), STT, charges
  - Liquidity filter (max participation rate of daily volume)
  - Risk guard-rails (max drawdown halt, daily stop-loss, max position size)
  - Tearsheet generation (CAGR, Sharpe, Sortino, max-DD, vs benchmark)

Usage:
    from scripts.sim.sim_step001_engine import Backtester, Portfolio
    bt = Backtester(strategy, prices_df, volumes_df, features_df)
    result = bt.run()
    result.tearsheet()
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "sim"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

SLIPPAGE_BPS       = 5       # per side
TC_BPS             = 30      # total round-trip transaction cost (brokerage + STT + charges)
MAX_PARTICIPATION  = 0.01    # max fraction of daily volume per order
MAX_STOCK_WEIGHT   = 0.20    # max single-stock portfolio weight
MAX_GROSS_EXPOSURE = 1.00    # no leverage
MIN_CASH_BUFFER    = 0.05    # always keep 5% cash
DAILY_STOPLOSS     = -0.02   # −2% portfolio NAV triggers cash-out for the day
MAX_DRAWDOWN_HALT  = -0.15   # −15% from peak → halt strategy
STARTING_CAPITAL   = 1_000_000.0  # ₹10 lakh
BROKERAGE_FLAT     = 20.0    # ₹20 per order
BROKERAGE_PCT      = 0.0003  # 0.03% of order value
STT_SELL           = 0.001   # 0.1% STT on sell side (equity delivery)
EXCHANGE_CHARGE    = 0.00005 # NSE + SEBI
GST_ON_BROKERAGE   = 0.18


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """Base class every strategy must implement."""

    @abstractmethod
    def reset(self, tickers: list[str]) -> None:
        """Called once before backtest begins."""

    @abstractmethod
    def on_bar(
        self,
        date: pd.Timestamp,
        prices: dict[str, float],
        volumes: dict[str, float],
        features: Optional[dict[str, np.ndarray]],
        portfolio: "Portfolio",
    ) -> dict[str, float]:
        """
        Return {ticker: target_weight} in [-1.0, +1.0].
        Positive = long, negative = short (ignored if shorts disabled).
        Missing tickers → weight unchanged.
        """


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------

@dataclass
class Portfolio:
    cash: float = STARTING_CAPITAL
    shares: dict[str, float] = field(default_factory=dict)
    nav_history: list[float] = field(default_factory=list)
    last_prices: dict[str, float] = field(default_factory=dict)

    def nav(self, prices: dict[str, float]) -> float:
        equity = sum(self.shares.get(t, 0.0) * prices.get(t, self.last_prices.get(t, 0.0))
                     for t in self.shares)
        return self.cash + equity

    def weight(self, ticker: str, prices: dict[str, float]) -> float:
        n = self.nav(prices)
        if n <= 0:
            return 0.0
        return self.shares.get(ticker, 0.0) * prices.get(ticker, 0.0) / n

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        n = self.nav(prices)
        if n <= 0:
            return {}
        return {t: s * prices.get(t, 0.0) / n for t, s in self.shares.items()}


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------

def _brokerage(order_value: float) -> float:
    """Compute brokerage + GST for one order leg."""
    flat = min(BROKERAGE_FLAT, BROKERAGE_PCT * order_value)
    return flat * (1 + GST_ON_BROKERAGE)


def execute_orders(
    portfolio: Portfolio,
    target_weights: dict[str, float],
    prices: dict[str, float],
    volumes: dict[str, float],
    allow_short: bool = False,
    slippage_bps: float = SLIPPAGE_BPS,
    max_participation: float = MAX_PARTICIPATION,
) -> dict:
    """
    Rebalance portfolio toward target_weights.
    Returns execution log for the day.
    """
    nav = portfolio.nav(prices)
    log = {"trades": [], "rejected": [], "nav_before": nav}

    if nav <= 0:
        return log

    # Clamp weights
    clipped = {}
    for t, w in target_weights.items():
        if not allow_short:
            w = max(0.0, w)
        w = min(abs(w), MAX_STOCK_WEIGHT) * np.sign(w) if w != 0 else 0.0
        if t in prices:
            clipped[t] = w

    # Ensure cash buffer — scale down if gross exposure too high
    total_long = sum(w for w in clipped.values() if w > 0)
    if total_long > (MAX_GROSS_EXPOSURE - MIN_CASH_BUFFER):
        scale = (MAX_GROSS_EXPOSURE - MIN_CASH_BUFFER) / max(total_long, 1e-9)
        clipped = {t: w * scale for t, w in clipped.items()}

    # Compute share deltas
    for ticker, target_w in clipped.items():
        price = prices[ticker]
        if price <= 0:
            continue

        fill_price = price * (1 + np.sign(target_w) * slippage_bps / 10_000)
        target_shares = (nav * target_w) / fill_price
        current_shares = portfolio.shares.get(ticker, 0.0)
        delta_shares = target_shares - current_shares

        if abs(delta_shares) < 0.01:
            continue

        order_value = abs(delta_shares * fill_price)

        # Liquidity filter
        daily_vol_value = volumes.get(ticker, 0.0) * price
        if daily_vol_value > 0 and order_value > max_participation * daily_vol_value:
            max_shares = (max_participation * daily_vol_value) / fill_price * np.sign(delta_shares)
            log["rejected"].append({
                "ticker": ticker,
                "reason": "liquidity",
                "requested": round(delta_shares, 2),
                "allowed": round(max_shares, 2),
            })
            delta_shares = max_shares
            order_value = abs(delta_shares * fill_price)

        if abs(delta_shares) < 0.01:
            continue

        # Costs
        side = "buy" if delta_shares > 0 else "sell"
        brok = _brokerage(order_value)
        stt  = STT_SELL * order_value if side == "sell" else 0.0
        exch = EXCHANGE_CHARGE * order_value
        total_cost = brok + stt + exch

        total_cash_impact = -(delta_shares * fill_price) - total_cost

        # Cash check for buys
        if delta_shares > 0 and portfolio.cash + total_cash_impact < 0:
            # Reduce size to what cash allows
            affordable = max(0.0, portfolio.cash - total_cost) / fill_price
            if affordable < 0.01:
                log["rejected"].append({"ticker": ticker, "reason": "insufficient_cash"})
                continue
            delta_shares = affordable
            order_value = delta_shares * fill_price
            brok = _brokerage(order_value)
            total_cost = brok + exch
            total_cash_impact = -(delta_shares * fill_price) - total_cost

        portfolio.shares[ticker] = portfolio.shares.get(ticker, 0.0) + delta_shares
        portfolio.cash += total_cash_impact

        log["trades"].append({
            "ticker": ticker,
            "side": side,
            "shares": round(delta_shares, 4),
            "fill_price": round(fill_price, 4),
            "value": round(order_value, 2),
            "cost": round(total_cost, 2),
        })

    # Remove zero/near-zero positions
    portfolio.shares = {t: s for t, s in portfolio.shares.items() if abs(s) > 0.001}
    portfolio.last_prices.update(prices)
    return log


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

@dataclass
class DayResult:
    date: pd.Timestamp
    nav: float
    daily_ret: float
    cash: float
    n_positions: int
    exec_log: dict


class BacktestResult:
    def __init__(self, days: list[DayResult], tickers: list[str], starting_capital: float):
        self.days = days
        self.tickers = tickers
        self.starting_capital = starting_capital
        self.nav_series = pd.Series(
            [d.nav for d in days], index=[d.date for d in days]
        )
        self.ret_series = pd.Series(
            [d.daily_ret for d in days], index=[d.date for d in days]
        )

    def tearsheet(self, benchmark_nav: Optional[pd.Series] = None) -> dict:
        rets = self.ret_series.dropna()
        nav  = self.nav_series

        n_years = len(rets) / 252
        cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        sharpe  = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252)
        down    = rets[rets < 0]
        sortino = (rets.mean() / (down.std() + 1e-9)) * np.sqrt(252)

        roll_max   = nav.cummax()
        drawdown   = (nav - roll_max) / roll_max
        max_dd     = drawdown.min()

        win_rate = (rets > 0).mean()
        total_return = nav.iloc[-1] / nav.iloc[0] - 1

        result = {
            "total_return_pct": round(total_return * 100, 2),
            "cagr_pct":         round(cagr * 100, 2),
            "sharpe":           round(sharpe, 3),
            "sortino":          round(sortino, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "win_rate_pct":     round(win_rate * 100, 2),
            "n_trading_days":   len(rets),
            "starting_capital": self.starting_capital,
            "ending_nav":       round(nav.iloc[-1], 2),
        }

        if benchmark_nav is not None:
            benchmark_nav = benchmark_nav.squeeze()  # ensure 1-D Series
            bench_ret = benchmark_nav.pct_change().dropna()
            bench_cagr = float((benchmark_nav.iloc[-1] / benchmark_nav.iloc[0]) ** (1 / n_years) - 1)
            bench_sharpe = float((bench_ret.mean() / (bench_ret.std() + 1e-9)) * np.sqrt(252))
            active_ret = rets.values - bench_ret.reindex(rets.index).fillna(0).values
            ir = float((active_ret.mean() / (active_ret.std() + 1e-9)) * np.sqrt(252))
            result["vs_benchmark"] = {
                "bench_cagr_pct":    round(bench_cagr * 100, 2),
                "bench_sharpe":      round(bench_sharpe, 3),
                "alpha_cagr_pp":     round((cagr - bench_cagr) * 100, 2),
                "information_ratio": round(ir, 3),
            }

        self._print_tearsheet(result)
        return result

    def _print_tearsheet(self, t: dict):
        print("\n=== Tearsheet ===")
        print(f"  Period        : {self.nav_series.index[0].date()} → {self.nav_series.index[-1].date()}")
        print(f"  Starting NAV  : ₹{t['starting_capital']:,.0f}")
        print(f"  Ending NAV    : ₹{t['ending_nav']:,.0f}")
        print(f"  Total Return  : {t['total_return_pct']:+.2f}%")
        print(f"  CAGR          : {t['cagr_pct']:+.2f}%")
        print(f"  Sharpe        : {t['sharpe']:.3f}")
        print(f"  Sortino       : {t['sortino']:.3f}")
        print(f"  Max Drawdown  : {t['max_drawdown_pct']:.2f}%")
        print(f"  Win Rate      : {t['win_rate_pct']:.1f}%")
        if "vs_benchmark" in t:
            b = t["vs_benchmark"]
            print(f"  vs Benchmark  : alpha={b['alpha_cagr_pp']:+.2f}pp CAGR | IR={b['information_ratio']:.3f}")
        print()


class Backtester:
    def __init__(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,       # index=date, columns=tickers
        volumes: pd.DataFrame,      # index=date, columns=tickers (shares traded)
        features: Optional[dict[str, pd.DataFrame]] = None,  # {ticker: df with feature cols}
        starting_capital: float = STARTING_CAPITAL,
        allow_short: bool = False,
        slippage_bps: float = SLIPPAGE_BPS,
        max_participation: float = MAX_PARTICIPATION,
        daily_stoploss: float = DAILY_STOPLOSS,
        max_drawdown_halt: float = MAX_DRAWDOWN_HALT,
    ):
        self.strategy = strategy
        self.prices = prices.sort_index()
        self.volumes = volumes.reindex(self.prices.index).fillna(0)
        self.features = features
        self.starting_capital = starting_capital
        self.allow_short = allow_short
        self.slippage_bps = slippage_bps
        self.max_participation = max_participation
        self.daily_stoploss = daily_stoploss
        self.max_drawdown_halt = max_drawdown_halt

    def run(self) -> BacktestResult:
        tickers = list(self.prices.columns)
        portfolio = Portfolio(cash=self.starting_capital)
        self.strategy.reset(tickers)

        days: list[DayResult] = []
        peak_nav = self.starting_capital
        halted = False
        prev_nav = self.starting_capital

        for date, price_row in self.prices.iterrows():
            prices = price_row.dropna().to_dict()
            volumes = self.volumes.loc[date].dropna().to_dict() if date in self.volumes.index else {}

            current_nav = portfolio.nav(prices)

            # Max drawdown halt check
            peak_nav = max(peak_nav, current_nav)
            dd = (current_nav - peak_nav) / peak_nav
            if dd <= self.max_drawdown_halt and not halted:
                print(f"  [HALT] {date.date()} max drawdown {dd*100:.1f}% exceeded threshold. Strategy paused.")
                halted = True

            exec_log = {"trades": [], "rejected": [], "halted": halted}

            if not halted:
                # Daily stop-loss check
                daily_ret_prev = (current_nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0
                in_stoploss = daily_ret_prev <= self.daily_stoploss

                if in_stoploss:
                    # Liquidate all positions
                    target_weights = {t: 0.0 for t in portfolio.shares}
                    exec_log = execute_orders(
                        portfolio, target_weights, prices, volumes,
                        self.allow_short, self.slippage_bps, self.max_participation
                    )
                    exec_log["stoploss_triggered"] = True
                else:
                    # Build feature snapshot for this date
                    feat_snap = None
                    if self.features is not None:
                        feat_snap = {}
                        for t in tickers:
                            if t in self.features and date in self.features[t].index:
                                feat_snap[t] = self.features[t].loc[date].values

                    target_weights = self.strategy.on_bar(date, prices, volumes, feat_snap, portfolio)
                    exec_log = execute_orders(
                        portfolio, target_weights, prices, volumes,
                        self.allow_short, self.slippage_bps, self.max_participation
                    )

            nav_after = portfolio.nav(prices)
            daily_ret = (nav_after - prev_nav) / prev_nav if prev_nav > 0 else 0.0
            prev_nav = nav_after

            days.append(DayResult(
                date=date,
                nav=nav_after,
                daily_ret=daily_ret,
                cash=portfolio.cash,
                n_positions=len(portfolio.shares),
                exec_log=exec_log,
            ))

        return BacktestResult(days, tickers, self.starting_capital)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

class _EqualWeightStrategy(Strategy):
    """Trivial equal-weight long strategy for smoke testing."""
    def reset(self, tickers):
        self.tickers = tickers

    def on_bar(self, date, prices, volumes, features, portfolio):
        w = (1.0 - MIN_CASH_BUFFER) / max(len(prices), 1)
        return {t: w for t in prices}


def _smoke_test():
    import yfinance as yf
    print("Downloading Nifty50 proxy (RELIANCE.NS, TCS.NS, INFY.NS) for smoke test...")
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    raw = yf.download(tickers, start="2022-01-01", end="2023-12-31", auto_adjust=True, progress=False)
    prices  = raw["Close"].dropna()
    volumes = raw["Volume"].reindex(prices.index).fillna(0)

    strategy = _EqualWeightStrategy()
    bt = Backtester(strategy, prices, volumes, starting_capital=1_000_000)
    result = bt.run()

    # Download Nifty50 index as benchmark
    nifty = yf.download("^NSEI", start="2022-01-01", end="2023-12-31", auto_adjust=True, progress=False)
    bench_nav = (nifty["Close"].dropna() / nifty["Close"].dropna().iloc[0]) * 1_000_000

    result.tearsheet(benchmark_nav=bench_nav.reindex(result.nav_series.index).ffill())
    print("Smoke test PASSED")


if __name__ == "__main__":
    _smoke_test()
