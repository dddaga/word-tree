# Simulation Engine — SPEC

**Prefix:** `sim_stepXXX`
**Location:** `scripts/sim/`, `learnings/sim/`
**Purpose:** Pluggable backtesting engine. Any signal model (SGNNET, LSTM, XGBoost, MA) plugs in via a common Strategy API. Results are directly comparable.

---

## Directory Layout

```
scripts/sim/
  sim_step001_engine.py        # Core order book + P&L engine
  sim_step002_benchmarks.py    # Nifty50 index, buy-and-hold, MA cross baselines
  sim_step003_liquidity.py     # Volume-weighted fill probability filter
  sim_step010_run_ml.py        # Run a ts_step model through the sim
  sim_step020_online_retrain.py # Rolling retrain + selective param update harness
  sim_step030_report.py        # Tearsheet generator (CAGR, Sharpe, max-DD vs index)

learnings/sim/
  SPEC.md                      # This file
  QUEUE.md                     # Experiment queue
  findings/                    # Session findings logs
```

---

## Strategy Interface

Every strategy is a class implementing:

```python
class Strategy:
    def reset(self, tickers: list[str]) -> None:
        """Called once before backtest begins."""

    def on_bar(
        self,
        date: str,
        prices: dict[str, float],       # {ticker: close}
        volumes: dict[str, float],      # {ticker: shares_traded}
        features: dict[str, np.ndarray],# {ticker: feature_vector}
        portfolio: "Portfolio",
    ) -> dict[str, float]:
        """Return {ticker: target_weight} in [-1.0, +1.0]. Sum need not be 1."""
```

Strategies are stateless between `on_bar` calls except via `self`. The engine calls `on_bar` once per trading day in chronological order.

---

## Execution Model

### Market hours
- NSE: 09:15–15:30 IST. Sim uses **closing price** for all fills.
- No intraday. All signals are end-of-day → execute at next day open (conservative) or same-day close (optimistic). Default: **same-day close**.

### Slippage model
```
fill_price = close * (1 + side * slippage_bps / 10000)
```
Default `slippage_bps = 5` (0.05%). Configurable per run.

### Brokerage
| Leg | Cost |
|---|---|
| Brokerage | ₹20 flat per order (Zerodha model) OR 0.03% whichever lower |
| STT (equity delivery) | 0.1% on sell side |
| Exchange + SEBI charges | ~0.005% |
| GST | 18% on brokerage |
| Total per round-trip (approx) | ~0.25–0.35% |

Default `transaction_cost_bps = 30` (0.30% round-trip). Configurable.

### Liquidity filter
Reject (or partial-fill) any order where:
```
order_value > max_participation_rate * daily_volume * close
```
Default `max_participation_rate = 0.01` (1% of daily volume). Unfilled portion is cancelled.

---

## Risk / Guard-Rails

| Rule | Default |
|---|---|
| Max single-stock weight | 20% of portfolio |
| Max gross exposure | 100% (no leverage) |
| Daily stop-loss | −2% portfolio NAV → exit all positions, sit in cash until next day |
| Max drawdown halt | −15% from peak → pause strategy, alert, require manual resume |
| Min cash buffer | 5% always in cash (liquidity reserve) |

---

## Portfolio Accounting

```
NAV(t) = cash(t) + sum(shares(i,t) * price(i,t))
daily_return(t) = NAV(t) / NAV(t-1) - 1
```

All P&L in INR. Starting capital: ₹10,00,000 (10 lakh). Configurable.

---

## Benchmarks (sim_step002)

| Benchmark | Description |
|---|---|
| **Nifty50 index** | Buy ^NSEI at start, hold. No rebalance. No transaction cost. |
| **Buy-and-hold universe** | Equal-weight all 56 tickers, rebalance monthly. |
| **MA crossover** | 21d/55d golden cross per stock; equal position sizing. |
| **Random signal** | Random ±1 direction per stock per day (Monte Carlo × 100 seeds). |

Any strategy that doesn't beat Nifty50 buy-and-hold (after costs) is not worth deploying.

---

## Evaluation Metrics (Tearsheet)

| Metric | Definition |
|---|---|
| CAGR | Compound annual growth rate over backtest period |
| Sharpe | annualized(mean_daily_ret / std_daily_ret) × √252 |
| Sortino | annualized(mean_daily_ret / std_downside_ret) × √252 |
| Max Drawdown | max(peak - trough) / peak over full period |
| Win rate | % of days with positive P&L |
| Avg trade holding | mean days between entry and exit per position |
| Turnover | mean daily portfolio turnover (%) |
| vs Nifty50 | CAGR delta, Sharpe delta, information ratio |

---

## Walk-Forward Splits (mirrors ts/ splits)

| Split | Dates | Purpose |
|---|---|---|
| Train | 2015-01-01 → 2021-12-31 | Strategy trained here |
| Validation | 2022-01-01 → 2023-12-31 | Hyperparameter selection |
| Test (held out) | 2024-01-01 → present | Final reported result — touch only once |

**Never optimize on test split.**

---

## Online Learning Harness (sim_step020)

For SGNNET-TS and other differentiable models:
- **Rolling retrain**: every N_retrain=21 days, fine-tune on the last W_window=252 days.
- **Selective update**: only update parameters with gradient magnitude > threshold (HYPOTHESIS — need ablation).
- **Regime detector**: if rolling 21d correlation of strategy vs market > 0.9, flag as "beta exposure" — likely not alpha.

---

## Experiment Queue

See `learnings/sim/QUEUE.md`.

---

## Key Rule

**ALWAYS run `bash scripts/slot_status.sh` before launching any training job.**
Sim engine scripts (data loading, backtesting, reporting) can run freely — they are CPU-only and short-lived.
