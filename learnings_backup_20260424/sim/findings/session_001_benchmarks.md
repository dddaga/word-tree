# sim_step002 — Benchmark Results (2026-04-19)

## Val Period: 2022-01-03 → 2023-12-29 (63 tickers, 493 days)

| Strategy | CAGR | Sharpe | Sortino | Max DD | Win Rate |
|---|---|---|---|---|---|
| MA Crossover 21d/55d | +13.59% | 1.112 | 1.438 | -17.4% | 49.1% |
| Buy-and-Hold equal-weight | +11.58% | 0.908 | 1.153 | -21.0% | 55.4% |
| Random signal (median 100 seeds) | +7.80% | 0.635 | 0.800 | -23.6% | 57.4% |
| Nifty50 proxy (equal-weight day-1) | -2.35% | -0.717 | -0.124 | -7.5% | 1.6% |

Note: NiftyIndexStrategy underperforms — local universe ≠ cap-weighted Nifty50 basket. Use actual ^NSEI download as reference.

## Bar to Clear

Any ML strategy must beat **Sharpe > 1.112 and CAGR > 13.59%** (MA crossover) after transaction costs to claim signal edge.

## Observations

- Both B&H and MA cross halted at −15% drawdown (April 2022, Ukraine shock), then recovered — guard-rail working correctly.
- MA cross leads on all risk-adjusted metrics despite lower win rate (49.1%) — momentum filtering cuts bad trades.
- Random median (null hypothesis) is +7.80% — 2022-2023 was a bull run for Indian equities; any long-biased strategy drifts positive.

## Implications for ML Models

1. Directional accuracy > 0.52 is necessary but NOT sufficient — need to beat MA cross Sharpe.
2. Transaction costs matter: MA cross turns over ~once per signal (low), ML models may turn over daily (high). Need to account for this.
3. Position sizing matters: MA cross uses equal-weight among longs. Full-Kelly or confidence-weighted sizing could lift Sharpe further.
