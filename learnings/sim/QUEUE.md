# Sim Engine — Experiment Queue

| Step | Script | Status | Description |
|---|---|---|---|
| sim_step001 | sim_step001_engine.py | PENDING | Core engine: order book, slippage, brokerage, risk guard-rails, tearsheet |
| sim_step002 | sim_step002_benchmarks.py | PENDING | Nifty50 index + buy-and-hold + MA crossover + random baselines |
| sim_step003 | sim_step003_liquidity.py | PENDING | Volume-weighted fill probability, partial fill logic |
| sim_step010 | sim_step010_run_ml.py | PENDING | Plug ts_step010 CNN+LSTM into sim engine, run full backtest |
| sim_step011 | sim_step011_run_transformer.py | PENDING | CNN+Transformer backtest |
| sim_step012 | sim_step012_run_xgboost.py | PENDING | XGBoost on MA features backtest (classical ML baseline) |
| sim_step020 | sim_step020_online_retrain.py | PENDING | Rolling retrain harness + selective param update |
| sim_step030 | sim_step030_run_sgnnet.py | PENDING | SGNNET-TS backtest — key paper result |
| sim_step040 | sim_step040_report.py | PENDING | Full tearsheet generator, comparison table vs all baselines |
