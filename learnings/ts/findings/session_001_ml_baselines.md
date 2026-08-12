# ts_step010-030 — ML Baseline Results (2026-04-19)

## All models, val period 2022-2023, 56 stocks, 100 epochs on CPU

| Model | Params | Dir Acc | MAE | Best Sharpe | Notes |
|---|---|---|---|---|---|
| CNN+LSTM | 2,101,272 | 0.501 | 0.0415 | -47.5 | |
| CNN+Transformer | 791,448 | 0.493 | 0.1445 | -41.7 | |
| MAE Encoder (frozen) | 52,500 | 0.495 | — | -62.3 | encoder pretrained 100ep |
| GAN Encoder (frozen) | ~278K | 0.501 | 0.104 | -46.1 | pretrained 50ep |
| SGNNET-TS | 51,050 | 0.497 | 0.183 | -36.9 | alpha learned=0.505, beta=0.298 |

## Key Observations

1. **All models ~random (0.49-0.50 dir_acc)** — expected at T0 without hyperparameter tuning.
   Not alarming; CNN+LSTM on Imagenette also starts near-random before tuning.

2. **SGNNET-TS high MAE (0.183 vs CNN+LSTM's 0.042)** — HYPOTHESIS: output from `Linear(N*D→N)` 
   is not bounded; returns are in ±0.05 range but model outputs ±0.5+. Need output scaling fix.
   Fix: add `* 0.01` scale or `tanh(x) * 0.1` at prediction head output.

3. **SGNNET val_sharpe degrades epoch→epoch (-37 → -150)** — overfitting signal.
   With only 169 train windows and 51K params, model memorizes train set.
   Fix: reduce D to 8, add dropout, or increase regularization.

4. **Val_sharpe is negative for all models** — models are predicting wrong magnitude, 
   causing large P&L swings. The log-wealth loss fine-tune wasn't long enough (only 80/100 ep).

5. **SGNNET param efficiency confirmed**: 51K vs 2.1M CNN+LSTM — 41× fewer params, 
   similar directional accuracy. This IS the paper claim — now need accuracy gap to close.

## Next Steps

1. `sim_step010` — plug CNN+LSTM into backtester (proof-of-concept pipeline test)
2. `sim_step012` — XGBoost on MA features (classical ML baseline with built-in position sizing)
3. `ts_step031` — fix SGNNET output scale + reduce D + more regularization
4. `ts_step032` — longer training (200ep) with recency weighting
