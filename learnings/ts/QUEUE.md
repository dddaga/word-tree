# SGNNET-TS Experiment Queue

## Phase 0 — Data + Universe (now)

| Step | Script | Device | Status | Goal |
|------|--------|--------|--------|------|
| ts_step001 | ts_step001_data_download.py | 5060ti_cpu | PENDING | 10y OHLCV for Nifty50 + screened small caps |
| ts_step002 | ts_step002_universe_selection.py | 5060ti_cpu | PENDING | Volatility sort + correlation cluster + stock_order.json |
| ts_step003 | ts_step003_ma_baseline.py | 5060ti_cpu | PENDING | MA windows [2..200], linear scorer, feature importance per stock |

## Phase 1 — DL Baselines

| Step | Script | Device | Status | Goal |
|------|--------|--------|--------|------|
| ts_step010 | ts_step010_cnn_lstm.py | studio_mps | PENDING | CNN+LSTM, walk-forward, Sharpe+log-wealth |
| ts_step011 | ts_step011_cnn_transformer.py | studio_mps | PENDING | CNN+Transformer, matched params |

## Phase 2 — Context Encoders

| Step | Script | Device | Status | Goal |
|------|--------|--------|--------|------|
| ts_step020 | ts_step020_mae_encoder.py | studio_mps | PENDING | MAE-style masked reconstruction pretraining |
| ts_step021 | ts_step021_gan_encoder.py | studio_mps | PENDING | GAN context encoder (Pathak-style), compare vs MAE |

## Phase 3 — SGNNET-TS

| Step | Script | Device | Status | Goal |
|------|--------|--------|--------|------|
| ts_step030 | ts_step030_sgnnet_ts_t0.py | 5060ti_cuda | PENDING | α=0 sanity (must recover stateless baseline) |
| ts_step031 | ts_step031_sgnnet_ts_alpha_scan.py | 5060ti_cuda | PENDING | α ∈ {0.1, 0.3, 0.5, 0.7, 0.9} sweep |
| ts_step032 | ts_step032_sgnnet_ts_t1.py | 5060ti_cuda | PENDING | Best α from step031, full walk-forward |

## Phase 4 — Liquid NN

| Step | Script | Device | Status | Goal |
|------|--------|--------|--------|------|
| ts_step050 | ts_step050_ltc_baseline.py | 5060ti_cuda | PENDING | LTC ODE, matched params, same splits |

## Phase 5 — Ablations / Paper

| Step | Script | Device | Status | Goal |
|------|--------|--------|--------|------|
| ts_step060 | ts_step060_encoder_ablation.py | TBD | PENDING | Frozen encoder vs fine-tuned encoder downstream |
| ts_step061 | ts_step061_log_wealth_vs_mse.py | TBD | PENDING | Confirm log-wealth fine-tuning improves Sharpe vs MSE-only |
