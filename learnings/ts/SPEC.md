# SGNNET-TS Shared Specification
# All agents and scripts must read this before writing code.
# Last updated: 2026-04-19

---

## 1. Goal

Predict next-day % return per stock. Use predictions as entry/exit signals for
medium-term positions (hold 2 weeks – 2 months). Secondary objective: maximise
total available capital (outside-market capital grows monotonically).

---

## 2. Directory Layout

```
neuro_graph/
  data/ts/
    raw/           # yfinance OHLCV parquet files, one per ticker
    processed/     # feature tensors, HDF5 keyed by split
  src/sgnnet_ts/
    __init__.py
    data/
      universe.py      # stock selection + correlation ordering
      pipeline.py      # fetch, cache, update (cron-safe)
      features.py      # all feature engineering (MA, spread_z, resid_z …)
      augmentation.py  # sliding window + masking utilities
    models/
      context_encoder_mae.py   # MAE-style reconstruction
      context_encoder_gan.py   # Pathak-style GAN reconstruction
      baseline_lstm.py         # CNN + LSTM
      baseline_transformer.py  # CNN + Transformer
      sgnnet_ts.py             # SGNNET with retention (Phase 3)
    losses/
      financial.py   # log_wealth, sharpe_loss, directional_accuracy
    eval/
      protocol.py    # walk-forward splits, metric aggregation
  scripts/ts/
    ts_step001_data_download.py
    ts_step002_universe_selection.py
    ts_step003_ma_baseline.py
    ts_step010_cnn_lstm.py
    ts_step011_cnn_transformer.py
    ts_step020_mae_encoder.py
    ts_step021_gan_encoder.py
    ts_step030_sgnnet_ts.py
  learnings/ts/
    SPEC.md           # this file
    QUEUE.md          # experiment queue
    concepts/         # design docs
    findings/         # per-step results
```

---

## 3. Stock Universe

### Primary: Nifty 50 (.NS tickers)

RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, HINDUNILVR, BAJFINANCE,
SBIN, BHARTIARTL, KOTAKBANK, LT, AXISBANK, ASIANPAINT, MARUTI,
TITAN, SUNPHARMA, ULTRACEMCO, WIPRO, ADANIENT, ONGC, POWERGRID,
NTPC, JSWSTEEL, TECHM, HCLTECH, BAJAJFINSV, M&M, TATAMOTORS,
TATASTEEL, INDUSINDBK, NESTLEIND, BRITANNIA, CIPLA, DRREDDY,
DIVISLAB, BPCL, COALINDIA, HINDALCO, GRASIM, APOLLOHOSP,
EICHERMOT, BAJAJ-AUTO, HEROMOTOCO, ADANIPORTS, TATACONSUM,
LTIM, SBILIFE, HDFCLIFE, UPL, VEDL

### Secondary: Small caps (screened)

Filters (applied via screener.in or yfinance fundamentals):
- ROE > 20% (trailing 12m)
- Annualised daily-return std > 0.30 (high volatility = opportunity)
- Market cap < 20,000 Cr
- Liquid: avg daily volume > 1L shares

Script: ts_step002 outputs `data/ts/universe_screened.csv`

### Correlation Ordering

Method: hierarchical clustering on `1 - |rolling_60d_corr|` distance matrix,
using Ward linkage + scipy `optimal_leaf_ordering`. Puts correlated stocks
adjacent in the 2D matrix — the synthetic spatial axis reviewers will ask about
(be honest in paper: this is a learned/sorted axis, not natural 2D geometry).

Output: `data/ts/stock_order.json` — canonical ordering used by ALL scripts.

---

## 4. Data Schema

### Raw (data/ts/raw/<TICKER>.parquet)
Columns: date, open, high, low, close, volume, adj_close
Frequency: daily (business days only)
History: as far back as yfinance provides (target ≥ 10 years, ~2500 rows)

### Processed features per stock (features.py)
All features are computed with shift(1) before any rolling window — NO lookahead.

| Feature | Formula | Window |
|---------|---------|--------|
| ret_1d | (close_t / close_{t-1}) - 1 | — |
| ma_ratio_{w} | close / MA(close, w) | w ∈ [2,3,5,8,13,21,34,55,89,144,200] |
| ma_cross_{f}_{s} | MA(f) / MA(s) - 1 | (f,s) ∈ [(5,21),(13,55),(21,89)] |
| vol_20 | rolling std of ret_1d | 20d |
| vol_60 | rolling std of ret_1d | 60d |
| spread_z | cointegration spread z-score vs sector peer | 60d OLS |
| resid_z | market-neutral residual z-score vs sector ETF | 60d OLS |
| delta_corr | corr_20d - corr_60d (mean over all peers) | — |
| lag1_ret | ret_1d.shift(1) | — |

### Target
target_ret = (close_{t+1} / close_t) - 1   # next-day % return
target_dir = sign(target_ret)               # direction (-1, 0, +1)

Compute AFTER all features — never use close_{t+1} in any feature.

### 2D Input Tensor
Shape: [B, T_lookback, N_stocks, F_features]
- B = batch size (32)
- T_lookback = 60 (trading days ≈ 3 months)
- N_stocks = size of ordered universe (target ~30-50)
- F_features = 16 (all features above)

Stock axis is ordered by correlation clustering (stock_order.json).
Time axis is causal: index 0 = oldest, index T-1 = most recent (t).

---

## 5. Data Augmentation Strategy

Problem: DL needs many samples; market regimes change — distant past may mislead.

Solutions (applied in augmentation.py):

### 5a. Sliding windows
Stride = 1 day. From D total trading days, generates (D - T_lookback - 1) samples.
From 10 years = 2500 days: ~2440 samples per dataset. Multiplied by N_stocks for
stock-specific targets. Dense enough for DL.

### 5b. Recency weighting
Exponential decay on sample weights: w_t = exp(λ · (t - t_max) / D)
λ = 1.0 by default (half-weight 5 years ago). Tunable hyperparameter.
Applied as sample_weight in loss, not by discarding old data.

### 5c. MAE / masked reconstruction augmentation
Each training step randomly samples a different mask → different training example
from the same window. Effective multiplier: ~5-10× with diverse masking patterns.

### 5d. Walk-forward validation (causal splits)
NEVER random split. Always time-ordered:
- Warmup: 2010-2014 (used for feature initialisation, not training)
- Train: 2015-2021 (7 years)
- Val: 2022-2023 (2 years)
- Test: 2024-2025-04 (held out; only for final paper numbers)

Walk-forward: re-train on expanding window every 6 months for production signals.

---

## 6. Loss Functions (losses/financial.py)

### Primary: log-wealth loss (differentiable P&L)
position_t = tanh(pred_t)           # maps prediction to [-1, +1]
log_wealth = sum(log(1 + position_t * actual_ret_t - tc))
L_wealth = -mean(log_wealth)        # minimise negative log-wealth
tc = 0.001                          # 0.1% round-trip transaction cost

### Secondary: MSE on returns (pretraining stability)
L_mse = MSE(pred_ret, actual_ret)

### Training protocol
Pretrain with L_mse for 20 epochs, then fine-tune with L_wealth + 0.1*L_mse.
Gradient clip: max_norm=1.0 (required for log-wealth stability).

### Evaluation metrics (all reported, per CLAUDE.md Pareto rule)
1. Directional accuracy (% correct sign prediction)
2. Sharpe ratio (annualised, on backtest P&L)
3. Max drawdown
4. MAE on % return
5. Params count
6. FLOPs per forward pass
7. Wall-time per step (B=32)

---

## 7. Models

### Phase 0-1: Baselines (no DL)
MA filter baseline:
- Windows: [2,3,5,8,13,21,34,55,89,144,200] (Fibonacci + standard)
- Feature: ma_ratio + ma_cross
- Scorer: LinearRegression (sklearn) + 1-layer MLP (PyTorch)
- Output: feature importance per stock → which horizon is most informative

### Phase 1: DL baselines
CNN feature extractor:
- Input: [B, F_features, T_lookback, N_stocks] (treat as 2D image)
- Conv2d layers: 3 × (Conv2d → BN → ReLU), kernel=(3,3), channels 16→32→64
- Output: [B, 64, T', N'] flattened to [B, T', 64*N'] → sequence for LSTM/Transformer

LSTM head: 2-layer LSTM, hidden=128, dropout=0.1
Transformer head: 4-head, 2-layer, d_model=128, feedforward=256

### Phase 2: Context Encoder (MAE-style, ts_step020)
Masking:
- Future: completely masked (indices T_known:T_lookback)
- Past/present: random 25% block mask
- Mask token: learnable vector (not zero — zero creates spurious signal)

Encoder: CNN (same as DL baseline above) → channel-wise FC (per Pathak: FC across
all spatial positions per channel, enables global information flow)
Decoder: transposed conv, mirror of encoder, outputs full [T, N, F] reconstruction

Loss: L_mae = MSE(pred[masked_positions], true[masked_positions])
Downstream: freeze encoder, train linear head on encoded features for return prediction.

### Phase 2b: GAN Context Encoder (ts_step021)
Same as MAE but add discriminator:
- Discriminator: 3-layer CNN on predicted vs real patches
- Loss: L_total = 0.999*L_mae + 0.001*L_adv
- Discriminator conditioned on generator output only (per Pathak sec 3.2)
Compare MAE vs GAN encoder quality on downstream task.

### Phase 3: SGNNET-TS with retention (ts_step030+)
Z_t = α · Z_{t-1} + (1-α) · route(seed(X_t) + β · Z_{t-1})
- α, β: learnable scalars, initialised α=0.5, β=0.1
- route(): SGNNET SmallWorld routing, K_iter=3 (reduced for efficiency)
- seed(): maps [N_stocks, F_features] → [N_hidden, D] (D=16)
- N_hidden=512 (smaller than vision model — sequence data is lower-dim)
- Import encoding.py, norm_masked.py from ../sgnnet/ (never edit those files)
- Gate-death mitigation: additive feedback (β*Z_{t-1} added to seed, not gated)

### Phase 4: Liquid NN comparison
Implement LTC: dx/dt = -x/τ(t) + f(x, I, θ) via fixed-step Euler (dt=1 trading day)
Match parameter budget to SGNNET-TS. Compare on same walk-forward splits.

---

## 8. Experiment Numbering

ts_step001 – ts_step009: data + universe
ts_step010 – ts_step019: DL baselines (LSTM, Transformer)
ts_step020 – ts_step029: context encoders
ts_step030 – ts_step049: SGNNET-TS retention variants
ts_step050 – ts_step059: Liquid NN comparison
ts_step060+: compound / ablation studies

---

## 9. Compute Assignment

Available slots (as of 2026-04-19):
- studio_mps: FREE → use for MAE/GAN training (MPS-compatible)
- 5060ti_cuda: FREE → primary for SGNNET-TS (CUDA, fastest)
- 5060ti_cpu: FREE → data prep + MA baseline (CPU-only fine)

Launch via existing scripts/launch_slot.sh. Session prefix: ts-dhiraj-<slot>-<step>.
ALWAYS run `bash scripts/slot_status.sh` first — colleagues share all slots.
Do NOT touch: mini_mps, mini_cpu (colleagues), studio_cpu (any indra session).
A slot is claimable only if slot_status.sh shows [FREE] and no NON-SGN session is on that machine.

---

## 10. Paper Framing

Title candidate: "SGNNET for Sequence Modelling: O(1) State Retention for
Efficient Auto-regressive Learning in Financial Time Series"

Contributions:
1. Adaptation of SGNNET to sequential data via learnable retention (Z_t mechanism)
2. Context-encoder pretraining on 2D stock×time matrix (MAE + GAN variants)
3. Correlation-clustered input representation (synthetic spatial axis)
4. First comparison of SGNNET retention vs LSTM/Transformer/LTC on NSE equity data
5. Differentiable log-wealth objective for end-to-end portfolio optimisation

Honest negatives to pre-register:
- Retention on static inputs hurts (step306) — only meaningful for sequences
- SGNNET trails Linear on text (step407/410) — scope is vision/time-series
- 6-month data is insufficient; results use 10y with walk-forward splits
