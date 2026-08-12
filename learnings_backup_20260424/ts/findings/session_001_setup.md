# Session 001 — Setup & Phase 0 (2026-04-19)

## Completed

- Shared spec written: `learnings/ts/SPEC.md`
- Directory structure created: `src/sgnnet_ts/`, `scripts/ts/`, `data/ts/`, `learnings/ts/`
- Data downloaded: 63 tickers (Nifty50 + small cap seeds), 2010–2026, ~4020 rows each (ts_step001)
- Scripts written: ts_step001 (data), ts_step002 (universe), ts_step003 (MA baseline),
  ts_step010 (CNN+LSTM), ts_step011 (CNN+Transformer), ts_step020 (MAE encoder)
- Losses module: `src/sgnnet_ts/losses/financial.py` (log-wealth, Sharpe, MSE, directional accuracy)

## Bug Fixed

**macOS dot-underbar files** (`._TICKER.parquet`): macOS HFS/APFS creates invisible metadata
sidecar files for every file. Python's `Path.glob("*.parquet")` picks these up; pyarrow fails
to parse them. Fix applied to all scripts:
```python
sorted(f for f in RAW_DIR.glob("*.parquet") if not f.stem.startswith("."))
```

## Pending (running)
- ts_step002: correlation clustering → stock_order.json
- ts_step003: MA baseline → feature importance per stock

## Architecture Decisions Locked

| Decision | Choice | Reason |
|---|---|---|
| MAE vs GAN | Test both | User confirmed both |
| Data history | 10y+ (2010–2026) | 6mo = 120 samples, fatal for DL |
| Recency handling | Exponential sample weighting λ=1.0 | Walk-forward + recency weight |
| Walk-forward splits | 2015-2021 train / 2022-2023 val / 2024+ test | Causal; no random split |
| Target | next-day % return = pct_change().shift(-1) | Regression + direction classification |
| P&L loss | log-wealth: -log(1 + tanh(pred)*actual - tc) | Differentiable, tc=0.1% |
| Stock ordering | Hierarchical Ward clustering on (1-|corr|) | Synthetic spatial axis; honest in paper |
| Slot rule | Always run slot_status.sh first; all 6 slots shared | Colleague safety |

## Slots (as of session start)
- studio_mps: FREE (available for training)
- 5060ti_cuda/cpu: grabbed by indra (step909)
- mini_mps/cpu: dhiraj colleague + indra colleague

## Next (Phase 1)
1. Read ts_step002 and ts_step003 results → decide final universe
2. Launch ts_step010 (CNN+LSTM) on studio_mps
3. Launch ts_step011 (CNN+Transformer) on 5060ti when free
4. Begin ts_step020 (MAE) after baselines give directional accuracy floor
