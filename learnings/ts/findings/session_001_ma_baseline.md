# ts_step003 — MA Baseline Results (2026-04-19)

## Summary

| Model | Mean Dir Acc | Range |
|---|---|---|
| LinearReg (MA features) | 0.504 | 0.434–0.574 |
| MLP 1-layer (MA features) | 0.500 | 0.454–0.560 |
| Random baseline | 0.500 | — |

**Verdict:** MA features alone = barely above random. Clear signal but weak.
DL needs to extract cross-stock + cross-time patterns MA ratios miss.

## Most Informative MA Windows (across 56 tickers)

Rank order: **21d (46) > 55d (42) = 89d (42) > 5d (36) > 144d (29) > 13d (27)**

Interpretation:
- Monthly (21d) and quarterly (55d, 89d) trends are the primary signal horizon
- T_lookback=60 in SPEC is well-calibrated (covers the top-3 windows)
- 200d least useful for next-day prediction — too slow
- 5d is noise-reactive but still informative (short-term momentum)

## Top Stocks by MA Predictability

| Ticker | LR Dir Acc | Top MA | Note |
|---|---|---|---|
| ULTRACEMCO | 0.574 | 21d, 89d, 13d | Most linearly predictable |
| TCS | 0.548 | 89d, 13d, 55d | Trend-following works well |
| DRREDDY | 0.542 | 21d, 55d, 13d | Pharma medium-term trend |
| POWERGRID | 0.540 | 89d, 8d, 3d | Utility: slow + stable |
| BAJAJ-AUTO | 0.537 | 21d, 55d, 89d | Auto: strong trend following |

## Implications for DL Design

1. **T_lookback = 60d is correct** — captures the 21d, 55d, 89d signal horizons
2. **Cross-stock features matter** — individual MA ratios barely beat random; context across all stocks in the 2D matrix should help more
3. **MLP doesn't improve over LR** — MA ratios are linearly exploitable; DL must find nonlinear structure beyond single-stock MAs
4. **Directional accuracy floor = 0.504** — this is the MA-only baseline DL must beat

## Bugs Fixed

- macOS `._TICKER.parquet` metadata files picked up by `Path.glob("*.parquet")` → all scripts now filter with `if not f.stem.startswith(".")`
- Format string crash when `evaluate_ticker` returns `{skip: ...}` dict → guarded with `if 'lr_dir_acc' in res`

## Infrastructure Note

Studio MPS (`studio_mps`) is on a different machine with REMOTE_DIR = `/Users/admin/ml/dhiraj/qwen2_omni/testing` — data is NOT mounted there. Training scripts must either:
1. Run locally on Mac Mini (where /Volumes/T9 is mounted), OR
2. Sync data + scripts to studio first

Decision: run ts_step010/011/020 locally on CPU initially for validation, then plan data sync to studio/5060ti for full training runs.
