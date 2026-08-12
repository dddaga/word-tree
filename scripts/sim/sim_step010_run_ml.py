"""sim_step010_run_ml.py — Plug trained ML models into the backtesting engine.

Models: CNN_LSTM (ts_step010), CNN_Transformer (ts_step011), SGNNETRecurrent (ts_step030).
Val period: 2022-01-01 → 2023-12-31.
Benchmark: ^NSEI (Nifty50).

Each model:
  1. Rebuilt from config stored in results/ts/*.json
  2. Weights loaded from data/ts/checkpoints/ts_stepXXX_best.pt (random if absent)
  3. Wrapped in MLStrategy — rolling [T, N, F] window, tanh(pred)*max_position weights
  4. Run through Backtester; tearsheet printed and saved

Feature computation:
  Precomputed for full val period via _compute_features from ts_step010.
  Strategy looks up current date in precomputed dict at each bar.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.sim.sim_step001_engine import (  # noqa: E402
    Backtester,
    Portfolio,
    Strategy,
    STARTING_CAPITAL,
)
from scripts.ts.ts_step010_cnn_lstm import (  # noqa: E402
    CNN_LSTM,
    _compute_features,
    FEATURE_COLS,
    T_LOOKBACK,
)
from scripts.ts.ts_step011_cnn_transformer import CNN_Transformer  # noqa: E402
from scripts.ts.ts_step030_sgnnet_ts import SGNNETRecurrent       # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

VAL_START = "2022-01-01"
VAL_END   = "2023-12-31"
MIN_ROWS  = 200

DATA_DIR   = ROOT / "data"  / "ts" / "raw"
CKPT_DIR   = ROOT / "data"  / "ts" / "checkpoints"
TS_RES_DIR = ROOT / "results" / "ts"
OUT_DIR    = ROOT / "results" / "sim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_POSITION = 0.15   # max_weight per stock = tanh(pred) * MAX_POSITION


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS_TO_RUN = [
    (
        "CNN_LSTM",
        CNN_LSTM,
        "ts_step010_best.pt",
        "ts_step010_cnn_lstm.json",
        lambda cfg: {"N_stocks": cfg["n_stocks"]},
    ),
    (
        "CNN_Transformer",
        CNN_Transformer,
        "ts_step011_best.pt",
        "ts_step011_cnn_transformer.json",
        lambda cfg: {
            "N_stocks": cfg["n_stocks"],
            "d_model":  cfg.get("d_model", 128),
            "nhead":    cfg.get("nhead", 4),
            "num_layers":      cfg.get("num_layers", 2),
            "dim_feedforward": cfg.get("dim_feedforward", 256),
        },
    ),
    (
        "SGNNETRecurrent",
        SGNNETRecurrent,
        "ts_step030_best.pt",
        "ts_step030_sgnnet_ts.json",
        lambda cfg: {
            "N": cfg["n_stocks"],
            "D": cfg.get("D", 16),
            "K_wiring": cfg.get("K_wiring", 4),
            "K_iter":   cfg.get("K_iter", 3),
            "dropout":  cfg.get("dropout", 0.1),
        },
    ),
]


# ---------------------------------------------------------------------------
# Feature precomputation helpers
# ---------------------------------------------------------------------------

def precompute_features(val_start: str, val_end: str) -> tuple[
    list[str],              # sorted tickers (universe)
    pd.DataFrame,           # prices [date × ticker]
    pd.DataFrame,           # volumes [date × ticker]
    dict[str, pd.DataFrame],  # feat_by_ticker: ticker → DataFrame[FEATURE_COLS]
]:
    """Load raw parquet files, compute features for the val period, return aligned universe."""
    parquets = sorted(f for f in DATA_DIR.glob("*.parquet") if not f.stem.startswith("."))
    if not parquets:
        raise FileNotFoundError(f"No parquet files in {DATA_DIR}")

    dfs_close: dict[str, pd.Series] = {}
    dfs_vol:   dict[str, pd.Series] = {}
    feat_by_ticker: dict[str, pd.DataFrame] = {}

    for pq in parquets:
        ticker = pq.stem
        df = pd.read_parquet(pq)
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_index()

        # Features need history before val_start for warm-up windows (rolling 89).
        # Load full series, then slice to val period for backtesting.
        feats = _compute_features(df)
        feats = feats.dropna()

        # Val slice
        mask = (df.index >= val_start) & (df.index <= val_end)
        sub = df.loc[mask]
        feat_sub = feats.loc[feats.index.isin(sub.index)]

        if len(feat_sub) >= MIN_ROWS:
            dfs_close[ticker] = sub["close"]
            dfs_vol[ticker]   = sub.get("volume", pd.Series(0, index=sub.index))
            feat_by_ticker[ticker] = feat_sub[FEATURE_COLS].astype(np.float32)

    prices  = pd.DataFrame(dfs_close).sort_index()
    volumes = pd.DataFrame(dfs_vol).sort_index()

    tickers = sorted(feat_by_ticker.keys())
    log.info("Universe: %d tickers, %d val trading days", len(tickers), len(prices))
    return tickers, prices, volumes, feat_by_ticker


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class MLStrategy(Strategy):
    """
    Wraps any ML model (CNN_LSTM / CNN_Transformer / SGNNETRecurrent) as a Strategy.

    Rolling buffer of T_LOOKBACK feature rows per ticker.
    On each bar:
      - look up precomputed features for current date
      - append to per-ticker deque
      - if warmup incomplete: return equal-weight
      - else: stack [T, N, F] → forward pass → tanh(pred) * MAX_POSITION weights
    """

    def __init__(
        self,
        name: str,
        model_cls,
        model_kwargs: dict,
        checkpoint_path: Optional[Path],
        tickers: list[str],
        feat_by_ticker: dict[str, pd.DataFrame],
        T: int = T_LOOKBACK,
        max_position: float = MAX_POSITION,
        device: str = "cpu",
    ):
        self.name = name
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.checkpoint_path = checkpoint_path
        self._tickers = tickers          # sorted, canonical order
        self.feat_by_ticker = feat_by_ticker
        self.T = T
        self.max_position = max_position
        self.device_str = device

        self.model: Optional[nn.Module] = None
        self._buf: Optional[dict[str, deque]] = None   # per-ticker rolling buffer
        self._N: int = 0

    # -----------------------------------------------------------------
    def reset(self, tickers: list[str]) -> None:
        # Canonical order = sorted intersection of requested tickers and our universe
        avail = set(self.feat_by_ticker.keys())
        self._active = sorted(t for t in tickers if t in avail)
        self._N = len(self._active)

        # Build model
        device = torch.device(self.device_str)
        self.model = self.model_cls(**self.model_kwargs).to(device)
        self.model.eval()

        # Load checkpoint if present
        if self.checkpoint_path and self.checkpoint_path.exists():
            state = torch.load(self.checkpoint_path, map_location=device)
            self.model.load_state_dict(state)
            log.info("[%s] Loaded checkpoint %s", self.name, self.checkpoint_path.name)
        else:
            log.warning(
                "[%s] No checkpoint at %s — using random weights (pipeline test)",
                self.name,
                self.checkpoint_path,
            )

        # Rolling buffers: deque of length T, each entry [N, F]
        self._buf = deque(maxlen=self.T)
        self._device = device

    # -----------------------------------------------------------------
    def on_bar(
        self,
        date: pd.Timestamp,
        prices: dict[str, float],
        volumes: dict[str, float],
        features: Optional[dict[str, np.ndarray]],   # ignored — we use precomputed
        portfolio: Portfolio,
    ) -> dict[str, float]:

        F = len(FEATURE_COLS)

        # Build feature row for this date: [N, F]
        row = np.zeros((self._N, F), dtype=np.float32)
        for i, ticker in enumerate(self._active):
            df_t = self.feat_by_ticker.get(ticker)
            if df_t is not None and date in df_t.index:
                vals = df_t.loc[date].values
                row[i] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

        self._buf.append(row)

        # Warmup: return equal-weight
        if len(self._buf) < self.T:
            w = (1.0 - 0.05) / max(self._N, 1)
            return {t: w for t in self._active if t in prices}

        # Stack buffer → [T, N, F] → [1, T, N, F]
        x = np.stack(list(self._buf), axis=0)            # [T, N, F]
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self._device)  # [1, T, N, F]

        with torch.no_grad():
            pred = self.model(x_tensor)   # [1, N]
        pred_np = pred.squeeze(0).cpu().numpy()   # [N]

        # Map predictions → weights via tanh * max_position
        weights = {}
        for i, ticker in enumerate(self._active):
            if ticker in prices:
                weights[ticker] = float(np.tanh(pred_np[i]) * self.max_position)

        return weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import yfinance as yf

    log.info("Precomputing features for val period %s → %s ...", VAL_START, VAL_END)
    tickers, prices, volumes, feat_by_ticker = precompute_features(VAL_START, VAL_END)

    # Benchmark: ^NSEI
    log.info("Downloading ^NSEI benchmark ...")
    nifty_raw = yf.download("^NSEI", start=VAL_START, end=VAL_END, auto_adjust=True, progress=False)
    nifty_close = nifty_raw["Close"].squeeze().dropna()
    nifty_nav = (nifty_close / nifty_close.iloc[0]) * STARTING_CAPITAL
    nifty_nav = nifty_nav.reindex(prices.index).ffill()

    all_results: dict[str, dict] = {}

    for name, model_cls, ckpt_fname, result_json, kwargs_fn in MODELS_TO_RUN:
        print(f"\n{'='*60}")
        print(f"  Model: {name}")
        print(f"{'='*60}")

        # Load config from result JSON
        res_path = TS_RES_DIR / result_json
        if res_path.exists():
            with open(res_path) as f:
                cfg = json.load(f)
            log.info("[%s] Config loaded: n_stocks=%s", name, cfg.get("n_stocks"))
        else:
            log.warning("[%s] Result JSON not found: %s — using defaults", name, res_path)
            cfg = {"n_stocks": len(tickers)}

        model_kwargs = kwargs_fn(cfg)

        # N must match universe actually available at val time
        # The result JSON records training N; use that so weights shapes match.
        # The strategy will intersect with active tickers at reset() time.
        ckpt_path = CKPT_DIR / ckpt_fname

        strategy = MLStrategy(
            name=name,
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            checkpoint_path=ckpt_path,
            tickers=tickers,
            feat_by_ticker=feat_by_ticker,
            T=T_LOOKBACK,
            max_position=MAX_POSITION,
            device="cpu",
        )

        bt = Backtester(
            strategy,
            prices,
            volumes,
            features=None,        # strategy uses precomputed feat_by_ticker internally
            starting_capital=STARTING_CAPITAL,
            allow_short=True,     # tanh can be negative
        )
        result = bt.run()

        ts = result.tearsheet(benchmark_nav=nifty_nav)
        ts["model"] = name
        ts["n_params"] = cfg.get("n_params", "unknown")
        ts["checkpoint_loaded"] = ckpt_path.exists()
        all_results[name] = ts

    # Save combined results
    out_path = OUT_DIR / "sim_step010_ml_backtest.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved to %s", out_path)

    # Summary table
    print("\n=== ML Backtest Summary ===")
    header = f"{'Model':<20} | {'CAGR%':>7} | {'Sharpe':>7} | {'MaxDD%':>8} | {'WinRate%':>9} | {'CkptLoaded':>10}"
    print(header)
    print("-" * len(header))
    for name, ts in all_results.items():
        print(
            f"{name:<20} | {ts['cagr_pct']:>7.2f} | {ts['sharpe']:>7.3f} | "
            f"{ts['max_drawdown_pct']:>8.2f} | {ts['win_rate_pct']:>9.1f} | "
            f"{str(ts['checkpoint_loaded']):>10}"
        )
    print()


if __name__ == "__main__":
    main()
