"""ts_step001 — Download 10+ years OHLCV for Nifty50 + screened small caps.

Output: data/ts/raw/<TICKER>.parquet (one file per ticker)
Cron-safe: re-running appends only new dates (no duplicates).

Usage:
    python scripts/ts/ts_step001_data_download.py
    python scripts/ts/ts_step001_data_download.py --update   # daily cron mode
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "ts" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Nifty 50 tickers (NSE suffix)
NIFTY50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "BAJFINANCE.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "ULTRACEMCO.NS", "WIPRO.NS", "ADANIENT.NS", "ONGC.NS",
    "POWERGRID.NS", "NTPC.NS", "JSWSTEEL.NS", "TECHM.NS", "HCLTECH.NS",
    "BAJAJFINSV.NS", "M&M.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "INDUSINDBK.NS",
    "NESTLEIND.NS", "BRITANNIA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS",
    "BPCL.NS", "COALINDIA.NS", "HINDALCO.NS", "GRASIM.NS", "APOLLOHOSP.NS",
    "EICHERMOT.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "ADANIPORTS.NS", "TATACONSUM.NS",
    "LTIM.NS", "SBILIFE.NS", "HDFCLIFE.NS", "UPL.NS", "VEDL.NS",
]

# High-volatility high-ROE small caps (manually curated seed; step002 will automate)
SMALL_CAP_SEED = [
    "ZOMATO.NS", "PAYTM.NS", "NYKAA.NS", "POLICYBZR.NS", "DELHIVERY.NS",
    "IRCTC.NS", "HAPPSTMNDS.NS", "TANLA.NS", "LATENTVIEW.NS", "MAPMYINDIA.NS",
    "ROUTE.NS", "CLEAN.NS", "KPITTECH.NS", "TATAELXSI.NS", "PERSISTENT.NS",
]

ALL_TICKERS = NIFTY50 + SMALL_CAP_SEED


def fetch_ticker(ticker: str, start: str, end: str, update_mode: bool) -> pd.DataFrame:
    out_path = RAW_DIR / f"{ticker.replace('.NS', '')}.parquet"

    if update_mode and out_path.exists():
        existing = pd.read_parquet(out_path)
        existing.index = pd.to_datetime(existing.index)
        last_date = existing.index.max()
        if last_date.date() >= datetime.today().date() - timedelta(days=1):
            print(f"  {ticker}: already up to date ({last_date.date()})")
            return existing
        start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  {ticker}: updating from {start}")

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            print(f"  {ticker}: NO DATA")
            return None

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        df.index = pd.to_datetime(df.index)
        df = df[["open", "high", "low", "close", "volume"]].dropna()

        if update_mode and out_path.exists():
            existing = pd.read_parquet(out_path)
            existing.index = pd.to_datetime(existing.index)
            df = pd.concat([existing, df]).sort_index().drop_duplicates()

        df.to_parquet(out_path)
        rows = len(df)
        span = f"{df.index.min().date()} → {df.index.max().date()}"
        print(f"  {ticker}: {rows} rows  [{span}]")
        return df

    except Exception as e:
        print(f"  {ticker}: ERROR — {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Append only new dates (cron mode)")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    print(f"ts_step001 | tickers={len(ALL_TICKERS)} | start={args.start} | end={args.end} | update={args.update}")
    print(f"Output dir: {RAW_DIR}\n")

    ok, fail = 0, 0
    for ticker in ALL_TICKERS:
        df = fetch_ticker(ticker, args.start, args.end, args.update)
        if df is not None:
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} ok / {fail} failed")
    print(f"Files: {len(list(RAW_DIR.glob('*.parquet')))} parquet files in {RAW_DIR}")


if __name__ == "__main__":
    main()
