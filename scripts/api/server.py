"""
FastAPI backend for the stock ML research dashboard.

Endpoints:
  GET  /api/universe
  GET  /api/ticker/{symbol}
  GET  /api/experiments
  GET  /api/experiments/{id}
  POST /api/experiments/run
  GET  /api/experiments/{id}/log
  GET  /api/experiments/{id}/status
  DELETE /api/experiments/{id}

SQLite DB auto-created at data/ts/experiments.db and seeded from
results/ts/*.json and results/sim/*.json on startup.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sqlite3
import subprocess
import threading
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "ts" / "raw"
DB_PATH = ROOT / "data" / "ts" / "experiments.db"
LOG_DIR = ROOT / "logs"
RESULTS_TS_DIR = ROOT / "results" / "ts"
RESULTS_SIM_DIR = ROOT / "results" / "sim"
SCRIPTS_TS_DIR = ROOT / "scripts" / "ts"

LOG_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step → script mapping
# ---------------------------------------------------------------------------

STEP_SCRIPTS: Dict[str, str] = {
    "ts_step010": "ts_step010_cnn_lstm.py",
    "ts_step011": "ts_step011_cnn_transformer.py",
    "ts_step020": "ts_step020_mae_encoder.py",
    "ts_step021": "ts_step021_gan_encoder.py",
    "ts_step030": "ts_step030_sgnnet_ts.py",
}

# ---------------------------------------------------------------------------
# In-memory PID registry
# ---------------------------------------------------------------------------

_running: Dict[str, subprocess.Popen] = {}   # id → Popen
_running_lock = threading.Lock()

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id          TEXT PRIMARY KEY,
                step        TEXT,
                model       TEXT,
                config      TEXT,
                status      TEXT,
                created_at  TEXT,
                started_at  TEXT,
                finished_at TEXT,
                metrics     TEXT,
                results     TEXT,
                log_path    TEXT
            )
        """)
        conn.commit()


def _extract_metrics(data: dict) -> dict:
    """Pull the canonical metrics fields from a result JSON."""
    return {
        "val_sharpe":   data.get("best_val_sharpe") or data.get("val_sharpe") or data.get("sharpe"),
        "dir_acc":      data.get("final_val_dir_acc") or data.get("val_directional_acc") or data.get("dir_acc"),
        "best_val_loss": data.get("best_val_loss") or data.get("final_val_mse"),
        "n_params":     data.get("n_params"),
    }


def _seed_from_results() -> None:
    """Seed DB from all existing results/ts/*.json and results/sim/*.json (status=done)."""
    conn = _get_conn()
    seeded = 0
    for results_dir in [RESULTS_TS_DIR, RESULTS_SIM_DIR]:
        if not results_dir.exists():
            continue
        for jf in sorted(results_dir.glob("[!.]*.json")):
            exp_id = jf.stem
            row = conn.execute("SELECT id FROM experiments WHERE id=?", (exp_id,)).fetchone()
            if row:
                continue
            try:
                data = json.loads(jf.read_text(errors="replace"))
            except Exception as e:
                logger.warning("Skip %s: %s", jf.name, e)
                continue
            # Skip list-shaped results (e.g. per-ticker MA baseline) — not experiment rows
            if not isinstance(data, dict):
                logger.info("Skip %s: top-level is %s, not dict", jf.name, type(data).__name__)
                continue
            metrics = _extract_metrics(data)
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """INSERT INTO experiments
                   (id, step, model, config, status, created_at, started_at, finished_at, metrics, results, log_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    exp_id,
                    data.get("step", ""),
                    data.get("model", ""),
                    json.dumps({}),
                    "done",
                    now,
                    now,
                    now,
                    json.dumps(metrics),
                    json.dumps(data),
                    "",
                ),
            )
            seeded += 1
    conn.commit()
    conn.close()
    logger.info("Seeded %d experiments from results JSONs", seeded)


# ---------------------------------------------------------------------------
# Feature computation (mirrors ts_step010 logic)
# ---------------------------------------------------------------------------

def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"].shift(1)
    volume = df["volume"].shift(1)
    high = df["high"].shift(1)
    low = df["low"].shift(1)

    out["ret_1d"] = close / close.shift(1) - 1
    for w in [5, 21, 55, 89]:
        out[f"ma_ratio_{w}"] = close / close.rolling(w).mean()
    for f, s in [(5, 21), (13, 55), (21, 89)]:
        out[f"ma_cross_{f}_{s}"] = close.rolling(f).mean() / close.rolling(s).mean() - 1
    ret = close / close.shift(1) - 1
    out["vol_20"] = ret.rolling(20).std()
    out["vol_60"] = ret.rolling(60).std()
    mu60 = close.rolling(60).mean()
    std60 = close.rolling(60).std()
    out["spread_z"] = (close - mu60) / (std60 + 1e-8)
    out["resid_z"] = (ret - ret.rolling(60).mean()) / (ret.rolling(60).std() + 1e-8)
    out["delta_corr"] = out["vol_20"] / (out["vol_60"] + 1e-8) - 1.0
    out["lag1_ret"] = ret.shift(1)
    out["high_low_ratio"] = (high - low) / (close + 1e-8)
    vol_mu = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    out["volume_z"] = (volume - vol_mu) / (vol_std + 1e-8)
    return out


# ---------------------------------------------------------------------------
# Background monitor — updates DB when subprocess finishes
# ---------------------------------------------------------------------------

def _monitor_subprocess(exp_id: str, proc: subprocess.Popen, result_json_path: Optional[Path]) -> None:
    proc.wait()
    returncode = proc.returncode
    finished_at = datetime.now(timezone.utc).isoformat()

    status = "done" if returncode == 0 else "failed"
    metrics = {}
    results_str = "{}"

    if status == "done" and result_json_path and result_json_path.exists():
        try:
            data = json.loads(result_json_path.read_text())
            metrics = _extract_metrics(data)
            results_str = json.dumps(data)
        except Exception as e:
            logger.warning("Could not parse result JSON for %s: %s", exp_id, e)

    with _get_conn() as conn:
        conn.execute(
            "UPDATE experiments SET status=?, finished_at=?, metrics=?, results=? WHERE id=?",
            (status, finished_at, json.dumps(metrics), results_str, exp_id),
        )
        conn.commit()

    with _running_lock:
        _running.pop(exp_id, None)

    logger.info("Experiment %s finished with status=%s", exp_id, status)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    _seed_from_results()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Stock ML Research Dashboard", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    step: str
    device: str = "cpu"
    epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for field in ("config", "metrics", "results"):
        if d.get(field):
            try:
                d[field] = json.loads(d[field])
            except Exception:
                pass
    return d


def _parquet_for(symbol: str) -> Path:
    p = DATA_DIR / f"{symbol}.parquet"
    if not p.exists():
        # case-insensitive fallback
        for f in DATA_DIR.glob("*.parquet"):
            if f.stem.upper() == symbol.upper():
                return f
        raise HTTPException(status_code=404, detail=f"Ticker {symbol} not found")
    return p


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/universe")
def get_universe():
    """List all tickers with row_count, start_date, end_date."""
    if not DATA_DIR.exists():
        raise HTTPException(status_code=503, detail="Data directory not found")
    result = []
    for pf in sorted(DATA_DIR.glob("*.parquet")):
        if pf.stem.startswith("."):
            continue
        try:
            df = pd.read_parquet(pf)
            # Resolve date column
            date_col = next((c for c in ["date", "Date"] if c in df.columns), None)
            if date_col:
                dates = pd.to_datetime(df[date_col])
            else:
                df.index = pd.to_datetime(df.index)
                dates = df.index
            result.append({
                "symbol": pf.stem,
                "row_count": len(df),
                "start_date": str(dates.min().date()),
                "end_date": str(dates.max().date()),
            })
        except Exception as e:
            logger.warning("Could not read %s: %s", pf.name, e)
    return result


@app.get("/api/ticker/{symbol}")
def get_ticker(
    symbol: str,
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    features: bool = Query(False),
):
    """Return OHLCV rows for a ticker, optionally with computed features."""
    pf = _parquet_for(symbol)
    df = pd.read_parquet(pf)

    # normalise index / date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        df.index = pd.to_datetime(df.index)

    df.index.name = "date"
    df = df.sort_index()

    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]

    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    if features:
        feats = _compute_features(df)
        df = df.join(feats, how="left")

    df = df.reset_index()
    df["date"] = df["date"].astype(str)
    return df.where(pd.notnull(df), None).to_dict(orient="records")


@app.get("/api/experiments")
def list_experiments(
    status: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    limit: int = Query(50),
):
    conn = _get_conn()
    clauses = []
    params: list = []
    if status:
        clauses.append("status=?")
        params.append(status)
    if model:
        clauses.append("model=?")
        params.append(model)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM experiments {where} ORDER BY created_at DESC LIMIT ?",
        params + [limit],
    ).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


@app.get("/api/experiments/{exp_id}")
def get_experiment(exp_id: str):
    conn = _get_conn()
    row = conn.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return _row_to_dict(row)


@app.post("/api/experiments/run", status_code=201)
def run_experiment(req: RunRequest):
    """Launch a training subprocess and track it."""
    step = req.step.strip()
    if step not in STEP_SCRIPTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown step '{step}'. Valid: {list(STEP_SCRIPTS.keys())}",
        )

    script_name = STEP_SCRIPTS[step]
    script_path = SCRIPTS_TS_DIR / script_name
    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"Script not found: {script_path}")

    exp_id = str(uuid.uuid4())[:8]
    log_path = LOG_DIR / f"{step}_api_{exp_id}.log"
    now = datetime.now(timezone.utc).isoformat()

    config = req.model_dump()

    # Determine expected result JSON path (scripts write to results/ts/<step>.json by default)
    result_json = ROOT / "results" / "ts" / f"{step}.json"

    cmd = [
        "python3", str(script_path),
        "--device", req.device,
        "--epochs", str(req.epochs),
        "--batch_size", str(req.batch_size),
        "--lr", str(req.lr),
        "--seed", str(req.seed),
    ]

    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO experiments
               (id, step, model, config, status, created_at, started_at, log_path)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                exp_id,
                step,
                script_name.replace(".py", ""),
                json.dumps(config),
                "running",
                now,
                now,
                str(log_path),
            ),
        )
        conn.commit()

    log_fp = open(str(log_path), "w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log_fp,
        stderr=subprocess.STDOUT,
    )

    with _running_lock:
        _running[exp_id] = proc

    monitor_thread = threading.Thread(
        target=_monitor_subprocess,
        args=(exp_id, proc, result_json),
        daemon=True,
    )
    monitor_thread.start()

    return {"id": exp_id, "log_path": str(log_path)}


@app.get("/api/experiments/{exp_id}/log")
def get_log(exp_id: str, lines: int = Query(100)):
    """Return last N lines of the log file."""
    conn = _get_conn()
    row = conn.execute("SELECT log_path FROM experiments WHERE id=?", (exp_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Experiment not found")
    log_path = row["log_path"]
    if not log_path or not Path(log_path).exists():
        return {"lines": []}
    buf: deque = deque(maxlen=lines)
    with open(log_path) as f:
        for line in f:
            buf.append(line.rstrip())
    return {"lines": list(buf)}


@app.get("/api/experiments/{exp_id}/status")
def get_status(exp_id: str):
    conn = _get_conn()
    row = conn.execute(
        "SELECT status, metrics FROM experiments WHERE id=?", (exp_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Experiment not found")
    metrics = {}
    if row["metrics"]:
        try:
            metrics = json.loads(row["metrics"])
        except Exception:
            pass
    return {"status": row["status"], "metrics": metrics}


@app.delete("/api/experiments/{exp_id}", status_code=200)
def delete_experiment(exp_id: str):
    """Kill subprocess if running, then delete DB row."""
    with _running_lock:
        proc = _running.pop(exp_id, None)
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()

    conn = _get_conn()
    row = conn.execute("SELECT id FROM experiments WHERE id=?", (exp_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Experiment not found")
    conn.execute("DELETE FROM experiments WHERE id=?", (exp_id,))
    conn.commit()
    conn.close()
    return {"deleted": exp_id}
