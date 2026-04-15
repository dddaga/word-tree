#!/usr/bin/env python3
"""
experiment_controller.py — SGNNET centralized experiment queue and scheduler.

HTTP server on 0.0.0.0:7433.  SQLite storage at .controller/queue.db.
Scheduler ticks every 30s, dispatches via scripts/launch_slot.sh.

Usage:
    python3 scripts/experiment_controller.py [--port 7433] [--tick 30] [--db .controller/queue.db]
"""

import argparse
import json
import logging
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / ".controller" / "queue.db"
DEFAULT_LOG = REPO_ROOT / "logs" / "experiment_controller.log"
DEFAULT_PORT = 7433
DEFAULT_TICK = 30

VALID_SLOTS = ["mini_mps", "mini_cpu", "studio_mps", "studio_cpu", "5060ti_cuda"]
VALID_DEVICE_PREFS = {"any", "cuda", "mps", "cpu"}
VALID_STATUSES = {"queued", "running", "done", "failed", "cancelled"}

SLOT_DEVICE: Dict[str, str] = {
    "mini_mps":    "mps",
    "mini_cpu":    "cpu",
    "studio_mps":  "mps",
    "studio_cpu":  "cpu",
    "5060ti_cuda": "cuda",
}

LAUNCH_SCRIPT = REPO_ROOT / "scripts" / "launch_slot.sh"
STATUS_SCRIPT = REPO_ROOT / "scripts" / "slot_status.sh"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> None:
    """Configure root logger to file + stderr."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stderr),
        ],
    )

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def open_db(db_path: Path) -> sqlite3.Connection:
    """Open SQLite connection with WAL mode enabled."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist. Idempotent — additive column migration."""
    # Additive migration for launched_slot on pre-existing DBs
    existing = conn.execute("PRAGMA table_info(queue_entries)").fetchall()
    existing_cols = {row[1] for row in existing}
    if existing and "launched_slot" not in existing_cols:
        conn.execute("ALTER TABLE queue_entries ADD COLUMN launched_slot TEXT")
        conn.commit()

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS queue_entries (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user         TEXT    NOT NULL,
            step_name    TEXT    NOT NULL,
            script_path  TEXT    NOT NULL,
            args         TEXT    NOT NULL DEFAULT '',
            device_pref  TEXT    NOT NULL DEFAULT 'any',
            slot_pref    TEXT,
            priority     INTEGER NOT NULL DEFAULT 0,
            status        TEXT    NOT NULL DEFAULT 'queued',
            submitted_at  TEXT    NOT NULL,
            launched_at   TEXT,
            launched_slot TEXT,
            finished_at   TEXT,
            result_path   TEXT
        );
        CREATE TABLE IF NOT EXISTS round_robin (
            id        INTEGER PRIMARY KEY CHECK (id = 1),
            last_user TEXT    NOT NULL DEFAULT ''
        );
        INSERT OR IGNORE INTO round_robin (id, last_user) VALUES (1, '');
    """)
    conn.commit()


def now_iso() -> str:
    """Return current UTC time as ISO8601 string."""
    return datetime.now(timezone.utc).isoformat()


def get_last_user(conn: sqlite3.Connection) -> str:
    """Retrieve the round-robin pointer (last dispatched user)."""
    row = conn.execute("SELECT last_user FROM round_robin WHERE id=1").fetchone()
    return row["last_user"] if row else ""


def set_last_user(conn: sqlite3.Connection, user: str) -> None:
    """Persist the round-robin pointer."""
    conn.execute("UPDATE round_robin SET last_user=? WHERE id=1", (user,))
    conn.commit()


def insert_entry(conn: sqlite3.Connection, data: dict) -> int:
    """Insert a new queue entry; return its id."""
    cur = conn.execute(
        """
        INSERT INTO queue_entries
            (user, step_name, script_path, args, device_pref, slot_pref, priority, submitted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data["user"],
            data["step_name"],
            data["script_path"],
            data.get("args", ""),
            data.get("device_pref", "any"),
            data.get("slot_pref") or None,
            int(data.get("priority", 0)),
            now_iso(),
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_queued_entries(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """All entries with status=queued, ordered by priority desc, submitted_at asc."""
    return conn.execute(
        """
        SELECT * FROM queue_entries
        WHERE status='queued'
        ORDER BY priority DESC, submitted_at ASC
        """
    ).fetchall()


def get_all_entries(conn: sqlite3.Connection, user: Optional[str] = None) -> List[sqlite3.Row]:
    """All entries, optionally filtered by user."""
    if user:
        return conn.execute(
            "SELECT * FROM queue_entries WHERE user=? ORDER BY priority DESC, submitted_at ASC",
            (user,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM queue_entries ORDER BY priority DESC, submitted_at ASC"
    ).fetchall()


def get_entry_by_id(conn: sqlite3.Connection, entry_id: int) -> Optional[sqlite3.Row]:
    """Fetch a single entry by id."""
    return conn.execute(
        "SELECT * FROM queue_entries WHERE id=?", (entry_id,)
    ).fetchone()


def set_entry_status(
    conn: sqlite3.Connection,
    entry_id: int,
    status: str,
    extra: Optional[dict] = None,
) -> None:
    """Update status and optional timestamp fields for an entry."""
    if extra:
        fields = ", ".join(f"{k}=?" for k in extra)
        vals = list(extra.values()) + [status, entry_id]
        conn.execute(
            f"UPDATE queue_entries SET {fields}, status=? WHERE id=?", vals
        )
    else:
        conn.execute(
            "UPDATE queue_entries SET status=? WHERE id=?", (status, entry_id)
        )
    conn.commit()

# ---------------------------------------------------------------------------
# Slot status parsing
# ---------------------------------------------------------------------------

def get_free_slots() -> List[str]:
    """Run slot_status.sh and return list of FREE slot names."""
    try:
        result = subprocess.run(
            ["bash", str(STATUS_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return parse_slot_status_output(result.stdout)
    except subprocess.TimeoutExpired:
        log.warning("slot_status.sh timed out")
        return []
    except Exception as exc:
        log.warning("slot_status.sh error: %s", exc)
        return []


def parse_slot_status_output(output: str) -> List[str]:
    """Parse slot_status.sh stdout and return FREE slot names."""
    free = []
    for line in output.splitlines():
        line = line.strip()
        for slot in VALID_SLOTS:
            if line.startswith(slot) and "[FREE]" in line:
                free.append(slot)
                break
    return free

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def device_compatible(device_pref: str, slot: str) -> bool:
    """Return True if device_pref is compatible with the slot's device."""
    if device_pref == "any":
        return True
    return SLOT_DEVICE.get(slot) == device_pref


def distinct_users(entries: List[sqlite3.Row]) -> List[str]:
    """Return ordered list of unique users from entries (insertion order)."""
    seen = {}
    for e in entries:
        seen[e["user"]] = True
    return list(seen.keys())


def round_robin_order(users: List[str], last_user: str) -> List[str]:
    """
    Return users in round-robin order starting after last_user.
    If last_user not in list, start from index 0.
    """
    if not users:
        return []
    if last_user not in users:
        return users[:]
    idx = users.index(last_user)
    start = (idx + 1) % len(users)
    return users[start:] + users[:start]


def pick_reserved_entry(
    queued: List[sqlite3.Row], slot: str, last_user: str
) -> Optional[sqlite3.Row]:
    """
    Among entries with slot_pref==slot, pick one in round-robin user order.
    Returns the highest-priority / earliest entry from the first eligible user.
    """
    candidates = [e for e in queued if e["slot_pref"] == slot]
    if not candidates:
        return None
    users = distinct_users(candidates)
    ordered = round_robin_order(users, last_user)
    for user in ordered:
        user_entries = [e for e in candidates if e["user"] == user]
        if user_entries:
            return user_entries[0]  # already sorted priority desc, submitted_at asc
    return None


def pick_any_entry(
    queued: List[sqlite3.Row], slot: str, last_user: str
) -> Optional[sqlite3.Row]:
    """
    Pick an any-pref (slot_pref IS NULL) entry compatible with slot's device,
    respecting round-robin across users.
    Returns (entry, user) or None.
    """
    compatible = [
        e for e in queued
        if e["slot_pref"] is None and device_compatible(e["device_pref"], slot)
    ]
    if not compatible:
        return None
    users = distinct_users(compatible)
    ordered = round_robin_order(users, last_user)
    for user in ordered:
        user_entries = [e for e in compatible if e["user"] == user]
        if user_entries:
            return user_entries[0]
    return None


def launch_entry(conn: sqlite3.Connection, entry: sqlite3.Row, slot: str) -> bool:
    """
    Invoke launch_slot.sh for the entry on the given slot.
    Returns True on success (exit 0), False otherwise.
    """
    cmd = ["bash", str(LAUNCH_SCRIPT), slot, entry["script_path"]]
    if entry["args"]:
        import shlex
        cmd.extend(shlex.split(entry["args"]))

    env = os.environ.copy()
    env["SGNNET_USER"] = entry["user"]

    log.info("Launching entry id=%d user=%s slot=%s script=%s",
             entry["id"], entry["user"], slot, entry["script_path"])

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            log.info("Launched OK: id=%d slot=%s", entry["id"], slot)
            set_entry_status(conn, entry["id"], "running",
                             {"launched_at": now_iso(), "launched_slot": slot})
            return True
        elif result.returncode == 2:
            log.warning("Slot occupied (exit 2): slot=%s entry id=%d — leaving queued", slot, entry["id"])
            return False
        else:
            log.error("launch_slot.sh failed (exit %d) for id=%d: %s",
                      result.returncode, entry["id"], result.stderr.strip())
            set_entry_status(conn, entry["id"], "failed", {"finished_at": now_iso()})
            return False
    except subprocess.TimeoutExpired:
        log.error("launch_slot.sh timeout for id=%d slot=%s", entry["id"], slot)
        return False
    except Exception as exc:
        log.error("launch_slot.sh exception for id=%d: %s", entry["id"], exc)
        set_entry_status(conn, entry["id"], "failed", {"finished_at": now_iso()})
        return False


def reconcile_running(conn: sqlite3.Connection, free_slots: List[str]) -> None:
    """
    Entries marked running whose slot is now FREE/DONE → transition to done.
    This handles the case where the controller was restarted mid-training.
    """
    running = conn.execute(
        "SELECT * FROM queue_entries WHERE status='running'"
    ).fetchall()

    # Use launched_slot (stored on dispatch) to identify which slot the entry
    # is holding. If that slot now shows FREE, the training finished; mark done.
    # Fall back to slot_pref for entries launched before the launched_slot column
    # existed (backward compat for pre-migration DBs).
    for entry in running:
        slot = entry["launched_slot"] or entry["slot_pref"]
        if slot and slot in free_slots:
            log.info("Reconciling: entry id=%d slot=%s now free — marking done",
                     entry["id"], slot)
            set_entry_status(conn, entry["id"], "done", {"finished_at": now_iso()})


def scheduler_tick(conn: sqlite3.Connection, db_lock: threading.Lock) -> None:
    """One scheduling tick: poll free slots, dispatch entries."""
    with db_lock:
        free_slots = get_free_slots()
        if not free_slots:
            return

        queued = get_queued_entries(conn)
        last_user = get_last_user(conn)

        reconcile_running(conn, free_slots)
        # Re-fetch after reconciliation in case statuses changed
        queued = get_queued_entries(conn)

        # Determine which free slots have ANY pending reservation
        reserved_slot_names = set()
        for slot in free_slots:
            has_reservation = any(e["slot_pref"] == slot for e in queued)
            if has_reservation:
                reserved_slot_names.add(slot)

        dispatched_user = None
        for slot in free_slots:
            if slot in reserved_slot_names:
                entry = pick_reserved_entry(queued, slot, last_user)
                if entry:
                    ok = launch_entry(conn, entry, slot)
                    if ok:
                        dispatched_user = entry["user"]
                        # Remove from in-memory queued list so we don't double-dispatch
                        queued = [e for e in queued if e["id"] != entry["id"]]
                # Else: slot stays free, waiting for the reservation to become ready
            else:
                entry = pick_any_entry(queued, slot, last_user)
                if entry:
                    ok = launch_entry(conn, entry, slot)
                    if ok:
                        dispatched_user = entry["user"]
                        queued = [e for e in queued if e["id"] != entry["id"]]
                        last_user = entry["user"]

        if dispatched_user:
            set_last_user(conn, dispatched_user)


def scheduler_loop(conn: sqlite3.Connection, db_lock: threading.Lock,
                   tick_seconds: int, stop_event: threading.Event) -> None:
    """Background thread: run scheduler_tick every tick_seconds until stopped."""
    log.info("Scheduler started (tick=%ds)", tick_seconds)
    while not stop_event.is_set():
        try:
            scheduler_tick(conn, db_lock)
        except Exception as exc:
            log.exception("Scheduler tick error: %s", exc)
        stop_event.wait(tick_seconds)
    log.info("Scheduler stopped")

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    """HTTP request handler for experiment controller API."""

    conn: sqlite3.Connection
    db_lock: threading.Lock

    def log_message(self, fmt, *args):  # silence default access log
        log.debug("HTTP %s", fmt % args)

    def send_json(self, code: int, data) -> None:
        """Send a JSON response."""
        body = json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self) -> Optional[dict]:
        """Read and parse JSON request body."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length))
        except Exception:
            return None

    def do_POST(self) -> None:
        """Route POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/submit":
            self._handle_submit()
        elif re.match(r"^/cancel/\d+$", path):
            entry_id = int(path.split("/")[-1])
            self._handle_cancel(entry_id)
        else:
            self.send_json(404, {"error": "not found"})

    def do_GET(self) -> None:
        """Route GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = parse_qs(parsed.query)

        if path == "/queue":
            user_filter = qs.get("user", [None])[0]
            self._handle_queue(user_filter)
        elif path == "/status":
            self._handle_status()
        else:
            self.send_json(404, {"error": "not found"})

    def _handle_submit(self) -> None:
        """POST /submit — add entry to queue."""
        data = self.read_json_body()
        if data is None:
            self.send_json(400, {"error": "invalid JSON"})
            return

        user = (data.get("user") or "").strip()
        if not user:
            self.send_json(400, {"error": "user field required"})
            return

        step_name = (data.get("step_name") or "").strip()
        script_path = (data.get("script_path") or "").strip()
        if not step_name or not script_path:
            self.send_json(400, {"error": "step_name and script_path required"})
            return

        device_pref = data.get("device_pref", "any")
        if device_pref not in VALID_DEVICE_PREFS:
            self.send_json(400, {"error": f"device_pref must be one of {sorted(VALID_DEVICE_PREFS)}"})
            return

        slot_pref = data.get("slot_pref") or None
        if slot_pref and slot_pref not in VALID_SLOTS:
            self.send_json(400, {"error": f"slot_pref must be one of {VALID_SLOTS}"})
            return

        with self.db_lock:
            entry_id = insert_entry(self.conn, data)

        log.info("Queued: id=%d user=%s step=%s device_pref=%s slot_pref=%s",
                 entry_id, user, step_name, device_pref, slot_pref)
        self.send_json(201, {"id": entry_id, "status": "queued"})

    def _handle_cancel(self, entry_id: int) -> None:
        """POST /cancel/<id> — cancel a queued entry."""
        with self.db_lock:
            entry = get_entry_by_id(self.conn, entry_id)
            if entry is None:
                self.send_json(404, {"error": "entry not found"})
                return
            if entry["status"] != "queued":
                self.send_json(409, {"error": f"cannot cancel entry with status={entry['status']}"})
                return
            set_entry_status(self.conn, entry_id, "cancelled", {"finished_at": now_iso()})

        log.info("Cancelled entry id=%d", entry_id)
        self.send_json(200, {"id": entry_id, "status": "cancelled"})

    def _handle_queue(self, user_filter: Optional[str]) -> None:
        """GET /queue[?user=x] — list entries."""
        with self.db_lock:
            entries = get_all_entries(self.conn, user_filter)
        self.send_json(200, [dict(e) for e in entries])

    def _handle_status(self) -> None:
        """GET /status — system overview."""
        with self.db_lock:
            running = self.conn.execute(
                "SELECT * FROM queue_entries WHERE status='running'"
            ).fetchall()
            depths = self.conn.execute(
                "SELECT user, COUNT(*) as n FROM queue_entries WHERE status='queued' GROUP BY user"
            ).fetchall()
            last_user = get_last_user(self.conn)

        self.send_json(200, {
            "running": [dict(e) for e in running],
            "queue_depth": {row["user"]: row["n"] for row in depths},
            "round_robin_pointer": last_user,
            "free_slots": get_free_slots(),
        })


def make_handler(conn: sqlite3.Connection, db_lock: threading.Lock):
    """Return a Handler class bound to the given db connection."""
    class BoundHandler(Handler):
        pass
    BoundHandler.conn = conn
    BoundHandler.db_lock = db_lock
    return BoundHandler

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="SGNNET experiment controller")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--tick", type=int, default=DEFAULT_TICK,
                   help="Scheduler tick interval in seconds")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    setup_logging(args.log)
    log.info("Starting experiment controller (port=%d tick=%ds db=%s)",
             args.port, args.tick, args.db)

    conn = open_db(args.db)
    init_schema(conn)

    db_lock = threading.Lock()
    stop_event = threading.Event()

    # Graceful shutdown on SIGTERM / SIGINT
    def shutdown(signum, frame):
        log.info("Signal %d received — shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # Start scheduler thread
    sched_thread = threading.Thread(
        target=scheduler_loop,
        args=(conn, db_lock, args.tick, stop_event),
        daemon=True,
        name="scheduler",
    )
    sched_thread.start()

    # Start HTTP server in a daemon thread
    handler_cls = make_handler(conn, db_lock)
    server = HTTPServer(("0.0.0.0", args.port), handler_cls)
    server.timeout = 1.0  # allow periodic check of stop_event

    http_thread = threading.Thread(
        target=_serve_forever,
        args=(server, stop_event),
        daemon=True,
        name="http",
    )
    http_thread.start()

    log.info("Controller ready on port %d", args.port)

    # Block until stop_event set
    stop_event.wait()
    log.info("Stopping HTTP server")
    server.server_close()
    sched_thread.join(timeout=5)
    log.info("Controller exited cleanly")


def _serve_forever(server: HTTPServer, stop_event: threading.Event) -> None:
    """Serve HTTP until stop_event is set."""
    while not stop_event.is_set():
        server.handle_request()


if __name__ == "__main__":
    main()
