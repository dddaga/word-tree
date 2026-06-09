"""Round-robin training scheduler for parallel research lines.

Lines (main, ffn_baseline, cnn_compress, ...) submit jobs via submit.py →
.scheduler/pending/*.json. Daemon loop: for each free slot, pick next job
round-robin ACROSS LINES (no line starves another), launch via launch_slot.sh.

Slots: 5060ti_cuda, mini_mps, mini_cpu (priority order).
Safety: skips slot if foreign (non-sgn) load detected — 5060ti needs >=6 GB GPU
free; mini needs >=8 GB RAM available. Never touches non-sgn tmux sessions.

Run inside tmux:  tmux new -d -s sgn-scheduler 'd_env/bin/python3 -u scripts/scheduler/scheduler.py'
Submit:           scripts/scheduler/submit.py <line> <script> [slots=a,b] [-- extra args]
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

ROOT  = Path(__file__).resolve().parent.parent.parent
SCHED = ROOT / ".scheduler"
SLOTS = ["5060ti_cuda", "mini_mps", "mini_cpu"]
POLL_S = 60

for d in ("pending", "running", "done", "failed"):
    (SCHED / d).mkdir(parents=True, exist_ok=True)


def sh(cmd: str, timeout: int = 30) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           errors="replace", timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def slot_busy_sgn(slot: str) -> bool:
    """Any live sgn-*-<slot>-* tmux session (any user)?"""
    tmux = {"5060ti_cuda": "ssh -o ConnectTimeout=5 5060ti /usr/bin/tmux"}.get(slot, "tmux")
    rc, out = sh(f"{tmux} list-sessions -F '#{{session_name}}' 2>/dev/null")
    return rc == 0 and any(f"-{slot}-" in s and s.startswith("sgn-")
                           for s in out.splitlines())


def slot_resources_ok(slot: str) -> bool:
    """Foreign-load guard: don't crash/slow other users' training."""
    if slot == "5060ti_cuda":
        rc, out = sh("ssh -o ConnectTimeout=5 5060ti nvidia-smi"
                     " --query-gpu=memory.free --format=csv,noheader,nounits")
        return rc == 0 and out.isdigit() and int(out) >= 6000
    rc, out = sh("vm_stat | awk '/Pages free|Pages inactive/ {s+=$NF} END {print s*16384/2^30}'")
    try:
        return rc == 0 and float(out) >= 8.0
    except ValueError:
        return False


def pending_jobs() -> list[dict]:
    jobs = []
    for f in sorted((SCHED / "pending").glob("*.json")):
        if f.name.startswith("."):  # AppleDouble ._* files on exFAT
            continue
        try:
            j = json.loads(f.read_text()); j["_file"] = f; jobs.append(j)
        except (json.JSONDecodeError, UnicodeDecodeError):
            f.rename(SCHED / "failed" / f.name)
    return jobs


def next_job_round_robin(jobs: list[dict], slot: str) -> dict | None:
    """Cycle lines so each research line gets fair slot share."""
    eligible = [j for j in jobs if slot in j.get("slots", SLOTS)]
    if not eligible:
        return None
    lines = sorted({j["line"] for j in eligible})
    cursor_f = SCHED / "line_cursor.txt"
    last = cursor_f.read_text().strip() if cursor_f.exists() else ""
    start = (lines.index(last) + 1) % len(lines) if last in lines else 0
    for i in range(len(lines)):
        line = lines[(start + i) % len(lines)]
        for j in eligible:
            if j["line"] == line:
                cursor_f.write_text(line)
                return j
    return None


def launch(job: dict, slot: str) -> bool:
    extra = " ".join(job.get("args", []))
    rc, out = sh(f"bash {ROOT}/scripts/launch_slot.sh {slot} {job['script']} {extra}",
                 timeout=120)
    print(f"[{time.strftime('%H:%M:%S')}] launch {job['script']} on {slot}: "
          f"rc={rc}\n{out}", flush=True)
    if rc == 0:
        job["slot"], job["launched_at"] = slot, time.strftime("%F %T")
        f = job.pop("_file")
        (SCHED / "running" / f.name).write_text(json.dumps(job, indent=2))
        f.unlink()
        return True
    if rc != 2:  # 2 = occupied (retry later); else mark failed
        f = job.pop("_file")
        job["error"] = out[-500:]
        (SCHED / "failed" / f.name).write_text(json.dumps(job, indent=2))
        f.unlink()
    return False


def reap_running():
    """Move finished jobs (sgn session gone) running/ → done/."""
    for f in (SCHED / "running").glob("*.json"):
        if f.name.startswith("."):
            continue
        job = json.loads(f.read_text())
        if not slot_busy_sgn(job["slot"]):
            job["finished_at"] = time.strftime("%F %T")
            (SCHED / "done" / f.name).write_text(json.dumps(job, indent=2))
            f.unlink()
            print(f"[{time.strftime('%H:%M:%S')}] done: {job['script']}", flush=True)


def main():
    print(f"scheduler up — slots={SLOTS}, poll={POLL_S}s", flush=True)
    while True:
        try:
            reap_running()
            jobs = pending_jobs()
            if jobs:
                for slot in SLOTS:
                    if slot_busy_sgn(slot) or not slot_resources_ok(slot):
                        continue
                    job = next_job_round_robin(jobs, slot)
                    if job and launch(job, slot):
                        jobs = pending_jobs()
        except Exception as e:  # daemon must survive transient ssh failures
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: {e}", flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
