"""Submit a training job to the round-robin scheduler.

Usage:
  scripts/scheduler/submit.py <line> <script_path> [--slots a,b] [-- extra args]

  line   — research line tag: main | ffn_baseline | cnn_compress | ...
  script — path relative to repo root, e.g. scripts/ffn_baseline/ffn_step002.py
  slots  — optional comma list; default = any of 5060ti_cuda,mini_mps,mini_cpu
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PENDING = ROOT / ".scheduler" / "pending"
PENDING.mkdir(parents=True, exist_ok=True)

def main():
    argv = sys.argv[1:]
    if len(argv) < 2:
        print(__doc__); sys.exit(1)
    line, script = argv[0], argv[1]
    rest = argv[2:]
    slots = ["5060ti_cuda", "mini_mps", "mini_cpu"]
    if rest and rest[0] == "--slots":
        slots = rest[1].split(","); rest = rest[2:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not (ROOT / script).exists():
        print(f"ERROR: {script} not found"); sys.exit(1)
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{line}_{Path(script).stem}.json"
    job = {"line": line, "script": script, "slots": slots, "args": rest,
           "submitted_at": time.strftime("%F %T")}
    (PENDING / name).write_text(json.dumps(job, indent=2))
    print(f"queued: {name}")

if __name__ == "__main__":
    main()
