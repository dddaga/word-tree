"""Generic sequential sweep runner for a single slot.

launch_slot.sh occupies one slot with one python process, but a T0 ablation is
arms x seeds. This runs the cells back-to-back in that one process so the slot
accounting stays honest (one session = one slot = one running script).

  python scripts/glam/glam_sweep.py --script scripts/glam/glam_step009_gap_gate_t0.py \
      --arms S0,S1,S2,S3 --seeds 42,43,44 --device mps
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--script", required=True)
parser.add_argument("--arms", required=True, help="comma list, each item 'ARM' or 'ARM:extra args'")
parser.add_argument("--seeds", default="42,43,44")
parser.add_argument("--device", default="auto")
parser.add_argument("--extra", default="", help="extra args appended to every cell")
args = parser.parse_args()

CELLS = [c for c in args.arms.split(",") if c]
SEEDS = [int(s) for s in args.seeds.split(",") if s]


def main() -> int:
    t0, fails = time.time(), []
    for cell in CELLS:
        arm, _, cell_extra = cell.partition(":")
        for seed in SEEDS:
            cmd = [sys.executable, "-u", str(ROOT / args.script),
                   "--arm", arm, "--seed", str(seed), "--device", args.device]
            cmd += cell_extra.split() + args.extra.split()
            print(f"\n{'#'*70}\n# {' '.join(cmd[2:])}\n{'#'*70}", flush=True)
            rc = subprocess.call(cmd, cwd=str(ROOT))
            if rc != 0:
                fails.append((arm, seed, rc))
                print(f"!! FAILED rc={rc}", flush=True)
    print(f"\nSWEEP DONE  {len(CELLS)*len(SEEDS)} cells  {time.time()-t0:.0f}s  "
          f"failures={fails or 'none'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
