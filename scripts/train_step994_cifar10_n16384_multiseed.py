"""Step 994: N=16384 CIFAR-10 T2 multi-seed variance.

Runs step986 script sequentially with seeds 43 and 44.
step986 seed=42 result: 84.60% @ep149.
Goal: establish ±variance for paper scaling claim at N=16384.

Paper claim target: N=16384 T2 = 84.60% ± Xpp (X from 3-seed variance).

Usage:
    python scripts/train_step994_cifar10_n16384_multiseed.py --device cuda
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

parser = argparse.ArgumentParser(description="step994: N=16384 multi-seed (seeds 43,44)")
parser.add_argument("--device",   default="auto")
parser.add_argument("--seeds",    default="43,44",
                    help="Comma-separated seeds to run (seed42 already done in step986)")
parser.add_argument("--data",     default="data/store_cifar10.h5")
args = parser.parse_args()

SEEDS   = [int(s) for s in args.seeds.split(",")]
SLOT    = os.environ.get("SGN_SLOT", "local")
STEP986 = ROOT / "scripts" / "train_step986_cifar10_n16384_t2.py"

print(f"step994 — N=16384 CIFAR-10 T2 multi-seed")
print(f"  seeds={SEEDS}  slot={SLOT}")
print(f"  seed=42 reference: 84.60% (step986 T2, already done)")

for seed in SEEDS:
    out_path = ROOT / "results" / f"train_step994_cifar10_n16384_t2_seed{seed}__{SLOT}.json"
    if out_path.exists():
        print(f"\n[seed={seed}] result already exists: {out_path}. Skipping.")
        continue

    cmd = [
        sys.executable, str(STEP986),
        "--device", args.device,
        "--seed",   str(seed),
        "--data",   args.data,
    ]
    env = os.environ.copy()
    env["SGN_SLOT"] = SLOT

    print(f"\n{'='*60}\nLaunching seed={seed} ...\n{'='*60}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"ERROR: seed={seed} failed with code {result.returncode}")
        sys.exit(result.returncode)

    # Rename step986 output to step994
    step986_out = ROOT / "results" / f"train_step986_cifar10_n16384_t2_seed{seed}__{SLOT}.json"
    if step986_out.exists() and not out_path.exists():
        import shutil
        shutil.copy(step986_out, out_path)
        print(f"  → Result also saved as {out_path.name}")

print(f"\nstep994 complete. All seeds: {SEEDS}")
print("Load results from results/train_step994_cifar10_n16384_t2_seed*__*.json")
