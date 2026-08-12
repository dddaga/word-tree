"""CNN distiller step005 — T2 multi-seed variance (150ep, 100% data).

Runs cnn_step004 with multiple seeds to establish ±variance for paper Pareto table.

Wait for F_wide T2 result (cnn_step004 mini_cpu, currently running) before deciding
which configs to multi-seed here.

T2 results (cnn_step004, seed=42):
  Ref       77.35%  (paper baseline)
  D_small_s 74.52%  (−2.83pp, 3.2× fewer MACs — efficiency Pareto)
  F_wide    TBD     (running)

Default: run Ref × 3 seeds to establish baseline ±variance for paper.
After F_wide completes, extend to D_small_s and F_wide as needed.

Usage:
    python scripts/cnn_distiller/train_cnn_step005_multiseed.py
    python scripts/cnn_distiller/train_cnn_step005_multiseed.py --configs Ref,D_small_s --seeds 1,2,3
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

parser = argparse.ArgumentParser(description="cnn_step005: CNN distiller multi-seed")
parser.add_argument("--device",   default="auto")
parser.add_argument("--configs",  default="Ref",
                    help="Comma-separated configs (from cnn_step004: Ref,F_wide,D_small_s)")
parser.add_argument("--seeds",    default="1,2,3",
                    help="Seeds to run (seed=42 already in step004)")
parser.add_argument("--epochs",   type=int, default=150)
parser.add_argument("--data_img", default="data/imagenette2-320")
parser.add_argument("--data_h5",  default="data/store.h5")
args = parser.parse_args()

SEEDS   = [int(s) for s in args.seeds.split(",")]
CONFIGS = args.configs
SLOT    = os.environ.get("SGN_SLOT", "local")
STEP004 = ROOT / "scripts" / "cnn_distiller" / "train_cnn_step004_t2.py"

print(f"cnn_step005 — CNN distiller multi-seed variance")
print(f"  configs={CONFIGS}  seeds={SEEDS}  slot={SLOT}")
print(f"  seed=42 reference: Ref=77.35%, D_small_s=74.52%, F_wide=TBD")

for seed in SEEDS:
    cmd = [
        sys.executable, str(STEP004),
        "--device",   args.device,
        "--seed",     str(seed),
        "--configs",  CONFIGS,
        "--epochs",   str(args.epochs),
        "--data_img", args.data_img,
        "--data_h5",  args.data_h5,
    ]
    env = os.environ.copy()
    env["SGN_SLOT"] = SLOT

    print(f"\n{'='*60}\nLaunching configs={CONFIGS} seed={seed} ...\n{'='*60}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"ERROR: seed={seed} failed with code {result.returncode}")
        sys.exit(result.returncode)

print(f"\ncnn_step005 complete. Seeds: {SEEDS}, configs: {CONFIGS}")
print("Results: results/cnn_step004_t2_seed{N}__*.json (reuses step004 output paths)")
