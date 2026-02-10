"""
Plot loss vs step from the phase-rad/mag-cond experiment CSV.
Run from repo root (word-tree):

  python fixed_io_nodes/experiments/phase_rad_mag_cond/plot_loss.py

Or pass a custom CSV path:

  python fixed_io_nodes/experiments/phase_rad_mag_cond/plot_loss.py path/to/loss.csv
"""

import sys
from pathlib import Path

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    print("Need pandas and matplotlib: pip install pandas matplotlib")
    sys.exit(1)

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[2]
_default_csv = _repo_root / "training_runs" / "phase_rad_mag_cond_experiment" / "loss.csv"


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_csv
    if not csv_path.is_absolute():
        csv_path = (_repo_root / csv_path).resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        print("Run the experiment first: python fixed_io_nodes/experiments/phase_rad_mag_cond/run_experiment.py")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df["loss"] = df["loss"].astype(float)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["step"], df["loss"], color="tab:blue", linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Phase-Radiation / Magnitude-Conduction Experiment — Loss vs Step")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = csv_path.parent / "loss_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
