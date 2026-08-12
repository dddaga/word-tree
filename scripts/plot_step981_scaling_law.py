#!/usr/bin/env python3
"""step981: Generate N-scaling law figure from existing results.

Plots accuracy vs N (log scale) for Imagenette and CIFAR-10,
fits saturating power law: acc = ceiling - a * N^(-b).
No training — pure analysis of existing results.

Usage:
    python scripts/plot_step981_scaling_law.py [--output figures/scaling_law.pdf]
"""
from __future__ import annotations
import argparse
import json
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="N-scaling law figure")
    p.add_argument("--output", default="figures/scaling_law.pdf",
                   help="Output path for figure (default: figures/scaling_law.pdf)")
    p.add_argument("--show", action="store_true", help="Show interactive plot")
    return p.parse_args()


IMAGENETTE_NOAUG = {
    1024: 90.57,   # step280 Ref T2
    2048: 95.52,   # step199 T2
    4096: 97.12,   # step273 Ref T2
    8192: 96.94,   # step276 Ref T2
}

IMAGENETTE_AUG = {
    1024: 91.11,   # step280 aug T2
    2048: 95.46,   # step269 aug T2
    4096: 97.30,   # step279 compound T2
    8192: 97.30,   # step282 compound T2
    16384: 97.20,  # step297 K_in=10+aug T2
}

CIFAR10 = {
    2048: 80.57,   # step980 T2 (multi-seed mean)
    4096: 82.53,   # step909
    8192: 83.55,   # step922
}

CIFAR10_LINEAR = 86.24  # Linear probe baseline


def fit_power_law(ns, accs, ceiling):
    """Fit: error = ceiling - acc = a * N^(-b) via log-linear regression."""
    errors = np.array([ceiling - a for a in accs])
    mask = errors > 0
    if mask.sum() < 2:
        return None, None
    log_n = np.log(np.array(ns)[mask])
    log_err = np.log(errors[mask])
    b, log_a = np.polyfit(log_n, log_err, 1)
    return np.exp(log_a), -b


def main():
    args = parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg" if not args.show else "TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib required. pip install matplotlib")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Imagenette
    for label, data, marker, color in [
        ("No aug", IMAGENETTE_NOAUG, "o", "#2196F3"),
        ("+Aug/K_in opt", IMAGENETTE_AUG, "s", "#4CAF50"),
    ]:
        ns = sorted(data.keys())
        accs = [data[n] for n in ns]
        ax1.plot(ns, accs, marker=marker, color=color, label=label,
                 linewidth=2, markersize=8)

    ax1.axhline(y=97.30, color="gray", linestyle="--", alpha=0.5, label="Ceiling (97.30%)")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("N (neurons)", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Imagenette T2", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_ylim(89, 98.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([1024, 2048, 4096, 8192, 16384])
    ax1.set_xticklabels(["1K", "2K", "4K", "8K", "16K"])

    # CIFAR-10
    ns_c = sorted(CIFAR10.keys())
    accs_c = [CIFAR10[n] for n in ns_c]
    ax2.plot(ns_c, accs_c, "o-", color="#F44336", linewidth=2, markersize=8,
             label="SGNNET")
    ax2.axhline(y=CIFAR10_LINEAR, color="gray", linestyle="--", alpha=0.5,
                label=f"Linear probe ({CIFAR10_LINEAR}%)")

    a, b = fit_power_law(ns_c, accs_c, CIFAR10_LINEAR)
    if a is not None:
        ns_fit = np.logspace(np.log2(1024), np.log2(32768), 50, base=2)
        accs_fit = CIFAR10_LINEAR - a * ns_fit ** (-b)
        ax2.plot(ns_fit, accs_fit, "--", color="#F44336", alpha=0.4,
                 label=f"Power law (b={b:.2f})")

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("N (neurons)", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("CIFAR-10 T2", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.set_ylim(78, 88)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([2048, 4096, 8192, 16384, 32768])
    ax2.set_xticklabels(["2K", "4K", "8K", "16K", "32K"])

    fig.suptitle("SGNNET Accuracy vs Network Size (D=16, K_iter=5)", fontsize=14, y=1.02)
    fig.tight_layout()

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", dpi=300)
    print(f"Saved: {out}")

    if args.show:
        plt.show()

    print("\nScaling summary:")
    print("  Imagenette: saturates at N=4096 (97.30%). D=16 ceiling.")
    print(f"  CIFAR-10: power-law b={b:.2f}. Gap closes with N but does not reach Linear.")
    print("  Paper story: log-linear gains at low N, diminishing returns at ceiling.")


if __name__ == "__main__":
    main()
