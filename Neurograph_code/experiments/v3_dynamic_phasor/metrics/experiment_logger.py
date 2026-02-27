"""
Experiment Logger
Tracks loss curves, performance metrics, issue-specific metrics, and generates plots.
"""

import json
import os
import time
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ExperimentLogger:
    """Logs metrics and generates plots for V3 experiment."""

    def __init__(self, log_dir: str = "logs/v3_experiment/", experiment_name: str = "v3_default"):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)

        # Loss curves
        self.primary_losses: List[float] = []
        self.diversity_losses: List[float] = []
        self.balance_losses: List[float] = []
        self.total_losses: List[float] = []

        # Performance metrics
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        self.test_metrics: Dict = {}

        # Issue-specific metrics (per epoch)
        self.issue_metrics: List[Dict] = []

        # Complexity
        self.complexity: Dict = {}

        # Timing
        self.start_time = time.time()

    def log_epoch(
        self,
        epoch: int,
        primary_loss: float,
        diversity_loss: float,
        balance_loss: float,
        total_loss: float,
        train_acc: float,
        val_acc: float,
        issue_metrics: Optional[Dict] = None,
    ) -> None:
        self.primary_losses.append(primary_loss)
        self.diversity_losses.append(diversity_loss)
        self.balance_losses.append(balance_loss)
        self.total_losses.append(total_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        if issue_metrics:
            self.issue_metrics.append(issue_metrics)

    def log_test_metrics(self, metrics: Dict) -> None:
        self.test_metrics = metrics

    def log_complexity(self, complexity: Dict) -> None:
        self.complexity = complexity

    def save_metrics(self) -> str:
        """Save all metrics to JSON file. Returns file path."""
        data = {
            "experiment_name": self.experiment_name,
            "total_time_s": time.time() - self.start_time,
            "loss_curves": {
                "primary": self.primary_losses,
                "diversity": self.diversity_losses,
                "balance": self.balance_losses,
                "total": self.total_losses,
            },
            "performance": {
                "train_accuracy": self.train_accuracies,
                "val_accuracy": self.val_accuracies,
                "test": self.test_metrics,
            },
            "issue_metrics": self.issue_metrics,
            "complexity": self.complexity,
        }

        path = os.path.join(self.log_dir, f"{self.experiment_name}_metrics.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Metrics saved to {path}")
        return path

    def plot_loss_curves(self) -> Optional[str]:
        """Generate and save loss curve plot."""
        if not HAS_MATPLOTLIB or not self.total_losses:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(self.total_losses) + 1)

        # Loss curves
        ax1.plot(epochs, self.total_losses, label="Total Loss", linewidth=2)
        ax1.plot(epochs, self.primary_losses, label="Primary (CE)", linewidth=1.5, alpha=0.8)
        if any(v > 0 for v in self.diversity_losses):
            ax1.plot(epochs, self.diversity_losses, label="Diversity", linewidth=1, alpha=0.7)
        if any(v > 0 for v in self.balance_losses):
            ax1.plot(epochs, self.balance_losses, label="Balance", linewidth=1, alpha=0.7)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curves")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Accuracy curves
        ax2.plot(epochs, self.train_accuracies, label="Train Acc", linewidth=2)
        if self.val_accuracies:
            ax2.plot(epochs, self.val_accuracies, label="Val Acc", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy Curves")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.log_dir, f"{self.experiment_name}_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plots saved to {path}")
        return path

    def plot_issue_metrics(self) -> Optional[str]:
        """Generate issue-specific metric plots."""
        if not HAS_MATPLOTLIB or not self.issue_metrics:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        epochs = range(1, len(self.issue_metrics) + 1)

        # Radiation entropy
        entropy_vals = [m.get("normalized_entropy", 0) for m in self.issue_metrics]
        axes[0, 0].plot(epochs, entropy_vals, linewidth=2)
        axes[0, 0].set_title("Radiation Diversity (Normalized Entropy)")
        axes[0, 0].set_ylabel("Entropy")
        axes[0, 0].grid(alpha=0.3)

        # Hit count variance
        variance_vals = [m.get("hit_count_variance", 0) for m in self.issue_metrics]
        axes[0, 1].plot(epochs, variance_vals, linewidth=2, color="orange")
        axes[0, 1].set_title("Load Balance (Hit Count Variance)")
        axes[0, 1].set_ylabel("Variance")
        axes[0, 1].grid(alpha=0.3)

        # Temperature
        temp_vals = [m.get("temperature", 1.0) for m in self.issue_metrics]
        axes[1, 0].plot(epochs, temp_vals, linewidth=2, color="green")
        axes[1, 0].set_title("Load Balancer Temperature")
        axes[1, 0].set_ylabel("Temperature")
        axes[1, 0].grid(alpha=0.3)

        # Unique targets
        unique_vals = [m.get("unique_targets", 0) for m in self.issue_metrics]
        axes[1, 1].plot(epochs, unique_vals, linewidth=2, color="red")
        axes[1, 1].set_title("Unique Radiation Targets")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].grid(alpha=0.3)

        for ax in axes.flat:
            ax.set_xlabel("Epoch")

        plt.tight_layout()
        path = os.path.join(self.log_dir, f"{self.experiment_name}_issue_metrics.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def print_summary(self) -> None:
        """Print a formatted summary table."""
        print("\n" + "=" * 60)
        print(f"  EXPERIMENT SUMMARY: {self.experiment_name}")
        print("=" * 60)

        if self.total_losses:
            print(f"  Final total loss:      {self.total_losses[-1]:.4f}")
            print(f"  Final primary loss:    {self.primary_losses[-1]:.4f}")

        if self.train_accuracies:
            print(f"  Final train accuracy:  {self.train_accuracies[-1]:.4f}")

        if self.val_accuracies:
            print(f"  Final val accuracy:    {self.val_accuracies[-1]:.4f}")

        if self.test_metrics:
            print(f"  Test accuracy:         {self.test_metrics.get('accuracy', 0):.4f}")
            print(f"  Test F1 (weighted):    {self.test_metrics.get('f1', 0):.4f}")
            print(f"  Test precision:        {self.test_metrics.get('precision', 0):.4f}")
            print(f"  Test recall:           {self.test_metrics.get('recall', 0):.4f}")

        if self.complexity:
            sc = self.complexity.get("space_complexity", {})
            tc = self.complexity.get("time_complexity", {})
            print(f"  Total parameters:      {sc.get('total_parameters', 0):,}")
            print(f"  Memory (MB):           {sc.get('memory_mb', 0):.2f}")
            if tc.get("inference_latency_us"):
                print(f"  Inference latency:     {tc['inference_latency_us']:.1f} us")

        elapsed = time.time() - self.start_time
        print(f"  Total wall time:       {elapsed:.1f}s")
        print("=" * 60)
