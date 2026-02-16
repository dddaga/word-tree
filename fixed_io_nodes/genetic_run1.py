"""
Entry point for GA run1: load run1 config, define train/val split (Iris), run genetic tuner.
"""

import signal
import sys
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, Subset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from distributed_training import IrisGNNModel
from genetic_algorithm import load_run_config, GeneticTuner


class Tee:
    """Write to a file and optionally to an original stream (e.g. stdout)."""

    def __init__(self, file, stream=None):
        self.file = file
        self.stream = stream

    def write(self, data):
        self.file.write(data)
        self.file.flush()
        if self.stream:
            self.stream.write(data)
            self.stream.flush()

    def flush(self):
        self.file.flush()
        if self.stream:
            self.stream.flush()

    def writable(self):
        return True


def _iris_dataset():
    """Iris in same format as main.load_iris_dataset (arccos, shape -1,1,4)."""
    iris_data = load_iris()
    X = iris_data.data
    y = iris_data.target
    X_tensor = torch.tensor(X, dtype=torch.float32)
    max_X = X_tensor.max(dim=0).values
    min_X = X_tensor.min(dim=0).values
    X_tensor = (X_tensor - min_X) / (max_X - min_X + 1e-8)
    X_tensor = torch.arccos(X_tensor).reshape(-1, 1, 4)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)


def get_train_val_datasets(config=None, seed=None, val_frac=None):
    """Return (train_dataset, val_dataset) as Subsets of Iris. If config is provided, use config['system']['random_seed'] and config['training']['validation_fraction']."""
    if config is not None:
        seed = config["system"].get("random_seed", 42) if seed is None else seed
        val_frac = config["training"].get("validation_fraction", 0.2) if val_frac is None else val_frac
    else:
        seed = 42 if seed is None else seed
        val_frac = 0.2 if val_frac is None else val_frac
    full = _iris_dataset()
    n = len(full)
    indices = list(range(n))
    labels = [full[i][1].item() for i in range(n)]
    if val_frac <= 0:
        train_dataset = Subset(full, indices)
        val_dataset = None
    else:
        train_idx, val_idx = train_test_split(indices, test_size=val_frac, random_state=seed, stratify=labels)
        train_dataset = Subset(full, train_idx)
        val_dataset = Subset(full, val_idx)
    return train_dataset, val_dataset


def main():
    def _sigint_handler(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _sigint_handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _sigint_handler)

    run_dir = "genetic_algorithm_runs/run2"
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(run_dir) / "log.txt"
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(log_file, orig_stdout)
    sys.stderr = Tee(log_file, orig_stderr)

    try:
        config_path = f"{run_dir}/config.yaml"
        base_config, search_space = load_run_config(config_path)
        if not search_space:
            raise ValueError("Run config must have at least one list-valued parameter (search space).")

        print("Initializing tuner...")
        tuner = GeneticTuner(
            base_config=base_config,
            search_space=search_space,
            model_class=IrisGNNModel,
            generations=8,
            population_size=50,
            elite_frac=0.2,
            crossover_rate=0.2,
            mutation_rate=0.2,
            top_k=8,
        )
        print("Tuner initialized")
        print(f"GA run: {run_dir} (generations={tuner.generations}, pop={tuner.population_size})")
        print(f"Log file: {log_path}")
        top = tuner.run(
            run_dir=run_dir,
            get_train_val_datasets=get_train_val_datasets,
            run_name="run1",
        )
        print("\nTop configs:")
        for i, cfg in enumerate(top, 1):
            print(f"  {i}. fitness={cfg.get('fitness', 0):.4f} gen={cfg.get('generation')} {cfg}")
        print(f"Saved to {run_dir}/best_configs.json")
        return top
    except KeyboardInterrupt:
        orig_stdout.write("\nInterrupted by user (Ctrl+C). Exiting.\n")
        orig_stdout.flush()
        return []
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()


if __name__ == "__main__":
    main()
