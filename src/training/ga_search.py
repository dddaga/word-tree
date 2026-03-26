"""Population-based GA hyperparameter search for SGNNET_Wave.

Per D-11: population=20, generations=10, top_k=5,
          partial_data_fraction=0.15, epochs_per_eval=15.
Per D-03: fitness = -final_loss * (params_min / model_params)^0.2
"""

from __future__ import annotations

import math
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.sgnnet.model_wave import SGNNET_Wave
from src.training.trainer import Trainer

# Search spaces (D-12): D=4 fixed, not swept
SEARCH_SPACE_AB: dict[str, dict] = {
    "K": {"type": "discrete", "values": [1, 2, 3, 4]},
    "N_hidden": {"type": "discrete", "values": [64, 128, 256]},
    "lr_Wpos": {"type": "continuous_log", "low": 1e-5, "high": 1e-2},
    "lambda_safety": {"type": "continuous", "low": 0.0, "high": 1.0},
    "batch_size": {"type": "discrete", "values": [64, 128, 256]},
}

SEARCH_SPACE_C: dict[str, dict] = {
    **SEARCH_SPACE_AB,
    "lr_Wphase": {"type": "continuous_log", "low": 1e-5, "high": 1e-2},
}


def _sample_param(spec: dict):
    """Sample a random value from a parameter spec."""
    if spec["type"] == "discrete":
        return random.choice(spec["values"])
    if spec["type"] == "continuous":
        return random.uniform(spec["low"], spec["high"])
    if spec["type"] == "continuous_log":
        return 10 ** random.uniform(math.log10(spec["low"]), math.log10(spec["high"]))
    raise ValueError(f"Unknown param type: {spec['type']}")


def _mutate_param(value, spec: dict):
    """Mutate a single parameter value, clipped to bounds."""
    if spec["type"] == "discrete":
        return random.choice(spec["values"])
    if spec["type"] == "continuous":
        new_val = value + random.gauss(0, 0.1 * (spec["high"] - spec["low"]))
        return max(spec["low"], min(spec["high"], new_val))
    if spec["type"] == "continuous_log":
        new_val = value * (10 ** random.gauss(0, 0.3))
        return max(spec["low"], min(spec["high"], new_val))
    raise ValueError(f"Unknown param type: {spec['type']}")


class GASearch:
    """GA hyperparameter search with efficiency-ratio fitness (D-03)."""

    def __init__(
        self,
        search_space: dict[str, dict],
        experiment_name: str,
        train_features: torch.Tensor,
        train_soft_labels: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: torch.Tensor,
        val_soft_labels: torch.Tensor,
        val_labels: torch.Tensor,
        population: int = 20,
        generations: int = 10,
        top_k: int = 5,
        partial_fraction: float = 0.15,
        epochs_per_eval: int = 15,
        n_in: int = 25088,
        device: str = "mps",
    ):
        self.search_space = search_space
        self.experiment_name = experiment_name
        self.population = population
        self.generations = generations
        self.top_k = top_k
        self.partial_fraction = partial_fraction
        self.epochs_per_eval = epochs_per_eval
        self.n_in = n_in
        self.device = device
        self.train_features = train_features
        self.train_soft_labels = train_soft_labels
        self.train_labels = train_labels
        self.val_features = val_features
        self.val_soft_labels = val_soft_labels
        self.val_labels = val_labels
        self.params_min = self._compute_params_min()

    def _compute_params_min(self) -> int:
        """Count params for smallest config (N_hidden=64, K=1)."""
        model = SGNNET_Wave(N_hidden=64, K=1, D=4, N_in=self.n_in)
        return sum(p.numel() for p in model.parameters())

    def _experiment_flags(self) -> dict[str, bool]:
        """Return use_proximity and use_wphase flags for experiment."""
        if self.experiment_name == "stageA":
            return {"use_proximity": False, "use_wphase": False}
        if self.experiment_name == "exp1":
            return {"use_proximity": True, "use_wphase": False}
        return {"use_proximity": True, "use_wphase": True}

    def _random_individual(self) -> dict:
        return {k: _sample_param(s) for k, s in self.search_space.items()}

    def _make_partial_loader(self, batch_size: int) -> DataLoader:
        """Random subset of training data (partial_fraction)."""
        n = self.train_features.shape[0]
        k = max(1, int(n * self.partial_fraction))
        idx = torch.randperm(n)[:k]
        ds = TensorDataset(
            self.train_features[idx],
            self.train_soft_labels[idx],
            self.train_labels[idx],
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def _make_val_loader(self, batch_size: int) -> DataLoader:
        ds = TensorDataset(self.val_features, self.val_soft_labels, self.val_labels)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def _evaluate(self, config: dict) -> float:
        """Train candidate and return efficiency-ratio fitness score."""
        flags = self._experiment_flags()
        bs = config.get("batch_size", 128)

        model = SGNNET_Wave(
            N_hidden=config["N_hidden"], K=config["K"],
            D=4, N_in=self.n_in, **flags,
        )
        model_params = sum(p.numel() for p in model.parameters())

        trainer = Trainer(
            model, self._make_partial_loader(bs), self._make_val_loader(bs),
            lr_wpos=config["lr_Wpos"],
            lr_wphase=config.get("lr_Wphase"),
            lambda_safety=config["lambda_safety"],
            device=self.device,
        )
        history = trainer.train(self.epochs_per_eval)

        # NaN disqualification (D-03)
        if any(h.get("nan_detected", False) for h in history):
            return -1e6

        losses = [h["train_loss"] for h in history if not math.isnan(h["train_loss"])]
        if not losses:
            return -1e6

        final_loss = sum(losses[-3:]) / len(losses[-3:])
        if math.isnan(final_loss) or math.isinf(final_loss):
            return -1e6

        score = -final_loss * (self.params_min / model_params) ** 0.2
        return score

    def _mutate(self, individual: dict) -> dict:
        """Mutate config with per-param probability 0.3."""
        return {
            k: _mutate_param(individual[k], self.search_space[k])
            if random.random() < 0.3 else individual[k]
            for k in self.search_space
        }

    def _select(self, scored: list[tuple[dict, float]]) -> list[dict]:
        """Keep top_k elites sorted by score descending."""
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        return [cfg for cfg, _ in ranked[:self.top_k]]

    def run(self) -> dict:
        """Run GA search. Returns best config, score, and history."""
        pop = [self._random_individual() for _ in range(self.population)]
        history: list[dict] = []
        best_config: dict = {}
        best_score = -float("inf")

        for gen in range(self.generations):
            scored = [(cfg, self._evaluate(cfg)) for cfg in pop]
            gen_best_cfg, gen_best_score = max(scored, key=lambda x: x[1])

            if gen_best_score > best_score:
                best_score = gen_best_score
                best_config = gen_best_cfg

            history.append({
                "generation": gen,
                "best_score": gen_best_score,
                "best_config": gen_best_cfg,
            })
            print(f"Gen {gen}: best_score={gen_best_score:.4f}")

            parents = self._select(scored)
            pop = list(parents)
            while len(pop) < self.population:
                pop.append(self._mutate(random.choice(parents)))

        return {"best_config": best_config, "best_score": best_score, "history": history}
