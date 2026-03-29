"""Training callbacks: EarlyStopping."""

from __future__ import annotations

import copy

import torch


class EarlyStopping:
    """Stop training when monitored loss stops improving.

    Monitors train_loss by default (val loss unreliable on small GA subsets).
    Restores best model weights on stop.

    Parameters
    ----------
    patience  : epochs to wait after last improvement before stopping
    min_delta : minimum improvement to count as progress
    restore_best : restore best weights when triggered
    """

    def __init__(
        self,
        patience: int = 25,
        min_delta: float = 1e-4,
        restore_best: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_loss: float = float("inf")
        self.best_weights = None
        self.counter: int = 0
        self.triggered: bool = False

    def step(self, loss: float, model: torch.nn.Module) -> bool:
        """Call after each epoch. Returns True if training should stop."""
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            if self.restore_best:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                if self.restore_best and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        return False
