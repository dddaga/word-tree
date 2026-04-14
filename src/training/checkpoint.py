"""Checkpoint utilities for SGNNET — save/resume training state.

Saves everything needed to resume training from any epoch:
  - model state_dict
  - optimizer state_dict
  - scheduler state_dict
  - scaler state_dict (AMP)
  - epoch number
  - training history
  - experiment config
  - best metrics

Usage:
    # Save
    save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, history, config)

    # Resume
    state = load_checkpoint(path, model, optimizer, scheduler, scaler)
    start_epoch = state["epoch"] + 1
    history = state["history"]
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    history: list[dict] | None = None,
    config: dict | None = None,
    best_top1: float = 0.0,
    diagnostics_history: list[dict] | None = None,
) -> Path:
    """Save full training state to a checkpoint file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_top1": best_top1,
        "history": history or [],
        "config": config or {},
        "diagnostics_history": diagnostics_history or [],
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()

    torch.save(state, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    device: str | torch.device = "cpu",
) -> dict:
    """Load training state from a checkpoint. Returns the full state dict.

    Model, optimizer, scheduler, scaler are loaded in-place.
    Returns dict with: epoch, history, config, best_top1, diagnostics_history.
    """
    path = Path(path)
    state = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])

    return {
        "epoch": state.get("epoch", 0),
        "history": state.get("history", []),
        "config": state.get("config", {}),
        "best_top1": state.get("best_top1", 0.0),
        "diagnostics_history": state.get("diagnostics_history", []),
    }


# ── Save policy helpers ──

class CheckpointPolicy:
    """Decides when to save checkpoints during training.

    Saves on:
      - New best val_top1 (always)
      - Every save_every epochs (periodic)
      - Final epoch (always)

    Naming: {prefix}_ep{epoch}_{top1:.4f}.pt
    Best checkpoint also saved as: {prefix}_best.pt
    """

    def __init__(self, save_dir: str | Path, prefix: str = "checkpoint",
                 save_every: int = 25):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.save_every = save_every
        self.best_top1 = 0.0

    def should_save(self, epoch: int, val_top1: float, is_final: bool = False) -> str | None:
        """Return save path if should save, else None."""
        is_best = val_top1 > self.best_top1
        is_periodic = self.save_every > 0 and epoch % self.save_every == 0

        if is_best or is_periodic or is_final:
            if is_best:
                self.best_top1 = val_top1
            return str(self.save_dir / f"{self.prefix}_ep{epoch}_{val_top1:.4f}.pt")
        return None

    def best_path(self) -> Path:
        """Path for the best checkpoint (symlink/copy target)."""
        return self.save_dir / f"{self.prefix}_best.pt"

    def step(self, epoch: int, val_top1: float, is_final: bool,
             model, optimizer=None, scheduler=None, scaler=None,
             history=None, config=None, diagnostics_history=None) -> str | None:
        """Check policy and save if needed. Returns path if saved, else None."""
        path = self.should_save(epoch, val_top1, is_final)
        if path is None:
            return None

        save_checkpoint(
            path, model, optimizer, scheduler, scaler,
            epoch=epoch, history=history, config=config,
            best_top1=self.best_top1,
            diagnostics_history=diagnostics_history,
        )

        # Also save as "best" if this is a new best
        if val_top1 >= self.best_top1:
            best_path = self.best_path()
            save_checkpoint(
                best_path, model, optimizer, scheduler, scaler,
                epoch=epoch, history=history, config=config,
                best_top1=self.best_top1,
                diagnostics_history=diagnostics_history,
            )

        return path
