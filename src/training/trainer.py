"""Shared training loop for SGNNET_Wave experiments.

Handles FP16 mixed precision on MPS, position clamping, gradient zeroing.
Loss per TRAIN-01: total = KL + lambda_safety * safety_valve + lambda_lb * load_balance.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.sgnnet.losses import load_balance_loss, safety_valve_loss


# -------------------------------------------------------------------
# FP16 support detection (D-09)
# -------------------------------------------------------------------

def _check_grad_scaler_support() -> bool:
    """Return True if PyTorch >= 2.3 (MPS GradScaler safe)."""
    parts = torch.__version__.split(".")[:2]
    major, minor = int(parts[0]), int(parts[1])
    return major > 2 or (major == 2 and minor >= 3)


# -------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------

class Trainer:
    """Shared training loop for SGNNET_Wave experiments.

    Handles FP16 mixed precision on MPS, position clamping (TRAIN-03),
    gradient zeroing for input neurons (TRAIN-02 -- by model design,
    only hidden+output positions are nn.Parameters).

    Parameters
    ----------
    model        : SGNNET_Wave instance
    train_loader : yields (features, soft_labels, labels)
    val_loader   : yields (features, soft_labels, labels)
    lr_wpos      : learning rate for W_pos
    lr_wphase    : learning rate for W_phase (Stage C only)
    lambda_safety: weight for safety valve loss
    lambda_lb    : weight for load balance loss
    box_size     : confining hypercube side length
    device       : torch device string
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        lr_wpos: float = 1e-3,
        lr_wphase: float | None = None,
        lambda_safety: float = 0.5,
        lambda_lb: float = 0.01,
        box_size: float = 1.0,
        device: str = "mps",
    ):
        self.model = model.to(device)
        self.device = device
        self.box_size = box_size
        self.lambda_safety = lambda_safety
        self.lambda_lb = lambda_lb
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Separate param groups: W_pos always, W_phase if present
        param_groups = [{"params": [model.W_pos], "lr": lr_wpos}]
        if model.W_phase is not None and lr_wphase is not None:
            param_groups.append({"params": [model.W_phase], "lr": lr_wphase})
        self.optimizer = torch.optim.Adam(param_groups)

        # FP16 GradScaler setup (D-09)
        self.use_grad_scaler = _check_grad_scaler_support()
        if self.use_grad_scaler:
            self.scaler = torch.amp.GradScaler(device)
        else:
            self.scaler = None

    # ---------------------------------------------------------------
    # Single training epoch
    # ---------------------------------------------------------------

    def train_epoch(self) -> dict:
        """Run one training epoch. Returns metric dict."""
        self.model.train()
        total_loss_sum = 0.0
        task_loss_sum = 0.0
        safety_loss_sum = 0.0
        lb_loss_sum = 0.0
        n_batches = 0

        for features, soft_labels, _labels in self.train_loader:
            features = features.to(self.device)
            soft_labels = soft_labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.autocast(self.device, dtype=torch.float16):
                scores = self.model(features)
                task_loss = F.kl_div(
                    F.log_softmax(scores, dim=-1),
                    soft_labels,
                    reduction="batchmean",
                )
                safety = safety_valve_loss(self.model.W_pos, self.box_size)
                lb_loss = load_balance_loss(scores.abs().sum(dim=0))
                loss = (
                    task_loss
                    + self.lambda_safety * safety
                    + self.lambda_lb * lb_loss
                )

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Position clamping (TRAIN-03)
            with torch.no_grad():
                self.model.W_pos.clamp_(0, self.box_size)

            total_loss_sum += loss.item()
            task_loss_sum += task_loss.item()
            safety_loss_sum += safety.item()
            lb_loss_sum += lb_loss.item()
            n_batches += 1

        n = max(n_batches, 1)
        return {
            "train_loss": total_loss_sum / n,
            "task_loss": task_loss_sum / n,
            "safety_loss": safety_loss_sum / n,
            "lb_loss": lb_loss_sum / n,
        }

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------

    def evaluate(self) -> dict:
        """Evaluate on validation set. Returns metric dict."""
        self.model.eval()
        all_scores = []
        all_labels = []
        val_loss_sum = 0.0
        n_batches = 0

        with torch.no_grad():
            for features, soft_labels, labels in self.val_loader:
                features = features.to(self.device)
                soft_labels = soft_labels.to(self.device)

                with torch.autocast(self.device, dtype=torch.float16):
                    scores = self.model(features)
                    task_loss = F.kl_div(
                        F.log_softmax(scores, dim=-1),
                        soft_labels,
                        reduction="batchmean",
                    )

                val_loss_sum += task_loss.item()
                all_scores.append(scores.cpu())
                all_labels.append(labels)
                n_batches += 1

        n = max(n_batches, 1)
        return {
            "val_loss": val_loss_sum / n,
            "scores": torch.cat(all_scores, dim=0),
            "labels": torch.cat(all_labels, dim=0),
        }

    # ---------------------------------------------------------------
    # Multi-epoch training
    # ---------------------------------------------------------------

    def train(
        self,
        n_epochs: int,
        log_fn: callable | None = None,
    ) -> list[dict]:
        """Train for n_epochs. Returns list of per-epoch metrics."""
        history: list[dict] = []

        for epoch in range(n_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate()

            combined = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "nan_detected": False,
            }

            if math.isnan(train_metrics["train_loss"]):
                combined["nan_detected"] = True
                history.append(combined)
                if log_fn:
                    log_fn(combined)
                break

            history.append(combined)
            if log_fn:
                log_fn(combined)

        return history
