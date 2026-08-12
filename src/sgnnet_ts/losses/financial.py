"""Differentiable financial losses.

All losses assume:
  pred:   [B, N_stocks] — predicted next-day % return per stock
  actual: [B, N_stocks] — actual next-day % return (as fraction, e.g. 0.01 = 1%)
  tc:     transaction cost as fraction (default 0.001 = 0.1% round-trip)
"""

import torch
import torch.nn.functional as F


def log_wealth_loss(pred: torch.Tensor, actual: torch.Tensor, tc: float = 0.001) -> torch.Tensor:
    """Negative mean log-wealth.

    Position: tanh(pred) maps prediction to [-1, +1] (long/short with continuous sizing).
    Wealth per step: 1 + position * actual_return - tc
    Objective: maximise sum of log-wealth → minimise negative mean.

    Gradient clipping (max_norm=1.0) required in the training loop — log can
    produce large gradients near position*actual ≈ -1.
    """
    position = torch.tanh(pred)           # [-1, +1]
    step_return = position * actual - tc  # [B, N_stocks]
    # Clamp to prevent log(0) or log(negative) from NaN
    wealth = torch.clamp(1.0 + step_return, min=1e-6)
    log_w = torch.log(wealth)
    return -log_w.mean()


def sharpe_loss(pred: torch.Tensor, actual: torch.Tensor, tc: float = 0.001,
                eps: float = 1e-8) -> torch.Tensor:
    """Negative Sharpe ratio loss (differentiable approximation).

    Useful as a regulariser alongside log_wealth_loss, or standalone.
    """
    position = torch.tanh(pred)
    step_return = (position * actual - tc).mean(dim=-1)  # [B]
    mu = step_return.mean()
    sigma = step_return.std() + eps
    return -(mu / sigma)


def mse_return_loss(pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """MSE on % return. Use for warmup (numerically stable)."""
    return F.mse_loss(pred, actual)


def directional_accuracy(pred: torch.Tensor, actual: torch.Tensor) -> float:
    """Fraction of correct sign predictions (non-differentiable, for logging only)."""
    with torch.no_grad():
        correct = (torch.sign(pred) == torch.sign(actual)).float()
        return correct.mean().item()


def compute_sharpe(daily_returns: torch.Tensor, eps: float = 1e-8) -> float:
    """Annualised Sharpe from a 1-D tensor of daily portfolio returns."""
    with torch.no_grad():
        mu = daily_returns.mean()
        sigma = daily_returns.std() + eps
        return (mu / sigma * (252 ** 0.5)).item()
