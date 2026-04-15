"""Shared training loop for SGNNET_Wave experiments.

Handles FP16 mixed precision on MPS, position clamping, gradient zeroing.
Loss per TRAIN-01: total = KL + lambda_safety * safety_valve + lambda_lb * load_balance.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.sgnnet.losses import load_balance_loss, safety_valve_loss
from src.training.callbacks import EarlyStopping


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
    """Training loop: FP16 AMP, AdamW, ReduceLROnPlateau, early stopping.

    Monitors train_loss for both scheduler and early stopping
    (val loss unreliable on small GA partial-data subsets).
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
        use_amp: bool = True,
        sched_type: str = "plateau",   # "plateau" | "cosine" | "warm_restarts"
        sched_patience: int = 10,
        sched_factor: float = 0.5,
        sched_cosine_T: int = 150,     # T_max for cosine annealing (= n_epochs)
        sched_T0: int = 10,            # T_0 for warm restarts (cycle length in epochs)
        sched_T_mult: int = 1,         # T_mult for warm restarts (1=constant, 2=doubling)
        min_lr: float = 1e-7,
        early_stop_patience: int = 25,
        early_stop_delta: float = 1e-4,
        grad_clip_norm: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.box_size = box_size
        self.lambda_safety = lambda_safety
        self.lambda_lb = lambda_lb
        self.grad_clip_norm = grad_clip_norm
        self.train_loader = train_loader
        self.val_loader = val_loader

        # W_pos: no weight decay — positions must explore [0,1]^D freely;
        # decay pulls coords toward 0, collapsing geometric spread.
        # W_phase (Stage C only): standard weight decay is fine.
        param_groups = [{"params": [model.W_pos], "lr": lr_wpos, "weight_decay": 0.0}]
        if model.W_phase is not None and lr_wphase is not None:
            param_groups.append({"params": [model.W_phase], "lr": lr_wphase})
        self.optimizer = torch.optim.AdamW(param_groups)

        # FP16 GradScaler setup (D-09); disabled when use_amp=False or on CPU/MPS.
        # AMP on CPU uses bfloat16 which is slower than float32 for small models.
        # MPS: GradScaler("mps") has a known hook-registration bug in PyTorch —
        # scaler.step() raises "No inf checks recorded" because MPS float16
        # autocast doesn't trigger the GradScaler backward hooks. Autocast is
        # still enabled on MPS (gives ~10% speedup via bf16 ops) but without
        # gradient scaling, which is not needed for bf16 anyway.
        _device_str = str(device)
        _is_cpu = _device_str == "cpu" or "cpu" in _device_str
        _is_mps = "mps" in _device_str
        self.use_amp = use_amp and not _is_cpu
        # GradScaler: CUDA only. MPS scaler hooks are unreliable.
        self.use_grad_scaler = self.use_amp and not _is_mps and _check_grad_scaler_support()
        if self.use_grad_scaler:
            self.scaler = torch.amp.GradScaler(device)
        else:
            self.scaler = None

        # LR scheduler: plateau (adaptive) | cosine (single decay) | warm_restarts (cyclic)
        self.sched_type = sched_type
        if sched_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=sched_cosine_T, eta_min=min_lr,
            )
        elif sched_type == "warm_restarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=sched_T0, T_mult=sched_T_mult, eta_min=min_lr,
            )
        elif sched_type == "none":
            self.scheduler = None   # constant LR — no decay
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=sched_factor,
                patience=sched_patience, min_lr=min_lr,
            )

        # Early stopping: monitors train_loss
        self.early_stopping = EarlyStopping(
            patience=early_stop_patience, min_delta=early_stop_delta,
        )

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

            _amp_ctx = (
                torch.autocast(str(self.device).split(":")[0], dtype=torch.float16)
                if self.use_amp
                else torch.autocast("cpu", enabled=False)
            )
            with _amp_ctx:
                scores = self.model(features)
                task_loss = F.kl_div(
                    F.log_softmax(scores, dim=-1),
                    soft_labels,
                    reduction="batchmean",
                )
                if self.lambda_safety > 0:
                    safety = safety_valve_loss(
                        self.model.W_pos, self.box_size, task_loss=task_loss
                    )
                else:
                    safety = torch.tensor(0.0, device=scores.device)
                if self.lambda_lb > 0:
                    lb_loss = load_balance_loss(scores.abs().sum(dim=0))
                else:
                    lb_loss = torch.tensor(0.0, device=scores.device)
                loss = (
                    task_loss
                    + self.lambda_safety * safety
                    + self.lambda_lb * lb_loss
                )

            # Only clip optimizer params — NOT all model params.
            # theta and W_phase are not in the optimizer so optimizer.zero_grad()
            # never clears their gradients; they accumulate across batches and
            # would dominate the norm, clipping W_pos gradient to near-zero.
            _opt_params = [p for g in self.optimizer.param_groups for p in g["params"]]

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                # Unscale before clipping so clip threshold is in real gradient units
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(_opt_params, self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_opt_params, self.grad_clip_norm)
                self.optimizer.step()

            # Sub-epoch phase graph rebuild (models that expose tick_step)
            if hasattr(self.model, "tick_step"):
                self.model.tick_step()

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

                _amp_ctx = (
                    torch.autocast(str(self.device).split(":")[0], dtype=torch.float16)
                    if self.use_amp
                    else torch.autocast("cpu", enabled=False)
                )
                with _amp_ctx:
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
        all_scores_cat = torch.cat(all_scores, dim=0)
        all_labels_cat = torch.cat(all_labels, dim=0)
        preds = all_scores_cat.argmax(dim=-1)
        val_top1 = (preds == all_labels_cat).float().mean().item()
        return {
            "val_loss": val_loss_sum / n,
            "val_top1": val_top1,
        }

    # ---------------------------------------------------------------
    # Multi-epoch training
    # ---------------------------------------------------------------

    def current_lr(self) -> float:
        """Return current learning rate for W_pos param group."""
        return self.optimizer.param_groups[0]["lr"]

    def train(
        self,
        n_epochs: int,
        log_fn: callable | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_prefix: str = "checkpoint",
        checkpoint_every: int = 25,
        resume_from: str | None = None,
        config: dict | None = None,
    ) -> list[dict]:
        """Train for n_epochs (may stop early). Returns per-epoch history.

        Args:
            checkpoint_dir: if set, saves checkpoints (best + periodic)
            checkpoint_prefix: filename prefix for checkpoints
            checkpoint_every: save every N epochs (0 = only best + final)
            resume_from: path to checkpoint to resume from
            config: experiment config dict to store in checkpoint
        """
        from src.training.checkpoint import CheckpointPolicy, load_checkpoint

        history: list[dict] = []
        start_epoch = 0

        # Resume from checkpoint if provided
        if resume_from is not None:
            state = load_checkpoint(
                resume_from, self.model, self.optimizer,
                self.scheduler, self.scaler, device=self.device,
            )
            start_epoch = state["epoch"] + 1
            history = state["history"]
            print(f"  Resumed from {resume_from} at epoch {start_epoch}")

        # Checkpoint policy
        ckpt_policy = None
        if checkpoint_dir is not None:
            ckpt_policy = CheckpointPolicy(
                checkpoint_dir, prefix=checkpoint_prefix,
                save_every=checkpoint_every,
            )
            # Inherit best_top1 from history
            if history:
                ckpt_policy.best_top1 = max(
                    h.get("val_top1", 0.0) for h in history)

        for epoch in range(start_epoch, n_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate()
            train_loss = train_metrics["train_loss"]

            combined = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "lr": self.current_lr(),
                "nan_detected": False,
                "stopped_early": False,
            }

            if math.isnan(train_loss):
                combined["nan_detected"] = True
                history.append(combined)
                if log_fn:
                    log_fn(combined)
                break

            # Topology reconnection (SGNNET_ProximityWave and similar models
            # that expose tick_epoch() to rebuild their conn_hh from W_pos)
            if hasattr(self.model, "tick_epoch"):
                self.model.tick_epoch()

            # LR scheduler step (no-op when sched_type="none")
            if self.scheduler is not None:
                if self.sched_type in ("cosine", "warm_restarts"):
                    self.scheduler.step()
                else:
                    self.scheduler.step(train_loss)

            history.append(combined)
            if log_fn:
                log_fn(combined)

            # Progress print every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                task  = train_metrics.get("task_loss", train_loss)
                safe  = train_metrics.get("safety_loss", 0.0)
                top1  = val_metrics.get("val_top1", 0.0)
                print(f"  e{epoch+1:3d}  loss={train_loss:.4f}  "
                      f"task={task:.4f}  safety={safe:.4f}  "
                      f"top1={top1:.4f}  lr={self.current_lr():.2e}")

            # Checkpoint: save on best, periodic, and final
            if ckpt_policy is not None:
                val_top1 = val_metrics.get("val_top1", 0.0)
                is_final = (epoch == n_epochs - 1)
                saved = ckpt_policy.step(
                    epoch, val_top1, is_final,
                    self.model, self.optimizer, self.scheduler, self.scaler,
                    history=history, config=config,
                )
                if saved:
                    print(f"  ��� Saved checkpoint → {saved}")

            # Early stopping check (monitors train_loss)
            if self.early_stopping.step(train_loss, self.model):
                combined["stopped_early"] = True
                print(f"  Early stop at epoch {epoch} — best train_loss={self.early_stopping.best_loss:.4f}")
                # Save final checkpoint on early stop
                if ckpt_policy is not None:
                    ckpt_policy.step(
                        epoch, val_metrics.get("val_top1", 0.0), True,
                        self.model, self.optimizer, self.scheduler, self.scaler,
                        history=history, config=config,
                    )
                break

        return history
