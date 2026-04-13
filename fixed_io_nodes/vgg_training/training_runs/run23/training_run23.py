"""
run23 — Re-run of run16 under new post-update LayerNorm (commit f11f768).

Goal: Measure effect of LN placement fix on the best known config.

LN change (run 17+):
  - Hardcoded mean-subtraction removed from update_activations (was γ=1, β=0 locked)
  - nn.LayerNorm moved from pre-update (source mag before aggregation) to post-update
    (new_mag immediately after update_activations, act_strength recomputed from LN'd mag)
  - γ, β now genuinely learnable and affect stored node state
  Full details: vgg_training/learnings/concepts/mag_normalization.md

Config identical to run16 (N=4146, T=4.0, 100% data, flat, no FFN, 40ep).
Comparison: run16 → 86.55% @ep40 (old pre-update LN).

Series:
  run23: N=4146, T=4.0, 100% data → primary validation at best config
  run18: N=4146, T=1.0, 100% data → check if T=1.0 starvation changes under new LN
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fixed_io_nodes.native import NativeNeurographLayer, NativeGNNOptimizer
from fixed_io_nodes.native.checkpoint import load_full_model, save_full_model
from fixed_io_nodes.main import load_config


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = THIS_DIR / "config.yaml"


def _save_training_state(optimizer, path, epoch, global_step):
    acc_states = []
    for acc in optimizer._accumulators:
        acc_states.append({
            "phase_grads": acc.phase_grads,
            "mag_grads": acc.mag_grads,
            "phase_grad_counts": acc.phase_grad_counts,
            "mag_grad_counts": acc.mag_grad_counts,
            "phase_exp_avg": acc.phase_exp_avg,
            "phase_exp_avg_sq": acc.phase_exp_avg_sq,
            "phase_state_steps": acc.phase_state_steps,
            "mag_exp_avg": acc.mag_exp_avg,
            "mag_exp_avg_sq": acc.mag_exp_avg_sq,
            "mag_state_steps": acc.mag_state_steps,
        })
    state = {
        "optimizer": optimizer._head_optimizer.state_dict(),
        "accumulators": acc_states,
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(state, str(path))


def _load_training_state(optimizer, path, device):
    state = torch.load(str(path), map_location=device, weights_only=False)
    optimizer._head_optimizer.load_state_dict(state["optimizer"])
    for acc, acc_state in zip(optimizer._accumulators, state["accumulators"]):
        acc.phase_grads = acc_state["phase_grads"].to(device)
        acc.mag_grads = acc_state["mag_grads"].to(device)
        acc.phase_grad_counts = acc_state["phase_grad_counts"].to(device)
        acc.mag_grad_counts = acc_state["mag_grad_counts"].to(device)
        acc.phase_exp_avg = acc_state["phase_exp_avg"].to(device)
        acc.phase_exp_avg_sq = acc_state["phase_exp_avg_sq"].to(device)
        acc.phase_state_steps = acc_state["phase_state_steps"].to(device)
        acc.mag_exp_avg = acc_state["mag_exp_avg"].to(device)
        acc.mag_exp_avg_sq = acc_state["mag_exp_avg_sq"].to(device)
        acc.mag_state_steps = acc_state["mag_state_steps"].to(device)
    return state.get("epoch", 0), state.get("global_step", 0)


def _save_scheduler(scheduler, path):
    torch.save(scheduler.state_dict(), str(path))


def _load_scheduler(scheduler, path, device):
    state = torch.load(str(path), map_location=device)
    scheduler.load_state_dict(state)


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _stratified_subsample(x, y, fraction=0.5, seed=42):
    """Subsample data with proportional class representation."""
    gen = torch.Generator().manual_seed(seed)
    classes = y.argmax(dim=1)
    keep = []
    for c in classes.unique():
        idx = (classes == c).nonzero(as_tuple=True)[0]
        n_keep = max(1, int(len(idx) * fraction))
        perm = torch.randperm(len(idx), generator=gen)[:n_keep]
        keep.append(idx[perm])
    keep = torch.cat(keep)
    shuffle = torch.randperm(len(keep), generator=gen)
    return x[keep[shuffle]], y[keep[shuffle]]


class CustomHybridModel(nn.Module):
    """
    run23 variant: no nn.Linear head.
    output_nodes=10; GNN act_strength values feed directly into CrossEntropyLoss.
    """
    def __init__(self, cfg):
        super().__init__()
        self.input_nodes = cfg["graph"]["input_nodes"]
        self.vector_dim = cfg["model"]["vector_dim"]
        self.output_nodes = cfg["graph"]["output_nodes"]

        assert self.input_nodes * self.vector_dim == 512 * 7 * 7, (
            "input_nodes * vector_dim must match VGG16 feature size (25088)"
        )
        assert self.output_nodes == 10, (
            f"run23 requires output_nodes=10, got {self.output_nodes}"
        )
        self.gnn = NativeNeurographLayer(cfg)

    def forward(self, x):
        batch_size = x.size(0)
        h = x.view(batch_size, self.input_nodes, self.vector_dim)
        # GNN returns (B, output_nodes=10) — used directly as class logits
        return self.gnn(h)


def main(config_path: str = None):
    cfg_path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH
    cfg = load_config(str(cfg_path))

    device = cfg.get("system", {}).get("device", "cpu")
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["lr"]

    # Resolve paths relative to vgg_training/ (parent of training_runs/)
    vgg_dir = THIS_DIR.parents[1]

    raw_log_dir = cfg.get("system", {}).get("tensorboard_dir", "training_runs/run23/tensorboard")
    log_dir = _resolve_path(raw_log_dir, vgg_dir)
    writer = SummaryWriter(log_dir=str(log_dir))

    raw_save_path = cfg.get("system", {}).get("weights_save_path", "training_runs/run23/run23_weights.pt")
    save_path = _resolve_path(raw_save_path, vgg_dir)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train = torch.load(str(_resolve_path("data/imagenette_train_data.pt", vgg_dir)))
    val = torch.load(str(_resolve_path("data/imagenette_val_data.pt", vgg_dir)))

    x_train = train["data"]
    y_train = train["label"]
    x_val = val["data"]
    y_val = val["label"]

    data_fraction = cfg.get("training", {}).get("data_fraction", 1.0)
    if data_fraction < 1.0:
        x_train, y_train = _stratified_subsample(x_train, y_train, fraction=data_fraction, seed=42)
        print(f"Using {data_fraction*100:.0f}% of training data: {len(x_train)} samples")
    else:
        print(f"Using 100% of training data: {len(x_train)} samples")

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CustomHybridModel(cfg).to(device)
    accumulation_steps = cfg.get("training", {}).get("accumulation_steps", batch_size)
    optimizer = NativeGNNOptimizer(model, lr=lr, accumulation_steps=accumulation_steps)

    optimizer_save_path = save_path.with_name(save_path.stem + "_optimizer" + save_path.suffix)

    scheduler_cfg = cfg.get("training", {})
    use_scheduler = "lr_decay_factor" in scheduler_cfg
    scheduler = None
    scheduler_save_path = None
    if use_scheduler:
        patience = scheduler_cfg.get("plateau_patience", 5)
        lr_factor = scheduler_cfg["lr_decay_factor"]
        min_lr = float(scheduler_cfg.get("min_lr", 1e-7))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=patience, min_lr=min_lr,
        )
        raw_sched_path = cfg.get("system", {}).get("scheduler_save_path")
        if raw_sched_path:
            scheduler_save_path = _resolve_path(raw_sched_path, vgg_dir)
        else:
            scheduler_save_path = save_path.with_name(save_path.stem + "_scheduler" + save_path.suffix)

    start_epoch = 0
    global_step = 0
    if save_path.exists():
        load_full_model(model, str(save_path), map_location=device)
        print(f"Loaded checkpoint from {save_path}")
        if optimizer_save_path.exists():
            loaded_epoch, loaded_step = _load_training_state(optimizer, optimizer_save_path, device)
            start_epoch = loaded_epoch + 1
            global_step = loaded_step
            print(f"Resuming from epoch {start_epoch}, step {global_step}")
        if scheduler is not None and scheduler_save_path.exists():
            _load_scheduler(scheduler, scheduler_save_path, device)

    criterion = nn.CrossEntropyLoss()
    total_steps = epochs * len(train_loader)

    print("Training started (run23 — N=4146, T=4.0, 100% data, no FFN, post-update LN)")
    for epoch in range(start_epoch, epochs):
        model.train()
        correct = 0
        total = 0
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            model.gnn.set_training_progress(global_step, total_steps)
            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            writer.add_scalar("Training/Loss", loss.item(), global_step)
            preds = logits.argmax(dim=1)
            ground_truth = y.argmax(dim=1)
            correct += (preds == ground_truth).sum().item()
            total += y.size(0)
            epoch_loss += loss.item() * y.size(0)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            global_step += 1

        avg_train_loss = epoch_loss / total
        train_acc = 100.0 * correct / total
        writer.add_scalar("Training/Accuracy", train_acc, epoch)
        writer.add_scalar("Training/EpochLoss", avg_train_loss, epoch)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * y.size(0)
                preds = logits.argmax(dim=1)
                ground_truth = y.argmax(dim=1)
                val_correct += (preds == ground_truth).sum().item()
                val_total += y.size(0)

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/Accuracy", val_acc, epoch)
        print(
            f"Epoch {epoch + 1} Summary: Train Loss: {avg_train_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}%, "
            f"Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}"
        )

        if scheduler is not None:
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < old_lr:
                print(f"LR decayed: {old_lr:.2e} -> {new_lr:.2e}")
            writer.add_scalar("Training/LR", new_lr, epoch)

        save_full_model(model, str(save_path))
        _save_training_state(optimizer, optimizer_save_path, epoch, global_step)
        print(f"Saved checkpoint to {save_path}")
        if scheduler is not None:
            _save_scheduler(scheduler, scheduler_save_path)

    print(f"Training complete. Final model at {save_path}")
    writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train run23 — N=4146, T=4.0, 100% data, no FFN, post-update LN")
    parser.add_argument("config", nargs="?", default=None, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
