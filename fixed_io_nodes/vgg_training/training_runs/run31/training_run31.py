"""
run31 — Uniform routing at C=50: cardinality sweep with new routing.

run29: uniform C=200 → 89.15% (+2.30pp vs softmax). run30: uniform C=2 → 18.39% (worse).
Now: C=50 — where softmax (run20) scored 71.64%. How much does uniform routing help?

Technique: Same monkey-patch as run29. core/ and native/ UNTOUCHED.
"""
import os
import sys
from pathlib import Path

# Add project root to path (same as run17)
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# UNIFORM ROUTING: drop-in replacement for core.custom_functions.update_activations
# ---------------------------------------------------------------------------
_EPSILON = 1e-8

def update_activations_uniform(
    phase_activations: torch.Tensor,
    mag_activations: torch.Tensor,
    phase_weights: torch.Tensor,
    mag_weights: torch.Tensor,
    activation_strengths: torch.Tensor,
    edge_index: torch.Tensor,
    weight_real: torch.Tensor = None,
    weight_imag: torch.Tensor = None,
    all_destinations: bool = False,
    temperature: float = 1.0,  # ignored — kept for API compatibility
):
    """Identical to update_activations but uses 1/degree instead of softmax.

    Removes: scatter_reduce_(amax), exp, temperature division.
    Replaces with: degree count, 1/degree uniform weight.
    All complex-number math (exp(mag), cos/sin, scatter_add, atan2, log) unchanged.
    """
    source, dest = edge_index[0], edge_index[1]

    source_phase_activations = phase_activations[source]
    source_mag_activations = mag_activations[source]

    # --- CHANGED: Uniform routing (1/degree) instead of softmax ---
    # Count incoming edges per destination
    ones = torch.ones(source.shape[0], device=source.device, dtype=phase_activations.dtype)
    degree = torch.zeros(phase_activations.shape[0], device=source.device,
                         dtype=phase_activations.dtype).scatter_add_(0, dest, ones)
    routing_weights = (1.0 / (degree[dest] + _EPSILON)).unsqueeze(-1)
    # --- END CHANGE ---

    # Complex superposition (identical to original)
    weighted_source_mag_activation = routing_weights * torch.exp(source_mag_activations)
    source_real = weighted_source_mag_activation * torch.cos(source_phase_activations)
    source_imaginary = weighted_source_mag_activation * torch.sin(source_phase_activations)

    dest_real_input = torch.zeros_like(phase_weights).scatter_add_(
        0, dest.unsqueeze(-1).expand_as(source_real), source_real)
    dest_imaginary_input = torch.zeros_like(mag_weights).scatter_add_(
        0, dest.unsqueeze(-1).expand_as(source_imaginary), source_imaginary)

    if weight_real is None:
        dest_real_weight = mag_weights * torch.cos(phase_weights)
        dest_imaginary_weight = mag_weights * torch.sin(phase_weights)
    else:
        dest_real_weight = weight_real
        dest_imaginary_weight = weight_imag

    dest_real_output = (dest_real_input * dest_real_weight
                        - dest_imaginary_input * dest_imaginary_weight)
    dest_imaginary_output = (dest_real_input * dest_imaginary_weight
                             + dest_imaginary_input * dest_real_weight)

    new_phase = torch.atan2(dest_imaginary_output, dest_real_output + _EPSILON)
    new_mag = 0.5 * torch.log(
        dest_real_output ** 2 + dest_imaginary_output ** 2 + _EPSILON)
    new_activation_strength = dest_real_output.sum(dim=-1)

    if all_destinations:
        return new_phase, new_mag, new_activation_strength

    mask_1d = torch.zeros(phase_activations.shape[0], dtype=torch.bool,
                          device=phase_activations.device)
    mask_1d[dest] = True
    mask_2d = mask_1d.unsqueeze(-1)

    phase_activations = torch.where(mask_2d, new_phase, phase_activations)
    mag_activations = torch.where(mask_2d, new_mag, mag_activations)
    activation_strengths = torch.where(mask_1d, new_activation_strength, activation_strengths)

    return phase_activations, mag_activations, activation_strengths


# ---------------------------------------------------------------------------
# MONKEY-PATCH: Replace update_activations in native.layer BEFORE import
# ---------------------------------------------------------------------------
# Import the layer module so the binding exists, then replace it.
import fixed_io_nodes.native.layer as _layer_module
_layer_module.update_activations = update_activations_uniform

# Now import everything else (NativeNeurographLayer will use our patched function)
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


class CustomHybridModel(nn.Module):
    """run29: no nn.Linear head. GNN act_strength as logits (same as run17)."""
    def __init__(self, cfg):
        super().__init__()
        self.input_nodes = cfg["graph"]["input_nodes"]
        self.vector_dim = cfg["model"]["vector_dim"]
        self.output_nodes = cfg["graph"]["output_nodes"]

        assert self.input_nodes * self.vector_dim == 512 * 7 * 7
        assert self.output_nodes == 10
        self.gnn = NativeNeurographLayer(cfg)

    def forward(self, x):
        batch_size = x.size(0)
        h = x.view(batch_size, self.input_nodes, self.vector_dim)
        return self.gnn(h)


def main(config_path: str = None):
    cfg_path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH
    cfg = load_config(str(cfg_path))

    device = cfg.get("system", {}).get("device", "cpu")
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["lr"]

    vgg_dir = THIS_DIR.parents[1]

    raw_log_dir = cfg.get("system", {}).get("tensorboard_dir", "training_runs/run29/tensorboard")
    log_dir = _resolve_path(raw_log_dir, vgg_dir)
    writer = SummaryWriter(log_dir=str(log_dir))

    raw_save_path = cfg.get("system", {}).get("weights_save_path", "training_runs/run29/run29_weights.pt")
    save_path = _resolve_path(raw_save_path, vgg_dir)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train = torch.load(str(_resolve_path("data/imagenette_train_data.pt", vgg_dir)))
    val = torch.load(str(_resolve_path("data/imagenette_val_data.pt", vgg_dir)))

    x_train = train["data"]
    y_train = train["label"]
    x_val = val["data"]
    y_val = val["label"]
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

    print("Training started (run31 -- UNIFORM ROUTING, N=4146, C=50, no FFN)")
    print("  Routing: 1/degree (uniform) instead of softmax")
    print("  C=50 — moderate sparsity, 4x FLOPs savings vs run17")
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
    parser = argparse.ArgumentParser(description="Train run29 — UNIFORM routing ablation vs run17")
    parser.add_argument("config", nargs="?", default=None, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
