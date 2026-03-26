import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fixed_io_nodes.native import NativeNeurographLayer, NativeGNNOptimizer
from fixed_io_nodes.native.checkpoint import load_full_model, save_full_model
from fixed_io_nodes.main import load_config


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = THIS_DIR / "training_runs" / "run1" / "config.yaml"


def _save_optimizer(optimizer, path):
    state = {"head_optimizer": optimizer._head_optimizer.state_dict()}
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
    state["accumulators"] = acc_states
    torch.save(state, str(path))


def _load_optimizer(optimizer, path, device):
    state = torch.load(str(path), map_location=device)
    optimizer._head_optimizer.load_state_dict(state["head_optimizer"])
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


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


class CustomHybridModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_nodes = cfg["graph"]["input_nodes"]
        self.vector_dim = cfg["model"]["vector_dim"]
        self.output_nodes = cfg["graph"]["output_nodes"]

        assert self.input_nodes * self.vector_dim == 512 * 7 * 7, (
            "input_nodes * vector_dim must match VGG16 feature size (25088)"
        )
        use_ln = cfg.get("system", {}).get("use_layer_norm", True)
        self.gnn = NativeNeurographLayer(cfg, use_layer_norm=use_ln)
        self.out = nn.Linear(self.output_nodes, 10)

    def forward(self, x):
        batch_size = x.size(0)
        h = x.view(batch_size, self.input_nodes, self.vector_dim)
        gnn_out = self.gnn(h)
        return self.out(gnn_out)


def main(config_path: str = None, layernorm_override: bool = None):
    cfg_path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH
    cfg = load_config(str(cfg_path))
    if layernorm_override is not None:
        cfg.setdefault("system", {})["use_layer_norm"] = layernorm_override

    device = cfg.get("system", {}).get("device", "cpu")
    epochs = cfg.get("training", {}).get("epochs", 10)
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    lr = cfg.get("training", {}).get("lr", 1e-3)

    raw_log_dir = cfg.get("system", {}).get(
        "tensorboard_dir", "training_runs/run1/tensorboard"
    )
    log_dir = _resolve_path(raw_log_dir, THIS_DIR)
    writer = SummaryWriter(log_dir=str(log_dir))

    raw_save_path = cfg.get("system", {}).get(
        "weights_save_path", "training_runs/run1/custom_model_weights.pt"
    )
    save_path = _resolve_path(raw_save_path, THIS_DIR)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train = torch.load(str(_resolve_path("data/imagenette_train_data.pt", THIS_DIR)))
    val = torch.load(str(_resolve_path("data/imagenette_val_data.pt", THIS_DIR)))

    x_train = train["data"]
    y_train = train["label"]
    x_val = val["data"]
    y_val = val["label"]

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CustomHybridModel(cfg).to(device)
    accumulation_steps = cfg.get("training", {}).get("accumulation_steps", batch_size)
    optimizer = NativeGNNOptimizer(model, lr=lr, accumulation_steps=accumulation_steps)

    optimizer_save_path = save_path.with_name(
        save_path.stem + "_optimizer" + save_path.suffix
    )

    global_step = 0
    if save_path.exists():
        load_full_model(model, str(save_path), map_location=device)
        print(f"Loaded checkpoint from {save_path}")
        if optimizer_save_path.exists():
            _load_optimizer(optimizer, optimizer_save_path, device)
            print(f"Loaded optimizer state from {optimizer_save_path}")

    criterion = nn.CrossEntropyLoss()
    total_steps = epochs * len(train_loader)

    patience = cfg.get("training", {}).get("plateau_patience", 5)
    min_delta = cfg.get("training", {}).get("plateau_min_delta", 1e-4)
    best_train_loss = float("inf")
    plateau_counter = 0

    print("Training started")
    for epoch in range(epochs):
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
            # print(logits.shape)
            # print(y.shape)
            optimizer.step()

            writer.add_scalar("Training/Loss", loss.item(), global_step)
            preds = logits.argmax(dim=1)
            ground_truth = y.argmax(dim=1) #We need to do this since `y` is also a soft-distribution, not one-hot
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

        save_full_model(model, str(save_path))
        _save_optimizer(optimizer, optimizer_save_path)
        print(f"Saved checkpoint to {save_path}")
        print(f"Saved optimizer state to {optimizer_save_path}")

        if best_train_loss - avg_train_loss > min_delta:
            best_train_loss = avg_train_loss
            plateau_counter = 0
        else:
            plateau_counter += 1
            print(
                f"Training loss plateau: no improvement for "
                f"{plateau_counter}/{patience} epochs"
            )
            if plateau_counter >= patience:
                print("Early stopping: training loss has plateaued.")
                break

    print(f"Training complete. Final model at {save_path}")
    writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CustomHybridModel")
    parser.add_argument("config", nargs="?", default=None, help="Path to config YAML")
    parser.add_argument("--config", dest="config_flag", default=None, help="Path to config YAML (alternative)")
    parser.add_argument(
        "--layernorm",
        type=lambda v: v.lower() not in ("false", "0", "no"),
        default=None,
        metavar="BOOL",
        help="Enable magnitude LayerNorm (default: use config value). Pass false/0/no to disable.",
    )
    args = parser.parse_args()
    resolved_config = args.config_flag or args.config
    main(resolved_config, layernorm_override=args.layernorm)

