import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fixed_io_nodes.distributed.gnn_optimizer import GNNAdam
from fixed_io_nodes.distributed.layer import DistributedNeurographLayer
from fixed_io_nodes.distributed.checkpoint import load_full_model, save_full_model
from fixed_io_nodes.main import load_config


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = THIS_DIR / "training_runs" / "run1" / "config.yaml"


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
        self.gnn = DistributedNeurographLayer(cfg)
        self.out = nn.Linear(self.output_nodes, 10)

    def forward(self, x):
        batch_size = x.size(0)
        h = x.view(batch_size, self.input_nodes, self.vector_dim)
        gnn_out = self.gnn(h)
        return self.out(gnn_out)


def main(config_path: str | None = None):
    cfg_path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH
    cfg = load_config(str(cfg_path))

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
    optimizer = GNNAdam(model, lr=lr)

    global_step = 0
    if save_path.exists():
        load_full_model(model, str(save_path), map_location=device)
        print(f"Loaded checkpoint from {save_path}")

    criterion = nn.CrossEntropyLoss()
    total_steps = epochs * len(train_loader)

    print("Training started")
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

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
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            global_step += 1

        train_acc = 100.0 * correct / total
        writer.add_scalar("Training/Accuracy", train_acc, epoch)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if hasattr(model.gnn, "_pool") and model.gnn._pool is not None:
                    model.gnn._pool.reset_workers()

                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * y.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/Accuracy", val_acc, epoch)
        print(
            f"Epoch {epoch + 1} Summary: Train Acc: {train_acc:.2f}%, "
            f"Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}"
        )

    save_full_model(model, str(save_path))
    print(f"Saved final model to {save_path}")
    writer.close()


if __name__ == "__main__":
    cli_config_path = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.startswith("--config="):
            cli_config_path = arg.split("=", 1)[1]
        else:
            cli_config_path = arg
    main(cli_config_path)

