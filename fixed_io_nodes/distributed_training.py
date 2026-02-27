import csv
import os
import time
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import torch.multiprocessing as mp

from distributed import DistributedNeurographLayer, GNNAdam
from main import load_config, load_iris_dataset, load_mnist_dataset

CONFIG_PATH = "training_runs/distributed_test/distributed.yaml"


class IrisGNNModel(nn.Module):
    """MLP(4 -> input_nodes*vector_dim) + tanh + reshape + GNN. Config-driven input_nodes, vector_dim."""

    def __init__(self, cfg):
        super().__init__()
        input_nodes = cfg["graph"]["input_nodes"]
        vector_dim = cfg["model"]["vector_dim"]
        self.input_nodes = input_nodes
        self.vector_dim = vector_dim
        self.linear = nn.Linear(4, input_nodes * vector_dim)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(cfg["graph"]["output_nodes"], 3)
        self.gnn = DistributedNeurographLayer(cfg)

    def forward(self, x):
        B = x.size(0)
        x = x.squeeze(1)
        h = self.tanh(self.linear(x))   
        h = h.view(B, self.input_nodes, self.vector_dim)
        return self.out(self.gnn(h))

class MNISTGNNModel(nn.Module):
    """MLP(784 -> input_nodes*vector_dim) + tanh + reshape + GNN. Config-driven input_nodes, vector_dim."""
    def __init__(self, cfg):
        super().__init__()
        input_nodes = cfg["graph"]["input_nodes"]
        vector_dim = cfg["model"]["vector_dim"]
        self.input_nodes = input_nodes
        self.vector_dim = vector_dim
        self.linear = nn.Linear(14*14, input_nodes * vector_dim)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(cfg["graph"]["output_nodes"], 10)
        self.gnn = DistributedNeurographLayer(cfg)

    def forward(self, x):
        B = x.size(0)
        x = x.squeeze(1)
        h = self.tanh(self.linear(x))
        h = h.view(B, self.input_nodes, self.vector_dim)
        return self.out(self.gnn(h))

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    cfg = load_config(CONFIG_PATH)
    torch.manual_seed(cfg["system"].get("random_seed", 42))

    log_cfg = cfg["system"]["logging"]
    log_path = log_cfg["log_path"]
    verbose = log_cfg.get("verbose", False)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_per_sample = log_cfg.get("log_per_sample", False)
    tensorboard_dir = log_cfg.get("tensorboard_dir")
    if tensorboard_dir is None:
        tensorboard_dir = os.path.join(os.path.dirname(log_path), "tensorboard")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    tensorboard_run_dir = os.path.join(tensorboard_dir, run_id)
    writer = SummaryWriter(log_dir=tensorboard_run_dir)
    if verbose:
        print(f"TensorBoard: {tensorboard_run_dir}")

    # Build model (layer owns gradient_sink, node_store, and accumulator; GNNAdam discovers them)
    model = IrisGNNModel(cfg)
    # model = MNISTGNNModel(cfg)
    device = cfg["system"]["device"]
    model = model.to(device)
    print("Initialized model")

    # Load weights if path exists (same path used for saving)
    load_path = cfg.get("system", {}).get("weights_save_path")
    if load_path and os.path.exists(load_path):
        model.gnn._node_store.load_weights(load_path)
        print(f"Loaded weights from {load_path}")

    optimizer = GNNAdam(
        model,
        lr=cfg["training"]["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    print("Initialized optimizer")

    dataset = load_iris_dataset()
    # dataset = load_mnist_dataset()
    batch_size = cfg["training"]["accumulation_steps"]
    validation_fraction = cfg["training"].get("validation_fraction", 0.0)
    seed = cfg["system"].get("random_seed", 42)
    if validation_fraction > 0:
        n = len(dataset)
        indices = list(range(n))
        labels = [dataset[i][1].item() for i in range(n)]
        train_idx, val_idx = train_test_split(
            indices, test_size=validation_fraction, random_state=seed, stratify=labels
        )
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = None
    print("Initialized dataset and dataloaders")

    criterion = nn.CrossEntropyLoss()
    criterion_noreduce = nn.CrossEntropyLoss(reduction="none") if log_per_sample else None
    epochs = cfg["training"]["epochs"]

    log_fields = ["iteration", "loss"]
    global_step = 0
    total_steps = epochs * len(train_loader)
    gnn_pool = getattr(model.gnn, "_pool", None)
    with open(log_path, "w", newline="") as log_f:
        log_writer = csv.DictWriter(log_f, fieldnames=log_fields)
        log_writer.writeheader()
        log_f.flush()
        print("Training started")
    
        for epoch in range(epochs):

            correct, total = 0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", total=len(train_loader))
            for x, y in pbar:
                if gnn_pool is not None:
                    gnn_pool.reset_workers()
                model.gnn.set_training_progress(global_step, total_steps)
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                log_writer.writerow({"iteration": global_step, "loss": loss.item()})
                log_f.flush()
                writer.add_scalar("Training/Loss", loss.item(), global_step)
                global_step += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}", "batch": global_step})

                if verbose:
                    if log_per_sample and criterion_noreduce is not None:
                        per_sample = criterion_noreduce(logits, y)
                        for b in range(y.size(0)):
                            print(f"Target {y[b].item()}: Loss: {per_sample[b].item():.4f}")
                    else:
                        print(f"Loss: {loss.item():.4f}")
                
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
            train_acc = 100.0 * correct / total
                
            pbar.close()

            val_acc = None
            if val_loader is not None:
                with torch.no_grad():
                    correct, total = 0, 0
                    for x, y in val_loader:
                        if gnn_pool is not None:
                            gnn_pool.reset_workers()
                        x, y = x.to(device), y.to(device)
                        logits = model(x)
                        correct += (logits.argmax(1) == y).sum().item()
                        total += y.size(0)
                    val_acc = 100.0 * correct / total
                writer.add_scalar("Validation/ValAcc", val_acc, epoch)
            if gnn_pool is not None:
                gnn_pool.reset_workers()
            model.train()
            writer.add_scalar("Training/TrainAcc", train_acc, epoch)
            msg = f"Epoch {epoch + 1}/{epochs}  train_acc={train_acc:.2f}%"
            if val_acc is not None:
                msg += f"  val_acc={val_acc:.2f}%"
            print(msg)

    writer.close()
    try:
        model.gnn.shutdown()
    except Exception:
        pass