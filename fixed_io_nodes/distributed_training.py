import csv
import os
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

from distributed import DistributedNeurographLayer, GNNAdam
from main import load_config, load_iris_dataset

CONFIG_PATH = "training_runs/distributed_test/distributed.yaml"


class IrisGNNModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear = nn.Linear(4, 16)
        self.tanh = nn.Tanh()
        self.gnn = DistributedNeurographLayer(cfg)

    def forward(self, x):
        B = x.size(0)
        x = x.squeeze(1)
        h = self.tanh(self.linear(x))
        h = h.view(B, 4, 4)
        return self.gnn(h)


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
    device = cfg["system"]["device"]
    model = model.to(device)

    optimizer = GNNAdam(
        model,
        lr=cfg["training"]["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    dataset = load_iris_dataset()
    batch_size = cfg["training"]["accumulation_steps"]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    criterion_noreduce = nn.CrossEntropyLoss(reduction="none") if log_per_sample else None
    epochs = cfg["training"]["epochs"]

    log_fields = ["iteration", "loss"]
    global_step = 0
    with open(log_path, "w", newline="") as log_f:
        log_writer = csv.DictWriter(log_f, fieldnames=log_fields)
        log_writer.writeheader()
        log_f.flush()
        for epoch in range(epochs):
            for x, y in dataloader:
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

                if verbose:
                    if log_per_sample and criterion_noreduce is not None:
                        per_sample = criterion_noreduce(logits, y)
                        for b in range(y.size(0)):
                            print(f"Target {y[b].item()}: Loss: {per_sample[b].item():.4f}")
                    else:
                        print(f"Loss: {loss.item():.4f}")

            if verbose:
                print(f"epoch {epoch + 1}/{epochs} loss: {loss.item():.4f}")

    writer.close()
