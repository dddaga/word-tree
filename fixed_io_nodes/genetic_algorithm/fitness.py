"""
Fitness evaluation: check if done, run training (distributed_training-style), run validation, write results.
"""

import csv
import os
import sys
from pathlib import Path


def _redirect_stdout_stderr_to_file(log_path):
    """
    Redirect stdout/stderr at fd level so all output (including C libs) goes to log_path.
    Returns (log_file, saved_fd1, saved_fd2) to pass to _restore_stdout_stderr.
    """
    saved_fd1 = os.dup(1)
    saved_fd2 = os.dup(2)
    log_file = open(log_path, "w", encoding="utf-8")
    os.dup2(log_file.fileno(), 1)
    os.dup2(log_file.fileno(), 2)
    sys.stdout = sys.stderr = log_file
    return log_file, saved_fd1, saved_fd2


def _restore_stdout_stderr(log_file, saved_fd1, saved_fd2):
    """Restore original stdout/stderr fds and close the log file."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os.dup2(saved_fd1, 1)
    os.dup2(saved_fd2, 2)
    os.close(saved_fd1)
    os.close(saved_fd2)
    log_file.close()

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from .config_resolver import inject_paths
from .run_folder import (
    get_candidate_dir,
    is_training_done,
    read_results,
    write_results,
)


def _run_training(resolved_config, train_dataset, candidate_dir):
    """Run distributed_training-style loop; save full model to candidate_dir/weights.pt."""
    import torch.multiprocessing as mp
    from distributed_training import IrisGNNModel
    from distributed import GNNAdam, save_full_model

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    torch.manual_seed(resolved_config["system"].get("random_seed", 42))
    device = resolved_config["system"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = IrisGNNModel(resolved_config)
    model = model.to(device)
    optimizer = GNNAdam(
        model,
        lr=resolved_config["training"]["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    batch_size = resolved_config["training"]["accumulation_steps"]
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    epochs = resolved_config["training"].get("epochs", 10)

    log_path = resolved_config["system"]["logging"]["log_path"]
    tensorboard_dir = resolved_config["system"]["logging"]["tensorboard_dir"]
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    candidate_log = Path(candidate_dir) / "log.txt"
    with open(candidate_log, "a", encoding="utf-8") as train_log:
        train_log.write(f"Training started (epochs={epochs})\n")
        train_log.flush()

    global_step = 0
    with open(log_path, "w", newline="") as log_f:
        log_writer = csv.DictWriter(log_f, fieldnames=["iteration", "loss"])
        log_writer.writeheader()
        log_f.flush()
        for epoch in range(epochs):
            epoch_loss_sum = 0.0
            epoch_batches = 0
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
                epoch_loss_sum += loss.item()
                epoch_batches += 1
                with open(candidate_log, "a", encoding="utf-8") as train_log:
                    train_log.write(f"step {global_step} loss: {loss.item():.4f}\n")
                    train_log.flush()
                global_step += 1
            mean_loss = epoch_loss_sum / epoch_batches if epoch_batches else 0.0
            print(f"epoch {epoch + 1}/{epochs} loss: {mean_loss:.4f}")
            with open(candidate_log, "a", encoding="utf-8") as train_log:
                train_log.write(f"epoch {epoch + 1}/{epochs} loss: {mean_loss:.4f}\n")
                train_log.flush()
    writer.close()

    weights_path = resolved_config["system"]["weights_save_path"]
    save_full_model(model, weights_path)
    try:
        model.gnn.shutdown()
    except Exception:
        pass


def _run_validation(resolved_config, val_dataset, candidate_dir):
    """Load full model, run on val set, return accuracy and loss."""
    from distributed_training import IrisGNNModel
    from distributed import load_full_model

    device = resolved_config["system"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = IrisGNNModel(resolved_config)
    model = model.to(device)
    weights_path = Path(candidate_dir) / "weights.pt"
    load_full_model(model, str(weights_path), map_location=device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    accuracy = correct / total if total else 0.0
    validation_loss = total_loss / total if total else 0.0
    return accuracy, validation_loss


def evaluate_fitness(
    run_dir,
    resolved_config,
    get_train_val_datasets,
    run_name="ga",
):
    """
    Evaluate fitness for one resolved config. If results.json exists, return cached validation_accuracy.
    Else run training, validation, write results.json, return validation_accuracy.
    get_train_val_datasets() -> (train_dataset, val_dataset).
    """
    candidate_dir = get_candidate_dir(run_dir, resolved_config)
    inject_paths(resolved_config, candidate_dir, run_name=run_name)

    config_path = Path(candidate_dir) / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)

    if is_training_done(candidate_dir):
        results = read_results(candidate_dir)
        return float(results["validation_accuracy"])

    train_dataset, val_dataset = get_train_val_datasets()
    candidate_log_path = Path(candidate_dir) / "log.txt"
    candidate_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file, saved_fd1, saved_fd2 = _redirect_stdout_stderr_to_file(candidate_log_path)
    try:
        _run_training(resolved_config, train_dataset, candidate_dir)
        accuracy, validation_loss = _run_validation(resolved_config, val_dataset, candidate_dir)
        print(f"validation accuracy: {accuracy:.4f} loss: {validation_loss:.4f}")
        with open(Path(candidate_dir) / "log.txt", "a", encoding="utf-8") as f:
            f.write(f"validation accuracy: {accuracy:.4f} loss: {validation_loss:.4f}\n")
            f.flush()
        write_results(
            candidate_dir,
            {"validation_accuracy": accuracy, "validation_loss": validation_loss},
        )
        return float(accuracy)
    finally:
        _restore_stdout_stderr(log_file, saved_fd1, saved_fd2)
