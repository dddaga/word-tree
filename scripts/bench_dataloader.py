"""Benchmark: in-memory tensor dataset vs HDF5 random-access DataLoader.

HDF5 with sample-level chunking (current) pays a seek + decompress cost per sample.
In-memory: pay once at startup, then pure RAM reads during training.

Measures: load time, per-epoch iteration time at batch_size=128, peak RAM.
"""

from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
import torch.utils.data
import resource


# ── Dataset variants ──────────────────────────────────────────────────────────

class H5Dataset(torch.utils.data.Dataset):
    """Current approach: open HDF5, seek per sample."""
    def __init__(self, split: str):
        self.path  = "data/store.h5"
        self.split = split
        with h5py.File(self.path, "r") as f:
            self.n = f[f"{split}/features"].shape[0]

    def __len__(self): return self.n

    def __getitem__(self, i):
        with h5py.File(self.path, "r") as f:
            x   = torch.from_numpy(f[f"{self.split}/features"][i])
            y   = torch.from_numpy(f[f"{self.split}/soft_labels"][i])
            lbl = int(f[f"{self.split}/labels"][i])
        return x, y, lbl


class InMemDataset(torch.utils.data.Dataset):
    """Load everything into RAM tensors once at init."""
    def __init__(self, split: str):
        t0 = time.time()
        with h5py.File("data/store.h5", "r") as f:
            self.X   = torch.from_numpy(f[f"{split}/features"][:])
            self.Y   = torch.from_numpy(f[f"{split}/soft_labels"][:])
            self.lbl = torch.from_numpy(f[f"{split}/labels"][:]).long()
        print(f"  [InMemDataset] loaded {split} in {time.time()-t0:.2f}s  "
              f"({self.X.nbytes / 1e6:.0f} MB)")

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i], self.lbl[i]


# ── Benchmark one epoch ───────────────────────────────────────────────────────

def bench_epoch(dataset, batch_size: int = 128, num_workers: int = 2, label: str = "") -> float:
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    t0 = time.time()
    n_batches = 0
    for x, y, lbl in loader:
        _ = x.shape   # ensure data is actually loaded
        n_batches += 1
    elapsed = time.time() - t0
    print(f"  {label:20s}  {n_batches} batches  {elapsed:.2f}s/epoch  "
          f"({elapsed/n_batches*1000:.1f} ms/batch)")
    return elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SPLIT      = "train"
    BATCH      = 128
    N_EPOCHS   = 3   # average over 3 epochs to reduce noise
    WORKERS    = 2

    print(f"\nDataLoader benchmark  split={SPLIT}  batch={BATCH}  workers={WORKERS}")
    print("="*60)

    # --- H5Dataset: open file per sample (current) ---
    # Note: this is very slow because h5py opens/closes file per __getitem__
    # A better H5 approach keeps the file handle open — test that too.
    print("\n[1] H5Dataset (open/close per sample — current):")
    h5_ds = H5Dataset(SPLIT)
    times_h5 = [bench_epoch(h5_ds, BATCH, WORKERS, "h5_per_sample") for _ in range(N_EPOCHS)]

    # --- InMemDataset: load all to RAM once ---
    print(f"\n[2] InMemDataset (all in RAM):")
    mem_ds = InMemDataset(SPLIT)
    times_mem = [bench_epoch(mem_ds, BATCH, WORKERS, "in_memory") for _ in range(N_EPOCHS)]

    # --- H5DatasetFast: numpy arrays loaded once ---

    print(f"\n[3] H5DatasetFast (numpy arrays loaded once, torch.from_numpy per item):")
    h5f_ds = H5DatasetFast(SPLIT)
    times_h5f = [bench_epoch(h5f_ds, BATCH, WORKERS, "h5_fast") for _ in range(N_EPOCHS)]

    # --- Summary ---
    print("\n── Summary (avg over epochs) ───────────────────────────")
    print(f"  h5_per_sample : {np.mean(times_h5):.2f}s/epoch")
    print(f"  h5_fast       : {np.mean(times_h5f):.2f}s/epoch")
    print(f"  in_memory     : {np.mean(times_mem):.2f}s/epoch")
    speedup = np.mean(times_h5) / np.mean(times_mem)
    print(f"\n  Speedup (in_memory vs h5_per_sample): {speedup:.1f}x")
    speedup2 = np.mean(times_h5f) / np.mean(times_mem)
    print(f"  Speedup (in_memory vs h5_fast):        {speedup2:.1f}x")
    print(f"\n  store.h5 train features: "
          f"{9469 * 25088 * 4 / 1e6:.0f} MB to hold in RAM")
