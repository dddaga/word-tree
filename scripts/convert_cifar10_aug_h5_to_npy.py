"""Convert store_cifar10_aug.h5 → flat .npy files for fast memmap loading.
Output: data/cifar10_aug_train_x.npy  [N_train, 25088] float32
        data/cifar10_aug_train_y.npy  [N_train]        int64
        data/cifar10_aug_val_x.npy    [N_val, 25088]   float32
        data/cifar10_aug_val_y.npy    [N_val]           int64
Chunked write — avoids loading 10GB at once; safe on 14GB RAM machines.
"""
import sys, time
from pathlib import Path
import numpy as np
import h5py

ROOT   = Path(__file__).parent.parent
SRC    = ROOT / "data" / "store_cifar10_aug.h5"
OUT    = ROOT / "data"
CHUNK  = 2000  # rows per chunk (~200MB per chunk)

if not SRC.exists():
    print(f"ERROR: {SRC} not found"); sys.exit(1)

for split in ("train", "val"):
    t0 = time.time()
    print(f"Converting {split}...", flush=True)
    with h5py.File(SRC, "r") as f:
        ds_x = f[f"{split}/features"]
        ds_y = f[f"{split}/labels"]
        N, D = ds_x.shape
        print(f"  Shape: {N}×{D}  dtype={ds_x.dtype}", flush=True)

        out_x = np.lib.format.open_memmap(
            str(OUT / f"cifar10_aug_{split}_x.npy"),
            mode="w+", dtype=np.float32, shape=(N, D)
        )
        out_y = np.lib.format.open_memmap(
            str(OUT / f"cifar10_aug_{split}_y.npy"),
            mode="w+", dtype=np.int64, shape=(N,)
        )

        for i in range(0, N, CHUNK):
            end = min(i + CHUNK, N)
            out_x[i:end] = ds_x[i:end].astype(np.float32)
            out_y[i:end] = ds_y[i:end].astype(np.int64)
            if i % 10000 == 0:
                pct = 100 * end / N
                print(f"  {end}/{N} ({pct:.0f}%) in {time.time()-t0:.0f}s", flush=True)

        out_x.flush(); out_y.flush()
    print(f"  Done {split} in {time.time()-t0:.0f}s", flush=True)

print("Done. Files written to data/cifar10_aug_{train,val}_{x,y}.npy")
