"""Convert store_cifar10_aug.h5 → flat .npy files for fast memmap loading.
Output: data/cifar10_aug_train_x.npy  [N_train, 25088] float32
        data/cifar10_aug_train_y.npy  [N_train]        int64
        data/cifar10_aug_val_x.npy    [N_val, 25088]   float32
        data/cifar10_aug_val_y.npy    [N_val]           int64
Run once; subsequent training uses memmap. ~10 GB uncompressed output.
"""
import sys, time
from pathlib import Path
import numpy as np
import h5py

ROOT   = Path(__file__).parent.parent
SRC    = ROOT / "data" / "store_cifar10_aug.h5"
OUT    = ROOT / "data"

if not SRC.exists():
    print(f"ERROR: {SRC} not found"); sys.exit(1)

for split in ("train", "val"):
    t0 = time.time()
    print(f"Converting {split}...", flush=True)
    with h5py.File(SRC, "r") as f:
        x = f[f"{split}/features"][:].astype(np.float32)
        y = f[f"{split}/labels"][:].astype(np.int64)
    print(f"  Loaded: x={x.shape} ({x.nbytes/1e9:.1f}GB)  y={y.shape}", flush=True)
    np.save(OUT / f"cifar10_aug_{split}_x.npy", x)
    np.save(OUT / f"cifar10_aug_{split}_y.npy", y)
    del x, y
    print(f"  Saved in {time.time()-t0:.0f}s", flush=True)

print("Done. Files written to data/cifar10_aug_{train,val}_{x,y}.npy")
