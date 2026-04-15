"""Step 604 (B1 teacher): Train K=5 SGNNET ΔW-proj teacher and cache trajectory.

MOTIVATION (Opus mediation candidate B1)
=========================================
step196 killed "K_iter distillation" using output-matching KD only.
Consistency-DEQ (arXiv:2602.03024, 2024) shows that matching the solver
trajectory — not just final output — succeeds where output KD fails.
SGNNET's K_iter loop IS structurally a DEQ fixed-point iteration.

This script:
  1. Trains a K=5 ΔW-proj teacher (step268 config, full 75ep Tier-1)
  2. Runs inference over the FULL dataset capturing:
       Z_5    : final latent state after K=5 iterations [N_samples, N_hidden, D]
       logits : pre-softmax output              [N_samples, N_out]
  3. Saves to data/teacher_trajectory_step604.h5

Storage estimate:
  train Z_final : 9469 × 2048 × 16 × 4 bytes ≈ 1.24 GB
  val   Z_final : 3925 × 2048 × 16 × 4 bytes ≈ 0.51 GB
  logits (both) : negligible

CONFIGS
=======
  Teacher: N=2048, D=16, K_hh=2, K_iter=5, ΔW proj, full data 75ep Tier-1

To run:
    python -u scripts/train_step604_b1_consistency_teacher.py
    python -u scripts/train_step604_b1_consistency_teacher.py --device cpu --epochs 3  # smoke
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders, H5Dataset

# --- CLI --------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Step 604 B1 consistency-DEQ teacher")
parser.add_argument("--device",       default="auto")
parser.add_argument("--epochs",       type=int, default=75)
parser.add_argument("--seed",         type=int, default=42)
parser.add_argument("--skip_cache",   action="store_true",
                    help="Skip trajectory cache (just train, for smoke test)")
parser.add_argument("--cache_fp16",   action="store_true",
                    help="Save Z_final as fp16 (halves storage ~850 MB vs 1.75 GB)")
args = parser.parse_args()

DEVICE = (torch.device("cuda")  if torch.cuda.is_available()
    else  torch.device("mps")   if torch.backends.mps.is_available()
    else  torch.device("cpu")   ) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = args.seed

# Efficiency config (step268 / step199 base)
N      = 2048;  N_IN  = 25088;  N_OUT = 10
D      = 16;    K_HH  = 2;      K_IN  = 25
K_ITER = 5
ALPHA_REFLECT = 0.5

DATA_PATH   = ROOT / "data" / "store.h5"
CACHE_PATH  = ROOT / "data" / "teacher_trajectory_step604.h5"
SLOT        = os.environ.get("SGN_SLOT", "local")
OUT_PATH    = ROOT / "results" / f"train_step604_b1_consistency_teacher_seed{SEED}__{SLOT}.json"


# ---------------------------------------------------------------------------
# Model — ΔW projection (identical to step268)
# ---------------------------------------------------------------------------

class SGNNET_DeltaProj(nn.Module):
    """ΔW projection routing — sign-invariant gate. Matches step268 exactly."""

    def __init__(self, N_hidden, N_out, D_, N_in, K_in, K_iter, K_local, K_random,
                 n_groups, alpha_reflect, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.base = SGNNET_SmallWorld(
            N_hidden=N_hidden, N_out=N_out, D=D_, N_in=N_in,
            K_in=K_in, K_iter=K_iter, K_local=K_local, K_random=K_random,
            n_groups=n_groups, norm_mode="l2", encoding_mode="fourier")
        self.resonant = SGNNET_Resonant(
            self.base, K_phase=8, alpha_reflect=alpha_reflect,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0)
        self.alpha_reflect = alpha_reflect

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase

    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()

    def forward(self, x):
        """Standard forward — used during training."""
        Z, _ = self._forward_with_z(x)
        return Z

    def _forward_with_z(self, x):
        """Forward returning (logits, Z_final_before_readout)."""
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh
        N_h = self.base.N_hidden
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h = self.base.W_pos[:N_h]
        delta_w  = W_h.unsqueeze(1) - W_h[conn_hh]           # [N, K_hh, D]
        dw_norm  = F.normalize(delta_w, dim=-1).unsqueeze(0)  # [1, N, K_hh, D]

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.base.K_iter):
            Z_fwd   = F.relu(Z - theta_pos)
            Z_nb    = Z_fwd[:, conn_hh, :]                    # [B, N, K_hh, D]
            proj    = (Z_nb * dw_norm).sum(-1, keepdim=True)  # [B, N, K_hh, 1]
            Z_nb    = Z_nb * proj.abs()                        # sign-invariant gate
            Z_struct = Z_nb.sum(dim=2)
            Z_remainder  = Z_fwd - Z
            Z_reflected  = self.alpha_reflect * Z_reflected + Z_remainder
            Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)
        # Z is the final latent state [B, N_hidden, D] BEFORE readout
        logits = self.base._readout(Z)
        return logits, Z


def build_teacher():
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    return SGNNET_DeltaProj(
        N_hidden=N, N_out=N_OUT, D_=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, alpha_reflect=ALPHA_REFLECT, seed=SEED)


# ---------------------------------------------------------------------------
# Training wrapper that routes forward() through a standard model interface
# ---------------------------------------------------------------------------

class ForwardWrapper(nn.Module):
    """Wraps SGNNET_DeltaProj so Trainer only sees standard forward → logits."""
    def __init__(self, inner: SGNNET_DeltaProj):
        super().__init__()
        self.inner = inner

    @property
    def W_pos(self):   return self.inner.W_pos
    @property
    def W_phase(self): return self.inner.W_phase

    def tick_epoch(self):
        self.inner.tick_epoch()

    def forward(self, x):
        return self.inner.forward(x)


# ---------------------------------------------------------------------------
# Trajectory cache extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_trajectory(model: SGNNET_DeltaProj, dataset, device, dtype=torch.float32,
                       desc="train") -> tuple[np.ndarray, np.ndarray]:
    """Run inference over dataset, return (Z_final, logits) as numpy arrays.

    Z_final shape: [N_samples, N_hidden, D]  fp32 or fp16 depending on dtype
    logits  shape: [N_samples, N_out]         fp32
    """
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=0)

    all_Z      = []
    all_logits = []

    n_batches = len(loader)
    t0 = time.time()
    for i, (x, _, _) in enumerate(loader):
        x = x.to(device)
        logits, Z = model._forward_with_z(x)   # logits [B, 10], Z [B, 2048, 16]
        all_Z.append(Z.to(dtype).cpu())
        all_logits.append(logits.to(torch.float32).cpu())
        if (i+1) % 20 == 0 or (i+1) == n_batches:
            pct = (i+1) / n_batches * 100
            print(f"  [{desc}] batch {i+1}/{n_batches} ({pct:.0f}%)  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    Z_all      = torch.cat(all_Z,      dim=0).numpy()   # [N_samples, N_hidden, D]
    logits_all = torch.cat(all_logits, dim=0).numpy()   # [N_samples, N_out]
    return Z_all, logits_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 604 B1 — Consistency-DEQ teacher (K=5 ΔW proj)")
    print(f"  seed={SEED}  epochs={EPOCHS}  device={DEVICE}  fp16={args.cache_fp16}")
    print(f"  data={DATA_PATH}")
    print(f"  cache={CACHE_PATH}")
    print(f"  results={OUT_PATH}")
    print(f"{'='*70}")

    # ── 1. Load data ──────────────────────────────────────────────────────
    tr, va = make_loaders(DATA_PATH, batch_size=BATCH, seed=SEED)
    print(f"  train={len(tr.dataset)}  val={len(va.dataset)}")

    # ── 2. Build + train teacher ──────────────────────────────────────────
    teacher = build_teacher()
    wrapper = ForwardWrapper(teacher).to(DEVICE)
    n_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    print(f"  params={n_params:,}  K_iter={K_ITER}")

    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=wrapper, train_loader=tr, val_loader=va, device=DEVICE, **kw)

    t0 = time.time()
    history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
        print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
        if (m['epoch']+1) % 10 == 0 else None
    ))
    elapsed_train = time.time() - t0

    top1h      = [h.get("val_top1", 0.0) for h in history]
    top1_best  = max(top1h) if top1h else 0.0
    best_ep    = int(np.argmax(top1h)) + 1 if top1h else 0
    top1_last  = top1h[-1] if top1h else 0.0
    print(f"\n  Teacher training done: best={top1_best:.4f} @ep{best_ep}  "
          f"last={top1_last:.4f}  elapsed={elapsed_train:.0f}s")

    result = {
        "step": "604",
        "role": "teacher",
        "K_iter": K_ITER,
        "N": N, "D": D, "K_hh": K_HH,
        "n_params": n_params,
        "top1_best": top1_best,
        "top1_last": top1_last,
        "best_epoch": best_ep,
        "elapsed_train_s": elapsed_train,
        "seed": SEED,
    }

    # ── 3. Cache trajectory ───────────────────────────────────────────────
    if not args.skip_cache:
        print(f"\n  Extracting trajectory cache → {CACHE_PATH}")
        cache_dtype = torch.float16 if args.cache_fp16 else torch.float32
        h5_dtype    = "float16"     if args.cache_fp16 else "float32"

        # Extract train split
        t_start = time.time()
        train_ds = H5Dataset(DATA_PATH, split="train")
        val_ds   = H5Dataset(DATA_PATH, split="val")

        print("  Extracting TRAIN split...")
        Z_train, logits_train = extract_trajectory(teacher.to(DEVICE), train_ds, DEVICE,
                                                    dtype=cache_dtype, desc="train")
        print(f"  train Z_final: {Z_train.shape}  dtype={Z_train.dtype}")

        print("  Extracting VAL split...")
        Z_val, logits_val = extract_trajectory(teacher.to(DEVICE), val_ds, DEVICE,
                                                dtype=cache_dtype, desc="val")
        print(f"  val   Z_final: {Z_val.shape}  dtype={Z_val.dtype}")

        # Save
        CACHE_PATH.parent.mkdir(exist_ok=True)
        t_write = time.time()
        with h5py.File(CACHE_PATH, "w") as f:
            f.attrs["teacher_top1"]  = float(top1_best)
            f.attrs["K_iter"]        = K_ITER
            f.attrs["N"]             = N
            f.attrs["D"]             = D
            f.attrs["seed"]          = SEED
            f.attrs["dtype_Z"]       = h5_dtype
            g_tr = f.create_group("train")
            g_tr.create_dataset("Z_final", data=Z_train, compression="lzf")
            g_tr.create_dataset("logits",  data=logits_train.astype("float32"))
            g_va = f.create_group("val")
            g_va.create_dataset("Z_final", data=Z_val,   compression="lzf")
            g_va.create_dataset("logits",  data=logits_val.astype("float32"))

        cache_gb = CACHE_PATH.stat().st_size / 1e9
        elapsed_cache = time.time() - t_start
        print(f"\n  Cache saved: {CACHE_PATH}")
        print(f"  Size on disk: {cache_gb:.2f} GB (lzf compressed)")
        print(f"  Uncompressed estimate: "
              f"{(Z_train.nbytes + Z_val.nbytes) / 1e9:.2f} GB Z + "
              f"{(logits_train.nbytes + logits_val.nbytes) / 1e6:.1f} MB logits")
        print(f"  Cache elapsed: {elapsed_cache:.0f}s")

        result["cache_path"]      = str(CACHE_PATH)
        result["cache_size_gb"]   = round(cache_gb, 3)
        result["cache_dtype"]     = h5_dtype
        result["n_train_samples"] = int(Z_train.shape[0])
        result["n_val_samples"]   = int(Z_val.shape[0])
        result["elapsed_cache_s"] = round(elapsed_cache, 1)
    else:
        print("  --skip_cache: trajectory extraction skipped.")
        result["cache_path"] = None

    # ── 4. Save result JSON ───────────────────────────────────────────────
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Step 604 teacher DONE")
    print(f"  top1_best={top1_best:.4f}  params={n_params:,}")
    print(f"  result → {OUT_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
