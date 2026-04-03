"""PCA input compression sweep: k={256, 512, 1024} on VGG16 25088-dim features.

ARM 5 Input Architecture experiment: Does PCA-compressed input beat random
K_in=50 sparse gather (current 75.24% baseline)?

Pipeline:
  1. Load train/val features from data/store.h5 (shape [N, 25088])
  2. Fit sklearn PCA on training features for each k
  3. Transform both train and val features to k dimensions
  4. For each k: build SGNNET_SmallWorld with N_in=k, wrap with Resonant + AntiHebb
  5. Train 150 epochs, batch=128, plateau LR
  6. Save results to results/pca_input_sweep.json

Key design:
  - K_in = min(50, k) since PCA output may be smaller than default 50
  - n_groups = max(8, k // 8) to keep groups meaningful for small k
  - All other params match step29 Config C (75.24% reference)
  - Records explained_variance_ratio per k for variance analysis

To reproduce:
    python -u scripts/pca_input_sweep.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time, os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import h5py
import torch
import torch.nn as nn

from sklearn.decomposition import PCA

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config  import trainer_kwargs, run_metadata

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS       = 150
BATCH        = 128
SEED         = 42
DATA         = "data/store.h5"
N            = 1024
D            = 64
K_ITER       = 8
ALPHA_AHEBB  = 0.7   # step29 Config C all-time best


# ── Data loading + PCA ────────────────────────────────────────────────────────

def load_raw_features(path: str):
    """Load raw features, soft_labels, labels from HDF5."""
    with h5py.File(path, "r") as f:
        tr_feat = f["train/features"][:]      # [N_train, 25088]
        tr_soft = f["train/soft_labels"][:]
        tr_lab  = f["train/labels"][:]
        va_feat = f["val/features"][:]
        va_soft = f["val/soft_labels"][:]
        va_lab  = f["val/labels"][:]
    return (tr_feat, tr_soft, tr_lab), (va_feat, va_soft, va_lab)


def pca_transform(train_feat, val_feat, k):
    """Fit PCA on training features, transform both splits."""
    print(f"  Fitting PCA(k={k}) on {train_feat.shape}...")
    t0 = time.time()
    pca = PCA(n_components=k, random_state=SEED)
    tr_pca = pca.fit_transform(train_feat)
    va_pca = pca.transform(val_feat)
    elapsed = time.time() - t0
    evr = pca.explained_variance_ratio_
    cum_var = float(evr.sum())
    print(f"  PCA(k={k}) done in {elapsed:.1f}s — cumulative explained variance: {cum_var:.4f}")
    return tr_pca, va_pca, evr, cum_var


class PCADataset(torch.utils.data.Dataset):
    """In-memory dataset for PCA-transformed features."""
    def __init__(self, features, soft_labels, labels):
        self.features    = torch.from_numpy(features).float()
        self.soft_labels = torch.from_numpy(soft_labels).float()
        self.labels      = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.soft_labels[idx], self.labels[idx]


def make_pca_loaders(tr_feat, tr_soft, tr_lab, va_feat, va_soft, va_lab):
    """Create DataLoaders from PCA-transformed features."""
    tr_ds = PCADataset(tr_feat, tr_soft, tr_lab)
    va_ds = PCADataset(va_feat, va_soft, va_lab)
    g = torch.Generator().manual_seed(SEED)
    tr_loader = torch.utils.data.DataLoader(
        tr_ds, batch_size=BATCH, shuffle=True, generator=g,
        num_workers=0, pin_memory=False,
    )
    va_loader = torch.utils.data.DataLoader(
        va_ds, batch_size=BATCH, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    return tr_loader, va_loader


# ── Model factory ─────────────────────────────────────────────────────────────

def make_model(N_in: int) -> nn.Module:
    """Build SmallWorld + Resonant + AntiHebb with PCA-sized input."""
    torch.manual_seed(SEED)
    K_in     = min(50, N_in)
    n_groups = max(8, N_in // 8)
    base = SGNNET_SmallWorld(
        N_in=N_in, N_hidden=N, N_out=10,
        K_local=4, K_random=2,
        K_in=K_in, K_iter=K_ITER,
        n_groups=n_groups,
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.5, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


# ── Run helper ────────────────────────────────────────────────────────────────

def run(label, model, tr_loader, va_loader, meta):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  total_params={total:,}")

    tk = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr_loader, val_loader=va_loader,
                      device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    best    = max(h["val_top1"] for h in history)
    best_ep = max(range(len(history)), key=lambda i: history[i]["val_top1"]) + 1
    frac    = best_ep / EPOCHS

    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{EPOCHS} ({frac:.0%})  t={elapsed:.0f}s")
    return {
        "label":        label,
        "top1_best":    best,
        "best_ep":      best_ep,
        "ep_frac":      round(frac, 3),
        "elapsed_s":    round(elapsed),
        "total_params": total,
        "top1_history": [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta": run_metadata(__file__, {**meta, "N": N, "D": D,
                                         "K_iter": K_ITER, "alpha_ahebb": ALPHA_AHEBB,
                                         "epochs": EPOCHS}),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

PCA_K_VALUES = [256, 512, 1024]

if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print(f"PCA Input Sweep: k={PCA_K_VALUES}")
    print(f"Base: D={D} N={N} K_iter={K_ITER} AntiHebb alpha={ALPHA_AHEBB} wpos")
    print(f"Reference: ~75.24% (step29 Config C, N_in=25088)")

    # Load raw data once
    (tr_feat, tr_soft, tr_lab), (va_feat, va_soft, va_lab) = load_raw_features(DATA)
    print(f"Raw features: train={tr_feat.shape}  val={va_feat.shape}")

    results = {}

    for k in PCA_K_VALUES:
        # PCA transform
        tr_pca, va_pca, evr, cum_var = pca_transform(tr_feat, va_feat, k)
        tr_loader, va_loader = make_pca_loaders(
            tr_pca, tr_soft, tr_lab, va_pca, va_soft, va_lab
        )

        # Build model with N_in=k
        model = make_model(N_in=k).to(DEVICE)
        label = f"PCA k={k}  N_in={k}  K_in={min(50,k)}  [cumvar={cum_var:.4f}]"
        meta  = {"pca_k": k, "N_in": k, "K_in": min(50, k),
                 "cum_explained_variance": round(cum_var, 6),
                 "n_groups": max(8, k // 8)}

        r = run(label, model, tr_loader, va_loader, meta)
        r["pca_k"]                   = k
        r["cum_explained_variance"]  = round(cum_var, 6)
        r["explained_variance_top5"] = [round(float(v), 6) for v in evr[:5]]
        results[f"pca_k{k}"] = r

        del model, tr_loader, va_loader, tr_pca, va_pca
        if str(DEVICE) == "cuda":
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    out = ROOT / "results" / "pca_input_sweep.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved -> {out}")

    REF_TOP1 = 0.7524  # step29 Config C (N_in=25088)
    print(f"\n-- PCA Input Sweep (Ref=75.24% @ N_in=25088) --------")
    print(f"  {'k':>6}  {'top1':>7}  {'vs Ref':>8}  {'cumvar':>8}  {'params':>10}")
    for k in PCA_K_VALUES:
        r   = results[f"pca_k{k}"]
        top = r["top1_best"]
        vs  = f"{(top - REF_TOP1)*100:+.2f}pp"
        cv  = r["cum_explained_variance"]
        print(f"  {k:>6}  {top:.4f}  {vs:>8}  {cv:.4f}  {r['total_params']:>10,}")

    print(f"\n  Interpretation:")
    print(f"  k=256 > Ref  -> PCA is better input mechanism (removes noise)")
    print(f"  k=256 < Ref  -> 25088-dim carries useful signal PCA discards")
    print(f"  k=1024 > k=256 -> more PCA components = better (diminishing returns?)")
    print(f"  cumvar < 0.90 at k=256 -> PCA loses too much variance")
