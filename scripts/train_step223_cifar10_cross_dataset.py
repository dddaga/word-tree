"""Step 223: CIFAR-10 cross-dataset generalization test.

MOTIVATION
==========
SGNNET achieved 95.52% on Imagenette (10-class subset of ImageNet).
Does the same efficiency config generalize to CIFAR-10 (50K train, 10K test, 10 classes)?
CIFAR-10 is a different domain: small images (32×32), same number of classes.
VGG16 pool5 features are 25088-dim — same as Imagenette (avgpool output flattened).

This test validates whether SGNNET is a general-purpose architecture or tuned to Imagenette.

Design:
  - Extract VGG16 pool5 features from CIFAR-10 train+test splits
  - Save as HDF5 in the same format as data/store.h5 (train/val groups)
  - Soft labels: temperature=3.0 on VGG16 logits (same as Imagenette pipeline)
  - Compare: SGNNET (step199 config) vs MLP vs linear probe on CIFAR-10

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep — Tier-0 scouts)
"""
from __future__ import annotations
import argparse, json, sys, time, math
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (e.g. Ref,MLP). Empty = all.")
parser.add_argument("--skip-extract", action="store_true",
                    help="Skip feature extraction even if HDF5 missing (for debugging).")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42
DATA_IMAGENETTE = "data/store.h5"
DATA_CIFAR10    = "data/cifar10_store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
SOFT_TEMP = 3.0   # KL temperature for soft labels — same as Imagenette pipeline

FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step223_cifar10_cross_dataset.json"


# ---------------------------------------------------------------------------
# CIFAR-10 feature extraction
# ---------------------------------------------------------------------------

def extract_cifar10_features(out_path: Path, device: torch.device, batch_size: int = 64):
    """Extract VGG16 pool5 features from CIFAR-10 and save as HDF5.

    Format matches data/store.h5:
        /train/features    [N_train, 25088]  float32
        /train/soft_labels [N_train, 10]     float32
        /train/labels      [N_train]         int64
        /val/features      [N_test, 25088]   float32
        /val/soft_labels   [N_test, 10]      float32
        /val/labels        [N_test]          int64

    CIFAR-10 images are 32×32; VGG16 expects 224×224.
    We resize via transforms to 224×224 (standard transfer learning approach).
    VGG16 avgpool output: [B, 512, 7, 7] → flatten → 25088-dim (same as Imagenette).
    """
    try:
        import h5py
        import torchvision
        import torchvision.transforms as T
        from torchvision.models import vgg16, VGG16_Weights
    except ImportError as e:
        raise RuntimeError(f"Missing dependency: {e}. Install torchvision.")

    print(f"\n  Extracting CIFAR-10 VGG16 features → {out_path}")
    print(f"  This may take 5-20 minutes on CPU, 1-5 min on MPS/GPU.")

    # Standard ImageNet normalization
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cifar_root = ROOT / "data" / "cifar10_raw"
    cifar_root.mkdir(parents=True, exist_ok=True)

    train_ds = torchvision.datasets.CIFAR10(
        root=str(cifar_root), train=True, download=True, transform=transform)
    test_ds  = torchvision.datasets.CIFAR10(
        root=str(cifar_root), train=False, download=True, transform=transform)

    # Load VGG16 with pretrained weights
    print("  Loading VGG16 pretrained weights...")
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
    vgg.eval()

    # Hook to capture avgpool output (25088-dim after flatten)
    features_hook = []
    def _hook(module, input, output):
        features_hook.append(output.detach().cpu().flatten(1))  # [B, 25088]
    handle = vgg.avgpool.register_forward_hook(_hook)

    def _extract_split(dataset, split_name):
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        all_feats = []
        all_logits = []
        all_labels = []
        n_batches = len(loader)

        with torch.no_grad():
            for i, (imgs, labels) in enumerate(loader):
                features_hook.clear()
                imgs = imgs.to(device)
                logits = vgg(imgs)  # [B, 1000] — triggers avgpool hook
                all_feats.append(features_hook[0])            # [B, 25088]
                all_logits.append(logits.detach().cpu())      # [B, 1000]
                all_labels.append(labels)
                if (i + 1) % 50 == 0:
                    print(f"    {split_name}: {i+1}/{n_batches} batches", flush=True)

        feats  = torch.cat(all_feats,  dim=0)   # [N, 25088]
        logits = torch.cat(all_logits, dim=0)   # [N, 1000]
        labels = torch.cat(all_labels, dim=0)   # [N]

        # Soft labels: take only CIFAR-10 class logits.
        # CIFAR-10 classes don't map 1:1 to ImageNet classes, so we use all 1000 dims
        # then reduce to 10 via temperature softmax over top-10 classes for each sample.
        # Alternative: use model's predicted class probabilities as soft labels (distillation).
        # We use the standard approach: softmax(logits / T) over all 1000 classes, then
        # aggregate to 10 CIFAR-10 classes by renormalising — but VGG16 isn't trained on
        # CIFAR-10, so its 1000-class logits don't align. Instead: use the true one-hot
        # labels as "soft" labels with temperature-smoothed one-hot (label smoothing).
        # This is consistent: we're training a classifier on features, not distilling.
        # Soft label = (1 - eps) * one_hot + eps/C, eps = 0.1 (standard label smoothing).
        eps = 0.1
        C = 10
        one_hot = torch.zeros(len(labels), C)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        soft = (1 - eps) * one_hot + eps / C

        return feats.numpy().astype("float32"), soft.numpy().astype("float32"), labels.numpy()

    import h5py
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("  Extracting train split (50000 images)...")
    tr_feats, tr_soft, tr_labels = _extract_split(train_ds, "train")
    print("  Extracting test split (10000 images)...")
    va_feats, va_soft, va_labels = _extract_split(test_ds, "test")

    handle.remove()

    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("train/features",    data=tr_feats,  compression="gzip")
        f.create_dataset("train/soft_labels", data=tr_soft,   compression="gzip")
        f.create_dataset("train/labels",      data=tr_labels)
        f.create_dataset("val/features",      data=va_feats,  compression="gzip")
        f.create_dataset("val/soft_labels",   data=va_soft,   compression="gzip")
        f.create_dataset("val/labels",        data=va_labels)

    print(f"  Saved: train={len(tr_labels)}, val={len(va_labels)} → {out_path}")


# ---------------------------------------------------------------------------
# MLP baseline (same as step222, ~67K params)
# ---------------------------------------------------------------------------

def _mlp_hidden_dim(target=67_000, n_in=N_IN, n_out=N_OUT):
    for h in range(1, 500):
        p = n_in * h + h * h + h * n_out + 8 * h + n_out
        if p >= target:
            if h > 1:
                p_prev = n_in * (h-1) + (h-1)**2 + (h-1)*n_out + 8*(h-1) + n_out
                return (h-1) if abs(p_prev - target) < abs(p - target) else h
            return h
    return 2


class MLPBaseline(nn.Module):
    def __init__(self, n_in=N_IN, n_out=N_OUT, hidden_dim=None):
        super().__init__()
        h = hidden_dim or _mlp_hidden_dim()
        self.net = nn.Sequential(
            nn.Linear(n_in, h, bias=True),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, h, bias=True),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, n_out, bias=True),
        )
        self._W_pos   = nn.Parameter(torch.zeros(1, 1))
        self._W_phase = nn.Parameter(torch.zeros(1, 1))

    @property
    def W_pos(self):   return self._W_pos
    @property
    def W_phase(self): return self._W_phase

    def forward(self, x):
        return self.net(x)


class LinearProbe(nn.Module):
    def __init__(self, n_in=N_IN, n_out=N_OUT):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out, bias=True)
        self._W_pos   = nn.Parameter(torch.zeros(1, 1))
        self._W_phase = nn.Parameter(torch.zeros(1, 1))

    @property
    def W_pos(self):   return self._W_pos
    @property
    def W_phase(self): return self._W_phase

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_sgnnet():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def build_model(config_key):
    torch.manual_seed(SEED)
    if config_key == "Ref":
        return build_sgnnet()
    elif config_key == "MLP":
        return MLPBaseline()
    elif config_key == "Linear":
        return LinearProbe()
    else:
        raise ValueError(f"Unknown config: {config_key}")


def main():
    all_keys = ["Ref", "MLP", "Linear"]
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]
    else:
        run_keys = all_keys

    # --- Feature extraction ---
    cifar10_path = ROOT / DATA_CIFAR10
    if not cifar10_path.exists() and not args.skip_extract:
        print(f"  CIFAR-10 store not found at {cifar10_path}. Extracting features...")
        extract_cifar10_features(cifar10_path, device=DEVICE)
    elif cifar10_path.exists():
        print(f"  CIFAR-10 store found: {cifar10_path}")
    else:
        print(f"  WARNING: --skip-extract set but {cifar10_path} missing. Will fail on load.")

    labels_map = {
        "Ref":    f"Ref: SGNNET step199 (N={N} D={D} K_hh={K_HH} K_iter={K_ITER})",
        "MLP":    f"MLP: 2-hidden ReLU+BN ~67K params",
        "Linear": f"Linear probe: {N_IN}→{N_OUT}",
    }

    print(f"\n{'='*70}")
    print(f"Step 223 — CIFAR-10 cross-dataset generalization (Tier-0 scouts)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"Dataset: CIFAR-10 (50K train / 10K test), VGG16 pool5 features")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(cifar10_path, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        print(f"\n{'─'*60}")
        print(f"Config {labels_map.get(key, key)}")
        print(f"{'─'*60}")

        model = build_model(key).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: None)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1

        results[key] = {
            "label": labels_map.get(key, key),
            "dataset": "cifar10",
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "epochs_run": len(history), "top1_history": top1h,
            "elapsed_s": round(elapsed, 1), "n_params": n_p,
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print("STEP 223 SUMMARY — CIFAR-10 cross-dataset generalization")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    imagenette_ref = 0.9552  # step199 Tier-2 result (95.52%)
    for key in run_keys:
        r = results[key]
        delta = f"  Δ_vs_Ref={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        imagenette_delta = f"  Δ_vs_Imagenette={r['top1_best']-imagenette_ref:+.4f}" if key == "Ref" else ""
        print(f"  {key}: {r['top1_best']:.4f}  params={r['n_params']:,}{delta}{imagenette_delta}")

    print(f"\n  Imagenette Ref (step199 Tier-2): 0.9552")
    print(f"  CIFAR-10 Ref (this run):         {results.get('Ref', {}).get('top1_best', 0):.4f}")
    print(f"  (CIFAR-10 is harder for VGG16 features — different domain)")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
