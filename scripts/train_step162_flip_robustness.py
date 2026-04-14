"""Step 162: Flip robustness — SGNNET vs VGG16-FC vs linear probe.

HYPOTHESIS
==========
SGNNET's K_iter message passing (12 routing steps, small-world graph) allows
global pattern re-assembly after a horizontal flip perturbs the VGG16 pool5
feature distribution. A linear FC classifier applies weights once and has no
such dynamic re-routing capacity.

If SGNNET is more robust to flip than a linear classifier trained on the same
features, that directly supports K_iter's role in building flip-invariant
representations.

EXPERIMENTAL DESIGN
===================
All models train on ORIGINAL features (store.h5 train split).
All models are evaluated on TWO val sets:
  1. Original val features  (in store.h5 — extracted from unflipped images)
  2. Flipped val features   (extracted inline: VGG16 on horizontally flipped val images)

Models compared:
  VGG16_direct : VGG16's own classification on flipped val images (argmax of
                 10-class logits). No training — uses VGG16's pretrained FC stack.
                 This is the "ceiling": the most powerful classifier on these features.
  FC_linear    : nn.Linear(N_IN→N_OUT). Minimal baseline. One matrix multiply.
  FC_mlp       : 3-layer MLP 25088→512→128→10. Standard deep classifier.
  SGNNET       : N=1024 D=16 K_hh=8 K_iter=8, standard AH routing.
                 Hypothesis: K_iter propagation → better flip robustness than FC.

METRICS
=======
  orig_acc   : top-1 on original val
  flip_acc   : top-1 on flipped val
  robustness : flip_acc / orig_acc  (1.0 = perfect, lower = more brittle)
  delta      : flip_acc - orig_acc  (pp drop)

To reproduce:
    python -u scripts/train_step162_flip_robustness.py --device mps
    python -u scripts/train_step162_flip_robustness.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs, topology_kwargs
from src.training.dataset             import make_loaders
from src.data.dataset                 import ImagenetteDataset
from src.data.extractor               import VGGExtractor

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75)
parser.add_argument("--configs", default="",
                    help="Comma-separated subset to run (e.g. FC_linear,SGNNET)")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
N_IN   = 25088
N_OUT  = 10

# SGNNET hypers (N=1024 current defaults)
N = 1024; D = 16; K_ITER = 8; K_HH = 8; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0


# ---------------------------------------------------------------------------
# Extract flipped val features from VGG16
# ---------------------------------------------------------------------------

def extract_flipped_val(device_str: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Run VGG16 on horizontally flipped val images.

    Returns:
        features    [N_val, 25088] — pool5 features of flipped images
        soft_labels [N_val, 10]   — VGG16's own 10-class predictions on flipped images
        hard_labels [N_val]       — ground-truth class indices (unchanged by flip)
    """
    print("\nExtracting flipped val features via VGG16...")
    flip_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),   # always flip
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset   = ImagenetteDataset(split="val", transform=flip_transform)
    loader    = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    extractor = VGGExtractor(device=device_str)
    feat, soft, labels = extractor.extract_all(loader, desc="  flipped val")
    print(f"  → {feat.shape}  soft_labels={soft.shape}  labels={labels.shape}")
    return feat, soft, labels


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class FC_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(N_IN, N_OUT)

    def forward(self, x):
        return self.fc(x)

    @property
    def W_phase(self): return self.fc.weight


class FC_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, N_OUT),
        )

    def forward(self, x):
        return self.net(x)

    @property
    def W_phase(self): return self.net[0].weight


def make_sgnnet(seed_offset=0):
    torch.manual_seed(SEED + seed_offset)
    K_random = max(1, K_HH // 4)
    K_local  = K_HH - K_random
    n_groups = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER,
        K_local=K_local, K_random=K_random, n_groups=n_groups,
        norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_top1(model: nn.Module, features: torch.Tensor, labels: torch.Tensor,
              device: torch.device, batch_size: int = 256) -> float:
    model.eval()
    correct = 0
    total   = 0
    for i in range(0, len(features), batch_size):
        x = features[i:i+batch_size].to(device)
        y = labels[i:i+batch_size]
        logits = model(x).cpu()
        correct += (logits.argmax(1) == y).sum().item()
        total   += len(y)
    return correct / total


@torch.no_grad()
def vgg16_direct_acc(soft_labels: torch.Tensor, hard_labels: torch.Tensor) -> float:
    """VGG16's own accuracy: argmax of its 10-class soft labels vs ground truth."""
    preds   = soft_labels.argmax(1)
    correct = (preds == hard_labels).sum().item()
    return correct / len(hard_labels)


# ---------------------------------------------------------------------------
# Shared data loaders (cached)
# ---------------------------------------------------------------------------

_loaders_cache = None

def get_loaders():
    global _loaders_cache
    if _loaders_cache is None:
        tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
        n   = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders_cache = (tr, va)
    return _loaders_cache


def load_val_tensors():
    """Load original val features, soft_labels, hard_labels as tensors."""
    _, va = get_loaders()
    feats, softs, hards = [], [], []
    for xb, sb, yb in va:
        feats.append(xb); softs.append(sb); hards.append(yb)
    return torch.cat(feats), torch.cat(softs), torch.cat(hards)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fc(model: nn.Module, n_epochs: int, key: str,
             val_feat: torch.Tensor, val_labels: torch.Tensor):
    """Train FC model with simple AdamW + cosine LR."""
    tr, _ = get_loaders()
    model  = model.to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit   = nn.CrossEntropyLoss()
    for ep in range(1, n_epochs + 1):
        model.train()
        for xb, _, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()
        if ep % 5 == 0 or ep == n_epochs:
            acc = eval_top1(model, val_feat, val_labels, DEVICE)
            print(f"  {key} ep{ep:3d}  orig={acc:.4f}")


def train_sgnnet(model: nn.Module, n_epochs: int):
    """Train SGNNET via Trainer."""
    tr, va = get_loaders()
    model  = model.to(DEVICE)
    kw     = trainer_kwargs(N, n_epochs=n_epochs)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **kw)
    history = trainer.train(n_epochs=n_epochs)
    return max(h.get("val_top1", 0.0) for h in history)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 162 — Flip Robustness: SGNNET vs VGG16-FC vs Linear Probe")
    print(f"Device={DEVICE}  Epochs={EPOCHS}")
    print(f"{'='*70}")
    print("""
Hypothesis: SGNNET's K_iter message passing provides better robustness to
horizontal flip than a linear FC classifier trained on the same features.

VGG16 pool5 features change under horizontal flip (VGG16 is not equivariant).
The question: can SGNNET's graph routing compensate, while FC cannot?
""")

    # ── Step 1: Load original val + extract flipped val features ─────────────
    orig_feat, orig_soft, orig_labels = load_val_tensors()
    device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    flip_feat, flip_soft, flip_labels = extract_flipped_val(device_str)

    print(f"\nOriginal val:  {orig_feat.shape[0]} samples")
    print(f"Flipped val:   {flip_feat.shape[0]} samples")

    # ── Step 2: VGG16 direct accuracy (no training) ───────────────────────────
    vgg_orig = vgg16_direct_acc(orig_soft, orig_labels)
    vgg_flip = vgg16_direct_acc(flip_soft, flip_labels)

    print(f"\n{'─'*60}")
    print(f"VGG16 direct (no training, N_params=138M)")
    print(f"  orig_acc={vgg_orig:.4f}  flip_acc={vgg_flip:.4f}  "
          f"delta={vgg_flip-vgg_orig:+.4f}  "
          f"robustness={vgg_flip/vgg_orig:.3f}")

    results = {
        "VGG16_direct": {
            "label": "VGG16 FC head (pretrained ImageNet, 138M params)",
            "orig_acc": round(vgg_orig, 4),
            "flip_acc": round(vgg_flip, 4),
            "delta":    round(vgg_flip - vgg_orig, 4),
            "robustness": round(vgg_flip / vgg_orig, 4) if vgg_orig > 0 else 0,
            "n_params": 138_357_544,
            "trained_on": "ImageNet",
        }
    }

    out_path = ROOT / "results" / "train_step162_flip_robustness.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    # ── Step 3: Train and evaluate FC_linear ──────────────────────────────────
    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []

    def should_run(key):
        return not cfg_filter or key in cfg_filter

    trainable_configs = [
        ("FC_linear", FC_Linear,  5,      "Linear(25088→10) — one matrix multiply", "fc"),
        ("FC_mlp",    FC_MLP,     20,     "MLP 25088→512→128→10 — deep but static",  "fc"),
        ("SGNNET",    make_sgnnet, EPOCHS, f"SGNNET N={N} D={D} K_iter={K_ITER} AH — graph routing", "sgnnet"),
    ]

    for key, model_fn, n_ep, label, mode in trainable_configs:
        if not should_run(key):
            continue

        print(f"\n{'─'*60}")
        torch.manual_seed(SEED)
        model    = model_fn()
        n_params = count_params(model)
        print(f"Config {key}: {label}")
        print(f"  params={n_params:,}  epochs={n_ep}")
        print(f"{'─'*60}")

        t0 = time.time()

        if mode == "fc":
            train_fc(model, n_ep, key, orig_feat, orig_labels)
        else:
            train_sgnnet(model, n_ep)

        # Final evaluation on both val sets
        orig_acc = eval_top1(model, orig_feat, orig_labels, DEVICE)
        flip_acc = eval_top1(model, flip_feat, flip_labels, DEVICE)
        elapsed  = time.time() - t0

        robustness = flip_acc / orig_acc if orig_acc > 0 else 0
        delta      = flip_acc - orig_acc

        print(f"\n  orig_acc={orig_acc:.4f}  flip_acc={flip_acc:.4f}  "
              f"delta={delta:+.4f}  robustness={robustness:.3f}  "
              f"elapsed={elapsed/60:.1f}min")

        results[key] = {
            "label":      label,
            "orig_acc":   round(orig_acc, 4),
            "flip_acc":   round(flip_acc, 4),
            "delta":      round(delta, 4),
            "robustness": round(robustness, 4),
            "n_params":   n_params,
            "epochs":     n_ep,
            "trained_on": "Imagenette original features",
        }
        out_path.write_text(json.dumps(results, indent=2))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"STEP 162 SUMMARY — Flip Robustness")
    print(f"{'='*70}")
    print(f"{'Model':15s}  {'orig':>7}  {'flip':>7}  {'delta':>8}  {'robust':>7}  {'params':>12}")
    print(f"{'─'*65}")
    for k, r in results.items():
        print(f"{k:15s}  {r['orig_acc']:.4f}   {r['flip_acc']:.4f}   "
              f"{r['delta']:+.4f}   {r['robustness']:.3f}   {r['n_params']:>12,}")

    print(f"""
Interpretation:
  robustness=1.0 → flip_acc == orig_acc (perfectly robust)
  robustness<1.0 → flip hurts (lower = more brittle)
  If SGNNET robustness > FC_linear robustness → hypothesis CONFIRMED
  If SGNNET robustness ≈ FC_linear robustness → K_iter propagation does not help
  If SGNNET robustness < FC_linear robustness → graph routing makes it MORE brittle
""")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
