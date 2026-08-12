"""Step 995: ResNet-18 backbone T0 scout (20ep, 50% data, N=2048).

Tests whether SGNNET routing transfers to ResNet-18 features.
POC insight (meditation 004): ResNet-18 layer4 = 512×7×7 = 25088 — same dim as VGG16.
The existing pipeline (N_IN=25088, K_in=25) works without modification.

If best >= 0.85 (vs VGG16 T0 ~0.91): backbone-agnostic routing confirmed.
If best < 0.70: VGG16 specific (feature structure matters).
Either result is paper-valuable.

Requires: ResNet-18 features extracted from imagenette2-320.
Phase 1: extract features → data/store_resnet18.h5
Phase 2: train SGNNET (same config as step199/step887 T0)

NOTE: mini-only (raw images required for feature extraction).
VGG16 h5 files on 5060ti don't include ResNet-18 features.

Usage:
    python scripts/train_step995_resnet18_backbone_t0.py
    python scripts/train_step995_resnet18_backbone_t0.py --skip_extract  # if h5 exists
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="step995: ResNet-18 backbone T0")
parser.add_argument("--device",       default="auto")
parser.add_argument("--epochs",       type=int, default=20)
parser.add_argument("--seed",         type=int, default=42)
parser.add_argument("--data_img",     default="data/imagenette2-320")
parser.add_argument("--h5_out",       default="data/store_resnet18.h5")
parser.add_argument("--skip_extract", action="store_true",
                    help="Skip feature extraction if h5 already exists")
args = parser.parse_args()

DEVICE = (
    __import__("torch").device("cuda") if __import__("torch").cuda.is_available()
    else __import__("torch").device("mps") if __import__("torch").backends.mps.is_available()
    else __import__("torch").device("cpu")
) if args.device == "auto" else __import__("torch").device(args.device)

SLOT    = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step995_resnet18_t0_seed{args.seed}__{SLOT}.json"

N = 2048; D = 16; K_IN = 25; K_HH = 2; K_ITER = 5; N_IN = 25088; N_OUT = 10
BATCH = 64; EPOCHS = args.epochs

print(f"\n{'='*70}")
print(f"step995 — ResNet-18 backbone T0 (20ep, 50% data, N=2048)")
print(f"  device={DEVICE}  seed={args.seed}  batch={BATCH}")
print(f"  N={N} D={D} K_in={K_IN} N_IN={N_IN} (same as VGG16)")
print(f"  Advance: T0 best >= 0.85 → backbone-agnostic CONFIRMED")
print(f"  Null: T0 best < 0.70 → VGG16 feature-specific")
print(f"{'='*70}\n")


def extract_resnet18_features(data_img_path: str, h5_out: str, device) -> None:
    """Extract ResNet-18 layer4 features → H5 store (same format as VGG16 store)."""
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    import h5py
    import numpy as np

    print(f"Extracting ResNet-18 features from {data_img_path} ...")

    r18 = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    extractor = nn.Sequential(*list(r18.children())[:-2])  # layer4 → 512×7×7
    extractor = extractor.to(device).eval()

    tfm = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    splits = {}
    for split in ("train", "val"):
        ds = ImageFolder(f"{data_img_path}/{split}", transform=tfm)
        dl = DataLoader(ds, batch_size=32, num_workers=2, shuffle=False)
        feats, labels = [], []
        with torch.no_grad():
            for x, y in dl:
                f = extractor(x.to(device)).cpu()  # [B, 512, 7, 7]
                feats.append(f.view(f.size(0), -1).numpy())  # [B, 25088]
                labels.append(y.numpy())
        splits[split] = (np.concatenate(feats), np.concatenate(labels))
        print(f"  {split}: {splits[split][0].shape}")

    Path(h5_out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_out, "w") as f:
        for split, (feats, labels) in splits.items():
            f.create_dataset(f"{split}/features", data=feats.astype("float32"))
            f.create_dataset(f"{split}/labels",   data=labels.astype("int64"))
    print(f"  → saved to {h5_out}")


def main():
    import torch
    import numpy as np

    h5_path = ROOT / args.h5_out
    if not args.skip_extract and not h5_path.exists():
        img_path = ROOT / args.data_img
        if not img_path.exists():
            print(f"ERROR: {img_path} not found — mini-only script"); sys.exit(1)
        extract_resnet18_features(str(img_path), str(h5_path), DEVICE)
    elif not h5_path.exists():
        print(f"ERROR: {h5_path} not found. Run without --skip_extract."); sys.exit(1)

    from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
    from src.sgnnet.model_resonant      import SGNNET_Resonant
    from src.training.trainer           import Trainer
    from src.training.experiment_config import trainer_kwargs
    from src.training.dataset           import make_loaders, make_subset_loader

    torch.manual_seed(args.seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    model = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=0.5, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params={n_p:,}")

    tr = make_subset_loader(str(h5_path), batch_size=BATCH, seed=args.seed, fraction=0.5)
    _, va = make_loaders(str(h5_path), batch_size=BATCH, seed=args.seed)

    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    t0 = time.time()
    history = Trainer(
        model=model, train_loader=tr, val_loader=va,
        device=DEVICE, **kw,
    ).train(
        n_epochs=EPOCHS,
        log_fn=lambda m: print(
            f"  e{m['epoch']+1:3d}  loss={m['train_loss']:.4f}  "
            f"top1={m['val_top1']:.4f}  lr={m['lr']:.2e}", flush=True,
        ),
    )

    elapsed = time.time() - t0
    top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
    best, best_ep = max(top1h), int(np.argmax(top1h)) + 1

    result = {
        "step": "step995",
        "tier": "T0",
        "backbone": "ResNet-18",
        "config": f"N={N} D={D} K_in={K_IN} K_hh={K_HH} K_iter={K_ITER}",
        "epochs": EPOCHS,
        "n_params": n_p,
        "best": round(best, 4),
        "best_ep": best_ep,
        "elapsed_s": round(elapsed, 1),
        "vgg16_t0_ref": 0.9157,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))

    print(f"\n{'='*70}")
    print(f"RESULT: best={best:.4f} @ep{best_ep}  elapsed={elapsed:.0f}s")
    print(f"VGG16 T0 ref ~91.57%")
    if best >= 0.85:
        print("  → BACKBONE-AGNOSTIC: R18 routing viable → extend Paper 1 claim")
    elif best >= 0.70:
        print("  → PARTIAL: R18 works but lower than VGG16 → architecture-sensitive")
    else:
        print("  → VGG16-SPECIFIC: routing requires VGG feature geometry")
    print(f"→ {OUT_PATH}")


if __name__ == "__main__":
    main()
