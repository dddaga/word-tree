"""cnn_step016 — GA search over EfficientVGG architecture space.

Motivation (CONFIRMED evidence):
  MultiScaleCNN family T1 ceiling: 37-42% (step007-013)
  EfficientVGG Ref T1: 73.17% (step003) — 31pp higher
  MultiScaleCNN GA (step006) searched wrong model family.

This search: EfficientVGG space, same fitness = val_acc × (183.2M/MACs)^0.1.
Goal: find EfficientVGG configs that beat Ref on efficiency (fewer MACs, same acc)
or beat Ref on accuracy (same MACs, higher acc).

Known anchors from step003 T1:
  D_small_s: 150K params, 57.5M MACs, T1=70.95%  (efficiency anchor)
  Ref:       419K params, 183M MACs,  T1=73.17%  (accuracy anchor)
  F_wide:    825K params, 560M MACs,  T1=74.42%  (max accuracy, expensive)

GA should find configs in the efficiency frontier between D_small_s and Ref.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import h5py, numpy as np, torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from scripts.cnn_distiller.ga_efficientvgg_arch import GAEfficientVGGSearch

IMAGENETTE_MAP = {
    "n01440764": 0, "n02102040": 1, "n02979186": 2, "n03000684": 3, "n03028079": 4,
    "n03394916": 5, "n03417042": 6, "n03425413": 7, "n03445777": 8, "n03888257": 9,
}

p = argparse.ArgumentParser()
p.add_argument("--device",      default="auto")
p.add_argument("--seed",        type=int, default=42)
p.add_argument("--data_img",    default="data/imagenette2-320")
p.add_argument("--data_h5",     default="data/store.h5")
p.add_argument("--population",  type=int, default=12)
p.add_argument("--generations", type=int, default=6)
p.add_argument("--top_k",       type=int, default=5)
p.add_argument("--eval_epochs", type=int, default=5)
p.add_argument("--eval_frac",   type=float, default=0.10)
p.add_argument("--batch",       type=int, default=32)
args = p.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
OUT  = ROOT / "results" / f"cnn_step016_ga_evgg_results__{SLOT}.json"

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class StudentDataset(Dataset):
    def __init__(self, img_root, h5_path, split, transform=None):
        split_dir = Path(img_root) / split
        self._folder = ImageFolder(str(split_dir), transform=transform)
        self._remap  = {v: IMAGENETTE_MAP[k] for k, v in self._folder.class_to_idx.items()}
        with h5py.File(h5_path, "r") as f:
            self.teacher_feat = torch.from_numpy(f[f"{split}/features"][:])
            self.soft_labels  = torch.from_numpy(f[f"{split}/soft_labels"][:])
        assert len(self._folder) == len(self.teacher_feat)

    def __len__(self): return len(self._folder)
    def __getitem__(self, i):
        img, fl = self._folder[i]
        return img, self.teacher_feat[i], self.soft_labels[i], self._remap[fl]


def main():
    img_root = ROOT / args.data_img
    h5_path  = ROOT / args.data_h5
    for path in (img_root, h5_path):
        if not path.exists(): print(f"ERROR: {path} not found"); sys.exit(1)

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    train_ds = StudentDataset(str(img_root), str(h5_path), "train", TRAIN_TF)
    val_ds   = StudentDataset(str(img_root), str(h5_path), "val",   VAL_TF)

    print(f"cnn_step016 EfficientVGG GA search")
    print(f"pop={args.population} gen={args.generations} top_k={args.top_k}")
    print(f"eval: {args.eval_epochs}ep × {args.eval_frac*100:.0f}% data")
    print(f"Device: {DEVICE} | Slot: {SLOT}")
    print(f"Anchors: D_small_s=70.95%(57.5M), Ref=73.17%(183M), F_wide=74.42%(560M)\n")

    ga = GAEfficientVGGSearch(
        train_ds=train_ds, val_ds=val_ds,
        population=args.population, generations=args.generations,
        top_k=args.top_k, eval_fraction=args.eval_frac,
        eval_epochs=args.eval_epochs, batch=args.batch,
        device=str(DEVICE),
    )
    results = ga.run()

    print(f"\n{'='*60}")
    print(f"EfficientVGG GA TOP {args.top_k} CONFIGS")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        c = r["config"]
        print(
            f"  [{i+1}] fit={r['ga_fitness']:.4f}  acc={r['ga_acc']:.4f}"
            f"  MACs={r['macs_M']:.1f}M  params={r['params']:,}"
            f"\n       C=({c['C1']},{c['C2']},{c['C3']})"
            f"  k={c['dw_kernel']}  exp={c['expansion']}"
            f"  side={c['use_side_branch']}  crelu={c['use_crelu_block3']}"
        )

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "ga_config": vars(args), "top_configs": results,
        "anchors": {
            "D_small_s": {"T1": 0.7095, "macs_M": 57.5, "params": 150000},
            "Ref":       {"T1": 0.7317, "macs_M": 183.0, "params": 419000},
            "F_wide":    {"T1": 0.7442, "macs_M": 560.0, "params": 825000},
        },
    }, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
