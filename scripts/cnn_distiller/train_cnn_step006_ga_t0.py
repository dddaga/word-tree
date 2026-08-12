"""cnn_step006 — GA architecture search + T0 proper eval of winners.

Phase 1 (GA): population=10, generations=6, 5ep/10% data — fast pre-filter.
  Fitness: val_acc * (183.2M / model_macs)^0.1  (accuracy × efficiency).
  Search: channels (C1,C2,C3), parallel dilation branches, CReLU per block,
          expansion factor.

Phase 2 (T0): top-4 GA configs trained 20ep, 50% data, full distil_loss.
  Advance rule: acc ≥ Ref − 0.5pp (efficiency candidates) OR > Ref + 0.5pp.
  Ref baseline: EfficientVGG Ref (77.35% T2 seed=42; T0 baseline ~70.3%).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from scripts.cnn_distiller.model_efficient_vgg import EfficientVGG, count_macs as _cmacs
from scripts.cnn_distiller.model_multiscale import MultiScaleCNN, count_macs
from scripts.cnn_distiller.ga_cnn_arch import GACNNSearch, _dil_rates, _build

parser = argparse.ArgumentParser()
parser.add_argument("--device",   default="auto")
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data_img", default="data/imagenette2-320")
parser.add_argument("--data_h5",  default="data/store.h5")
parser.add_argument("--batch",    type=int, default=32)
parser.add_argument("--ga_pop",   type=int, default=10)
parser.add_argument("--ga_gen",   type=int, default=6)
parser.add_argument("--top_k",    type=int, default=4)
parser.add_argument("--skip_ga",  action="store_true", help="Skip GA, use --ga_configs JSON")
parser.add_argument("--ga_configs", default=None, help="Path to ga_results JSON to use")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)
SEED   = args.seed
SLOT   = os.environ.get("SGN_SLOT", "local")
GA_OUT = ROOT / "results" / f"cnn_step006_ga_results__{SLOT}.json"
T0_OUT = ROOT / "results" / f"cnn_step006_t0_seed{SEED}__{SLOT}.json"

IMAGENETTE_MAP = {
    "n01440764": 0, "n02102040": 1, "n02979186": 2, "n03000684": 3, "n03028079": 4,
    "n03394916": 5, "n03417042": 6, "n03425413": 7, "n03445777": 8, "n03888257": 9,
}
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
        self._remap = {v: IMAGENETTE_MAP[k] for k, v in self._folder.class_to_idx.items()}
        with h5py.File(h5_path, "r") as f:
            self.teacher_feat = torch.from_numpy(f[f"{split}/features"][:])
            self.soft_labels  = torch.from_numpy(f[f"{split}/soft_labels"][:])
        assert len(self._folder) == len(self.teacher_feat)

    def __len__(self): return len(self._folder)
    def __getitem__(self, i):
        img, fl = self._folder[i]
        return img, self.teacher_feat[i], self.soft_labels[i], self._remap[fl]


def distil_loss(logits, sfeat, tfeat, soft, labels, wf=0.5, wd=0.3, wc=0.2):
    feat = (1 - F.cosine_similarity(sfeat, tfeat, dim=1)).mean()
    ce   = F.cross_entropy(logits, labels)
    kl   = F.kl_div(F.log_softmax(logits / 4, -1), soft.clamp(1e-8), reduction="batchmean")
    return wf * feat + wd * 16 * kl + wc * ce


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); correct = total = 0
    for img, _, _, lbl in loader:
        img, lbl = img.to(DEVICE), lbl.to(DEVICE)
        logits, _ = model(img)
        correct += (logits.argmax(1) == lbl).sum().item(); total += len(lbl)
    return correct / total if total else 0.0


def train_config(model, tr, va, epochs=20):
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best  = 0.0; t0 = time.time()
    model.to(DEVICE); model.train()
    for ep in range(1, epochs + 1):
        for img, tf, sl, lbl in tr:
            img, tf, sl, lbl = img.to(DEVICE), tf.to(DEVICE), sl.to(DEVICE), lbl.to(DEVICE)
            logits, sfeat = model(img)
            loss = distil_loss(logits, sfeat, tf, sl, lbl)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        acc = evaluate(model, va)
        best = max(best, acc)
        if ep % 5 == 0:
            print(f"  ep{ep:3d}  val={acc:.4f}  best={best:.4f}", flush=True)
    return best, time.time() - t0


def main():
    img_root = ROOT / args.data_img; h5_path = ROOT / args.data_h5
    for p in (img_root, h5_path):
        if not p.exists(): print(f"ERROR: {p} not found"); sys.exit(1)
    torch.manual_seed(SEED); np.random.seed(SEED)

    train_full = StudentDataset(str(img_root), str(h5_path), "train", TRAIN_TF)
    val_ds     = StudentDataset(str(img_root), str(h5_path), "val",   VAL_TF)
    n = len(train_full)
    g = torch.Generator().manual_seed(SEED)
    sub_idx = torch.randperm(n, generator=g)[:n // 2].tolist()
    train_sub = Subset(train_full, sub_idx)
    nw = 2 if DEVICE.type == "cuda" else 0; pin = DEVICE.type == "cuda"
    tr = DataLoader(train_sub, batch_size=args.batch, shuffle=True, num_workers=nw, pin_memory=pin)
    va = DataLoader(val_ds,    batch_size=args.batch, shuffle=False, num_workers=nw, pin_memory=pin)

    # Phase 1 — GA search
    if args.skip_ga and args.ga_configs:
        with open(args.ga_configs) as f: ga_winners = json.load(f)["winners"]
        print(f"Loaded {len(ga_winners)} configs from {args.ga_configs}")
    else:
        print(f"\n{'='*60}\nPhase 1: GA search  pop={args.ga_pop}  gen={args.ga_gen}\n{'='*60}")
        dev_str = str(DEVICE)
        ga = GACNNSearch(train_full, val_ds, population=args.ga_pop,
                         generations=args.ga_gen, top_k=args.top_k,
                         eval_fraction=0.10, eval_epochs=5, batch=args.batch, device=dev_str)
        ga_winners = ga.run()
        GA_OUT.parent.mkdir(exist_ok=True)
        GA_OUT.write_text(json.dumps({"winners": ga_winners}, indent=2))
        print(f"\nGA done. Winners saved → {GA_OUT}")
        for i, w in enumerate(ga_winners):
            print(f"  #{i+1}: fit={w['ga_fitness']:.4f} MACs={w['macs_M']:.1f}M cfg={w['config']}")

    # Phase 2 — T0 proper eval (20ep, 50% data, full distil_loss)
    print(f"\n{'='*60}\nPhase 2: T0 proper eval (20ep, 50% data)\n{'='*60}")
    # Reference baseline
    torch.manual_seed(SEED)
    ref_model = EfficientVGG()
    ref_macs  = _cmacs(ref_model) / 1e6
    ref_np    = sum(p.numel() for p in ref_model.parameters())
    print(f"\nRef (EfficientVGG): {ref_np:,} params  {ref_macs:.1f}M MACs")
    ref_best, _ = train_config(ref_model, tr, va, epochs=20)
    results = {"Ref": {"best": ref_best, "params": ref_np, "macs_M": ref_macs, "delta": 0.0}}

    for i, w in enumerate(ga_winners):
        cfg = w["config"]
        tag = f"GA_{i+1}"
        torch.manual_seed(SEED)
        model = _build(cfg)
        macs  = count_macs(model) / 1e6
        np_   = sum(p.numel() for p in model.parameters())
        print(f"\n{'-'*50}\n{tag}: {np_:,} params  {macs:.1f}M MACs  cfg={cfg}")
        best, elapsed = train_config(model, tr, va, epochs=20)
        delta = best - ref_best
        results[tag] = {"best": best, "params": np_, "macs_M": macs,
                        "delta": delta, "config": cfg, "elapsed_s": int(elapsed)}
        print(f"  -> best={best:.4f}  Δ_ref={delta:+.4f}  {elapsed:.0f}s")

    T0_OUT.parent.mkdir(exist_ok=True)
    T0_OUT.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*60}\ncnn_step006 T0 SUMMARY\n{'='*60}")
    for tag, r in results.items():
        adv = "ADVANCE" if r.get("delta", 0) > -0.005 else "KILL"
        print(f"  {tag:12s}  {r['params']:>7,}p  {r['macs_M']:>6.1f}M  "
              f"best={r['best']:.4f}  Δ={r.get('delta',0):+.4f}  {adv}")
    print(f"\n-> {T0_OUT}")


if __name__ == "__main__":
    main()
