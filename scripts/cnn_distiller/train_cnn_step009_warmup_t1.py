"""cnn_step009 — GA_2 T1 with LR warmup (diagnostic for step007 failure).

step007 failed: best=35.44%@ep8, then degraded 67 epochs. Ref T1 (same lr=3e-4)
reached 73.17% — so general T1 training works. Hypothesis: CReLU at C=32/64 with
LN over few channels is unstable at high initial LR.

Fix: 10ep linear warmup (lr: 1e-5 → 3e-4) then cosine annealing to 1e-5 over 65ep.
Advance threshold: same as step007 (≥76.85%). This is diagnostic — compare final best.
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from scripts.cnn_distiller.model_multiscale import MultiScaleCNN, count_macs

GA2_CFG = dict(channels=(32, 64, 384), dil_rates=(1,), expansion=1,
               crelu_mask=(True, True, False))

import argparse
p = argparse.ArgumentParser()
p.add_argument("--device",   default="auto")
p.add_argument("--seed",     type=int, default=42)
p.add_argument("--data_img", default="data/imagenette2-320")
p.add_argument("--data_h5",  default="data/store.h5")
p.add_argument("--batch",    type=int, default=32)
p.add_argument("--epochs",   type=int, default=75)
p.add_argument("--warmup",   type=int, default=10)
args = p.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
OUT  = ROOT / "results" / f"cnn_step009_warmup_t1_seed{args.seed}__{SLOT}.json"

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


def main():
    img_root = ROOT / args.data_img; h5_path = ROOT / args.data_h5
    for path in (img_root, h5_path):
        if not path.exists(): print(f"ERROR: {path} not found"); sys.exit(1)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    train_full = StudentDataset(str(img_root), str(h5_path), "train", TRAIN_TF)
    val_ds     = StudentDataset(str(img_root), str(h5_path), "val",   VAL_TF)
    n = len(train_full)
    g = torch.Generator().manual_seed(args.seed)
    sub_idx   = torch.randperm(n, generator=g)[:n // 2].tolist()
    train_sub = Subset(train_full, sub_idx)
    nw = 0; pin = False
    tr = DataLoader(train_sub, batch_size=args.batch, shuffle=True,  num_workers=nw, pin_memory=pin)
    va = DataLoader(val_ds,    batch_size=args.batch, shuffle=False, num_workers=nw, pin_memory=pin)

    model = MultiScaleCNN(**GA2_CFG).to(DEVICE)
    macs  = count_macs(model) / 1e6
    n_par = sum(p.numel() for p in model.parameters())
    cos_eps = args.epochs - args.warmup
    print(f"cnn_step009 GA_2 T1+warmup | {n_par:,} params | {macs:.1f}M MACs")
    print(f"Schedule: {args.warmup}ep warmup (1e-5→3e-4) + {cos_eps}ep cosine (→1e-5)")
    print(f"Device: {DEVICE} | Slot: {SLOT}")
    print(f"Advance threshold (vs Ref T2=77.35%): >= 76.85%\n")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1/30, end_factor=1.0, total_iters=args.warmup)
    cos_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cos_eps, eta_min=1e-5)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cos_sched], milestones=[args.warmup])

    best = 0.0; best_ep = 0; t0 = time.time()
    model.train()
    for ep in range(1, args.epochs + 1):
        for img, tf, sl, lbl in tr:
            img, tf, sl, lbl = img.to(DEVICE), tf.to(DEVICE), sl.to(DEVICE), lbl.to(DEVICE)
            logits, sfeat = model(img)
            loss = distil_loss(logits, sfeat, tf, sl, lbl)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        acc = evaluate(model, va)
        if acc > best: best, best_ep = acc, ep
        lr_now = sched.get_last_lr()[0]
        if ep % 5 == 0:
            print(f"  ep{ep:3d}  val={acc:.4f}  best={best:.4f}@ep{best_ep}  lr={lr_now:.2e}", flush=True)

    elapsed = time.time() - t0
    adv = "ADVANCE → T2" if best >= 0.7685 else "BELOW_THRESHOLD"
    print(f"\n{'='*60}")
    print(f"cnn_step009 GA_2 T1+warmup RESULT")
    print(f"  best={best:.4f} @ep{best_ep}  elapsed={elapsed:.0f}s")
    print(f"  MACs={macs:.1f}M  params={n_par:,}")
    print(f"  vs step007 (no warmup): best=35.44% — delta={best*100-35.44:+.2f}pp")
    print(f"  vs Ref T2=77.35%: {best*100-77.35:+.2f}pp  → {adv}")
    print(f"{'='*60}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "config": GA2_CFG, "best": best, "best_ep": best_ep,
        "macs_M": macs, "params": n_par, "elapsed_s": int(elapsed),
        "warmup_eps": args.warmup, "tier": "T1", "advance": adv,
        "vs_step007_delta_pp": round(best * 100 - 35.44, 2),
    }, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
