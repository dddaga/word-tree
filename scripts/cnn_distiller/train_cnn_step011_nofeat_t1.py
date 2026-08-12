"""cnn_step011 — GA_2 T1, feat loss disabled (wf=0). Diagnostic for T1 failure.

Evidence so far:
  T0 (20ep, wf=0.5): GA_2 → 41.32%  [WORKS]
  step007 T1 (75ep, wf=0.5): GA_2 → 35.44%@ep8, degrades to ~24%  [FAILS]
  step009 T1+warmup (75ep, wf=0.5): GA_2 → 33.07%@ep24, degrades to 23%  [FAILS]
  step010 T1+SGDR (75ep, wf=0.5): RUNNING

Competing hypotheses:
  A: AdamW optimizer momentum trap — v_t accumulates at high LR, steers away from basin
  B2: CReLU-VGG16 feature incompatibility — feat cosine loss (wf=0.5) pushes CReLU
      features toward VGG16 (ReLU-only) space over 75ep, killing negative-half expressivity.
      T0 survives because 20ep insufficient for feat loss to converge to degenerate state.

This experiment: wf=0, wd=0.3 (×16 = 4.8), wc=0.2 — identical to step007 except feat loss off.
If training stabilizes and accuracy improves monotonically → B2 confirmed.
If training still peaks early and degrades → A is the culprit.
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import h5py, numpy as np, torch, torch.nn.functional as F
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
args = p.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
OUT  = ROOT / "results" / f"cnn_step011_nofeat_t1_seed{args.seed}__{SLOT}.json"

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


def distil_loss_nofeat(logits, soft, labels, wd=0.3, wc=0.2):
    """KL + CE only — no feature cosine loss (wf=0)."""
    ce  = F.cross_entropy(logits, labels)
    kl  = F.kl_div(F.log_softmax(logits / 4, -1), soft.clamp(1e-8), reduction="batchmean")
    return wd * 16 * kl + wc * ce


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
    n_par = sum(q.numel() for q in model.parameters())
    print(f"cnn_step011 GA_2 T1+nofeat | {n_par:,} params | {macs:.1f}M MACs")
    print(f"Loss: wf=0 (NO FEAT COSINE), wd=0.3 (×16 KL), wc=0.2 (CE)")
    print(f"Schedule: cosine T_max={args.epochs} eta_min=1e-5")
    print(f"Device: {DEVICE} | Slot: {SLOT}")
    print(f"Baseline: step007 (wf=0.5) best=35.44%@ep8. If monotone → B2 confirmed.\n")

    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    best  = 0.0; best_ep = 0; t0 = time.time()
    model.train()
    for ep in range(1, args.epochs + 1):
        for img, tf, sl, lbl in tr:
            img, sl, lbl = img.to(DEVICE), sl.to(DEVICE), lbl.to(DEVICE)
            logits, _ = model(img)
            loss = distil_loss_nofeat(logits, sl, lbl)
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
    print(f"cnn_step011 GA_2 T1+nofeat RESULT")
    print(f"  best={best:.4f} @ep{best_ep}  elapsed={elapsed:.0f}s")
    print(f"  MACs={macs:.1f}M  params={n_par:,}")
    print(f"  vs step007 (wf=0.5): {best*100-35.44:+.2f}pp  (B2 signal: positive=feat-loss caused degradation)")
    print(f"  vs T0 baseline (41.32%): {best*100-41.32:+.2f}pp")
    print(f"  vs Ref T2=77.35%: {best*100-77.35:+.2f}pp  → {adv}")
    monotone_flag = "MONOTONE (B2 confirmed)" if best_ep >= 60 else f"PEAKED@ep{best_ep} (A+B2 mixed)"
    print(f"  Training pattern: {monotone_flag}")
    print(f"{'='*60}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "config": GA2_CFG, "best": best, "best_ep": best_ep,
        "macs_M": macs, "params": n_par, "elapsed_s": int(elapsed),
        "loss_weights": {"wf": 0.0, "wd": 0.3, "wc": 0.2},
        "tier": "T1", "advance": adv,
        "vs_step007_pp": round(best * 100 - 35.44, 2),
        "vs_t0_baseline_pp": round(best * 100 - 41.32, 2),
        "monotone_flag": monotone_flag,
    }, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
