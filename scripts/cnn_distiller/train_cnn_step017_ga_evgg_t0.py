"""CNN distillation step017 — T0 validation of top-4 GA configs (step016).

GA search (step016) found EfficientVGG configs on efficiency frontier:
  GA1: C=(32,64,384) k=3 exp=1 side=F crelu=T  85.6M  471K  GA-fit=0.4761
  GA2: C=(32,64,384) k=3 exp=2 side=F crelu=T  98.5M  482K  GA-fit=0.4684
  GA3: C=(32,96,384) k=3 exp=2 side=F crelu=T 118.0M  516K  GA-fit=0.4670
  GA4: C=(32,64,256) k=3 exp=1 side=F crelu=F  62.9M  305K  GA-fit=0.4664

Known anchors (T0 from step003):
  D_small_s: 57.5M  150K  T0=61.0%
  Ref:      183.0M  419K  T0=70.3%

T0 advance rule: T0_acc >= Ref T0 (70.3%) → strong; >=D_small_s T0 (61.0%)
  at <183M MACs → efficiency-Pareto advance.
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
from scripts.cnn_distiller.model_efficient_vgg import EfficientVGG, count_macs

p = argparse.ArgumentParser()
p.add_argument("--device",   default="auto")
p.add_argument("--epochs",   type=int, default=20)
p.add_argument("--seed",     type=int, default=42)
p.add_argument("--data_img", default="data/imagenette2-320")
p.add_argument("--data_h5",  default="data/store.h5")
p.add_argument("--configs",  default="GA1,GA2,GA3,GA4")
p.add_argument("--batch",    type=int, default=32)
args = p.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
OUT  = ROOT / "results" / f"cnn_step017_ga_t0_seed{args.seed}__{SLOT}.json"

CONFIGS = {
    "GA1": ((32, 64, 384), 3, 1, False, True,  "GA#1: C=(32,64,384) k=3 exp=1 crelu"),
    "GA2": ((32, 64, 384), 3, 2, False, True,  "GA#2: C=(32,64,384) k=3 exp=2 crelu"),
    "GA3": ((32, 96, 384), 3, 2, False, True,  "GA#3: C=(32,96,384) k=3 exp=2 crelu"),
    "GA4": ((32, 64, 256), 3, 1, False, False, "GA#4: C=(32,64,256) k=3 exp=1"),
}
REF_T0, DSMALL_T0 = 0.7034, 0.6099

IMAGENETTE_MAP = {
    "n01440764": 0, "n02102040": 1, "n02979186": 2, "n03000684": 3, "n03028079": 4,
    "n03394916": 5, "n03417042": 6, "n03425413": 7, "n03445777": 8, "n03888257": 9,
}
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.1), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class StudentDataset(Dataset):
    def __init__(self, img_root, h5_path, split, tf=None):
        d = Path(img_root) / split
        self._f = ImageFolder(str(d), transform=tf)
        self._r = {v: IMAGENETTE_MAP[k] for k, v in self._f.class_to_idx.items()}
        with h5py.File(h5_path, "r") as f:
            self.feat = torch.from_numpy(f[f"{split}/features"][:])
            self.soft = torch.from_numpy(f[f"{split}/soft_labels"][:])
        assert len(self._f) == len(self.feat)
    def __len__(self): return len(self._f)
    def __getitem__(self, i):
        img, fl = self._f[i]
        return img, self.feat[i], self.soft[i], self._r[fl]


def _std(logits, T):
    mu = logits.mean(-1, keepdim=True); sig = logits.std(-1, keepdim=True).clamp(1e-6)
    return (logits - mu) / sig / T


def dkd_loss(logits, soft, labels, T=4.0, a=1.0, b=2.0):
    B, C = logits.shape
    ps = F.softmax(_std(logits, T), -1)
    tl = torch.log(soft.clamp(1e-8))
    tm, ts = tl.mean(-1, keepdim=True), tl.std(-1, keepdim=True).clamp(1e-6)
    pt = F.softmax((tl - tm) / ts / T, -1)
    mask = F.one_hot(labels, C).float()
    psy = (ps * mask).sum(-1).clamp(1e-8, 1-1e-8); pty = (pt * mask).sum(-1).clamp(1e-8, 1-1e-8)
    tckd = -(pty * psy.log() + (1 - pty) * (1 - psy).log()).mean()
    ps_nt = (ps * (1 - mask)) / (1 - psy.unsqueeze(-1) + 1e-8)
    pt_nt = (pt * (1 - mask)) / (1 - pty.unsqueeze(-1) + 1e-8)
    nckd = F.kl_div((ps_nt + 1e-8).log(), pt_nt, reduction="batchmean")
    return T * T * (a * tckd + b * nckd)


def distil_loss(logits, sf, tf, soft, labels):
    feat = (1.0 - F.cosine_similarity(sf, tf, dim=1)).mean()
    return 0.50 * feat + 0.30 * dkd_loss(logits, soft, labels) + 0.20 * F.cross_entropy(logits, labels)


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); correct = total = 0
    for img, _, _, lbl in loader:
        img, lbl = img.to(DEVICE), lbl.to(DEVICE)
        correct += (model(img)[0].argmax(1) == lbl).sum().item(); total += len(lbl)
    return correct / total


def make_model(key):
    torch.manual_seed(args.seed)
    ch, k, exp, side, crelu, _ = CONFIGS[key]
    return EfficientVGG(channels=ch, dw_kernel=k, expansion=exp,
                        use_side_branch=side, use_crelu_block3=crelu)


def main():
    img_root, h5 = ROOT / args.data_img, ROOT / args.data_h5
    for p_ in (img_root, h5):
        if not p_.exists(): print(f"ERROR: {p_} not found"); sys.exit(1)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    full  = StudentDataset(str(img_root), str(h5), "train", TRAIN_TF)
    val   = StudentDataset(str(img_root), str(h5), "val",   VAL_TF)
    idx   = torch.randperm(len(full), generator=torch.Generator().manual_seed(args.seed))[:len(full)//2]
    tr_ds = Subset(full, idx.tolist())
    nw = 2 if DEVICE.type == "cuda" else 0
    va = DataLoader(val, batch_size=args.batch, shuffle=False, num_workers=nw)

    keys = [k for k in args.configs.split(",") if k.strip() in CONFIGS]
    print(f"cnn_step017 EfficientVGG GA T0 | {args.epochs}ep 50%data | device={DEVICE} slot={SLOT}")
    print(f"Ref T0={REF_T0:.1%}  D_small_s T0={DSMALL_T0:.1%}  | configs: {keys}\n")

    results = {}
    for key in keys:
        m0 = make_model(key)
        n_p = sum(p_.numel() for p_ in m0.parameters()); macs = count_macs(m0) / 1e6
        ch, k, exp, side, crelu, desc = CONFIGS[key]
        print(f"{'─'*60}\n{key}: {desc}\n  params={n_p:,}  MACs={macs:.1f}M")

        model = m0.to(DEVICE)
        tr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=nw)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
        best = 0.0; t0 = time.time()

        for ep in range(args.epochs):
            model.train()
            for img, tf_, soft, lbl in tr:
                img, tf_, soft, lbl = img.to(DEVICE), tf_.to(DEVICE), soft.to(DEVICE), lbl.to(DEVICE)
                logits, sf = model(img)
                loss = distil_loss(logits, sf, tf_, soft, lbl)
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
            if (ep + 1) % 5 == 0:
                acc = evaluate(model, va); best = max(best, acc)
                print(f"  ep{ep+1:3d}  val={acc:.4f}  best={best:.4f}  lr={sch.get_last_lr()[0]:.2e}", flush=True)

        elapsed = time.time() - t0
        tag = "STRONG" if best >= REF_T0 else ("EFF-PARETO" if best >= DSMALL_T0 and macs < 183 else "WEAK")
        print(f"  -> best={best:.4f}  {tag}  ({elapsed:.0f}s)")
        results[key] = {"best": best, "params": n_p, "macs_M": macs, "tag": tag, "elapsed": elapsed}

    print(f"\n{'='*60}\ncnn_step017 SUMMARY\n{'='*60}")
    for k, r in results.items():
        print(f"  {k}: {r['best']:.4f}  {r['macs_M']:.1f}M  {r['params']:,}p  {r['tag']}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"configs": results, "ref_t0": REF_T0, "dsmall_t0": DSMALL_T0}, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
