"""CNN distillation step034 — GA5 C=(48,96,384) k=5 T0 scout.
Hypothesis: k=7 depthwise is baseline for Ref/D_small_s (both STRONG at k=7).
GA5 uses k=3 (efficiency). k=5 = intermediate receptive field, minimal Pareto hit.
Expected: ~531K params (+8.4K), ~146M MACs (+4.8M). Single variable change.
Prior: GA5 k=3 T2=77.79% (step030). Baseline: GA5+Mixup T1=74.73% (step029).
Advance>=75.23% (+0.5pp). T0: 20ep 50%data seed=42.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from scripts.cnn_distiller.model_efficient_vgg import EfficientVGG, count_macs

p = argparse.ArgumentParser()
p.add_argument("--device", default="auto")
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--mixup_alpha", type=float, default=0.2)
p.add_argument("--data_img", default="data/imagenette2-320")
p.add_argument("--data_h5", default="data/store.h5")
p.add_argument("--batch", type=int, default=32)
args = p.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
OUT = ROOT / "results" / f"cnn_step034_ga5k5_t0__{SLOT}.json"

GA5_CH, GA5_K, GA5_EXP, GA5_SIDE, GA5_CRELU = (48, 96, 384), 5, 2, False, True
GA5_K3_T1 = 0.7473   # step029 GA5 k=3 Mixup T1
ADVANCE = 0.7523      # +0.50pp

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
    def __init__(self, img_root, h5_path, split, tf=None, fraction=1.0, rng=None):
        d = Path(img_root) / split
        self._f = ImageFolder(str(d), transform=tf)
        self._r = {v: IMAGENETTE_MAP[k] for k, v in self._f.class_to_idx.items()}
        with h5py.File(h5_path, "r") as f:
            self.feat = torch.from_numpy(f[f"{split}/features"][:])
            self.soft = torch.from_numpy(f[f"{split}/soft_labels"][:])
        assert len(self._f) == len(self.feat)
        if fraction < 1.0 and rng is not None:
            n = int(len(self._f) * fraction)
            self._idx = rng.choice(len(self._f), n, replace=False).tolist()
        else:
            self._idx = list(range(len(self._f)))

    def __len__(self): return len(self._idx)

    def __getitem__(self, i):
        j = self._idx[i]
        img, fl = self._f[j]
        return img, self.feat[j], self.soft[j], self._r[fl]


def mixup_batch(img, feat, soft, lbl, alpha):
    if alpha <= 0.0:
        return img, feat, soft, lbl, lbl, torch.ones(len(lbl), device=img.device)
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(img.size(0), device=img.device)
    return (lam * img + (1 - lam) * img[idx], lam * feat + (1 - lam) * feat[idx],
            lam * soft + (1 - lam) * soft[idx], lbl, lbl[idx],
            torch.full((img.size(0),), lam, device=img.device))


def dkd_loss(logits, soft, labels, T=4.0, a=1.0, b=2.0):
    B, C = logits.shape
    mu = logits.mean(-1, keepdim=True); sig = logits.std(-1, keepdim=True).clamp(1e-6)
    ps = F.softmax((logits - mu) / sig / T, -1)
    tl = torch.log(soft.clamp(1e-8)); tm = tl.mean(-1, keepdim=True); ts = tl.std(-1, keepdim=True).clamp(1e-6)
    pt = F.softmax((tl - tm) / ts / T, -1)
    mask = F.one_hot(labels, C).float()
    psy = (ps * mask).sum(-1).clamp(1e-8, 1 - 1e-8); pty = (pt * mask).sum(-1).clamp(1e-8, 1 - 1e-8)
    tckd = -(pty * psy.log() + (1 - pty) * (1 - psy).log()).mean()
    nckd = F.kl_div(((ps * (1 - mask)) / (1 - psy.unsqueeze(-1) + 1e-8) + 1e-8).log(),
                    (pt * (1 - mask)) / (1 - pty.unsqueeze(-1) + 1e-8), reduction="batchmean")
    return T * T * (a * tckd + b * nckd)


def distil_loss(logits, sf, tf, soft, lbl_a, lbl_b, lam):
    feat = (1.0 - F.cosine_similarity(sf, tf, dim=1)).mean()
    kd = dkd_loss(logits, soft, lbl_a)
    ce = lam.mean() * F.cross_entropy(logits, lbl_a) + (1 - lam.mean()) * F.cross_entropy(logits, lbl_b)
    return 0.50 * feat + 0.30 * kd + 0.20 * ce


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); correct = total = 0
    for img, _, _, lbl in loader:
        img, lbl = img.to(DEVICE), lbl.to(DEVICE)
        correct += (model(img)[0].argmax(1) == lbl).sum().item(); total += len(lbl)
    return correct / total


def main():
    img_root, h5 = ROOT / args.data_img, ROOT / args.data_h5
    for x in (img_root, h5):
        if not x.exists(): print(f"ERROR: {x} not found"); sys.exit(1)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    sub = StudentDataset(str(img_root), str(h5), "train", TRAIN_TF, fraction=0.5, rng=rng)
    va  = StudentDataset(str(img_root), str(h5), "val",   VAL_TF)
    nw = 2 if DEVICE.type == "cuda" else 0
    tr = DataLoader(sub, batch_size=args.batch, shuffle=True,  num_workers=nw)
    vl = DataLoader(va,  batch_size=args.batch, shuffle=False, num_workers=nw)
    model = EfficientVGG(GA5_CH, GA5_K, GA5_EXP, GA5_SIDE, GA5_CRELU).to(DEVICE)
    n_p = sum(q.numel() for q in model.parameters())
    macs = count_macs(EfficientVGG(GA5_CH, GA5_K, GA5_EXP, GA5_SIDE, GA5_CRELU)) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    print(f"cnn_step034 GA5 k=5 Mixup T0 | {args.epochs}ep 50%data | alpha={args.mixup_alpha}")
    print(f"  {DEVICE}  slot={SLOT}  params={n_p:,}  MACs={macs:.1f}M")
    print(f"  k=5 vs GA5 k=3 ({GA5_K3_T1:.4f} T1). Advance>={ADVANCE:.4f}")
    best = 0.0; best_ep = 0; t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for img, tf_, soft, lbl in tr:
            img, tf_, soft, lbl = img.to(DEVICE), tf_.to(DEVICE), soft.to(DEVICE), lbl.to(DEVICE)
            img, tf_, soft, lbl_a, lbl_b, lam = mixup_batch(img, tf_, soft, lbl, args.mixup_alpha)
            logits, sf = model(img)
            loss = distil_loss(logits, sf, tf_, soft, lbl_a, lbl_b, lam)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        if (ep + 1) % 5 == 0:
            acc = evaluate(model, vl)
            if acc > best: best = acc; best_ep = ep + 1
            print(f"  ep{ep+1:3d}  val={acc:.4f}  best={best:.4f}  lr={sch.get_last_lr()[0]:.2e}", flush=True)
    elapsed = time.time() - t0
    delta = best - GA5_K3_T1
    verdict = "ADVANCE" if best >= ADVANCE else "NO-GAIN"
    print(f"\n{'='*60}\ncnn_step034 RESULT — GA5 k=5 Mixup T0\n{'='*60}")
    print(f"  best={best:.4f} @ep{best_ep}  Δ_GA5k3={delta:+.4f}  {verdict}  ({elapsed:.0f}s)")
    adv = "ADVANCE → T1 GA5 k=5 (step035)" if verdict == "ADVANCE" else "NO-GAIN → k=5 no improvement; try k=7 or depth"
    print(f"  -> {adv}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"best": best, "best_ep": best_ep, "ga5k3_t1": GA5_K3_T1,
                               "delta": delta, "verdict": verdict, "elapsed": elapsed,
                               "dw_kernel": GA5_K, "channels": list(GA5_CH),
                               "params": n_p, "macs_M": macs}, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
