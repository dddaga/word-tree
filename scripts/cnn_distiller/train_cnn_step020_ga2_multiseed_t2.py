"""CNN distillation step020 — GA2 multi-seed T2 variance (seeds 1, 2, 3).
seed=42 done in step019: best=77.25% @ep25. Paper needs mean±std ≥4 seeds.
Config: GA#2 C=(32,64,384) k=3 exp=2 crelu=(T,T,F)
Output: results/cnn_step020_ga2_multiseed_t2__{SLOT}.json
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
p.add_argument("--device",   default="auto")
p.add_argument("--epochs",   type=int, default=150)
p.add_argument("--seeds",    default="1,2,3")
p.add_argument("--data_img", default="data/imagenette2-320")
p.add_argument("--data_h5",  default="data/store.h5")
p.add_argument("--batch",    type=int, default=32)
args = p.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
OUT  = ROOT / "results" / f"cnn_step020_ga2_multiseed_t2__{SLOT}.json"

GA2_CH, GA2_K, GA2_EXP, GA2_SIDE, GA2_CRELU = (32, 64, 384), 3, 2, False, True
SEED42_BEST = 0.7725
REF_T2, DSMALL_T2 = 0.7761, 0.7582

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
        img, fl = self._f[i]; return img, self.feat[i], self.soft[i], self._r[fl]


def _std(logits, T):
    mu = logits.mean(-1, keepdim=True)
    sig = logits.std(-1, keepdim=True).clamp(1e-6)
    return (logits - mu) / sig / T


def dkd_loss(logits, soft, labels, T=4.0, a=1.0, b=2.0):
    B, C = logits.shape
    ps = F.softmax(_std(logits, T), -1)
    tl = torch.log(soft.clamp(1e-8))
    tm, ts = tl.mean(-1, keepdim=True), tl.std(-1, keepdim=True).clamp(1e-6)
    pt = F.softmax((tl - tm) / ts / T, -1)
    mask = F.one_hot(labels, C).float()
    psy = (ps * mask).sum(-1).clamp(1e-8, 1 - 1e-8)
    pty = (pt * mask).sum(-1).clamp(1e-8, 1 - 1e-8)
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
        correct += (model(img)[0].argmax(1) == lbl).sum().item()
        total += len(lbl)
    return correct / total


def main():
    img_root, h5 = ROOT / args.data_img, ROOT / args.data_h5
    for p_ in (img_root, h5):
        if not p_.exists(): print(f"ERROR: {p_} not found"); sys.exit(1)

    full = StudentDataset(str(img_root), str(h5), "train", TRAIN_TF)
    val  = StudentDataset(str(img_root), str(h5), "val",   VAL_TF)
    nw = 2 if DEVICE.type == "cuda" else 0
    va = DataLoader(val, batch_size=args.batch, shuffle=False, num_workers=nw)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    n_p = sum(p_.numel() for p_ in EfficientVGG(GA2_CH, GA2_K, GA2_EXP,
                                                  GA2_SIDE, GA2_CRELU).parameters())
    macs = count_macs(EfficientVGG(GA2_CH, GA2_K, GA2_EXP, GA2_SIDE, GA2_CRELU)) / 1e6
    print(f"cnn_step020 GA2 multi-seed T2 | {args.epochs}ep 100%data | device={DEVICE} slot={SLOT}")
    print(f"GA#2: C=(32,64,384) k=3 exp=2 crelu  params={n_p:,}  MACs={macs:.1f}M")
    print(f"seed42 (step019): {SEED42_BEST:.4f}  new seeds: {seeds}\n")

    results = {}
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = EfficientVGG(GA2_CH, GA2_K, GA2_EXP, GA2_SIDE, GA2_CRELU).to(DEVICE)
        tr = DataLoader(full, batch_size=args.batch, shuffle=True, num_workers=nw)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
        best = 0.0; best_ep = 0; t_start = time.time()
        print(f"{'─'*60}\nSeed={seed}")

        for ep in range(args.epochs):
            model.train()
            for img, tf_, soft, lbl in tr:
                img, tf_, soft, lbl = (img.to(DEVICE), tf_.to(DEVICE),
                                       soft.to(DEVICE), lbl.to(DEVICE))
                logits, sf = model(img)
                loss = distil_loss(logits, sf, tf_, soft, lbl)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
            if (ep + 1) % 5 == 0:
                acc = evaluate(model, va)
                if acc > best: best = acc; best_ep = ep + 1
                print(f"  ep{ep+1:3d}  val={acc:.4f}  best={best:.4f}"
                      f"  lr={sch.get_last_lr()[0]:.2e}", flush=True)

        elapsed = time.time() - t_start
        print(f"  -> seed={seed}  best={best:.4f} @ep{best_ep}  ({elapsed:.0f}s)")
        results[seed] = {"best": best, "best_ep": best_ep, "elapsed": elapsed}

    accs = [results[s]["best"] for s in seeds] + [SEED42_BEST]
    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
    print(f"\n{'='*60}\ncnn_step020 SUMMARY — GA2 multi-seed T2 (4 seeds: 42,{','.join(str(s) for s in seeds)})\n{'='*60}")
    print(f"  seed=42 (step019): {SEED42_BEST:.4f}")
    for s in seeds:
        r = results[s]
        print(f"  seed={s}: {r['best']:.4f} @ep{r['best_ep']}  ({r['elapsed']:.0f}s)")
    print(f"  mean={mean_acc:.4f}  std={std_acc:.4f}  n={len(accs)}")
    tag = ("STRONG" if mean_acc >= REF_T2 else
           "EFF-PARETO" if mean_acc >= DSMALL_T2 else "WEAK")
    print(f"  -> GA2 T2: {mean_acc:.2%} ±{std_acc:.2%}  {tag}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"seeds": results, "seed42": SEED42_BEST,
                               "mean": mean_acc, "std": std_acc,
                               "n_seeds": len(accs), "tag": tag}, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
