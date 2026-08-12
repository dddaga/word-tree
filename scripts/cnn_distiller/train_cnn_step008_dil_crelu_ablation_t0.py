"""cnn_step008 — 2×2 ablation: dilation × CReLU placement.

GA search never tested true dilation for n_br=1 configs (_dil_rates returns (1,)
when n_br==1 regardless of max_dil). This ablation fixes that gap.

Design: C=(32,64,384), br=1, exp=1 — GA_2's winning channel profile.
  Factor A: dil_rates = (1,) vs (4,)   [real dilation effect]
  Factor B: crelu_mask = (T,T,F) vs (F,T,T)  [early vs late CReLU]

Configs:
  ctrl_dil1_early : dil=1 crelu=(T,T,F) = GA_2 actual  [T0 known: 41.32%]
  exp_dil4_early  : dil=4 crelu=(T,T,F) = true dilation + early CReLU
  ctrl_dil1_late  : dil=1 crelu=(F,T,T) = GA_1 actual  [T0 known: 37.99%]
  exp_dil4_late   : dil=4 crelu=(F,T,T) = true dilation + late CReLU

Ref T0 = 36.82%. Advance: Δ > −0.5pp.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import h5py, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from scripts.cnn_distiller.model_multiscale import MultiScaleCNN, count_macs

ABLATION_CONFIGS = {
    "dil1_crelu_early": dict(channels=(32,64,384), dil_rates=(1,), expansion=1, crelu_mask=(True,True,False)),
    "dil4_crelu_early": dict(channels=(32,64,384), dil_rates=(4,), expansion=1, crelu_mask=(True,True,False)),
    "dil1_crelu_late":  dict(channels=(32,64,384), dil_rates=(1,), expansion=1, crelu_mask=(False,True,True)),
    "dil4_crelu_late":  dict(channels=(32,64,384), dil_rates=(4,), expansion=1, crelu_mask=(False,True,True)),
}
REF_T0 = 0.3682

p = argparse.ArgumentParser()
p.add_argument("--device",   default="auto")
p.add_argument("--seed",     type=int, default=42)
p.add_argument("--data_img", default="data/imagenette2-320")
p.add_argument("--data_h5",  default="data/store.h5")
p.add_argument("--batch",    type=int, default=32)
p.add_argument("--epochs",   type=int, default=20)
args = p.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
OUT  = ROOT / "results" / f"cnn_step008_ablation_seed{args.seed}__{SLOT}.json"

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


def run_config(name, cfg, tr, va):
    torch.manual_seed(args.seed)
    model = MultiScaleCNN(**cfg).to(DEVICE)
    macs  = count_macs(model) / 1e6
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\n{'─'*55}\n{name}: {n_par:,}p  {macs:.1f}M MACs  dil={cfg['dil_rates']}  crelu={cfg['crelu_mask']}")
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    best  = 0.0; t0 = time.time()
    model.train()
    for ep in range(1, args.epochs + 1):
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
    delta = best - REF_T0
    adv = "ADVANCE" if delta > -0.005 else "KILL"
    print(f"  -> best={best:.4f}  Δ_ref={delta:+.4f}  {adv}  ({time.time()-t0:.0f}s)")
    return {"best": best, "delta": delta, "macs_M": macs, "params": n_par,
            "config": {k: str(v) for k, v in cfg.items()}, "verdict": adv}


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

    print(f"cnn_step008 2×2 ablation: dilation × CReLU placement")
    print(f"Device={DEVICE}  epochs={args.epochs}  50% data  Ref_T0={REF_T0:.4f}")
    print(f"Prior results: dil1_crelu_early=41.32%  dil1_crelu_late=37.99%")

    results = {}
    for name, cfg in ABLATION_CONFIGS.items():
        results[name] = run_config(name, cfg, tr, va)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*55}")
    print(f"cnn_step008 ABLATION SUMMARY  (Ref_T0={REF_T0:.4f})")
    print(f"{'='*55}")
    for name, r in results.items():
        print(f"  {name:22s}  {r['macs_M']:6.1f}M  best={r['best']:.4f}  Δ={r['delta']:+.4f}  {r['verdict']}")
    # 2×2 effect sizes
    a = results.get("dil1_crelu_early", {}).get("best", 0)
    b = results.get("dil4_crelu_early", {}).get("best", 0)
    c = results.get("dil1_crelu_late",  {}).get("best", 0)
    d = results.get("dil4_crelu_late",  {}).get("best", 0)
    print(f"\n  Dilation effect (early crelu): dil4−dil1 = {b-a:+.4f}")
    print(f"  Dilation effect (late crelu):  dil4−dil1 = {d-c:+.4f}")
    print(f"  CReLU placement (dil=1): early−late = {a-c:+.4f}")
    print(f"  CReLU placement (dil=4): early−late = {b-d:+.4f}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
