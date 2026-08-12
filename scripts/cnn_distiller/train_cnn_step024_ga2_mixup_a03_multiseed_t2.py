"""CNN distillation step024 — GA2 + Mixup T2 multi-seed variance.
Launch if step021 (Mixup T1) ADVANCE >=73.01%.
Hypothesis: Mixup stabilizes ep25 peak, pushes mean above Ref=77.61%.
Config: GA#2 C=(32,64,384) k=3 exp=2 crelu=(T,T,F) mixup_alpha=0.3
seed=42 T2+Mixup best from step021 passed via --seed42_best or auto-loaded.
Output: results/cnn_step024_ga2_mixup_multiseed_t2__{SLOT}.json
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
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--seeds", default="1,2,3")
p.add_argument("--seed42_best", type=float, default=None)
p.add_argument("--mixup_alpha", type=float, default=0.3)
p.add_argument("--data_img", default="data/imagenette2-320")
p.add_argument("--data_h5", default="data/store.h5")
p.add_argument("--batch", type=int, default=32)
args = p.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
OUT = ROOT / "results" / f"cnn_step024_ga2_mixup_multiseed_t2__{SLOT}.json"
GA2_CH, GA2_K, GA2_EXP, GA2_SIDE, GA2_CRELU = (32, 64, 384), 3, 2, False, True
REF_T2, DSMALL_T2 = 0.7761, 0.7582
IMAGENETTE_MAP = {"n01440764":0,"n02102040":1,"n02979186":2,"n03000684":3,"n03028079":4,
                  "n03394916":5,"n03417042":6,"n03425413":7,"n03445777":8,"n03888257":9}
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.1), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
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


def mixup_batch(img, feat, soft, lbl, alpha):
    if alpha <= 0.0: return img, feat, soft, lbl, lbl, torch.ones(len(lbl), device=img.device)
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(img.size(0), device=img.device)
    return (lam*img+(1-lam)*img[idx], lam*feat+(1-lam)*feat[idx],
            lam*soft+(1-lam)*soft[idx], lbl, lbl[idx],
            torch.full((img.size(0),), lam, device=img.device))


def dkd_loss(logits, soft, labels, T=4.0, a=1.0, b=2.0):
    B, C = logits.shape
    mu = logits.mean(-1, keepdim=True); sig = logits.std(-1, keepdim=True).clamp(1e-6)
    ps = F.softmax((logits-mu)/sig/T, -1)
    tl = torch.log(soft.clamp(1e-8)); tm = tl.mean(-1,keepdim=True); ts = tl.std(-1,keepdim=True).clamp(1e-6)
    pt = F.softmax((tl-tm)/ts/T, -1)
    mask = F.one_hot(labels, C).float()
    psy = (ps*mask).sum(-1).clamp(1e-8,1-1e-8); pty = (pt*mask).sum(-1).clamp(1e-8,1-1e-8)
    tckd = -(pty*psy.log()+(1-pty)*(1-psy).log()).mean()
    nckd = F.kl_div(((ps*(1-mask))/(1-psy.unsqueeze(-1)+1e-8)+1e-8).log(),
                    (pt*(1-mask))/(1-pty.unsqueeze(-1)+1e-8), reduction="batchmean")
    return T*T*(a*tckd+b*nckd)


def distil_loss_mixup(logits, sf, tf, soft, lbl_a, lbl_b, lam):
    feat = (1.0-F.cosine_similarity(sf, tf, dim=1)).mean()
    kd = dkd_loss(logits, soft, lbl_a)
    ce = lam.mean()*F.cross_entropy(logits,lbl_a)+(1-lam.mean())*F.cross_entropy(logits,lbl_b)
    return 0.50*feat+0.30*kd+0.20*ce


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); correct = total = 0
    for img, _, _, lbl in loader:
        img, lbl = img.to(DEVICE), lbl.to(DEVICE)
        correct += (model(img)[0].argmax(1)==lbl).sum().item(); total += len(lbl)
    return correct/total


def load_seed42():
    slot = os.environ.get("SGN_SLOT", "local")
    for name in [f"cnn_step021_ga2_mixup_t1__{slot}.json", "cnn_step021_ga2_mixup_t1__mini_mps.json"]:
        f = ROOT/"results"/name
        if f.exists(): return json.loads(f.read_text()).get("best")
    return None


def main():
    img_root, h5 = ROOT/args.data_img, ROOT/args.data_h5
    for x in (img_root, h5):
        if not x.exists(): print(f"ERROR: {x} not found"); sys.exit(1)
    seed42 = args.seed42_best if args.seed42_best is not None else load_seed42()
    if seed42 is None: print("WARNING: seed=42 result missing; pass --seed42_best"); seed42 = float("nan")
    full = StudentDataset(str(img_root), str(h5), "train", TRAIN_TF)
    val  = StudentDataset(str(img_root), str(h5), "val",   VAL_TF)
    nw = 2 if DEVICE.type=="cuda" else 0
    va = DataLoader(val, batch_size=args.batch, shuffle=False, num_workers=nw)
    seeds = [int(s) for s in args.seeds.split(",")]
    macs = count_macs(EfficientVGG(GA2_CH,GA2_K,GA2_EXP,GA2_SIDE,GA2_CRELU))/1e6
    n_p  = sum(p_.numel() for p_ in EfficientVGG(GA2_CH,GA2_K,GA2_EXP,GA2_SIDE,GA2_CRELU).parameters())
    print(f"cnn_step024 GA2+Mixup T2 multi-seed | {args.epochs}ep 100%data | alpha={args.mixup_alpha}")
    print(f"  {DEVICE}  slot={SLOT}  params={n_p:,}  MACs={macs:.1f}M  seed42={seed42:.4f}  seeds={seeds}\n")
    results = {}
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = EfficientVGG(GA2_CH,GA2_K,GA2_EXP,GA2_SIDE,GA2_CRELU).to(DEVICE)
        tr = DataLoader(full, batch_size=args.batch, shuffle=True, num_workers=nw)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
        best = 0.0; best_ep = 0; t0 = time.time()
        print(f"{'─'*50}\nSeed={seed}")
        for ep in range(args.epochs):
            model.train()
            for img, tf_, soft, lbl in tr:
                img,tf_,soft,lbl = img.to(DEVICE),tf_.to(DEVICE),soft.to(DEVICE),lbl.to(DEVICE)
                img,tf_,soft,lbl_a,lbl_b,lam = mixup_batch(img,tf_,soft,lbl,args.mixup_alpha)
                logits,sf = model(img)
                loss = distil_loss_mixup(logits,sf,tf_,soft,lbl_a,lbl_b,lam)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            sch.step()
            if (ep+1)%5==0:
                acc = evaluate(model,va)
                if acc>best: best=acc; best_ep=ep+1
                print(f"  ep{ep+1:3d}  val={acc:.4f}  best={best:.4f}  lr={sch.get_last_lr()[0]:.2e}",flush=True)
        elapsed = time.time()-t0
        print(f"  -> seed={seed}  best={best:.4f} @ep{best_ep}  ({elapsed:.0f}s)")
        results[seed] = {"best":best,"best_ep":best_ep,"elapsed":elapsed}
    accs = [results[s]["best"] for s in seeds]
    if not np.isnan(seed42): accs.append(seed42)
    mean_acc,std_acc = float(np.mean(accs)),float(np.std(accs))
    tag = "STRONG" if mean_acc>=REF_T2 else "EFF-PARETO" if mean_acc>=DSMALL_T2 else "WEAK"
    print(f"\n{'='*60}\ncnn_step024 SUMMARY — GA2+Mixup T2 (alpha={args.mixup_alpha})\n{'='*60}")
    print(f"  seed=42: {seed42:.4f}")
    for s in seeds: r=results[s]; print(f"  seed={s}: {r['best']:.4f} @ep{r['best_ep']}  ({r['elapsed']:.0f}s)")
    print(f"  mean={mean_acc:.4f}  std={std_acc:.4f}  n={len(accs)}")
    print(f"  -> {mean_acc:.2%} ±{std_acc:.2%}  {tag}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"seeds":results,"seed42":seed42,"mean":mean_acc,
                               "std":std_acc,"n_seeds":len(accs),"tag":tag,
                               "mixup_alpha":args.mixup_alpha},indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
