"""CNN distillation step030 — GA5 C=(48,96,384) + Mixup(α=0.2) T2 multi-seed.
Launch after step029 (GA5 T1) ADVANCE (74.73% @ep75, Δ=+1.12pp vs GA2).
Hypothesis: wider C1/C2 capacity breaks GA2 regularization ceiling; T2 mean > GA2 77.32%.
Config: GA5 C=(48,96,384) k=3 exp=2 crelu=True side=False mixup_alpha=0.2
Loss: 0.50·feat_cos + 0.30·dkd + 0.20·ce
Thresholds: STRONG≥77.61% (Ref T2 mean), EFF-PARETO≥75.82% (D_small_s mean)
GA2 T2 mean: 77.32% ±0.56% — this is the GA5 target to beat.
Output: results/cnn_step030_ga5_mixup_t2__seed{SEED}__{SLOT}.json
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
p.add_argument("--seeds", default="1,2,3,42")
p.add_argument("--mixup_alpha", type=float, default=0.2)
p.add_argument("--data_img", default="data/imagenette2-320")
p.add_argument("--data_h5", default="data/store.h5")
p.add_argument("--batch", type=int, default=32)
args = p.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
GA5_CH, GA5_K, GA5_EXP, GA5_SIDE, GA5_CRELU = (48, 96, 384), 3, 2, False, True
REF_T2, DSMALL_T2 = 0.7761, 0.7582
GA2_T2_MEAN = 0.7732
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


def run_seed(seed, full_ds, val_ds, n_params, macs):
    torch.manual_seed(seed); np.random.seed(seed)
    model = EfficientVGG(GA5_CH, GA5_K, GA5_EXP, GA5_SIDE, GA5_CRELU).to(DEVICE)
    nw = 2 if DEVICE.type == "cuda" else 0
    tr = DataLoader(full_ds, batch_size=args.batch, shuffle=True, num_workers=nw)
    va = DataLoader(val_ds,  batch_size=args.batch, shuffle=False, num_workers=nw)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    best = 0.0; best_ep = 0; t0 = time.time()
    print(f"\n{'─'*50}\nSeed={seed}  params={n_params:,}  MACs={macs:.1f}M")
    for ep in range(args.epochs):
        model.train()
        for img, tf_, soft, lbl in tr:
            img,tf_,soft,lbl = img.to(DEVICE),tf_.to(DEVICE),soft.to(DEVICE),lbl.to(DEVICE)
            img,tf_,soft,lbl_a,lbl_b,lam = mixup_batch(img,tf_,soft,lbl,args.mixup_alpha)
            logits, sf = model(img)
            loss = distil_loss_mixup(logits,sf,tf_,soft,lbl_a,lbl_b,lam)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        if (ep+1) % 5 == 0:
            acc = evaluate(model, va)
            if acc > best: best = acc; best_ep = ep+1
            print(f"  ep{ep+1:3d}  val={acc:.4f}  best={best:.4f}  lr={sch.get_last_lr()[0]:.2e}", flush=True)
    elapsed = time.time() - t0
    print(f"  -> seed={seed}  best={best:.4f} @ep{best_ep}  ({elapsed:.0f}s)")
    out = ROOT/"results"/f"cnn_step030_ga5_mixup_t2__seed{seed}__{SLOT}.json"
    out.parent.mkdir(exist_ok=True)
    payload = {"seed":seed,"best":best,"best_ep":best_ep,"elapsed":elapsed,
               "mixup_alpha":args.mixup_alpha,"channels":list(GA5_CH),
               "params":n_params,"macs_M":macs}
    out.write_text(json.dumps(payload, indent=2))
    print(f"  -> {out}")
    return best, best_ep, elapsed


def main():
    img_root, h5 = ROOT/args.data_img, ROOT/args.data_h5
    for x in (img_root, h5):
        if not x.exists(): print(f"ERROR: {x} not found"); sys.exit(1)
    _model = EfficientVGG(GA5_CH, GA5_K, GA5_EXP, GA5_SIDE, GA5_CRELU)
    macs = count_macs(_model) / 1e6
    n_params = sum(p_.numel() for p_ in _model.parameters())
    full = StudentDataset(str(img_root), str(h5), "train", TRAIN_TF)
    val  = StudentDataset(str(img_root), str(h5), "val",   VAL_TF)
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"cnn_step030 GA5+Mixup T2 multi-seed | {args.epochs}ep 100%data | alpha={args.mixup_alpha}")
    print(f"  {DEVICE}  slot={SLOT}  params={n_params:,}  MACs={macs:.1f}M  seeds={seeds}")
    print(f"  Thresholds: STRONG≥{REF_T2:.2%}  EFF-PARETO≥{DSMALL_T2:.2%}  GA2-target≥{GA2_T2_MEAN:.2%}\n")
    results = {}
    for seed in seeds:
        best, best_ep, elapsed = run_seed(seed, full, val, n_params, macs)
        results[seed] = {"best": best, "best_ep": best_ep, "elapsed": elapsed}
    accs = [results[s]["best"] for s in seeds]
    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
    tag = "STRONG" if mean_acc >= REF_T2 else "EFF-PARETO" if mean_acc >= DSMALL_T2 else "WEAK"
    beats_ga2 = mean_acc > GA2_T2_MEAN
    print(f"\n{'='*60}")
    print(f"cnn_step030 SUMMARY — GA5+Mixup(α={args.mixup_alpha}) T2")
    print(f"{'='*60}")
    for s in seeds:
        r = results[s]; print(f"  seed={s}: {r['best']:.4f} @ep{r['best_ep']}  ({r['elapsed']:.0f}s)")
    print(f"  mean={mean_acc:.4f}  std={std_acc:.4f}  n={len(accs)}")
    print(f"  -> {mean_acc:.2%} ±{std_acc:.2%}  {tag}  beats_GA2={beats_ga2} (GA2={GA2_T2_MEAN:.2%})")
    summary_out = ROOT/"results"/f"cnn_step030_ga5_mixup_t2__summary__{SLOT}.json"
    summary_out.write_text(json.dumps({"seeds":results,"mean":mean_acc,"std":std_acc,
                                        "n_seeds":len(accs),"tag":tag,"beats_ga2":beats_ga2,
                                        "ga2_t2_mean":GA2_T2_MEAN,"mixup_alpha":args.mixup_alpha,
                                        "channels":list(GA5_CH),"params":n_params,"macs_M":macs},
                                       indent=2))
    print(f"-> {summary_out}")


if __name__ == "__main__":
    main()
