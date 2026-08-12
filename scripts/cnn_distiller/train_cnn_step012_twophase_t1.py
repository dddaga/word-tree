"""cnn_step012 — GA_2 T1, two-phase LR schedule. Diagnostic for T1 failure.

Evidence:
  T0 (20ep, cosine 3e-4→1e-5): GA_2 → 41.32%  [WORKS — fast convergence]
  step007 T1 (75ep, cosine 3e-4→1e-5): GA_2 → 35.44%@ep8, degrades  [FAILS]
  step009 T1+warmup: GA_2 → 33.07%@ep24, degrades  [FAILS]
  step010 T1+SGDR: cycle1=41.32%@ep20, cycle2 val drops 40.99%→27.46%  [FAILS]

Pattern: every restart or sustained high LR disrupts the GA_2 basin.
Hypothesis A_prime: GA_2 CReLU+LN architecture is basin-sensitive — model can only
  remain at the good basin under VERY LOW LR. High LR (≥1e-4) destroys basin.

Fix: Phase 1 = T0 schedule (cosine 3e-4→1e-5, 20ep). Phase 2 = fixed LR=1e-5, 55ep.
  Never raise LR after Phase 1. Tests: can the model maintain and refine the T0 basin?

Expected: best ≥41.32% (maintains T0), may improve to ~42-44% under gentle fine-tune.
If monotone or flat → A_prime confirmed (high LR was the only problem).
If still degrades even at 1e-5 → feat loss (B2) is the culprit (test step011 result).
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
p.add_argument("--device",     default="auto")
p.add_argument("--seed",       type=int, default=42)
p.add_argument("--data_img",   default="data/imagenette2-320")
p.add_argument("--data_h5",    default="data/store.h5")
p.add_argument("--batch",      type=int, default=32)
p.add_argument("--epochs",     type=int, default=75)
p.add_argument("--phase1_eps", type=int, default=20, help="T0 cosine phase length")
p.add_argument("--phase2_lr",  type=float, default=1e-5, help="fixed LR for phase 2")
args = p.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
OUT  = ROOT / "results" / f"cnn_step012_twophase_t1_seed{args.seed}__{SLOT}.json"

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
    n_par = sum(q.numel() for q in model.parameters())
    phase2_eps = args.epochs - args.phase1_eps
    print(f"cnn_step012 GA_2 T1+twophase | {n_par:,} params | {macs:.1f}M MACs")
    print(f"Phase 1: {args.phase1_eps}ep cosine 3e-4→1e-5 (T0 schedule)")
    print(f"Phase 2: {phase2_eps}ep fixed LR={args.phase2_lr:.0e} (no restart)")
    print(f"Device: {DEVICE} | Slot: {SLOT}")
    print(f"T0 baseline: 41.32%. If best improves in Phase 2 → A_prime confirmed.\n")

    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.phase1_eps, eta_min=1e-5)

    best = 0.0; best_ep = 0; phase1_best = None; t0 = time.time()
    model.train()
    for ep in range(1, args.epochs + 1):
        # Phase 2: freeze LR at phase2_lr (no scheduler step)
        if ep == args.phase1_eps + 1:
            for pg in opt.param_groups:
                pg['lr'] = args.phase2_lr
            print(f"  ── Phase 2 start: LR set to {args.phase2_lr:.0e} ──", flush=True)

        for img, tf, sl, lbl in tr:
            img, tf, sl, lbl = img.to(DEVICE), tf.to(DEVICE), sl.to(DEVICE), lbl.to(DEVICE)
            logits, sfeat = model(img)
            loss = distil_loss(logits, sfeat, tf, sl, lbl)
            opt.zero_grad(); loss.backward(); opt.step()

        if ep <= args.phase1_eps:
            sched.step()
        acc = evaluate(model, va)
        if acc > best: best, best_ep = acc, ep
        lr_now = opt.param_groups[0]['lr']
        if ep % 5 == 0 or ep == args.phase1_eps:
            phase_tag = f"P1@ep{ep}" if ep <= args.phase1_eps else f"P2@ep{ep-args.phase1_eps}"
            print(f"  ep{ep:3d} [{phase_tag}]  val={acc:.4f}  best={best:.4f}@ep{best_ep}  lr={lr_now:.2e}", flush=True)
        if ep == args.phase1_eps:
            phase1_best = best
            print(f"  → Phase 1 done: best={best:.4f} vs T0 baseline=0.4132 "
                  f"({'MATCH' if abs(best-0.4132)<0.002 else 'DIVERGED'})", flush=True)

    elapsed = time.time() - t0
    adv = "ADVANCE → T2" if best >= 0.7685 else "BELOW_THRESHOLD"
    phase2_delta = best - phase1_best if phase1_best else 0
    pattern = "MONOTONE/FLAT (A_prime confirmed)" if best_ep >= 55 else f"PEAKED@ep{best_ep}"
    print(f"\n{'='*60}")
    print(f"cnn_step012 GA_2 T1+twophase RESULT")
    print(f"  best={best:.4f} @ep{best_ep}  elapsed={elapsed:.0f}s")
    print(f"  Phase 1 best={phase1_best:.4f}  Phase 2 delta={phase2_delta:+.4f}")
    print(f"  vs T0 baseline (41.32%): {best*100-41.32:+.2f}pp")
    print(f"  vs Ref T2=77.35%: {best*100-77.35:+.2f}pp  → {adv}")
    print(f"  Pattern: {pattern}")
    print(f"{'='*60}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "config": GA2_CFG, "best": best, "best_ep": best_ep,
        "macs_M": macs, "params": n_par, "elapsed_s": int(elapsed),
        "phase1_eps": args.phase1_eps, "phase2_lr": args.phase2_lr,
        "phase1_best": phase1_best, "phase2_delta": phase2_delta,
        "tier": "T1", "advance": adv,
        "vs_t0_baseline_pp": round(best * 100 - 41.32, 2),
        "pattern": pattern,
    }, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
