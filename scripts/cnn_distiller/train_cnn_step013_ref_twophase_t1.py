"""cnn_step013 — MultiScaleCNN Ref T1, two-phase LR schedule.

Key discovery from step012: step003 Ref (73.17% T1) uses EfficientVGG, NOT
MultiScaleCNN. The GA (step006) searched over MultiScaleCNN configs only.

step006 T0 results (MultiScaleCNN):
  Ref (C=64,128,256, default): 36.82%  [lowest T0]
  GA_1 (C=32,64,384, crelu):  37.99%
  GA_2 (C=32,64,384, crelu):  41.32%  ← GA winner
  GA_3 (C=64,192,384, crelu): 38.04%

step012 T1 result (GA_2 + two-phase): best=41.68% — CAPS AT T0 VALUE.
Hypothesis: GA selected GA_2 because it fast-converges in 20ep, but that
fast-convergence means it has ALREADY HIT ITS CAPACITY CEILING at T0.
MultiScaleCNN Ref (T0=36.82%) may still be climbing at ep20 → higher T1 ceiling.

This experiment tests: does MultiScaleCNN Ref scale from 36.82% T0 to competitive T1?
If it reaches 60%+: GA_2 was a FALSE winner (optimized T0, not T1 scalability).
If it also caps near T0: MultiScaleCNN family has fundamental capacity issues.

Two-phase used because step012 confirmed cosine T_max=75 destroys basins.
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

# MultiScaleCNN default Ref config — step006 T0=36.82%, T1 unknown
REF_CFG = dict(channels=(64, 128, 256), dil_rates=(1,), expansion=2,
               crelu_mask=(False, False, True))

import argparse
p = argparse.ArgumentParser()
p.add_argument("--device",     default="auto")
p.add_argument("--seed",       type=int, default=42)
p.add_argument("--data_img",   default="data/imagenette2-320")
p.add_argument("--data_h5",    default="data/store.h5")
p.add_argument("--batch",      type=int, default=32)
p.add_argument("--epochs",     type=int, default=75)
p.add_argument("--phase1_eps", type=int, default=20, help="cosine phase length")
p.add_argument("--phase2_lr",  type=float, default=1e-5)
args = p.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
OUT  = ROOT / "results" / f"cnn_step013_ref_twophase_t1_seed{args.seed}__{SLOT}.json"

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

    model = MultiScaleCNN(**REF_CFG).to(DEVICE)
    macs  = count_macs(model) / 1e6
    n_par = sum(q.numel() for q in model.parameters())
    phase2_eps = args.epochs - args.phase1_eps
    print(f"cnn_step013 MultiScaleCNN Ref T1+twophase | {n_par:,} params | {macs:.1f}M MACs")
    print(f"Config: C=(64,128,256) exp=2 crelu=(F,F,T) — step006 T0 baseline: 36.82%")
    print(f"Phase 1: {args.phase1_eps}ep cosine 3e-4→1e-5")
    print(f"Phase 2: {phase2_eps}ep fixed LR={args.phase2_lr:.0e}")
    print(f"Device: {DEVICE} | Slot: {SLOT}")
    print(f"Diagnostic: GA_2 T1=41.68% (caps at T0). Ref T0=36.82% — still climbing?\n")

    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.phase1_eps, eta_min=1e-5)

    best = 0.0; best_ep = 0; phase1_best = None; t0 = time.time()
    model.train()
    for ep in range(1, args.epochs + 1):
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
            ref_t0 = 0.3682
            print(f"  → Phase 1 done: best={best:.4f} vs Ref T0={ref_t0:.4f} "
                  f"({'ABOVE' if best > ref_t0 else 'BELOW'} T0 baseline)", flush=True)

    elapsed = time.time() - t0
    ga2_t1 = 0.4168
    phase2_delta = best - phase1_best if phase1_best else 0
    scaling_verdict = "SCALES (GA_2 was false winner)" if best > ga2_t1 + 0.05 else "CAPS EARLY (MultiScaleCNN limited)"
    print(f"\n{'='*60}")
    print(f"cnn_step013 MultiScaleCNN Ref T1+twophase RESULT")
    print(f"  best={best:.4f} @ep{best_ep}  elapsed={elapsed:.0f}s")
    print(f"  Phase 1 best={phase1_best:.4f}  Phase 2 delta={phase2_delta:+.4f}")
    print(f"  vs Ref T0 baseline (36.82%): {best*100-36.82:+.2f}pp")
    print(f"  vs GA_2 T1 (41.68%): {best*100-41.68:+.2f}pp")
    print(f"  Scaling verdict: {scaling_verdict}")
    print(f"{'='*60}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "config": REF_CFG, "best": best, "best_ep": best_ep,
        "macs_M": macs, "params": n_par, "elapsed_s": int(elapsed),
        "phase1_eps": args.phase1_eps, "phase2_lr": args.phase2_lr,
        "phase1_best": phase1_best, "phase2_delta": phase2_delta,
        "tier": "T1",
        "vs_ref_t0_pp": round(best * 100 - 36.82, 2),
        "vs_ga2_t1_pp": round(best * 100 - 41.68, 2),
        "scaling_verdict": scaling_verdict,
    }, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
