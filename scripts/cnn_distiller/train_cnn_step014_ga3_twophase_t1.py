"""cnn_step014 — GA_3 T1, two-phase LR schedule.

GA step006 results (T0, MultiScaleCNN):
  GA_2: C=(32,64,384), crelu=(T,T,F), exp=1 → T0=41.32%, T1=41.68% (CAPS AT T0)
  GA_3: C=(64,192,384), crelu=(T,T,T), exp=2 → T0=38.04%, T1=UNKNOWN

GA_3 has C1=64 (same as Ref), C2=192 (wider than Ref=128), C3=384.
exp=2 parameter is IGNORED for CReLU blocks — effective expansion=1 (C//2 bottleneck).

Hypothesis: GA_3 may scale better than GA_2 at T1 because:
  1. C1=64 (vs GA_2 C1=32) — 2× wider first stage, richer early features
  2. C2=192 (vs GA_2 C2=64) — 3× wider middle stage
  3. T0=38.04% < 41.32% — may not have hit its ceiling yet at ep20

If GA_3 T1 >> GA_2 T1 (41.68%): early-stage width is the key bottleneck in GA_2.
If GA_3 T1 ≈ GA_2 T1: CReLU at all stages limits capacity regardless of width.
Compare with step013 (Ref, no CReLU stages 1-2): isolates CReLU vs GELU effect.
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

# GA_3: highest-fitness MultiScaleCNN Ref-MAC config from step006
GA3_CFG = dict(channels=(64, 192, 384), dil_rates=(1,), expansion=2,
               crelu_mask=(True, True, True))

import argparse
p = argparse.ArgumentParser()
p.add_argument("--device",     default="auto")
p.add_argument("--seed",       type=int, default=42)
p.add_argument("--data_img",   default="data/imagenette2-320")
p.add_argument("--data_h5",    default="data/store.h5")
p.add_argument("--batch",      type=int, default=32)
p.add_argument("--epochs",     type=int, default=75)
p.add_argument("--phase1_eps", type=int, default=20)
p.add_argument("--phase2_lr",  type=float, default=1e-5)
args = p.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
OUT  = ROOT / "results" / f"cnn_step014_ga3_twophase_t1_seed{args.seed}__{SLOT}.json"

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

    model = MultiScaleCNN(**GA3_CFG).to(DEVICE)
    macs  = count_macs(model) / 1e6
    n_par = sum(q.numel() for q in model.parameters())
    phase2_eps = args.epochs - args.phase1_eps
    print(f"cnn_step014 GA_3 T1+twophase | {n_par:,} params | {macs:.1f}M MACs")
    print(f"Config: C=(64,192,384) crelu=(T,T,T) exp=2(ignored) — step006 T0=38.04%")
    print(f"Phase 1: {args.phase1_eps}ep cosine 3e-4→1e-5  Phase 2: {phase2_eps}ep LR={args.phase2_lr:.0e}")
    print(f"Device: {DEVICE} | Slot: {SLOT}")
    print(f"Key test: GA_3 T1 vs GA_2 T1=41.68% and step013 Ref T1=?\n")

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
            print(f"  → Phase 1 done: best={best:.4f} vs GA_3 T0=0.3804 "
                  f"({'MATCH' if abs(best-0.3804)<0.003 else 'DIVERGED'})", flush=True)

    elapsed = time.time() - t0
    ga2_t1 = 0.4168
    phase2_delta = best - phase1_best if phase1_best else 0
    verdict = "SCALES (C1=64 fixes GA_2 gap)" if best > ga2_t1 + 0.05 else "CAPS EARLY (CReLU limits regardless of width)"
    print(f"\n{'='*60}")
    print(f"cnn_step014 GA_3 T1+twophase RESULT")
    print(f"  best={best:.4f} @ep{best_ep}  elapsed={elapsed:.0f}s")
    print(f"  Phase 1 best={phase1_best:.4f}  Phase 2 delta={phase2_delta:+.4f}")
    print(f"  vs GA_3 T0 (38.04%): {best*100-38.04:+.2f}pp")
    print(f"  vs GA_2 T1 (41.68%): {best*100-41.68:+.2f}pp")
    print(f"  Verdict: {verdict}")
    print(f"{'='*60}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "config": GA3_CFG, "best": best, "best_ep": best_ep,
        "macs_M": macs, "params": n_par, "elapsed_s": int(elapsed),
        "phase1_eps": args.phase1_eps, "phase2_lr": args.phase2_lr,
        "phase1_best": phase1_best, "phase2_delta": phase2_delta,
        "tier": "T1",
        "vs_ga3_t0_pp": round(best * 100 - 38.04, 2),
        "vs_ga2_t1_pp": round(best * 100 - 41.68, 2),
        "verdict": verdict,
    }, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
