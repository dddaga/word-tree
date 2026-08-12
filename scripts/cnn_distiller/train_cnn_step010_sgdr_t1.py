"""cnn_step010 — GA_2 T1 with SGDR (CosineAnnealingWarmRestarts, T_0=20).

Evidence so far:
  T0 (20ep, T_max=20): GA_2 → 41.32%  [WORKS — LR drops fast]
  step007 T1 (75ep, T_max=75): GA_2 → 35.44%  [FAILS — LR too high too long]
  step009 T1+warmup (75ep, 10ep warmup+T_max=65): GA_2 → 33.07%@ep24  [FAILS same root]

Root cause hypothesis: LayerNorm (ChanLN) lacks BatchNorm's running-stats stabilizer;
sensitive to sustained high LR. GA_2 finds a basin at ep~8-24 but large gradient steps
from still-high LR kick it out before it converges.

Fix: SGDR with T_0=20 — same fast-decay period as T0, repeated 3× over 60ep +
a 15ep settling phase. Each restart re-explores; best checkpoint preserved.
Expected: exceed T0 best (41.32%) by visiting multiple optima.
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
p.add_argument("--T0",       type=int, default=20, help="SGDR cycle length")
p.add_argument("--T_mult",   type=int, default=1,  help="SGDR T multiplier")
args = p.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
OUT  = ROOT / "results" / f"cnn_step010_sgdr_t1_seed{args.seed}__{SLOT}.json"

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
    n_cycles = args.epochs // args.T0
    print(f"cnn_step010 GA_2 T1+SGDR | {n_par:,} params | {macs:.1f}M MACs")
    print(f"Schedule: SGDR T_0={args.T0} T_mult={args.T_mult} → {n_cycles} cycles over {args.epochs}ep")
    print(f"Device: {DEVICE} | Slot: {SLOT}")
    print(f"T0 baseline: 41.32% (20ep). Advance: >= 76.85% (Ref T2 77.35% −0.5pp)\n")

    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=args.T0, T_mult=args.T_mult, eta_min=1e-5)

    best = 0.0; best_ep = 0; t0 = time.time()
    cycle_bests = []
    model.train()
    for ep in range(1, args.epochs + 1):
        for img, tf, sl, lbl in tr:
            img, tf, sl, lbl = img.to(DEVICE), tf.to(DEVICE), sl.to(DEVICE), lbl.to(DEVICE)
            logits, sfeat = model(img)
            loss = distil_loss(logits, sfeat, tf, sl, lbl)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        acc = evaluate(model, va)
        if acc > best: best, best_ep = acc, ep
        lr_now = sched.get_last_lr()[0]
        # log at end of each cycle and every 5ep
        at_cycle_end = (ep % args.T0 == 0)
        if ep % 5 == 0 or at_cycle_end:
            tag = " ← cycle end" if at_cycle_end else ""
            print(f"  ep{ep:3d}  val={acc:.4f}  best={best:.4f}@ep{best_ep}  lr={lr_now:.2e}{tag}", flush=True)
        if at_cycle_end:
            cycle_bests.append({"cycle": ep // args.T0, "ep": ep, "best_so_far": best})

    elapsed = time.time() - t0
    adv = "ADVANCE → T2" if best >= 0.7685 else "BELOW_THRESHOLD"
    print(f"\n{'='*60}")
    print(f"cnn_step010 GA_2 T1+SGDR RESULT")
    print(f"  best={best:.4f} @ep{best_ep}  elapsed={elapsed:.0f}s")
    print(f"  MACs={macs:.1f}M  params={n_par:,}")
    print(f"  vs T0 baseline (41.32%): {best*100-41.32:+.2f}pp")
    print(f"  vs step007 (35.44%): {best*100-35.44:+.2f}pp")
    print(f"  vs step009 (33.07%): {best*100-33.07:+.2f}pp")
    cb_str = [(c['cycle'], f"{c['best_so_far']:.4f}") for c in cycle_bests]
    print(f"  cycle bests: {cb_str}")
    print(f"  vs Ref T2=77.35%: {best*100-77.35:+.2f}pp  → {adv}")
    print(f"{'='*60}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "config": GA2_CFG, "best": best, "best_ep": best_ep,
        "macs_M": macs, "params": n_par, "elapsed_s": int(elapsed),
        "T0": args.T0, "T_mult": args.T_mult,
        "cycle_bests": cycle_bests, "tier": "T1", "advance": adv,
        "vs_t0_baseline_pp": round(best * 100 - 41.32, 2),
        "vs_step007_pp": round(best * 100 - 35.44, 2),
    }, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
