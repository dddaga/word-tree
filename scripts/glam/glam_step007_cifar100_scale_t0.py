"""glam_step007: locality champion on CIFAR-100 features T0 (2026-08-13).

Scale-invariance rung. Same frozen VGG16 backbone as Imagenette (10 cls) and
CIFAR-10 (10 cls), class count now 100 — the cleanest on-disk test of whether
the locality-vs-dense gap WIDENS with class count. No download: reuses
data/store_cifar100.h5 already on disk. Backbone held constant, only the readout
width and the label set change, so this is a pure scale rung, not a new setup.

Arms (--arm):
  LIN   dense Linear head 25088->100                (within-tier Pareto anchor, ~2.5M params)
  LOC   GLAMNet L=1, gsz=1 additive locality        (the 97.68% Imagenette champion)

Prior rungs (same backbone, VGG16 25088-dim):
  Imagenette (10 cls): LOC 96.96 T0, matches dense ceiling (saturated).
  CIFAR-10   (10 cls): dense LIN 86.90 @ 250K, LOC 85.66 @ 47K (gap -1.24pp, 5.3x fewer params).
Question under test: does the -1.24pp gap hold, widen, or shrink at 100 classes?
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import torch
import torch.nn.functional as F

from src.sgnnet.model_glam import GLAMNet

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="LOC", help="LIN / LOC")
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--data_frac", type=float, default=0.5)
parser.add_argument("--d_out", type=int, default=8)
parser.add_argument("--M", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--store", default="data/store_cifar100.h5")
parser.add_argument("--smoke_test", action="store_true")
parser.add_argument("--tag", default="t0")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS, SEED, BATCH = args.epochs, args.seed, 512
N_IN, N_OUT, FC_REF = 25088, 100, 119_586_826
SLOT = os.environ.get("SGN_SLOT", "local")


def out_path() -> Path:
    return ROOT / "results" / "glam" / f"glam_step007_{args.arm}_{args.tag}_seed{SEED}__{SLOT}.json"


def build():
    if args.arm == "LIN":
        return torch.nn.Linear(N_IN, N_OUT)  # dense same-tier Pareto anchor
    return GLAMNet(in_dim=N_IN, n_out=N_OUT, L=1, P=512, d_out=args.d_out, M=args.M,
                   G=512, gsz=1, within_op="add", across_op="concat",
                   selectivity=False, seed=SEED, collapse_last=False)


def load_data():
    with h5py.File(ROOT / args.store, "r") as f:
        tr_x = torch.tensor(f["train/features"][:], dtype=torch.float32)
        tr_y = torch.tensor(f["train/labels"][:], dtype=torch.long)
        va_x = torch.tensor(f["val/features"][:], dtype=torch.float32)
        va_y = torch.tensor(f["val/labels"][:], dtype=torch.long)
    if args.data_frac >= 1.0:
        return tr_x, tr_y, va_x, va_y
    g = torch.Generator().manual_seed(SEED)
    n = int(len(tr_x) * args.data_frac)
    idx = torch.randperm(len(tr_x), generator=g)[:n]
    return tr_x[idx], tr_y[idx], va_x, va_y


def _set_slope(model, s):
    if hasattr(model, "set_slope"):
        model.set_slope(s)


def evaluate(model, va_x, va_y):
    model.eval()
    _set_slope(model, 1e-5)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(va_x), BATCH):
            out = model(va_x[i:i + BATCH].to(DEVICE))
            correct += (out.argmax(1).cpu() == va_y[i:i + BATCH]).sum().item()
    return correct / len(va_y)


def train_arm(data) -> dict:
    tr_x, tr_y, va_x, va_y = data
    model = build().to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    n_batches = (len(tr_x) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=EPOCHS * n_batches, pct_start=0.1, anneal_strategy="cos")
    print(f"\n{'─'*60}\n{args.arm}  params={n_p:,} ({100*n_p/FC_REF:.4f}% FC)  "
          f"store={args.store}  device={DEVICE}", flush=True)
    best, best_ep, t0 = 0.0, 0, time.time()
    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        model.train()
        _set_slope(model, 0.01)
        perm = torch.randperm(len(tr_x), generator=g)
        for i in range(0, len(perm), BATCH):
            bidx = perm[i:i + BATCH]
            bx, by = tr_x[bidx].to(DEVICE), tr_y[bidx].to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(bx), by).backward()
            opt.step(); sched.step()
        acc = evaluate(model, va_x, va_y)
        if acc > best:
            best, best_ep = acc, ep + 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  e{ep+1:3d}/{EPOCHS}  val={acc:.4f}  best={best:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    elapsed = time.time() - t0
    print(f"  DONE: best={best:.4f} @ep{best_ep}  params={n_p:,}  {elapsed:.0f}s")
    return {"arm": args.arm, "M": args.M, "d_out": args.d_out, "n_params": n_p,
            "pct_fc": round(100 * n_p / FC_REF, 4), "best": round(best, 4),
            "best_ep": best_ep, "dense_flops": 2 * n_p, "elapsed_s": round(elapsed, 1),
            "store": args.store, "seed": SEED}


def main():
    if args.smoke_test:
        m = build().to(DEVICE)
        m.train(); _set_slope(m, 0.01)
        out = m(torch.randn(4, N_IN, device=DEVICE))
        n_p = sum(p.numel() for p in m.parameters())
        fits = out.shape == (4, N_OUT) and not out.isnan().any()
        print(f"  {args.arm} params={n_p:,} out={tuple(out.shape)} {'OK' if fits else 'FAIL'}")
        sys.exit(0 if fits else 1)

    data = load_data()
    print(f"\n{'='*70}\nglam_step007 — {args.arm} CIFAR-100 scale rung T0 "
          f"({EPOCHS}ep, {int(args.data_frac*100)}%)")
    print(f"  device={DEVICE}  100 classes, VGG16 25088-dim (same backbone)\n{'='*70}")
    result = train_arm(data)
    p = out_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"step": "glam_step007", **result}, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
