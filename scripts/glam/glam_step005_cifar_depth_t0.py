"""glam_step005: depth-below-ceiling on CIFAR-10 features T0 (2026-08-13).

DECISIVE open arm. Imagenette VGG16 features are SATURATED (~97.3% ceiling); GLAM
locality ALONE hits it and every added mechanism (affinity, mul, key×value, depth)
converges to ~96.9% — no headroom to differentiate. CIFAR-10 VGG16 features have
headroom (Linear head = 86.24%, gap to ceiling is real), so this is the only place
mixing / affinity / depth CAN separate from plain locality.

Same 25088-dim VGG16 head, same pipeline/split, store swapped to CIFAR-10. Arms:
  LOC     GLAMNet L=1, gsz=1 additive locality        (the 97.68% Imagenette champion)
  KV      GLAMKeyValNet L=1, sigmoid(key)*value       (user's faithful mechanism)
  LOCL2   GLAMNet L=2, gsz=1 depth-stacked locality   (depth-over-breadth probe)

Ref (same store/pipeline): Linear head CIFAR-10 = 86.24%. Question under test: does
ANY mechanism (KV mixing, or depth) beat plain locality where the ceiling has room?
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
from src.sgnnet.model_glam_keyval import GLAMKeyValNet

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="LOC", help="LOC / KV / LOCL2")
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--data_frac", type=float, default=0.5)
parser.add_argument("--d_out", type=int, default=8)
parser.add_argument("--M", type=int, default=16, help="LOC pool; KV forces >=34")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--store", default="data/store_cifar10.h5")
parser.add_argument("--smoke_test", action="store_true")
parser.add_argument("--tag", default="t0")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS, SEED, BATCH = args.epochs, args.seed, 512
N_IN, N_OUT, FC_REF = 25088, 10, 119_586_826
SLOT = os.environ.get("SGN_SLOT", "local")


def out_path() -> Path:
    return ROOT / "results" / "glam" / f"glam_step005_{args.arm}_{args.tag}_seed{SEED}__{SLOT}.json"


def build():
    if args.arm == "LIN":
        return torch.nn.Linear(N_IN, N_OUT)  # dense same-tier Pareto anchor (250K params)
    if args.arm == "KV":
        M = max(args.M, 34)  # combinatorial floor MC2 >= P=512
        return GLAMKeyValNet(in_dim=N_IN, n_out=N_OUT, L=1, P=512, d_out=args.d_out,
                             M=M, key_act="sigmoid", across_op="concat",
                             seed=SEED, collapse_last=False)
    L = 2 if args.arm == "LOCL2" else 1
    return GLAMNet(in_dim=N_IN, n_out=N_OUT, L=L, P=512, d_out=args.d_out, M=args.M,
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
    print(f"\n{'='*70}\nglam_step005 — {args.arm} CIFAR-10 depth-below-ceiling T0 "
          f"({EPOCHS}ep, {int(args.data_frac*100)}%)")
    print(f"  device={DEVICE}  Linear-head CIFAR-10 ref = 86.24%\n{'='*70}")
    result = train_arm(data)
    p = out_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"step": "glam_step005", **result}, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
