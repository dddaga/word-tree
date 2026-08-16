"""glam_step004: the user's faithful key×value mechanism T0 (2026-08-13).

Each chunk = one group, attended by two pooled weights (distinct pair): key & value.
    key = sigmoid(E_k·x_c + b_k);  val = leakyReLU(E_v·x_c + b_v);  y_c = key * val
Asymmetric, applied once. Pool floor is combinatorial (MC2 >= P) -> M>=33 for P=512.

Isolates the design against the additive locality champion (step002 A1, 96.96/97.15/
97.68) and separates the sigmoid from the mere asymmetric product:
  KV  key_act=sigmoid  -> user's faithful design (bounded [0,1] key gate, once)
  PV  key_act=none     -> raw asymmetric product (isolates the sigmoid's effect)

Ref (same pipeline/data/split): additive locality A1 = 96.96% T0 @ 47,370p.
FFN-head = 93.2% @ 1.11M.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import torch
import torch.nn.functional as F

from src.sgnnet.model_glam_keyval import GLAMKeyValNet

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="KV", help="KV(sigmoid key) / PV(raw asym product)")
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--data_frac", type=float, default=0.5)
parser.add_argument("--d_out", type=int, nargs="+", default=[8])
parser.add_argument("--M", type=int, nargs="+", default=[34, 64])
parser.add_argument("--L", type=int, nargs="+", default=[1])
parser.add_argument("--seed", type=int, default=42)
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

ARMS = {"KV": "sigmoid", "PV": "none"}


def out_path() -> Path:
    return ROOT / "results" / "glam" / f"glam_step004_{args.arm}_{args.tag}_seed{SEED}__{SLOT}.json"


def build(d_out: int, M: int, L: int = 1):
    return GLAMKeyValNet(in_dim=N_IN, n_out=N_OUT, L=L, P=512, d_out=d_out, M=M,
                         key_act=ARMS[args.arm], across_op="concat",
                         seed=SEED, collapse_last=False).to(DEVICE)


def load_data():
    with h5py.File(ROOT / "data" / "store.h5", "r") as f:
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


def evaluate(model, va_x, va_y):
    model.eval()
    model.set_slope(1e-5)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(va_x), BATCH):
            out = model(va_x[i:i + BATCH].to(DEVICE))
            correct += (out.argmax(1).cpu() == va_y[i:i + BATCH]).sum().item()
    return correct / len(va_y)


def train_cell(d_out: int, M: int, L: int, data) -> dict:
    tr_x, tr_y, va_x, va_y = data
    model = build(d_out, M, L)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    n_batches = (len(tr_x) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=EPOCHS * n_batches, pct_start=0.1, anneal_strategy="cos")
    print(f"\n{'─'*60}\n{args.arm} L={L} d_out={d_out} M={M}  params={n_p:,} "
          f"({100*n_p/FC_REF:.4f}% FC)  key_act={ARMS[args.arm]}  device={DEVICE}", flush=True)
    best, best_ep, t0 = 0.0, 0, time.time()
    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        model.train()
        model.set_slope(0.01)
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
    return {"arm": args.arm, "L": L, "d_out": d_out, "M": M, "n_params": n_p,
            "pct_fc": round(100 * n_p / FC_REF, 4), "best": round(best, 4),
            "best_ep": best_ep, "dense_flops": 2 * n_p, "elapsed_s": round(elapsed, 1)}


def main():
    if args.smoke_test:
        ok = True
        for L in args.L:
            for d_out in args.d_out:
                for M in args.M:
                    m = build(d_out, M, L)
                    m.train(); m.set_slope(0.01)
                    out = m(torch.randn(4, N_IN, device=DEVICE))
                    n_p = sum(p.numel() for p in m.parameters())
                    fits = out.shape == (4, N_OUT) and not out.isnan().any()
                    print(f"  {args.arm} L={L} d_out={d_out} M={M} params={n_p:,} "
                          f"out={tuple(out.shape)} {'OK' if fits else 'FAIL'}")
                    ok &= fits
        sys.exit(0 if ok else 1)

    data = load_data()
    grid = [(L, d, M) for L in args.L for d in args.d_out for M in args.M]
    print(f"\n{'='*70}\nglam_step004 — {args.arm} key×value T0 "
          f"({EPOCHS}ep, {int(args.data_frac*100)}%)  grid={grid}")
    print(f"  device={DEVICE}  additive-locality ref 96.96% @ 47K; FFN-head 93.2% @ 1.11M\n{'='*70}")
    results = {"step": "glam_step004", "seed": SEED, "arm": args.arm, "cells": []}
    for L, d_out, M in grid:
        results["cells"].append(train_cell(d_out, M, L, data))
        p = out_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*70}\nglam_step004 {args.arm} DONE (loc ref 96.96%, FFN 93.2%)")
    for c in results["cells"]:
        print(f"  L={c['L']} d_out={c['d_out']:3d} M={c['M']:3d}  {c['best']:.4f}  "
              f"{c['n_params']:,} ({c['pct_fc']}%)")
    print(f"\n-> {out_path()}")


if __name__ == "__main__":
    main()
