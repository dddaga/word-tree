"""glam_step008: cross-chunk INFORMATION MERGING on CIFAR-100 T0 (2026-08-13).

/goal innovation rung. The LOC champion leaves a STRUCTURAL -3.14pp gap vs dense at
100 classes (capacity recovers only ~1pp then saturates) — a missing-merging
signature: LOC compresses each channel independently, the only cross-channel merge
is the readout. This rung inserts one cheap low-rank cross-channel merge (P->r->P,
residual, zero-init) BETWEEN the locality projection and the readout, and asks:
does structured information merging recover the structural gap at low param cost?

Arms (--arm):
  LIN     dense Linear 25088->100                     (dense anchor, ~2.5M params)
  LOC     GLAMNet gsz=1 additive locality             (champion control, ~416K, 65.35 T2 / ~64 T0)
  MERGE   LOC + low-rank cross-channel merge (--rank) (the new mechanism, +2*P*rank params)

rank=0 in MERGE reproduces LOC. Same VGG16 backbone, same recipe as step007 so the
LOC/LIN numbers are directly comparable; only the merge block is added.
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
from src.sgnnet.model_glam_merge import GLAMMergeNet

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="MERGE", help="LIN / LOC / MERGE")
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--data_frac", type=float, default=0.5)
parser.add_argument("--d_out", type=int, default=8)
parser.add_argument("--M", type=int, default=16)
parser.add_argument("--rank", type=int, default=32)
parser.add_argument("--merge_act", default="gelu", help="gelu / relu / linear")
parser.add_argument("--n_merge", type=int, default=1)
parser.add_argument("--merge_where", default="post", help="post (codes) / pre (raw chunks)")
parser.add_argument("--merge_kind", default="add", help="add (linear low-rank) / bilinear (2nd-order q⊙k)")
parser.add_argument("--readout_rank", type=int, default=0, help="0=dense readout; r>0=low-rank 4096->r->100")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--seeds", default="", help="comma list; overrides --seed, loops in-process")
parser.add_argument("--store", default="data/store_cifar100.h5")
parser.add_argument("--smoke_test", action="store_true")
parser.add_argument("--tag", default="t0")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS, BATCH = args.epochs, 512
N_IN, N_OUT, FC_REF = 25088, 100, 119_586_826
SLOT = os.environ.get("SGN_SLOT", "local")
SEEDS = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]


def out_path(seed: int) -> Path:
    sfx = f"_{args.merge_kind}_{args.merge_where}_r{args.rank}_ro{args.readout_rank}" if args.arm == "MERGE" else ""
    return ROOT / "results" / "glam" / f"glam_step008_{args.arm}{sfx}_{args.tag}_seed{seed}__{SLOT}.json"


def build(seed: int):
    if args.arm == "LIN":
        return torch.nn.Linear(N_IN, N_OUT)
    if args.arm == "LOC":
        return GLAMNet(in_dim=N_IN, n_out=N_OUT, L=1, P=512, d_out=args.d_out, M=args.M,
                       G=512, gsz=1, within_op="add", across_op="concat",
                       selectivity=False, seed=seed, collapse_last=False)
    return GLAMMergeNet(in_dim=N_IN, n_out=N_OUT, P=512, d_out=args.d_out, M=args.M,
                        rank=args.rank, merge_act=args.merge_act, n_merge=args.n_merge,
                        merge_where=args.merge_where, merge_kind=args.merge_kind,
                        readout_rank=args.readout_rank, seed=seed)


def load_data(seed: int):
    with h5py.File(ROOT / args.store, "r") as f:
        tr_x = torch.tensor(f["train/features"][:], dtype=torch.float32)
        tr_y = torch.tensor(f["train/labels"][:], dtype=torch.long)
        va_x = torch.tensor(f["val/features"][:], dtype=torch.float32)
        va_y = torch.tensor(f["val/labels"][:], dtype=torch.long)
    if args.data_frac >= 1.0:
        return tr_x, tr_y, va_x, va_y
    g = torch.Generator().manual_seed(seed)
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


def train_arm(data, seed: int) -> dict:
    tr_x, tr_y, va_x, va_y = data
    model = build(seed).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    n_batches = (len(tr_x) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=EPOCHS * n_batches, pct_start=0.1, anneal_strategy="cos")
    tag = f"{args.arm}" + (f" r={args.rank} act={args.merge_act}" if args.arm == "MERGE" else "")
    print(f"\n{'─'*60}\n{tag}  params={n_p:,} ({100*n_p/FC_REF:.4f}% FC)  "
          f"store={args.store}  device={DEVICE}", flush=True)
    best, best_ep, t0 = 0.0, 0, time.time()
    g = torch.Generator().manual_seed(seed)
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
    return {"arm": args.arm, "M": args.M, "d_out": args.d_out, "rank": args.rank,
            "merge_act": args.merge_act, "n_merge": args.n_merge, "n_params": n_p,
            "pct_fc": round(100 * n_p / FC_REF, 4), "best": round(best, 4),
            "best_ep": best_ep, "dense_flops": 2 * n_p, "elapsed_s": round(elapsed, 1),
            "store": args.store, "seed": seed}


def main():
    if args.smoke_test:
        m = build(SEEDS[0]).to(DEVICE)
        m.train(); _set_slope(m, 0.01)
        out = m(torch.randn(4, N_IN, device=DEVICE))
        n_p = sum(p.numel() for p in m.parameters())
        fits = out.shape == (4, N_OUT) and not out.isnan().any()
        print(f"  {args.arm} params={n_p:,} out={tuple(out.shape)} {'OK' if fits else 'FAIL'}")
        sys.exit(0 if fits else 1)

    print(f"\n{'='*70}\nglam_step008 — {args.arm} CIFAR-100 cross-chunk merge T0 "
          f"({EPOCHS}ep, {int(args.data_frac*100)}%)  seeds={SEEDS}")
    print(f"  device={DEVICE}  100 classes, VGG16 25088-dim (same backbone)\n{'='*70}")
    accs = []
    for seed in SEEDS:
        data = load_data(seed)
        result = train_arm(data, seed)
        p = out_path(seed)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"step": "glam_step008", **result}, indent=2))
        print(f"-> {p}")
        accs.append(result["best"])
    if len(accs) > 1:
        mean = sum(accs) / len(accs)
        sd = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
        print(f"\n{args.arm} SUMMARY: mean={mean:.4f} sd={sd:.4f}  n={len(accs)}  seeds={SEEDS}")


if __name__ == "__main__":
    main()
