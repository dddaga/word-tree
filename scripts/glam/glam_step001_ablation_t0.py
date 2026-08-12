"""glam_step001: GLAM layer ablation ladder — T0 (20ep, 50% Imagenette).

GLAM = a new layer type (peer to SGNNET): partition input into chunks, per-chunk
weights from a SHARED pool (local ops -> param reduction), GLOBAL random second-order
grouping (recover pruned info), WITHIN-group multiply, ACROSS-group add + re-frame.

One script, --arm selects the rung. Ref = same-tier T0 Ref_dw ~94.0-94.1% (A0).
Every arm reports the full Pareto row (acc, params, dense_FLOPs). See
learnings/concepts/glam_grouped_multiplicative_routing.md.

VGG16 FC block = 119,586,826 params. Budgets: 1%=1,195,868 | 5%=5,979,341.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_glam import GLAMNet

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="all", help="'all' or one of A1,A2,A3,A4,A4b,A5,A7det,A8")
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--data_frac", type=float, default=0.5)
parser.add_argument("--M", type=int, default=16, help="shared expert pool size")
parser.add_argument("--L", type=int, default=1, help="depth (stacked GLAM layers)")
parser.add_argument("--slope_epochs", type=int, default=0, help="0 = use total epochs")
parser.add_argument("--decor_w", type=float, default=1e-2, help="anti-Hebbian penalty weight")
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


def out_path(arm: str) -> Path:
    return ROOT / "results" / "glam" / f"glam_step001_{arm}_{args.tag}_seed{SEED}__{SLOT}.json"

# arm -> GLAMNet kwargs. within/across ops + selectivity + AH + depth are the knobs.
# gsz=1 => no cross-chunk pairing (pure local); gsz=2 => global 2nd-order pairing.
ARMS = {
    "A1":    dict(gsz=1, within_op="add", selectivity=False),              # locality alone
    "A2":    dict(gsz=2, within_op="mul", selectivity=False),              # + multiplicative pairing
    "A3":    dict(gsz=2, within_op="add", selectivity=False),              # + additive pairing (mix, no mul)
    "A4":    dict(gsz=2, within_op="mul", selectivity=True),               # + selectivity, AH off (decor_w=0)
    "A4b":   dict(gsz=2, within_op="mul", selectivity=True),               # + AH on (decor_w>0)
    "A5":    dict(gsz=2, within_op="mul", selectivity=True),               # full, L=1
    "A7det": dict(gsz=2, within_op="mul", selectivity=True),               # full + slope anneal (det)
    "A8":    dict(gsz=2, within_op="mul", selectivity=True),               # full, depth L>=2
}
ANNEAL_ARMS = {"A7det", "A7rand"}     # log-decay leaky slope 1.0 -> 1e-5
NO_AH_ARMS = {"A4"}                    # selectivity but AH penalty off


def slope_at(ep: int) -> float:
    """Log-linear negative-slope schedule 1.0 -> 1e-5 over training fraction."""
    horizon = args.slope_epochs or EPOCHS
    t = min(1.0, ep / max(1, horizon - 1))
    return math.exp(math.log(1.0) + (math.log(1e-5) - math.log(1.0)) * t)


def build_model():
    kw = dict(ARMS[args.arm])
    # L>1: intermediate layers stay wide (concat), GLAMNet collapses the LAST layer (add).
    # L==1: single layer collapses directly (add) -> small readout.
    kw.update(L=args.L, M=args.M, seed=SEED, in_dim=N_IN, n_out=N_OUT,
              P=512, d_out=8, G=512, across_op="concat" if args.L > 1 else "add")
    return GLAMNet(**kw).to(DEVICE)


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
    if hasattr(model, "set_slope"):
        model.set_slope(1e-5)          # inference = hard-ReLU limit
    correct = 0
    with torch.no_grad():
        for i in range(0, len(va_x), BATCH):
            out = model(va_x[i:i + BATCH].to(DEVICE))
            correct += (out.argmax(1).cpu() == va_y[i:i + BATCH]).sum().item()
    return correct / len(va_y)


def train_run(data) -> dict:
    tr_x, tr_y, va_x, va_y = data
    model = build_model()
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    decor_w = 0.0 if args.arm in NO_AH_ARMS else args.decor_w
    anneal = args.arm in ANNEAL_ARMS
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    n_batches = (len(tr_x) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=EPOCHS * n_batches, pct_start=0.1, anneal_strategy="cos")
    print(f"\n{'─'*60}\n{args.arm}: {ARMS[args.arm]}  L={args.L} M={args.M} "
          f"anneal={anneal} decor_w={decor_w}\n  params={n_p:,} ({100*n_p/FC_REF:.4f}% FC)  device={DEVICE}", flush=True)

    best, best_ep, t0 = 0.0, 0, time.time()
    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        model.train()
        if hasattr(model, "set_slope"):
            model.set_slope(slope_at(ep) if anneal else 0.01)
        perm = torch.randperm(len(tr_x), generator=g)
        for i in range(0, len(perm), BATCH):
            bidx = perm[i:i + BATCH]
            bx, by = tr_x[bidx].to(DEVICE), tr_y[bidx].to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            if decor_w and hasattr(model, "aux_loss"):
                loss = loss + decor_w * model.aux_loss()
            loss.backward()
            opt.step(); sched.step()
        acc = evaluate(model, va_x, va_y)
        if acc > best:
            best, best_ep = acc, ep + 1
        if (ep + 1) % 5 == 0 or ep == 0:
            extra = f"  slope={slope_at(ep):.1e}" if anneal else ""
            print(f"  e{ep+1:3d}/{EPOCHS}  val={acc:.4f}  best={best:.4f}{extra}  [{time.time()-t0:.0f}s]", flush=True)

    elapsed = time.time() - t0
    print(f"  DONE: best={best:.4f} @ep{best_ep}  params={n_p:,}  {elapsed:.0f}s")
    return {"arm": args.arm, "config": ARMS[args.arm], "L": args.L, "M": args.M,
            "anneal": anneal, "decor_w": decor_w, "n_params": n_p,
            "pct_fc": round(100 * n_p / FC_REF, 4), "best": round(best, 4),
            "best_ep": best_ep, "dense_flops": 2 * n_p, "elapsed_s": round(elapsed, 1)}


def main():
    if args.smoke_test:
        ok = True
        for arm in ARMS:
            args.arm = arm
            args.L = 2 if arm == "A8" else 1
            m = build_model()
            m.train()
            if hasattr(m, "set_slope"):
                m.set_slope(0.01)
            out = m(torch.randn(4, N_IN, device=DEVICE))
            n_p = sum(p.numel() for p in m.parameters())
            fits = out.shape == (4, N_OUT) and not out.isnan().any()
            print(f"  {arm:<6} params={n_p:,} out={tuple(out.shape)} {'OK' if fits else 'FAIL'}")
            ok &= fits
        sys.exit(0 if ok else 1)

    data = load_data()
    arms = list(ARMS) if args.arm == "all" else [args.arm]
    print(f"\n{'='*70}\nglam_step001 — arms {arms} T0 ({EPOCHS}ep, {int(args.data_frac*100)}% Imagenette)")
    print(f"  device={DEVICE}  Ref_dw~94.0%\n{'='*70}")
    for arm in arms:
        args.arm = arm
        args.L = 2 if (arm == "A8" and args.L == 1) else args.L   # A8 depth default
        result = {"step": "glam_step001", "seed": SEED, "arm": arm,
                  "result": train_run(data)}
        p = out_path(arm)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2))
        print(f"  -> {p}")
    print(f"\n{'='*70}\nglam_step001 LADDER DONE (Ref_dw~94.0%)")


if __name__ == "__main__":
    main()
