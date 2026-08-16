"""glam_step006: mechanism 4 — slope-anneal ENERGY arm, T0 (2026-08-13).

The LAST open GLAM arm. Scored on ACTIVATION-ZERO FRACTION (real dormancy → NVIDIA
zero-mul skipping) + wall-time, NOT accuracy. Accuracy is a guard, not the objective:
the arm earns its place only if anneal pushes zeros materially above the plain-ReLU
reference (~0.50–0.52 per layer, ffn_baseline) for <=0.5pp accuracy cost.

Champion under test = GLAM locality (L=1, gsz=1, additive, concat, M=16, d_out=8) —
the 97.68% Imagenette T2 head. Slope schedule is log-linear in training fraction t:
    alpha(t) = exp(log a0 + (log a1 - log a0) * t),  a0=1.0 (identity), a1=1e-5 (hard).
t = ep/(EPOCHS-1), so the same schedule is valid at any tier (T0/T1/T2), no rewrite.

Arms (--arm):
  CTRL    constant slope 0.01 train / 1e-5 eval        (the champion recipe, control)
  A7DET   deterministic LeakyReLU(alpha(t)) each epoch (mechanism 4, det)
  A7RAND  slope ~ U[alpha(t)/2, alpha(t)*2] per epoch  (stochastic around schedule)

Ref (same store/pipeline): GLAM-LOC constant-slope = 96.96% T0 @ 47,370p; plain-ReLU
activation-zero fraction ~0.50–0.52 per layer (learnings/ffn_baseline/QUEUE.md:41).
Joules methodology (meditation-005) is measured on survivors only — if this T0 shows
zeros do not rise or accuracy craters, the arm is dead and no Joules run is warranted.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import torch
import torch.nn.functional as F

from src.sgnnet.model_glam import GLAMNet

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="A7DET", help="CTRL / A7DET / A7RAND")
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--data_frac", type=float, default=0.5)
parser.add_argument("--d_out", type=int, default=8)
parser.add_argument("--M", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--store", default="data/store.h5")
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
A0, A1 = 1.0, 1e-5                       # slope schedule endpoints (identity -> hard)
ZERO_EPS = 1e-4                          # |activation| below this counts as a real zero
SLOT = os.environ.get("SGN_SLOT", "local")


def out_path() -> Path:
    return ROOT / "results" / "glam" / f"glam_step006_{args.arm}_{args.tag}_seed{SEED}__{SLOT}.json"


def alpha_at(t: float) -> float:
    """Log-linear slope schedule; t in [0,1] = fraction of training done."""
    return math.exp(math.log(A0) + (math.log(A1) - math.log(A0)) * t)


def train_slope(ep: int) -> float:
    """Negative slope used DURING training for this epoch, per arm."""
    if args.arm == "CTRL":
        return 0.01
    t = ep / max(1, EPOCHS - 1)
    a = alpha_at(t)
    if args.arm == "A7RAND":
        g = torch.Generator().manual_seed(SEED * 1000 + ep)
        return float(torch.empty(1).uniform_(a * 0.5, a * 2.0, generator=g))
    return a                              # A7DET


def build():
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


def evaluate(model, va_x, va_y):
    """Return (accuracy, activation-zero-fraction) at hard-ReLU eval (slope 1e-5)."""
    model.eval()
    model.set_slope(1e-5)
    captured = {}
    h = model.layers[0].register_forward_hook(lambda m, i, o: captured.update(z=o.detach()))
    correct, zeros, total = 0, 0, 0
    with torch.no_grad():
        for i in range(0, len(va_x), BATCH):
            out = model(va_x[i:i + BATCH].to(DEVICE))
            correct += (out.argmax(1).cpu() == va_y[i:i + BATCH]).sum().item()
            z = captured["z"]
            zeros += (z.abs() < ZERO_EPS).sum().item()
            total += z.numel()
    h.remove()
    return correct / len(va_y), zeros / total


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
    best, best_ep, best_zf, t0 = 0.0, 0, 0.0, time.time()
    per_epoch = []
    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        model.train()
        slope = train_slope(ep)
        model.set_slope(slope)
        perm = torch.randperm(len(tr_x), generator=g)
        for i in range(0, len(perm), BATCH):
            bidx = perm[i:i + BATCH]
            bx, by = tr_x[bidx].to(DEVICE), tr_y[bidx].to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(bx), by).backward()
            opt.step(); sched.step()
        acc, zf = evaluate(model, va_x, va_y)
        per_epoch.append({"ep": ep + 1, "train_slope": round(slope, 6),
                          "val": round(acc, 4), "zero_frac": round(zf, 4)})
        if acc > best:
            best, best_ep, best_zf = acc, ep + 1, zf
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  e{ep+1:3d}/{EPOCHS}  slope={slope:.4g}  val={acc:.4f}  "
                  f"zero_frac={zf:.4f}  best={best:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    elapsed = time.time() - t0
    final_zf = per_epoch[-1]["zero_frac"]
    print(f"  DONE: best={best:.4f} @ep{best_ep}  final_zero_frac={final_zf:.4f}  "
          f"params={n_p:,}  {elapsed:.0f}s")
    return {"arm": args.arm, "M": args.M, "d_out": args.d_out, "n_params": n_p,
            "pct_fc": round(100 * n_p / FC_REF, 4), "best": round(best, 4),
            "best_ep": best_ep, "zero_frac_at_best": round(best_zf, 4),
            "final_zero_frac": round(final_zf, 4), "dense_flops": 2 * n_p,
            "elapsed_s": round(elapsed, 1), "store": args.store, "seed": SEED,
            "per_epoch": per_epoch}


def main():
    if args.smoke_test:
        m = build().to(DEVICE)
        m.train(); m.set_slope(train_slope(0))
        out = m(torch.randn(4, N_IN, device=DEVICE))
        n_p = sum(p.numel() for p in m.parameters())
        fits = out.shape == (4, N_OUT) and not out.isnan().any()
        print(f"  {args.arm} params={n_p:,} slope0={train_slope(0):.4g} "
              f"out={tuple(out.shape)} {'OK' if fits else 'FAIL'}")
        sys.exit(0 if fits else 1)

    data = load_data()
    print(f"\n{'='*70}\nglam_step006 — {args.arm} slope-anneal ENERGY T0 "
          f"({EPOCHS}ep, {int(args.data_frac*100)}%)")
    print(f"  device={DEVICE}  ref: GLAM-LOC 96.96% @47K; plain-ReLU zero_frac ~0.50-0.52\n{'='*70}")
    result = train_arm(data)
    p = out_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"step": "glam_step006", **result}, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
