"""glam_step010: randomized-leaky-ReLU on the LOC readout WEIGHTS, Imagenette T0.

/goal energy rung (post GLAM merging-TERMINUS). The LOC champion's params live 97%
in the dense readout Linear(4096,C). This rung asks: does a train-soft / infer-HARD
randomized-leaky magnitude gate on that readout buy REAL weight-zeros (energy +
effective-param cut) at <=0.5pp accuracy cost vs the dense champion?

Prior-art synthesis (RReLU / LTP / Reizinger / BNN-STE): skeleton proven, infer-hard
is the unvalidated novel leg, accuracy prior is negative -> scored as ENERGY, not acc.
This is a REJECTION FILTER: advance any sparsity that holds accuracy while zeroing.

Arms (--arm):
  R0   dense LOC champion readout                (control; 0% zeros)
  R1   leaky readout, deterministic α anneal 1->~0, infer-hard   (mechanism)
  R2   R1 but α randomized per-step (RReLU transfer)             (isolates stochasticity)
Sweep --sparsity {0.5,0.7,0.9}. Same LOC backbone/recipe as step008 -> comparable.
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
from src.sgnnet.weight_leaky import attach_leaky_readout

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="R1", help="R0 / R1 / R2")
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--data_frac", type=float, default=0.5)
parser.add_argument("--d_out", type=int, default=8)
parser.add_argument("--M", type=int, default=16)
parser.add_argument("--sparsity", type=float, default=0.7)
parser.add_argument("--rand_ratio", type=float, default=2.0)
parser.add_argument("--a0", type=float, default=1.0, help="anneal start slope α₀")
parser.add_argument("--a1", type=float, default=1e-3, help="anneal end slope α₁")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--seeds", default="", help="comma list; overrides --seed")
parser.add_argument("--store", default="data/store.h5")
parser.add_argument("--sweep", action="store_true", help="run full ladder R0+R1/R2×sparsities")
parser.add_argument("--sparsities", default="0.5,0.7,0.9", help="sweep sparsity set (CSV)")
parser.add_argument("--smoke_test", action="store_true")
parser.add_argument("--tag", default="t0")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS, BATCH = args.epochs, 512
N_IN, FC_REF = 25088, 119_586_826
SLOT = os.environ.get("SGN_SLOT", "local")
SEEDS = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]


def out_path(seed: int) -> Path:
    sfx = "" if args.arm == "R0" else f"_s{args.sparsity}"
    return ROOT / "results" / "glam" / f"glam_step010_{args.arm}{sfx}_{args.tag}_seed{seed}__{SLOT}.json"


def build(seed: int, n_out: int):
    net = GLAMNet(in_dim=N_IN, n_out=n_out, L=1, P=512, d_out=args.d_out, M=args.M,
                  G=512, gsz=1, within_op="add", across_op="concat",
                  selectivity=False, seed=seed, collapse_last=False)
    gate = None
    if args.arm in ("R1", "R2"):
        gate = attach_leaky_readout(net, sparsity=args.sparsity,
                                    randomized=(args.arm == "R2"), rand_ratio=args.rand_ratio)
    return net, gate


def slope_at(ep: int) -> float:
    # geometric in log-space over epoch fraction; α₀ -> α₁ (readout weight gate only).
    t = ep / max(1, EPOCHS - 1)
    return math.exp(math.log(args.a0) + (math.log(args.a1) - math.log(args.a0)) * t)


def load_data(seed: int):
    with h5py.File(ROOT / args.store, "r") as f:
        tr_x = torch.tensor(f["train/features"][:], dtype=torch.float32)
        tr_y = torch.tensor(f["train/labels"][:], dtype=torch.long)
        va_x = torch.tensor(f["val/features"][:], dtype=torch.float32)
        va_y = torch.tensor(f["val/labels"][:], dtype=torch.long)
    n_out = int(tr_y.max()) + 1
    if args.data_frac >= 1.0:
        return (tr_x, tr_y, va_x, va_y), n_out
    g = torch.Generator().manual_seed(seed)
    n = int(len(tr_x) * args.data_frac)
    idx = torch.randperm(len(tr_x), generator=g)[:n]
    return (tr_x[idx], tr_y[idx], va_x, va_y), n_out


def evaluate(model, gate, va_x, va_y):
    model.eval()  # gate.forward branch -> infer-HARD (real zeros)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(va_x), BATCH):
            out = model(va_x[i:i + BATCH].to(DEVICE))
            correct += (out.argmax(1).cpu() == va_y[i:i + BATCH]).sum().item()
    return correct / len(va_y)


def train_arm(data, n_out, seed: int) -> dict:
    tr_x, tr_y, va_x, va_y = data
    model, gate = build(seed, n_out)
    model = model.to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    n_batches = (len(tr_x) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=EPOCHS * n_batches, pct_start=0.1, anneal_strategy="cos")
    print(f"\n{'─'*60}\n{args.arm} sparsity={args.sparsity} rand={args.arm=='R2'}  "
          f"params={n_p:,} ({100*n_p/FC_REF:.4f}% FC)  device={DEVICE}", flush=True)
    best, best_ep, best_zf, t0 = 0.0, 0, 0.0, time.time()
    g = torch.Generator().manual_seed(seed)
    for ep in range(EPOCHS):
        model.train()
        model.set_slope(0.01)  # GLAM-layer activation slope, held (orthogonal to readout gate)
        if gate is not None:
            gate.set_slope(slope_at(ep))
        perm = torch.randperm(len(tr_x), generator=g)
        for i in range(0, len(perm), BATCH):
            bidx = perm[i:i + BATCH]
            bx, by = tr_x[bidx].to(DEVICE), tr_y[bidx].to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(bx), by).backward()
            opt.step(); sched.step()
        acc = evaluate(model, gate, va_x, va_y)
        zf = gate.zero_fraction() if gate is not None else 0.0
        if acc > best:
            best, best_ep, best_zf = acc, ep + 1, zf
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  e{ep+1:3d}/{EPOCHS}  val={acc:.4f}  best={best:.4f}  "
                  f"zero_frac={zf:.3f}  α={slope_at(ep):.4g}  [{time.time()-t0:.0f}s]", flush=True)
    elapsed = time.time() - t0
    # zero_frac is a fraction of the READOUT weight only, not of all params.
    ro_w = gate.weight.numel() if gate is not None else 0
    eff_p = n_p - int(best_zf * ro_w)
    print(f"  DONE: best={best:.4f} @ep{best_ep}  zero_frac={best_zf:.3f}  "
          f"eff_params={eff_p:,}  {elapsed:.0f}s")
    return {"arm": args.arm, "sparsity": args.sparsity, "randomized": args.arm == "R2",
            "n_params": n_p, "eff_params": eff_p, "zero_frac": round(best_zf, 4),
            "pct_fc": round(100 * n_p / FC_REF, 4), "best": round(best, 4),
            "best_ep": best_ep, "elapsed_s": round(elapsed, 1), "seed": seed}


def run_one(seed: int, data, n_out):
    result = train_arm(data, n_out, seed)
    p = out_path(seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"step": "glam_step010", **result}, indent=2))
    print(f"-> {p}")
    return result


def main():
    if args.smoke_test:
        n_out = 10
        m, gate = build(SEEDS[0], n_out)
        m = m.to(DEVICE); m.train(); m.set_slope(0.01)
        if gate is not None:
            gate.set_slope(0.5)
        out = m(torch.randn(4, N_IN, device=DEVICE))
        m.eval(); out_e = m(torch.randn(4, N_IN, device=DEVICE))
        zf = gate.zero_fraction() if gate is not None else 0.0
        n_p = sum(p.numel() for p in m.parameters())
        ok = out.shape == (4, n_out) and not out.isnan().any() and not out_e.isnan().any()
        print(f"  {args.arm} params={n_p:,} out={tuple(out.shape)} zero_frac={zf:.3f} "
              f"{'OK' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

    print(f"\n{'='*70}\nglam_step010 — weight-RReLU readout Imagenette T0 "
          f"({EPOCHS}ep, {int(args.data_frac*100)}%)  sweep={args.sweep}  seeds={SEEDS}")
    print(f"  device={DEVICE}  store={args.store}\n{'='*70}")
    # ladder: control + mechanism (det) + stochastic, across the sparsity sweep
    ladder = [("R0", None)] if args.sweep else [(args.arm, args.sparsity)]
    if args.sweep:
        for s in (float(x) for x in args.sparsities.split(",")):
            ladder += [("R1", s), ("R2", s)]
    for seed in SEEDS:
        (data, n_out) = load_data(seed)
        for arm, s in ladder:
            args.arm = arm
            if s is not None:
                args.sparsity = s
            run_one(seed, data, n_out)


if __name__ == "__main__":
    main()
