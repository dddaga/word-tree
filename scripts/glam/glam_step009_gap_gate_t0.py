"""glam_step009: FgSegNet_v2-style GAP gate on the locality head, CIFAR-100 T0 (2026-08-14).

Motivation (user, FgSegNet_v2 source read 2026-08-14). Its decoder does
    b = GAP(conv1x1(shallow));  x = x + x*b        # i.e. x * (1 + b)
— an UNBOUNDED gate that survives, because a RESIDUAL keeps the identity path alive.
That is a SECOND escape from gate-death, distinct from the one we already found:
    A2  key*value  symmetric unbounded, no identity -> 94.84 (L=1) -> 89.29 (L=2)  DIES
    s004 sigmoid(key)*value  BOUNDED, applied once  -> 96.91                       LIVES
    here x*(1+g)             UNBOUNDED but RESIDUAL                                 ?

HYPOTHESIS under test: gate-death is loss of the IDENTITY PATH, not multiplication.
Boundedness is one guarantee of it; a residual wrapper is a stronger one.
S1 vs S3 is the load-bearing pair — both add ZERO params, so neither is confounded
with capacity, and they separate "residual" from "bounded" as the surviving ingredient.

Gate axis: x is [B, P=512, chunk_dim=49] = channels x spatial. GAP over chunk_dim
gives a per-channel global-energy vector, the exact analogue of FgSegNet's GAP.

Arms (--arm):
  S0  LOC control, no gate                              (=glam_step007 LOC d8 T0, 64.12% ref)
  S1  y * (1 + GAP(x))          residual, unbounded     +0 params
  S2  S1 with a learned per-channel scale on the gate   +1,024 params (0.25%)
  S3  y * sigmoid(GAP(x))       bounded, NO residual    +0 params

No param-matched control arm: the largest arm adds 1,024 params on a 416K head
(0.25%), so a capacity confound is not available as an explanation.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import torch
import torch.nn.functional as F

from src.sgnnet.model_glam import GLAMLayer, GLAMNet

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="S0", help="S0 / S1 / S2 / S3")
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
GATE = {"S0": None, "S1": "residual", "S2": "residual_scaled", "S3": "sigmoid"}[args.arm]


class GatedGLAMLayer(GLAMLayer):
    """GLAMLayer + a GAP-derived per-channel gate on the chunk projections.

    gate source g = mean over chunk_dim of the layer input -> [B, P], the analogue of
    FgSegNet_v2's GlobalAveragePooling2D over the spatial axes.
    """

    def __init__(self, *a, gate: str | None = None, **kw):
        super().__init__(*a, **kw)
        self.gate = gate
        if gate == "residual_scaled":
            self.gate_w = torch.nn.Parameter(torch.ones(self.P))
            self.gate_b = torch.nn.Parameter(torch.zeros(self.P))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate is None:
            return super().forward(x)
        B = x.shape[0]
        xc = x.view(B, self.P, self.chunk_dim)
        Ec = self.E[self.chunk_assign]
        y = torch.einsum("bpc,pcd->bpd", xc, Ec) + self.bias[self.chunk_assign]

        g = xc.mean(dim=-1)                                          # [B, P] GAP
        if self.gate == "residual":
            y = y * (1.0 + g.unsqueeze(-1))
        elif self.gate == "residual_scaled":
            y = y * (1.0 + (g * self.gate_w + self.gate_b).unsqueeze(-1))
        else:                                                        # sigmoid, no residual
            y = y * torch.sigmoid(g).unsqueeze(-1)

        y = self._act(y)
        yg = y[:, self.groups, :]
        gg = yg.prod(dim=2) if self.within_op == "mul" else yg.sum(dim=2)
        if self.across_op == "add":
            agg = F.layer_norm(gg.sum(dim=1), (self.d_out,))
            return self._act(agg)
        return gg.reshape(B, self.G * self.d_out)


def build():
    net = GLAMNet(in_dim=N_IN, n_out=N_OUT, L=1, P=512, d_out=args.d_out, M=args.M,
                  G=512, gsz=1, within_op="add", across_op="concat",
                  selectivity=False, seed=SEED, collapse_last=False)
    if GATE is not None:
        lyr = GatedGLAMLayer(N_IN, P=512, d_out=args.d_out, M=args.M, G=512, gsz=1,
                             within_op="add", across_op="concat", selectivity=False,
                             seed=SEED, gate=GATE)
        net.layers[0] = lyr
    return net


def load_data():
    with h5py.File(ROOT / args.store, "r") as f:
        tr_x = torch.tensor(f["train/features"][:], dtype=torch.float32)
        tr_y = torch.tensor(f["train/labels"][:], dtype=torch.long)
        va_x = torch.tensor(f["val/features"][:], dtype=torch.float32)
        va_y = torch.tensor(f["val/labels"][:], dtype=torch.long)
    if args.data_frac >= 1.0:
        return tr_x, tr_y, va_x, va_y
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(tr_x), generator=g)[:int(len(tr_x) * args.data_frac)]
    return tr_x[idx], tr_y[idx], va_x, va_y


def evaluate(model, va_x, va_y):
    model.eval(); model.set_slope(1e-5)
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
    print(f"\n{'-'*60}\n{args.arm} gate={GATE}  params={n_p:,} ({100*n_p/FC_REF:.4f}% FC)  "
          f"device={DEVICE}", flush=True)
    best, best_ep, t0 = 0.0, 0, time.time()
    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        model.train(); model.set_slope(0.01)
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
            print(f"  e{ep+1:3d}/{EPOCHS}  val={acc:.4f}  best={best:.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    elapsed = time.time() - t0
    print(f"  DONE: best={best:.4f} @ep{best_ep}  params={n_p:,}  {elapsed:.0f}s")
    return {"arm": args.arm, "gate": GATE, "d_out": args.d_out, "M": args.M,
            "n_params": n_p, "pct_fc": round(100 * n_p / FC_REF, 4),
            "best": round(best, 4), "best_ep": best_ep, "elapsed_s": round(elapsed, 1),
            "store": args.store, "seed": SEED}


def main():
    if args.smoke_test:
        m = build().to(DEVICE)
        m.train(); m.set_slope(0.01)
        out = m(torch.randn(4, N_IN, device=DEVICE))
        n_p = sum(p.numel() for p in m.parameters())
        ok = out.shape == (4, N_OUT) and not out.isnan().any()
        print(f"  {args.arm} gate={GATE} params={n_p:,} out={tuple(out.shape)} "
              f"{'OK' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

    data = load_data()
    print(f"\n{'='*70}\nglam_step009 — {args.arm} GAP gate CIFAR-100 T0 "
          f"({EPOCHS}ep, {int(args.data_frac*100)}%)\n{'='*70}")
    result = train_arm(data)
    p = ROOT / "results" / "glam" / f"glam_step009_{args.arm}_{args.tag}_seed{SEED}__{SLOT}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"step": "glam_step009", **result}, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
