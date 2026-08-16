"""glam_step010: CReLU / RReLU activations on the locality head, CIFAR-100 T0 (2026-08-14).

Motivation (user, 2026-08-14). The CIFAR-100 gap is tagged STRUCTURAL: the shared pool
compresses 25088 -> P*d_out before the readout, and dense is full-rank. CReLU
concat[relu(y), relu(-y)] DOUBLES the rank leaving the projection at ZERO extra
projection params — it keeps the negative phase that ReLU discards. That is the right
shape of tool for a rank-limited bottleneck (unlike a rank-1 channel rescale).

Params are not free at the READOUT, which dominates: P*d_out*n_cls = 512*8*100 = 409,600
of the 416K total. CReLU doubles the dims reaching the readout, so CReLU at d_out=k costs
the same as plain ReLU at d_out=2k. Both of those ReLU points are ALREADY MEASURED
(glam_step007 capacity sweep), so every arm here has a free iso-param control:

  A1 CReLU d_out=4  (~416K)  vs  ReLU d_out=8   64.12% (T0)   <- does sign-phase beat rank?
  A2 CReLU d_out=8  (~819K)  vs  ReLU d_out=16  66.27% (T0)   <- same question, 2x capacity

Energy cost, to be reported not omitted (mechanism 4, glam_step006 measured plain-ReLU
zero_frac 0.398-0.41): CReLU pins zero_frac near 0.50 but at 2x the activations, so
absolute nonzeros are unchanged; RReLU leaks and drives zero_frac to ~0, a near-total
loss of activation sparsity. Judge these on accuracy-per-param and log zero_frac.

Arms (--arm):
  A0  ReLU control at --d_out                      (reproduces glam_step007 LOC)
  A1  CReLU   concat[relu(y), relu(-y)]
  A2  RReLU   randomised leaky slope U(l,u) train, (l+u)/2 eval
  A3  MIX     concat[relu(y), rrelu(-y)]  — sparse positive phase, random-leaky negative
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
parser.add_argument("--arm", default="A0", help="A0 / A1 / A2 / A3")
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--data_frac", type=float, default=0.5)
parser.add_argument("--d_out", type=int, default=8)
parser.add_argument("--M", type=int, default=16)
parser.add_argument("--rrelu_lower", type=float, default=0.05)
parser.add_argument("--rrelu_upper", type=float, default=0.33)
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
DOUBLES = args.arm in ("A1", "A3")   # arms whose activation concatenates both phases


class ActGLAMLayer(GLAMLayer):
    """GLAMLayer with a swappable activation; CReLU-family arms double d_out downstream."""

    def __init__(self, *a, act: str = "A0", lower: float = 0.05, upper: float = 0.33, **kw):
        super().__init__(*a, **kw)
        self.act, self.lower, self.upper = act, lower, upper
        if act in ("A1", "A3"):
            self.out_dim = self.G * self.d_out * 2

    def _act(self, z: torch.Tensor) -> torch.Tensor:
        if self.act == "A0":
            return F.leaky_relu(z, negative_slope=self.slope)
        if self.act == "A1":
            return torch.cat([F.relu(z), F.relu(-z)], dim=-1)
        if self.act == "A2":
            return F.rrelu(z, lower=self.lower, upper=self.upper, training=self.training)
        return torch.cat([F.relu(z),
                          F.rrelu(-z, lower=self.lower, upper=self.upper,
                                  training=self.training)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        xc = x.view(B, self.P, self.chunk_dim)
        Ec = self.E[self.chunk_assign]
        y = torch.einsum("bpc,pcd->bpd", xc, Ec) + self.bias[self.chunk_assign]
        y = self._act(y)                                  # [B, P, d_out or 2*d_out]
        yg = y[:, self.groups, :]
        gg = yg.prod(dim=2) if self.within_op == "mul" else yg.sum(dim=2)
        return gg.reshape(B, gg.shape[1] * gg.shape[2])


def zero_frac(model, x) -> float:
    """Fraction of exactly-zero activations leaving layer 0 — mechanism-4 energy proxy."""
    lyr = model.layers[0]
    with torch.no_grad():
        xc = x.view(x.shape[0], lyr.P, lyr.chunk_dim)
        y = torch.einsum("bpc,pcd->bpd", xc, lyr.E[lyr.chunk_assign]) + lyr.bias[lyr.chunk_assign]
        a = lyr._act(y)
    return (a == 0).float().mean().item()


def build():
    net = GLAMNet(in_dim=N_IN, n_out=N_OUT, L=1, P=512, d_out=args.d_out, M=args.M,
                  G=512, gsz=1, within_op="add", across_op="concat",
                  selectivity=False, seed=SEED, collapse_last=False)
    lyr = ActGLAMLayer(N_IN, P=512, d_out=args.d_out, M=args.M, G=512, gsz=1,
                       within_op="add", across_op="concat", selectivity=False,
                       seed=SEED, act=args.arm, lower=args.rrelu_lower,
                       upper=args.rrelu_upper)
    net.layers[0] = lyr
    net.readout = torch.nn.Linear(lyr.out_dim, N_OUT)
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
    print(f"\n{'-'*60}\n{args.arm} d_out={args.d_out} doubles={DOUBLES}  params={n_p:,} "
          f"({100*n_p/FC_REF:.4f}% FC)  device={DEVICE}", flush=True)
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
    model.eval(); model.set_slope(1e-5)
    zf = zero_frac(model, va_x[:512].to(DEVICE))
    elapsed = time.time() - t0
    print(f"  DONE: best={best:.4f} @ep{best_ep}  params={n_p:,}  zero_frac={zf:.3f}  {elapsed:.0f}s")
    return {"arm": args.arm, "d_out": args.d_out, "M": args.M, "doubles": DOUBLES,
            "n_params": n_p, "pct_fc": round(100 * n_p / FC_REF, 4),
            "best": round(best, 4), "best_ep": best_ep, "zero_frac": round(zf, 4),
            "rrelu": [args.rrelu_lower, args.rrelu_upper],
            "elapsed_s": round(elapsed, 1), "store": args.store, "seed": SEED}


def main():
    if args.smoke_test:
        m = build().to(DEVICE)
        m.train(); m.set_slope(0.01)
        out = m(torch.randn(4, N_IN, device=DEVICE))
        n_p = sum(p.numel() for p in m.parameters())
        ok = out.shape == (4, N_OUT) and not out.isnan().any()
        print(f"  {args.arm} d_out={args.d_out} params={n_p:,} out={tuple(out.shape)} "
              f"{'OK' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

    data = load_data()
    print(f"\n{'='*70}\nglam_step010 — {args.arm} activation CIFAR-100 T0 "
          f"({EPOCHS}ep, {int(args.data_frac*100)}%)\n{'='*70}")
    result = train_arm(data)
    p = (ROOT / "results" / "glam" /
         f"glam_step010_{args.arm}_d{args.d_out}_{args.tag}_seed{SEED}__{SLOT}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"step": "glam_step010", **result}, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
