"""ffn_step003: Anneal-curriculum squish-then-MIX FFN — T0 (20ep, 50% Imagenette).

The user's original untested activation schedule (distinct from step001 variant B,
which was RReLU-throughout and lost −0.4pp). Here activations are RReLU (randomized
leaky, slope ~U[1/8,1/3] on negatives) for the EARLY epochs — a regularizer/noise
schedule — then switch to HARD ReLU for the final ANNEAL_FRAC of epochs so the net
settles on the deterministic function it will use at inference. Inference is always
hard ReLU. Applied to the step002 squish-then-MIX arch (bias variant, its best head).

Hypothesis: the early RReLU noise regularizes, the late hard-ReLU removes train/test
mismatch → beats step002's plain-ReLU ~93% plateau. If it also caps at ~93%, the
FFN-head ceiling is activation-independent (strengthens SGNNET's mechanistic lead).

VGG16 FC block = 119,586,826 params. Budgets: 1%=1,195,868 | 5%=5,979,341.
MPS lacks aten::rrelu_with_noise, so RReLU is applied manually (torch.where).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--data_frac", type=float, default=0.5, help="train subsample (T0/T1=0.5, T2=1.0)")
parser.add_argument("--anneal_frac", type=float, default=0.3, help="final fraction of epochs on HARD ReLU")
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--smoke_test", action="store_true")
parser.add_argument("--tag", default="t0", help="output filename tag (t0/t1/t2)")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS, SEED, BATCH = args.epochs, args.seed, 512
N_CH, CH_DIM, N_OUT = 512, 49, 10
FC_REF = 119_586_826
# (squish_dim, mix_hidden_widths) — mix stack = [N_CH*squish] + hidden + [N_OUT]
BUDGETS = {"b1": (0.01, 8, [240]), "b5": (0.05, 8, [1152, 128])}
SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / "ffn_baseline" / f"ffn_step003_anneal_{args.tag}_seed{SEED}__{SLOT}.json"


class AnnealSquishMixFFN(nn.Module):
    """squish-then-MIX (bias head) with RReLU-early -> hard-ReLU-late activation anneal."""
    def __init__(self, squish_dim: int, mix_hidden: list):
        super().__init__()
        torch.manual_seed(SEED)
        self.hard = False                                   # set True by train loop past anneal boundary
        self.squish_w = nn.Parameter(
            torch.randn(N_CH, CH_DIM, squish_dim) * (2.0 / CH_DIM) ** 0.5)
        self.squish_b = nn.Parameter(torch.zeros(N_CH, squish_dim))
        dims = [N_CH * squish_dim] + list(mix_hidden) + [N_OUT]
        self.mix = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

    def _act(self, z):
        # hard ReLU at inference OR after the anneal boundary; else manual RReLU (MPS lacks kernel)
        if self.hard or not self.training:
            return F.relu(z)
        slope = torch.empty_like(z).uniform_(1.0 / 8, 1.0 / 3)
        return torch.where(z >= 0, z, z * slope)

    def forward(self, x):
        z = x.view(-1, N_CH, CH_DIM)                        # (B, 512, 49)
        z = torch.einsum("bcw,cwo->bco", z, self.squish_w)  # per-channel squish
        z = F.layer_norm(z, (z.shape[-1],))
        z = z + self.squish_b
        z = self._act(z)
        h = z.flatten(1)                                    # CONCAT -> cross-channel from here
        for li, lin in enumerate(self.mix):
            h = lin(h)
            if li < len(self.mix) - 1:
                h = F.layer_norm(h, (h.shape[-1],))
                h = self._act(h)
        return h


def load_data():
    with h5py.File(ROOT / "data" / "store.h5", "r") as f:
        tr_x = torch.tensor(f["train/features"][:], dtype=torch.float32)
        tr_y = torch.tensor(f["train/labels"][:], dtype=torch.long)
        va_x = torch.tensor(f["val/features"][:], dtype=torch.float32)
        va_y = torch.tensor(f["val/labels"][:], dtype=torch.long)
    if args.data_frac >= 1.0:                               # T2 = full data
        return tr_x, tr_y, va_x, va_y
    g = torch.Generator().manual_seed(SEED)                # T0/T1 = subsample
    n = int(len(tr_x) * args.data_frac)
    idx = torch.randperm(len(tr_x), generator=g)[:n]
    return tr_x[idx], tr_y[idx], va_x, va_y


def evaluate(model, va_x, va_y):
    model.eval()                                           # eval() -> _act forces hard ReLU
    correct = 0
    with torch.no_grad():
        for i in range(0, len(va_x), BATCH):
            out = model(va_x[i:i + BATCH].to(DEVICE))
            correct += (out.argmax(1).cpu() == va_y[i:i + BATCH]).sum().item()
    return correct / len(va_y)


def train_config(name, squish_dim, mix_hidden, data) -> dict:
    tr_x, tr_y, va_x, va_y = data
    model = AnnealSquishMixFFN(squish_dim, mix_hidden).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    n_batches = (len(tr_x) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=EPOCHS * n_batches, pct_start=0.1, anneal_strategy="cos")
    anneal_ep = EPOCHS - int(EPOCHS * args.anneal_frac)     # first hard-ReLU epoch (0-indexed)
    print(f"\n{'─'*60}\n{name}: squish={squish_dim} mix={mix_hidden}  "
          f"params={n_p:,} ({100*n_p/FC_REF:.3f}% FC)  hard@ep{anneal_ep+1}  device={DEVICE}", flush=True)

    best, best_ep, t0 = 0.0, 0, time.time()
    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        model.train()
        model.hard = ep >= anneal_ep                       # RReLU early, hard ReLU final anneal_frac
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
        if (ep + 1) % 5 == 0 or ep == 0 or ep + 1 == anneal_ep + 1:
            tag = " <hard>" if model.hard else ""
            print(f"  e{ep+1:3d}/{EPOCHS}  val={acc:.4f}  best={best:.4f}{tag}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    elapsed = time.time() - t0
    print(f"  DONE: best={best:.4f} @ep{best_ep}  params={n_p:,}  {elapsed:.0f}s")
    return {"squish_dim": squish_dim, "mix_hidden": mix_hidden, "anneal_frac": args.anneal_frac,
            "n_params": n_p, "pct_fc": round(100 * n_p / FC_REF, 4),
            "best": round(best, 4), "best_ep": best_ep,
            "dense_flops": 2 * n_p, "elapsed_s": round(elapsed, 1)}


def main():
    configs = {b: (sq, mh) for b, (_, sq, mh) in BUDGETS.items()}
    if args.smoke_test:
        ok = True
        for name, (sq, mh) in configs.items():
            m = AnnealSquishMixFFN(sq, mh)
            m.train(); m.hard = False
            out = m(torch.randn(4, 25088))                 # exercises RReLU branch
            n_p = sum(p.numel() for p in m.parameters())
            budget = int(FC_REF * BUDGETS[name][0])
            fits = n_p <= budget and out.shape == (4, N_OUT) and not out.isnan().any()
            print(f"  {name:<6} params={n_p:,} budget={budget:,} {'OK' if fits else 'FAIL'}")
            ok &= fits
        sys.exit(0 if ok else 1)

    data = load_data()
    print(f"\n{'='*70}\nffn_step003 — anneal-curriculum squish-MIX FFN {args.tag} "
          f"({EPOCHS}ep, {int(args.data_frac*100)}% Imagenette, anneal_frac={args.anneal_frac})")
    print(f"  device={DEVICE}  configs={list(configs)}\n{'='*70}")
    results = {"step": "ffn_step003", "seed": SEED, "anneal_frac": args.anneal_frac, "configs": {}}
    for name, (sq, mh) in configs.items():
        results["configs"][name] = train_config(name, sq, mh, data)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}\nffn_step003 SUMMARY (refs: step002 plain-ReLU mix=93.2%, "
          f"no-mix step001=93.0%, SGNNET=95.95% @35K)")
    for name, r in results["configs"].items():
        print(f"  {name:<6} {r['best']:.4f}  params={r['n_params']:,} ({r['pct_fc']}%)")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
