"""ffn_step002: Squish-then-MIX per-channel FFN — T0 (20ep, 50% Imagenette).

Corrects ffn_step001, which applied the FULL depth per-channel (einsum, channel
kept separate every layer) with only a terminal LINEAR readout — i.e. NO nonlinear
cross-channel mixing. That plateaued at 93.0% and saturated at 1% budget because all
capacity went into isolated per-channel MLPs.

This is the INTENDED design: (1) squish each channel 49->s per-channel, (2) CONCAT to
512*s, (3) run a nonlinear mixing FFN on the concat vector -> 10. Most params now live
in the MIXING layers. Tests whether post-concat mixing beats the no-mix 93.0% plateau
and approaches SGNNET champion (95.95% @35K) / D=16 ceiling (97.30%).

VGG16 FC block = 119,586,826 params. Budgets: 1%=1,195,868 | 5%=5,979,341.
Variants: A_norm_relu (LN no-affine -> ReLU), C_norm_bias_relu (+ per-channel squish bias).
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
VARIANTS = ["A_norm_relu", "C_norm_bias_relu"]
SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / "ffn_baseline" / f"ffn_step002_squishmix_{args.tag}_seed{SEED}__{SLOT}.json"


class SquishMixFFN(nn.Module):
    """Per-channel squish -> concat -> nonlinear mixing FFN -> logits."""
    def __init__(self, squish_dim: int, mix_hidden: list, variant: str):
        super().__init__()
        torch.manual_seed(SEED)
        self.variant = variant
        self.squish_w = nn.Parameter(
            torch.randn(N_CH, CH_DIM, squish_dim) * (2.0 / CH_DIM) ** 0.5)
        self.squish_b = (nn.Parameter(torch.zeros(N_CH, squish_dim))
                         if variant == "C_norm_bias_relu" else None)
        dims = [N_CH * squish_dim] + list(mix_hidden) + [N_OUT]
        self.mix = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

    def forward(self, x):
        z = x.view(-1, N_CH, CH_DIM)                        # (B, 512, 49)
        z = torch.einsum("bcw,cwo->bco", z, self.squish_w)  # per-channel squish
        z = F.layer_norm(z, (z.shape[-1],))
        if self.squish_b is not None:
            z = z + self.squish_b
        z = F.relu(z)
        h = z.flatten(1)                                    # CONCAT -> cross-channel from here
        for li, lin in enumerate(self.mix):
            h = lin(h)
            if li < len(self.mix) - 1:                       # hidden layers: norm + act (mixing)
                h = F.layer_norm(h, (h.shape[-1],))
                h = F.relu(h)
        return h


def load_data():
    with h5py.File(ROOT / "data" / "store.h5", "r") as f:
        tr_x = torch.tensor(f["train/features"][:], dtype=torch.float32)
        tr_y = torch.tensor(f["train/labels"][:], dtype=torch.long)
        va_x = torch.tensor(f["val/features"][:], dtype=torch.float32)
        va_y = torch.tensor(f["val/labels"][:], dtype=torch.long)
    g = torch.Generator().manual_seed(SEED)                 # T0 = 50% data
    idx = torch.randperm(len(tr_x), generator=g)[: len(tr_x) // 2]
    return tr_x[idx], tr_y[idx], va_x, va_y


def evaluate(model, va_x, va_y):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(va_x), BATCH):
            out = model(va_x[i:i + BATCH].to(DEVICE))
            correct += (out.argmax(1).cpu() == va_y[i:i + BATCH]).sum().item()
    return correct / len(va_y)


def train_config(name, squish_dim, mix_hidden, variant, data) -> dict:
    tr_x, tr_y, va_x, va_y = data
    model = SquishMixFFN(squish_dim, mix_hidden, variant).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    n_batches = (len(tr_x) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=EPOCHS * n_batches, pct_start=0.1, anneal_strategy="cos")
    print(f"\n{'─'*60}\n{name}: squish={squish_dim} mix={mix_hidden}  "
          f"params={n_p:,} ({100*n_p/FC_REF:.3f}% FC)  device={DEVICE}", flush=True)

    best, best_ep, t0 = 0.0, 0, time.time()
    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        model.train()
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
    return {"squish_dim": squish_dim, "mix_hidden": mix_hidden, "variant": variant,
            "n_params": n_p, "pct_fc": round(100 * n_p / FC_REF, 4),
            "best": round(best, 4), "best_ep": best_ep,
            "dense_flops": 2 * n_p, "elapsed_s": round(elapsed, 1)}


def build_configs():
    return {f"{b}_{v}": (sq, mh, v)
            for b, (_, sq, mh) in BUDGETS.items() for v in VARIANTS}


def main():
    configs = build_configs()
    if args.smoke_test:
        ok = True
        for name, (sq, mh, v) in configs.items():
            m = SquishMixFFN(sq, mh, v)
            out = m(torch.randn(4, 25088))
            n_p = sum(p.numel() for p in m.parameters())
            budget = int(FC_REF * BUDGETS[name[:2]][0])
            fits = n_p <= budget and out.shape == (4, N_OUT) and not out.isnan().any()
            print(f"  {name:<24} params={n_p:,} budget={budget:,} {'OK' if fits else 'FAIL'}")
            ok &= fits
        sys.exit(0 if ok else 1)

    data = load_data()
    print(f"\n{'='*70}\nffn_step002 — squish-then-MIX per-channel FFN T0 (20ep, 50% Imagenette)")
    print(f"  device={DEVICE}  configs={list(configs)}\n{'='*70}")
    results = {"step": "ffn_step002", "seed": SEED, "configs": {}}
    for name, (sq, mh, v) in configs.items():
        results["configs"][name] = train_config(name, sq, mh, v, data)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}\nffn_step002 SUMMARY (refs: no-mix step001=93.0%, SGNNET=95.95% @35K, ceiling=97.30%)")
    for name, r in results["configs"].items():
        print(f"  {name:<24} {r['best']:.4f}  params={r['n_params']:,} ({r['pct_fc']}%)")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
