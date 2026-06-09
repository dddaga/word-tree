"""ffn_step001: Per-channel depth-wise FFN baseline — T0 (20ep, 50% Imagenette).

PARALLEL LINE (learnings/ffn_baseline/) — standard components only, isolated
from main SGNNET line. Hypothesis: norm→ReLU sparsity in per-channel FFNs at
1%/5% of VGG16 FC budget gets close to SGNNET accuracy.

VGG16 FC block = 119,586,826 params. Budgets: 1%=1,195,868 | 5%=5,979,341.
(B, 25088) → (B, 512, 49); each channel its own MLP (einsum); concat → readout.

Variants: A_norm_relu (LN→ReLU always), B_norm_rrelu (LN→RReLU train/ReLU infer),
C_norm_bias_rrelu (LN+bias→RReLU train/ReLU infer).
Targets: SGNNET step605 95.95% @35K params; D=16 ceiling 97.30%.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
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
BUDGETS = {"b1": (0.01, [49, 28, 20, 8]), "b5": (0.05, [49, 84, 60, 32, 8])}
VARIANTS = ["A_norm_relu", "B_norm_rrelu", "C_norm_bias_rrelu"]
SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / "ffn_baseline" / f"ffn_step001_perchannel_{args.tag}_seed{SEED}__{SLOT}.json"


class PerChannelFFN(nn.Module):
    def __init__(self, widths: list, variant: str):
        super().__init__()
        torch.manual_seed(SEED)
        self.variant = variant
        self.weights = nn.ParameterList(
            nn.Parameter(torch.randn(N_CH, wi, wo) * (2.0 / wi) ** 0.5)
            for wi, wo in zip(widths[:-1], widths[1:]))
        self.biases = nn.ParameterList(
            nn.Parameter(torch.zeros(N_CH, wo)) for wo in widths[1:]
        ) if variant == "C_norm_bias_rrelu" else None
        self.readout = nn.Linear(N_CH * widths[-1], N_OUT)
        self.track_sparsity = False
        self.zeros, self.totals = [], []

    def _act(self, z):
        if self.variant == "A_norm_relu" or not self.training:
            return F.relu(z)
        # manual RReLU — aten::rrelu_with_noise missing on MPS
        slope = torch.empty_like(z).uniform_(1.0 / 8, 1.0 / 3)
        return torch.where(z >= 0, z, z * slope)

    def forward(self, x):
        z = x.view(-1, N_CH, CH_DIM)
        for li, W in enumerate(self.weights):
            z = torch.einsum("bcw,cwo->bco", z, W)
            z = F.layer_norm(z, (z.shape[-1],))
            if self.biases is not None:
                z = z + self.biases[li]
            z = self._act(z)
            if self.track_sparsity:
                self.zeros[li] += (z == 0).sum().item()
                self.totals[li] += z.numel()
        return self.readout(z.flatten(1))


def load_data():
    with h5py.File(ROOT / "data" / "store.h5", "r") as f:
        tr_x = torch.tensor(f["train/features"][:], dtype=torch.float32)
        tr_y = torch.tensor(f["train/labels"][:], dtype=torch.long)
        va_x = torch.tensor(f["val/features"][:], dtype=torch.float32)
        va_y = torch.tensor(f["val/labels"][:], dtype=torch.long)
    g = torch.Generator().manual_seed(SEED)               # T0 = 50% data
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


def train_config(name, widths, variant, data) -> dict:
    tr_x, tr_y, va_x, va_y = data
    model = PerChannelFFN(widths, variant).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    n_batches = (len(tr_x) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=EPOCHS * n_batches,
        pct_start=0.1, anneal_strategy="cos")
    print(f"\n{'─'*60}\n{name}: widths={widths}  params={n_p:,} "
          f"({100*n_p/FC_REF:.3f}% FC)  device={DEVICE}", flush=True)

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

    n_layers = len(model.weights)
    model.track_sparsity = True
    model.zeros, model.totals = [0] * n_layers, [0] * n_layers
    final = evaluate(model, va_x, va_y)
    sparsity = [round(z / t, 4) for z, t in zip(model.zeros, model.totals)]
    dense_flops = 2 * n_p
    # hidden-layer matmul FLOPs scale with fraction of nonzero inputs
    eff = 2 * N_CH * widths[0] * widths[1] + 2 * model.readout.in_features * N_OUT
    for li in range(1, n_layers):
        eff += int(2 * N_CH * widths[li] * widths[li + 1] * (1 - sparsity[li - 1]))
    elapsed = time.time() - t0
    print(f"  DONE: best={best:.4f} @ep{best_ep}  final={final:.4f}  {elapsed:.0f}s")
    print(f"  sparsity/layer={sparsity}  dense={dense_flops/1e6:.2f}M  eff={eff/1e6:.2f}M")
    return {"widths": widths, "variant": variant, "n_params": n_p,
            "pct_fc": round(100 * n_p / FC_REF, 4), "best": round(best, 4),
            "best_ep": best_ep, "final": round(final, 4),
            "sparsity_per_layer": sparsity, "dense_flops": dense_flops,
            "effective_flops": eff, "elapsed_s": round(elapsed, 1)}


def main():
    configs = {f"{b}_{v}": (w, v) for b, (_, w) in BUDGETS.items() for v in VARIANTS}
    if args.smoke_test:
        ok = True
        for name, (w, v) in configs.items():
            m = PerChannelFFN(w, v)
            out = m(torch.randn(4, 25088))
            n_p = sum(p.numel() for p in m.parameters())
            budget = int(FC_REF * BUDGETS[name[:2]][0])
            fits = n_p <= budget and out.shape == (4, N_OUT) and not out.isnan().any()
            print(f"  {name:<22} params={n_p:,} budget={budget:,} {'OK' if fits else 'FAIL'}")
            ok &= fits
        sys.exit(0 if ok else 1)

    data = load_data()
    print(f"\n{'='*70}\nffn_step001 — per-channel FFN T0 (20ep, 50% Imagenette)")
    print(f"  device={DEVICE}  configs={list(configs)}\n{'='*70}")
    results = {"step": "ffn_step001", "seed": SEED, "configs": {}}
    for name, (w, v) in configs.items():
        results["configs"][name] = train_config(name, w, v, data)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}\nffn_step001 SUMMARY  (SGNNET refs: step605=95.95% @35K, ceiling=97.30%)")
    for name, r in results["configs"].items():
        print(f"  {name:<22} {r['best']:.4f}  params={r['n_params']:,} "
              f"({r['pct_fc']}%)  eff_flops={r['effective_flops']/1e6:.2f}M")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
