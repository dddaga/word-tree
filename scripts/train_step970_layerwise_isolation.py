"""Step 970: Layer-wise vs full backprop isolation on a tiny MLP.

QUESTION: Does layer-wise sequential unfreezing (greedy layer-wise pretraining)
converge differently from end-to-end backprop? Does partial-grad give speedup?

SPEEDUP ANSWER (answered here empirically):
  zero_grad-after-backward = NO speedup (backward already ran).
  requires_grad=False BEFORE backward = YES speedup (PyTorch skips those layers).
  This script measures wall-time per epoch for both methods.

ARCHITECTURE: VGG FC features → TinyMLP (5 hidden layers × 10 neurons = 50 hidden)
  L0: Linear(25088, 10) + ReLU
  L1: Linear(10, 10)    + ReLU
  L2: Linear(10, 10)    + ReLU
  L3: Linear(10, 10)    + ReLU
  L4: Linear(10, 10)    + ReLU
  L5: Linear(10, 10)    ← output head (no ReLU, 10 classes)

VARIANTS:
  full_bp    — standard end-to-end AdamW, all layers trainable every epoch
  layer_seq  — each phase trains exactly ONE layer (others frozen with requires_grad=False)
               Schedule: L5→L4→L3→L2→L1→L0, EP_PER_LAYER epochs each
  layer_cum  — greedy layer-wise: L5 first, then unfreeze L4+L5 together, etc.
               (Hinton/Bengio 2006 greedy pretraining variant)

METRICS per epoch:
  acc, loss, epoch_wall_ms (actual backward speed with/without frozen layers)

ADVANCE CRITERION:
  Any layer-wise variant reaches full_bp accuracy within 2pp
  AND (epoch_wall_ms < full_bp epoch_wall_ms) → efficiency gain confirmed.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_aug.h5")
parser.add_argument("--frac",    type=float, default=0.5)
parser.add_argument("--ep_per_layer", type=int, default=20,
                    help="Epochs per layer phase in layer_seq/layer_cum")
parser.add_argument("--configs", default="full_bp,layer_seq,layer_cum")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SEED   = args.seed
BATCH  = 64
N_IN   = 25088
N_OUT  = 10
HIDDEN = [10, 10, 10, 10, 10]   # 5 hidden layers × 10 neurons = 50 hidden

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step970_layerwise_seed{SEED}__{SLOT}.json"


# ── TinyMLP ───────────────────────────────────────────────────────────────────
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        dims   = [N_IN] + HIDDEN + [N_OUT]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)
        self.n_layers = len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.n_layers - 1:
                x = F.relu(x)
        return x

    def set_trainable_layer(self, active_idx: int | None):
        """
        Freeze all layers except active_idx.
        active_idx=None → all trainable (full_bp mode).
        Uses requires_grad=False for actual backward speedup.
        """
        for i, layer in enumerate(self.layers):
            trainable = (active_idx is None) or (i == active_idx)
            for p in layer.parameters():
                p.requires_grad = trainable

    def set_trainable_from(self, min_idx: int):
        """Unfreeze layers[min_idx:], freeze layers[:min_idx]."""
        for i, layer in enumerate(self.layers):
            trainable = (i >= min_idx)
            for p in layer.parameters():
                p.requires_grad = trainable


def build_model() -> TinyMLP:
    torch.manual_seed(SEED)
    return TinyMLP().to(DEVICE)


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found"); sys.exit(1)
    with h5py.File(data_path, "r") as f:
        tr_x = torch.tensor(f["train/features"][:], dtype=torch.float32)
        tr_y = torch.tensor(f["train/labels"][:],   dtype=torch.long)
        va_x = torch.tensor(f["val/features"][:],   dtype=torch.float32)
        va_y = torch.tensor(f["val/labels"][:],     dtype=torch.long)
    n_tr = int(len(tr_x) * args.frac)
    idx  = torch.randperm(len(tr_x), generator=torch.Generator().manual_seed(SEED))[:n_tr]
    tr_x, tr_y = tr_x[idx], tr_y[idx]
    tr = DataLoader(TensorDataset(tr_x, tr_y), batch_size=BATCH, shuffle=True,  num_workers=2)
    va = DataLoader(TensorDataset(va_x, va_y), batch_size=256, shuffle=False, num_workers=2)
    print(f"  Data: {len(tr_x)} train ({args.frac:.0%}) / {len(va_x)} val")
    return tr, va


@torch.no_grad()
def evaluate(model, va) -> float:
    model.eval()
    correct = total = 0
    for x, y in va:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def make_opt(model, lr=3e-3) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)


def run_one_epoch(model, tr, lr=3e-3) -> tuple[float, float]:
    """Train one epoch; returns (avg_loss, wall_ms). Optimizer created fresh (for frozen-layer support)."""
    model.train()
    opt   = make_opt(model, lr)
    total_loss = 0.0
    t0 = time.perf_counter()
    for bx, by in tr:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt.zero_grad()
        loss = F.cross_entropy(model(bx), by)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    wall_ms = 1000 * (time.perf_counter() - t0)
    return total_loss / len(tr), wall_ms


def train_config(key: str, tr, va, results: dict):
    model  = build_model()
    n_p    = sum(p.numel() for p in model.parameters())
    n_lay  = model.n_layers  # 6 layers (L0..L5)
    ep_pl  = args.ep_per_layer
    total_epochs = n_lay * ep_pl

    hist_ep, hist_acc, hist_loss, hist_ms, hist_frozen = [], [], [], [], []
    best_acc = 0.0
    t_start  = time.time()

    print(f"\n{'─'*60}")
    print(f"{key}: {n_lay} layers  ep_per_layer={ep_pl}  total_ep={total_epochs}  params={n_p:,}", flush=True)

    if key == "full_bp":
        model.set_trainable_layer(None)  # all trainable
        for ep in range(1, total_epochs + 1):
            avg_loss, wall_ms = run_one_epoch(model, tr)
            acc = evaluate(model, va)
            if acc > best_acc: best_acc = acc
            hist_ep.append(ep); hist_acc.append(round(acc, 4))
            hist_loss.append(round(avg_loss, 4)); hist_ms.append(round(wall_ms, 1))
            hist_frozen.append(0)
            print(f"  ep{ep:3d}  acc={acc:.4f}  loss={avg_loss:.4f}  {wall_ms:.0f}ms/ep  [all_trainable]", flush=True)

    elif key == "layer_seq":
        # Train one layer at a time, last-to-first
        for phase, layer_idx in enumerate(range(n_lay - 1, -1, -1)):
            model.set_trainable_layer(layer_idx)
            n_frozen = n_lay - 1
            print(f"  PHASE {phase+1}: training L{layer_idx} only  ({n_frozen} layers frozen)", flush=True)
            for local_ep in range(1, ep_pl + 1):
                ep = phase * ep_pl + local_ep
                avg_loss, wall_ms = run_one_epoch(model, tr)
                acc = evaluate(model, va)
                if acc > best_acc: best_acc = acc
                hist_ep.append(ep); hist_acc.append(round(acc, 4))
                hist_loss.append(round(avg_loss, 4)); hist_ms.append(round(wall_ms, 1))
                hist_frozen.append(n_frozen)
                print(f"  ep{ep:3d}  acc={acc:.4f}  loss={avg_loss:.4f}  {wall_ms:.0f}ms/ep  [L{layer_idx} active]", flush=True)

    elif key == "layer_cum":
        # Greedy cumulative: unfreeze one more layer each phase (last→first)
        for phase in range(n_lay):
            min_idx = n_lay - 1 - phase   # L5, L4, L3, L2, L1, L0
            model.set_trainable_from(min_idx)
            n_frozen = min_idx
            n_active = n_lay - min_idx
            print(f"  PHASE {phase+1}: training L{min_idx}..L{n_lay-1}  ({n_frozen} frozen, {n_active} active)", flush=True)
            for local_ep in range(1, ep_pl + 1):
                ep = phase * ep_pl + local_ep
                avg_loss, wall_ms = run_one_epoch(model, tr)
                acc = evaluate(model, va)
                if acc > best_acc: best_acc = acc
                hist_ep.append(ep); hist_acc.append(round(acc, 4))
                hist_loss.append(round(avg_loss, 4)); hist_ms.append(round(wall_ms, 1))
                hist_frozen.append(n_frozen)
                print(f"  ep{ep:3d}  acc={acc:.4f}  loss={avg_loss:.4f}  {wall_ms:.0f}ms/ep  [{n_active} layers active]", flush=True)

    elapsed = time.time() - t_start
    avg_ms  = sum(hist_ms) / len(hist_ms) if hist_ms else 0
    print(f"  DONE: best={best_acc:.4f}  avg_ep_ms={avg_ms:.0f}ms  {elapsed:.1f}s total")

    results[key] = {
        "best_acc": round(best_acc, 4),
        "avg_ep_ms": round(avg_ms, 1),
        "hist_ep": hist_ep, "hist_acc": hist_acc,
        "hist_loss": hist_loss, "hist_ms": hist_ms,
        "hist_frozen_layers": hist_frozen,
        "elapsed_s": round(elapsed, 1),
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    return best_acc


def main():
    tr, va = load_data()
    n_p = sum(p.numel() for p in build_model().parameters())

    print(f"\n{'='*70}")
    print(f"step970 — Layer-wise vs Full Backprop Isolation")
    print(f"  device={DEVICE}  seed={SEED}  frac={args.frac}")
    print(f"  Architecture: 25088→{'→'.join(str(h) for h in HIDDEN)}→10  ({n_p:,} params)")
    print(f"  ep_per_layer={args.ep_per_layer}  total_ep={model.n_layers * args.ep_per_layer if False else (len(HIDDEN)+1) * args.ep_per_layer}")
    print(f"  SPEEDUP TEST: requires_grad=False → PyTorch skips backward for frozen layers")
    print(f"{'='*70}")

    keys    = [k.strip() for k in args.configs.split(",")]
    results = {}

    for key in keys:
        train_config(key, tr, va, results)

    print(f"\n{'='*70}")
    print(f"STEP 970 SUMMARY")
    print(f"{'='*70}")
    full_ms = results.get("full_bp", {}).get("avg_ep_ms", None)
    for k, r in results.items():
        speedup = f"{full_ms/r['avg_ep_ms']:.2f}×" if full_ms and k != "full_bp" else "1.00×"
        print(f"  {k:<12}  best={r['best_acc']:.4f}  avg_ep={r['avg_ep_ms']:.0f}ms  speedup={speedup}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
