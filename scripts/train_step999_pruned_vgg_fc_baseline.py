"""Step 999 — Pruned VGG16 FC baseline at 34,976 params (SGNNET-matched budget).

Motivation (meditation 005): the paper explicitly flags this baseline as MISSING
("Pruned VGG16 FC at 34,976 params — NOT YET — Blocks FLOPs-efficient head comparison",
sec3). It is a reviewer's first question: "is SGNNET beating a *fairly pruned* FC head,
or just a full one?" This trains the VGG16 FC head, magnitude-prunes it to exactly the
champion's 34,976-param budget (99.97% sparsity), fine-tunes, and reports accuracy.

Expected: at 99.97% unstructured sparsity the FC head collapses far below SGNNET's
95.95% — giving the paper a clean iso-param win. If it holds up, that is a finding too.

Usage:
    d_env/bin/python3 scripts/train_step999_pruned_vgg_fc_baseline.py --epochs 40 --target 34976
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

parser = argparse.ArgumentParser(description="Step 999 pruned VGG16 FC baseline @ matched params")
parser.add_argument("--epochs",  type=int, default=40)
parser.add_argument("--target",  type=int, default=34976, help="Target param budget (champion-matched)")
parser.add_argument("--device",  default="auto")
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--slot",    default="studio_mps")
args = parser.parse_args()

N_IN, N_OUT = 25088, 10
ROOT = Path(__file__).resolve().parents[1]


def pick_device(torch):
    if args.device != "auto":
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_vgg_fc(nn):
    return nn.Sequential(
        nn.Linear(N_IN, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, N_OUT))


def global_magnitude_mask(torch, model, keep):
    """Global unstructured magnitude prune → boolean masks keeping top-`keep` weights."""
    weights = [m.weight for m in model if hasattr(m, "weight")]
    allw = torch.cat([w.abs().flatten() for w in weights])
    thresh = torch.kthvalue(allw, allw.numel() - keep).values
    return {id(w): (w.abs() >= thresh).float() for w in weights}, weights


def main():
    import torch, torch.nn as nn, torch.nn.functional as F
    import h5py, numpy as np
    torch.manual_seed(args.seed)
    device = pick_device(torch)
    model = build_vgg_fc(nn).to(device)

    # Budget accounting: distribute target params across weight tensors by magnitude (global).
    total_w = sum(m.weight.numel() for m in model if hasattr(m, "weight"))
    keep = min(args.target, total_w)
    masks, weights = global_magnitude_mask(torch, model, keep)
    kept = int(sum(m.sum().item() for m in masks.values()))
    print(f"VGG_FC total weights={total_w:,}  target={args.target:,}  kept={kept:,} "
          f"(sparsity={1 - kept / total_w:.5f})")

    # Load Imagenette VGG features (data/store.h5) — same store the FFN/SGNNET lines use.
    store = ROOT / "data" / "store.h5"
    if not store.exists():
        print(f"[WARN] {store} missing — smoke path only; skipping train.")
        return
    with h5py.File(store, "r") as f:
        tr_x = torch.tensor(np.asarray(f["train/features"]), dtype=torch.float32)
        tr_y = torch.tensor(np.asarray(f["train/labels"]), dtype=torch.long)
        va_x = torch.tensor(np.asarray(f["val/features"]), dtype=torch.float32)
        va_y = torch.tensor(np.asarray(f["val/labels"]), dtype=torch.long)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(tr_x.shape[0])
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            xb, yb = tr_x[idx].to(device), tr_y[idx].to(device)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
            with torch.no_grad():  # enforce prune mask (fixed sparsity fine-tune)
                for m in model:
                    if hasattr(m, "weight"):
                        m.weight.mul_(masks[id(m.weight)])
        model.eval()
        with torch.no_grad():
            acc = (model(va_x.to(device)).argmax(1).cpu() == va_y).float().mean().item()
        best = max(best, acc)
        if ep % 5 == 0:
            print(f"  ep{ep:3d} val={acc:.4f} best={best:.4f}")

    print(f"BEST pruned VGG_FC @ {kept:,} params = {best:.4f}  (vs SGNNET champion 95.95%)")
    out = ROOT / "results" / f"train_step999_pruned_vgg_fc__{args.slot}.json"
    out.write_text(json.dumps(dict(config=vars(args), kept_params=kept,
                                   best_acc=best), indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
