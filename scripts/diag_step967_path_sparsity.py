"""Step 967: Path sparsity diagnostic — does SGNNET already form class-specific pathways?

QUESTION
========
Before investing in backward reward scoring (step966), check whether class-specific
activation pathways ALREADY exist in a standard trained model. If yes, step966
reinforces an existing structure (favorable). If no, step966 is building from
scratch (harder, higher risk).

METRICS
=======
For each class c, collect active-node masks across test inputs:
  mask[sample] = (Z_final.norm(dim=-1) > theta) : bool[N]

(a) act_frac[c]       — mean fraction of N nodes active per class
(b) intra_jaccard[c]  — mean pairwise Jaccard within class c (same-class consistency)
(c) inter_jaccard     — mean pairwise Jaccard across classes (cross-class overlap)
(d) separation[c]     — intra_jaccard[c] - mean inter_jaccard

INTERPRETATION
==============
separation > 0.10  → pathways are ALREADY class-specific. step966 reinforces.
separation ≈ 0.0   → routing is class-agnostic. step966 builds from scratch.
separation < 0.0   → cross-class activation is more consistent than within-class (unlikely).

Also computes MI(class; active_node) per node to identify "class-selective nodes."

USAGE
=====
  # Train fresh and diagnose:
  python scripts/diag_step967_path_sparsity.py --device mps

  # Diagnose existing checkpoint:
  python scripts/diag_step967_path_sparsity.py --checkpoint checkpoints/foo.pt --device mps
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant   import SGNNET_Resonant

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device",     default="auto")
parser.add_argument("--checkpoint", default=None, help="Path to saved model .pt")
parser.add_argument("--epochs",     type=int, default=20,
                    help="Training epochs if no checkpoint provided")
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--dataset",    default="imagenette",
                    choices=["imagenette", "cifar10"])
parser.add_argument("--data_root",  default="data")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SEED   = args.seed
N      = 2048
D      = 16
N_IN   = 512    # VGG16 FC input (after flatten, before FC layers)
N_OUT  = 10
K_HH   = 2
K_ITER = 5
ALPHA_REFLECT = 0.5
BATCH  = 64

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"diag_step967_path_sparsity_seed{SEED}__{SLOT}.json"

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model() -> nn.Module:
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=25, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )

    class CanonicalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = resonant

        def forward(self, x):
            return self.m(x)

        def forward_with_Z(self, x):
            """Returns (logits, Z_final) for diagnostic use."""
            base_m = self.m.base
            Z = base_m._seed(x)
            theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
            conn_hh   = base_m.conn_hh
            W_h = self.m.W_pos[:N]
            dw  = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
            Z_ref = torch.zeros_like(Z)
            for _ in range(K_ITER):
                Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
                Z_nb  = Z_fwd[:, conn_hh, :]
                c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
                Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
                Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
                Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
            return base_m._readout(Z), Z

    return CanonicalModel()


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    import torchvision

    if args.dataset == "imagenette":
        # Try 160 then 320 resolution
        data_path = ROOT / args.data_root / "imagenette2-160"
        if not data_path.exists():
            data_path = ROOT / args.data_root / "imagenette2-320"
        if not data_path.exists():
            print(f"ERROR: imagenette2-160 or imagenette2-320 not found under {ROOT / args.data_root}"); sys.exit(1)
        tfm = T.Compose([
            T.Resize(160), T.CenterCrop(128),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tr_ds = ImageFolder(data_path / "train", transform=tfm)
        va_ds = ImageFolder(data_path / "val",   transform=tfm)
    else:
        data_path = ROOT / args.data_root / "cifar10"
        tfm = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
        tr_ds = torchvision.datasets.CIFAR10(data_path, train=True,  transform=tfm, download=True)
        va_ds = torchvision.datasets.CIFAR10(data_path, train=False, transform=tfm, download=True)

    tr = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
    va = torch.utils.data.DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=2)
    return tr, va


# ── VGG feature extractor (same as canonical pipeline) ───────────────────────
def get_vgg_extractor():
    import torchvision.models as models
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg.eval()
    extractor = nn.Sequential(vgg.features, vgg.avgpool,
                              nn.Flatten(), vgg.classifier[:1])
    for p in extractor.parameters(): p.requires_grad_(False)
    # VGG adaptive_avg_pool2d on MPS errors when input not divisible → run on CPU
    _dev = torch.device("cpu") if DEVICE.type == "mps" else DEVICE
    return extractor.to(_dev)


# ── Training ──────────────────────────────────────────────────────────────────
def train_and_save(model, tr, va, extractor, epochs, save_path):
    ext_dev = next(extractor.parameters()).device
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best_acc = 0.0
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            y = y.to(DEVICE)
            with torch.no_grad():
                x = extractor(x.to(ext_dev)).to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in va:
                y = y.to(DEVICE)
                x = extractor(x.to(ext_dev)).to(DEVICE)
                correct += (model(x).argmax(1) == y).sum().item()
                total   += y.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save({"model_state": model.state_dict(), "epoch": ep+1, "acc": acc},
                       save_path)
        print(f"  e{ep+1:3d}  top1={acc:.4f}", flush=True)
    print(f"  Checkpoint saved: {save_path}  best={best_acc:.4f}")
    return save_path


# ── Jaccard helpers ──────────────────────────────────────────────────────────
def pairwise_jaccard(masks: list[torch.Tensor]) -> float:
    """Mean pairwise Jaccard over a list of boolean tensors."""
    if len(masks) < 2:
        return 0.0
    scores = []
    masks_stack = torch.stack(masks).float()  # [M, N]
    n = len(masks)
    for i in range(min(n, 50)):   # cap at 50 samples per class to keep fast
        a = masks_stack[i]
        inter = (a.unsqueeze(0) * masks_stack[i+1:]).sum(dim=1)
        union = ((a.unsqueeze(0) + masks_stack[i+1:]) > 0).float().sum(dim=1)
        j = (inter / union.clamp(min=1)).mean().item()
        scores.append(j)
    return float(np.mean(scores)) if scores else 0.0


# ── Main diagnostic ──────────────────────────────────────────────────────────
def run_diagnostic(model, va, extractor):
    model.eval()
    ext_dev = next(extractor.parameters()).device
    masks_by_class = {c: [] for c in range(N_OUT)}
    correct = total = 0

    print("  Collecting activation masks...")
    with torch.no_grad():
        for x, y in va:
            y = y.to(DEVICE)
            x = extractor(x.to(ext_dev)).to(DEVICE)
            logits, Z = model.forward_with_Z(x)

            # Active mask: norm > theta (use mean theta)
            theta = model.m.theta.abs().mean().item()
            active = (Z.norm(dim=-1) > theta)   # [B, N]

            correct += (logits.argmax(1) == y).sum().item()
            total   += y.size(0)

            for b in range(y.size(0)):
                c = y[b].item()
                masks_by_class[c].append(active[b].cpu())

    val_acc = correct / total
    print(f"  Val acc: {val_acc:.4f}")

    # Per-class activation fraction
    act_frac = {}
    for c in range(N_OUT):
        if masks_by_class[c]:
            act_frac[c] = torch.stack(masks_by_class[c]).float().mean().item()

    # Intra-class Jaccard
    intra = {}
    for c in range(N_OUT):
        intra[c] = pairwise_jaccard(masks_by_class[c])

    # Inter-class Jaccard (sample 3 pairs per class)
    inter_scores = []
    classes = list(range(N_OUT))
    for c1 in classes:
        for c2 in classes:
            if c1 >= c2: continue
            m1 = masks_by_class[c1][:10]
            m2 = masks_by_class[c2][:10]
            if m1 and m2:
                combined = m1 + m2
                # cross-Jaccard: compare m1 samples with m2 samples
                a_stack = torch.stack(m1).float()
                b_stack = torch.stack(m2).float()
                for a in a_stack:
                    inter = (a.unsqueeze(0) * b_stack).sum(dim=1)
                    union = ((a.unsqueeze(0) + b_stack) > 0).float().sum(dim=1)
                    inter_scores.append((inter / union.clamp(min=1)).mean().item())

    mean_inter = float(np.mean(inter_scores)) if inter_scores else 0.0

    # Per-class separation
    separation = {c: intra[c] - mean_inter for c in range(N_OUT)}
    mean_sep = float(np.mean(list(separation.values())))

    # Class-selective nodes: nodes where P(active|class=c) >> P(active|other)
    # Compute per-node, per-class activation rate
    node_rates = {}
    all_masks = []
    all_labels = []
    for c in range(N_OUT):
        for m in masks_by_class[c][:20]:
            all_masks.append(m)
            all_labels.append(c)

    if all_masks:
        M = torch.stack(all_masks).float()    # [S, N]
        L = torch.tensor(all_labels)          # [S]
        class_rates = torch.zeros(N_OUT, N)
        class_counts = torch.zeros(N_OUT)
        for c in range(N_OUT):
            idx = (L == c).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                class_rates[c] = M[idx].mean(0)
                class_counts[c] = len(idx)

        overall_rate = M.mean(0)  # [N]
        # Selectivity = max_c(rate[c]) / (overall_rate + eps)
        selectivity = class_rates.max(0).values / (overall_rate + 1e-6)
        top_selective = selectivity.topk(20).indices.tolist()
    else:
        top_selective = []

    return {
        "val_acc":          round(val_acc, 4),
        "mean_act_frac":    round(float(np.mean(list(act_frac.values()))), 4),
        "act_frac_by_class": {str(c): round(v, 4) for c, v in act_frac.items()},
        "intra_jaccard_by_class": {str(c): round(v, 4) for c, v in intra.items()},
        "mean_intra_jaccard": round(float(np.mean(list(intra.values()))), 4),
        "mean_inter_jaccard": round(mean_inter, 4),
        "mean_separation":   round(mean_sep, 4),
        "separation_by_class": {str(c): round(v, 4) for c, v in separation.items()},
        "top20_selective_nodes": top_selective,
        "verdict": (
            "PATHWAYS_EXIST (sep>0.10) — step966 reinforces existing structure"
            if mean_sep > 0.10 else
            "PARTIAL (0.02<sep<0.10) — weak class specificity"
            if mean_sep > 0.02 else
            "CLASS_AGNOSTIC (sep≤0.02) — step966 builds from scratch"
        ),
    }


def main():
    print(f"\n{'='*70}")
    print(f"step967 — Path sparsity diagnostic")
    print(f"  device={DEVICE}  dataset={args.dataset}  seed={SEED}")
    print(f"{'='*70}\n")

    model = build_model().to(DEVICE)
    extractor = get_vgg_extractor()

    tr, va = load_data()

    ckpt_path = args.checkpoint
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        if ckpt_path:
            print(f"  Checkpoint not found: {ckpt_path}")
        print(f"  Training {args.epochs}ep model first...")
        save_path = ROOT / "checkpoints" / f"step967_train_seed{SEED}.pt"
        save_path.parent.mkdir(exist_ok=True)
        train_and_save(model, tr, va, extractor, args.epochs, str(save_path))
        ckpt = torch.load(str(save_path), map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])

    print("\n  Running path sparsity diagnostic...")
    metrics = run_diagnostic(model, va, extractor)

    print(f"\n{'='*70}")
    print(f"STEP 967 RESULTS — Path Sparsity Diagnostic")
    print(f"  val_acc={metrics['val_acc']:.4f}")
    print(f"  mean_act_frac={metrics['mean_act_frac']:.4f}  "
          f"(fraction of {N} nodes active per input)")
    print(f"  mean_intra_jaccard={metrics['mean_intra_jaccard']:.4f}  "
          f"(same-class activation overlap)")
    print(f"  mean_inter_jaccard={metrics['mean_inter_jaccard']:.4f}  "
          f"(cross-class activation overlap)")
    print(f"  mean_separation={metrics['mean_separation']:+.4f}  "
          f"(intra − inter)")
    print(f"\n  VERDICT: {metrics['verdict']}")
    print(f"\n  Per-class separation:")
    for c, sep in metrics["separation_by_class"].items():
        intra_v = metrics["intra_jaccard_by_class"][c]
        print(f"    class {c}: intra={intra_v:.3f}  sep={sep:+.3f}")
    print(f"\n-> {OUT_PATH}")
    print(f"{'='*70}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
