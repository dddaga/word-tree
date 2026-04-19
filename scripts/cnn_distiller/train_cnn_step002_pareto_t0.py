"""CNN distillation step002 — Pareto T0 sweep (20ep, 50% data).

Sweeps channel widths, kernel sizes, expansion factors, and optional
side-branch + CReLU across 8 configs to map accuracy vs FLOPs/params.

All configs use DKD + logit standardization (CVPR 2022 + CVPR 2024).
All configs use MaxPool throughout (no AvgPool).

Pareto configs:
  G_ultra   : channels=(8,16,32),  k=3, exp=2 → ~28K params, ~70M MACs
  A_tiny    : channels=(16,32,64), k=3, exp=2 → ~55K params, ~190M MACs
  B_tiny_k7 : channels=(16,32,64), k=7, exp=2 → ~57K params, ~280M MACs
  C_small   : channels=(32,64,128),k=7, exp=2 → ~150K params, ~530M MACs
  D_small_s : channels=(32,64,128),k=7, exp=2, side=True → ~175K params
  Ref       : channels=(64,128,256),k=7,exp=2, side=True → ~419K params
  E_expand4 : channels=(64,128,256),k=7,exp=4, side=True → ~830K params
  F_wide    : channels=(128,256,256),k=7,exp=2, side=True→ ~1.3M params

Launch suggestion:
  mini_cpu  : G_ultra,A_tiny,B_tiny_k7,C_small   (small models, CPU OK)
  mini_mps  : D_small_s,Ref,E_expand4,F_wide      (larger, need MPS)
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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from scripts.cnn_distiller.model_efficient_vgg import EfficientVGG, count_macs


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--device",   default="auto")
parser.add_argument("--epochs",   type=int, default=20)
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data_img", default="data/imagenette2-320")
parser.add_argument("--data_h5",  default="data/store.h5")
parser.add_argument("--configs",  default="G_ultra,A_tiny,B_tiny_k7,C_small,D_small_s,Ref,E_expand4,F_wide")
parser.add_argument("--batch",    type=int, default=32)
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps")  if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = args.batch
SEED   = args.seed
SLOT   = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"cnn_step002_pareto_t0_seed{SEED}__{SLOT}.json"

IMAGENETTE_FOLDER_TO_IDX = {
    "n01440764": 0, "n02102040": 1, "n02979186": 2, "n03000684": 3, "n03028079": 4,
    "n03394916": 5, "n03417042": 6, "n03425413": 7, "n03445777": 8, "n03888257": 9,
}

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Pareto config table
# (channels, dw_kernel, expansion, side_branch, crelu_block3, description)
# ---------------------------------------------------------------------------

CONFIGS: dict[str, tuple] = {
    "G_ultra":   ((8,  16,  32),  3, 2, False, False, "ultra-tiny: 8/16/32 k=3"),
    "A_tiny":    ((16, 32,  64),  3, 2, False, True,  "tiny: 16/32/64 k=3"),
    "B_tiny_k7": ((16, 32,  64),  7, 2, False, True,  "tiny: 16/32/64 k=7 (larger RF)"),
    "C_small":   ((32, 64,  128), 7, 2, False, True,  "small: 32/64/128 k=7"),
    "D_small_s": ((32, 64,  128), 7, 2, True,  True,  "small + side branch: 32/64/128"),
    "Ref":       ((64, 128, 256), 7, 2, True,  True,  "ref: 64/128/256 k=7 (step001 Ref)"),
    "E_expand4": ((64, 128, 256), 7, 4, True,  True,  "ref channels + expansion=4"),
    "F_wide":    ((128,256, 256), 7, 2, True,  True,  "wide early: 128/256/256 k=7"),
}


# ---------------------------------------------------------------------------
# Joint dataset (image + teacher features)
# ---------------------------------------------------------------------------

class StudentDataset(Dataset):
    def __init__(self, img_root: str, h5_path: str, split: str, transform=None):
        split_dir = Path(img_root) / split
        self._folder = ImageFolder(str(split_dir), transform=transform)
        self._remap  = {
            v: IMAGENETTE_FOLDER_TO_IDX[k]
            for k, v in self._folder.class_to_idx.items()
        }
        with h5py.File(h5_path, "r") as f:
            self.teacher_feat = torch.from_numpy(f[f"{split}/features"][:])
            self.soft_labels  = torch.from_numpy(f[f"{split}/soft_labels"][:])
        assert len(self._folder) == len(self.teacher_feat)

    def __len__(self):
        return len(self._folder)

    def __getitem__(self, idx):
        image, folder_label = self._folder[idx]
        label = self._remap[folder_label]
        return image, self.teacher_feat[idx], self.soft_labels[idx], label


# ---------------------------------------------------------------------------
# DKD + logit standardization loss (CVPR 2022 + CVPR 2024)
# ---------------------------------------------------------------------------

def _std_logits(logits: torch.Tensor, T: float) -> torch.Tensor:
    mu  = logits.mean(-1, keepdim=True)
    sig = logits.std(-1,  keepdim=True).clamp(min=1e-6)
    return (logits - mu) / sig / T


def dkd_loss(logits, soft_labels, labels, T=4.0, alpha=1.0, beta=2.0):
    B, C = logits.shape
    p_s = F.softmax(_std_logits(logits, T), dim=-1)
    t_log = torch.log(soft_labels.clamp(1e-8))
    t_mu  = t_log.mean(-1, keepdim=True)
    t_sig = t_log.std(-1,  keepdim=True).clamp(1e-6)
    p_t   = F.softmax((t_log - t_mu) / t_sig / T, dim=-1)

    mask  = F.one_hot(labels, C).float()
    p_s_y = (p_s * mask).sum(-1).clamp(1e-8, 1 - 1e-8)
    p_t_y = (p_t * mask).sum(-1).clamp(1e-8, 1 - 1e-8)

    tckd = -(p_t_y * p_s_y.log() + (1 - p_t_y) * (1 - p_s_y).log()).mean()
    p_s_nt = (p_s * (1 - mask)) / (1 - p_s_y.unsqueeze(-1) + 1e-8)
    p_t_nt = (p_t * (1 - mask)) / (1 - p_t_y.unsqueeze(-1) + 1e-8)
    nckd   = F.kl_div((p_s_nt + 1e-8).log(), p_t_nt, reduction="batchmean")
    return T * T * (alpha * tckd + beta * nckd)


def distil_loss(logits, student_feat, teacher_feat, soft_labels, labels,
                w_feat=0.50, w_dkd=0.30, w_ce=0.20, T=4.0, alpha=1.0, beta=2.0):
    feat = (1.0 - F.cosine_similarity(student_feat, teacher_feat, dim=1)).mean()
    dkd  = dkd_loss(logits, soft_labels, labels, T=T, alpha=alpha, beta=beta)
    ce   = F.cross_entropy(logits, labels)
    return w_feat * feat + w_dkd * dkd + w_ce * ce


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for image, _, _, label in loader:
        logits, _ = model(image.to(DEVICE))
        correct  += (logits.argmax(1) == label.to(DEVICE)).sum().item()
        total    += label.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    img_root = ROOT / args.data_img
    h5_path  = ROOT / args.data_h5
    for p in (img_root, h5_path):
        if not p.exists():
            print(f"ERROR: {p} not found"); sys.exit(1)

    torch.manual_seed(SEED); np.random.seed(SEED)

    train_ds_full = StudentDataset(str(img_root), str(h5_path), "train", TRAIN_TRANSFORM)
    val_ds        = StudentDataset(str(img_root), str(h5_path), "val",   VAL_TRANSFORM)

    n_full  = len(train_ds_full)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    train_ds = Subset(train_ds_full, sub_idx.tolist())

    pin = (DEVICE.type == "cuda")
    nw  = 2 if DEVICE.type == "cuda" else 0
    tr  = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=nw, pin_memory=pin)
    va  = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=nw, pin_memory=pin)

    print(f"\n{'='*72}")
    print(f"CNN step002 Pareto T0  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  train={len(train_ds)}  val={len(val_ds)}  batch={BATCH}")
    print(f"{'='*72}\n")
    print(f"  {'config':<14} {'params':>8} {'MACs(M)':>9} {'%VGG16conv':>11} {'desc'}")
    print(f"  {'-'*70}")

    VGG16_CONV_MACS = 14_700_000_000  # ~14.7B MACs (VGG16 conv stages total)
    VGG16_CONV_PAR  = 14_700_000

    keys     = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results  = {}
    ref_acc  = None

    # Pre-measure all models before training (avoids mixing train/eval state)
    cfg_info = {}
    for key in keys:
        ch, k, exp, side, crelu, desc = CONFIGS[key]
        torch.manual_seed(SEED)
        m   = EfficientVGG(channels=ch, dw_kernel=k, expansion=exp,
                            use_side_branch=side, use_crelu_block3=crelu)
        n_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
        n_m = count_macs(m)
        cfg_info[key] = (n_p, n_m, desc)
        print(f"  {key:<14} {n_p:>8,} {n_m/1e6:>9.1f} {100*n_m/VGG16_CONV_MACS:>10.3f}%  {desc}")

    print(f"  {'-'*70}")
    print(f"  {'VGG16-conv':14} {VGG16_CONV_PAR:>8,} {VGG16_CONV_MACS/1e6:>9.1f} {'100.000%':>11}  (teacher reference)")
    print()

    # Training loop
    for key in keys:
        ch, k, exp, side, crelu, desc = CONFIGS[key]
        n_p, n_m, _ = cfg_info[key]

        torch.manual_seed(SEED)
        model = EfficientVGG(channels=ch, dw_kernel=k, expansion=exp,
                             use_side_branch=side, use_crelu_block3=crelu).to(DEVICE)

        print(f"{'─'*60}")
        print(f"{key}: {desc}")
        print(f"  params={n_p:,}  MACs={n_m/1e6:.1f}M ({100*n_p/VGG16_CONV_PAR:.2f}% params, {100*n_m/VGG16_CONV_MACS:.3f}% FLOPs)")

        opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

        best = 0.0; best_ep = 0
        t0   = time.time()

        for ep in range(EPOCHS):
            model.train()
            for image, teacher_feat, soft_labels, label in tr:
                image        = image.to(DEVICE)
                teacher_feat = teacher_feat.to(DEVICE)
                soft_labels  = soft_labels.to(DEVICE)
                label        = label.to(DEVICE)

                opt.zero_grad()
                logits, student_feat = model(image)
                loss = distil_loss(logits, student_feat, teacher_feat, soft_labels, label)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            val_acc = evaluate(model, va)
            if val_acc > best:
                best = val_acc; best_ep = ep + 1
            if (ep + 1) % 5 == 0:
                print(f"  ep{ep+1:3d}  val={val_acc:.4f}  best={best:.4f}", flush=True)

        elapsed = time.time() - t0
        if key == "Ref": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else best)

        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "desc":         desc,
            "channels":     list(ch),
            "dw_kernel":    k,
            "expansion":    exp,
            "side_branch":  side,
            "crelu_block3": crelu,
            "n_params":     n_p,
            "n_macs":       n_m,
            "params_pct":   round(100 * n_p / VGG16_CONV_PAR, 3),
            "macs_pct":     round(100 * n_m / VGG16_CONV_MACS, 4),
            "best":         round(best, 4),
            "best_ep":      best_ep,
            "delta_vs_ref": round(delta, 4),
            "elapsed_s":    round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # Pareto summary table
    print(f"\n{'='*72}")
    print(f"CNN STEP002 PARETO SUMMARY — T0 (20ep, 50% data)")
    print(f"{'='*72}")
    print(f"  {'config':<14} {'params':>8} {'MACs(M)':>9} {'acc':>7} {'Δ_Ref':>8}  efficiency")
    print(f"  {'-'*70}")
    for k, r in sorted(results.items(), key=lambda x: x[1]["n_macs"]):
        eff = f"acc/MACs={r['best']*100/max(r['n_macs']/1e6,0.1):.3f} %/M"
        print(f"  {k:<14} {r['n_params']:>8,} {r['n_macs']/1e6:>9.1f} {r['best']:>7.4f} "
              f"{r['delta_vs_ref']*100:>+7.2f}pp  {eff}")

    # Find Pareto front (non-dominated in accuracy × MACs)
    print(f"\n  Pareto front (non-dominated):")
    sorted_by_mac = sorted(results.items(), key=lambda x: x[1]["n_macs"])
    best_so_far = -1.0
    for k, r in sorted_by_mac:
        if r["best"] >= best_so_far:
            best_so_far = r["best"]
            print(f"    ★ {k:<14} acc={r['best']:.4f}  MACs={r['n_macs']/1e6:.1f}M  params={r['n_params']:,}")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
