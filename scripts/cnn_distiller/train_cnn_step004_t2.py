"""CNN distillation step004 — T2 Pareto validation (150ep, 100% data).

Promotes T1 paper-critical configs for final Pareto table validation.

T1 results (step003, 75ep, 50% data — mini_mps):
  Ref       419K  183M  73.17%  baseline, always advances
  F_wide    825K  560M  74.42%  (+1.25pp vs Ref — exceeds +0.5pp threshold)
  D_small_s 149K   57M  70.96%  (−2.22pp vs Ref — misses accuracy threshold,
                                  BUT advances for paper efficiency Pareto table:
                                  3.2× fewer MACs, 2.8× fewer params; honest
                                  tradeoff documented as accuracy-vs-efficiency.)

H2 CONFIRMED: F_wide maintains +1.25pp advantage at T1 (was +2.39pp at T0).
H3 CONFIRMED: D_small_s gap narrowed from 9.4pp (T0) → 2.22pp (T1). Strong
              convergence improvement; T2 will confirm whether gap stabilises.
H1, H4: pending (B_tiny_k7 + C_small still running on mini_cpu at T1).

T2 advance decision:
  - Ref: always (baseline)
  - F_wide: STRONG accuracy advance (+1.25pp), advance to T2
  - D_small_s: CONDITIONAL advance for Pareto story. Does NOT meet +0.5pp
    accuracy criterion. Advances because: the paper's efficiency Pareto table
    requires a sub-200M-MACs CNN point; D_small_s at 57.5M MACs (3.2× fewer)
    at only −2.22pp is the best efficiency-frontier candidate. Tag as
    "efficiency-Pareto advance" in paper, not accuracy advance.

NOTE — runtime estimate on mini_mps (T1 timings as proxy, ×4 for T2):
  Ref:      ~2,015s × 4 ≈ 2.2h
  F_wide:   ~3,226s × 4 ≈ 3.6h
  D_small_s: ~1,806s × 4 ≈ 2.0h
  Total sequential: ~7.8h on one GPU slot.
  Recommended split:
    --configs Ref,D_small_s   → mini_mps  (start immediately, ~4.2h)
    --configs F_wide          → mini_cpu  (when free, ~3.6h)
  Or run all three sequentially if slot is free for a full day.

CNN experiments are mini-only (raw imagenette2-320 images required).
Remote machines (studio, 5060ti) have only .h5 files.

Usage:
  python scripts/cnn_distiller/train_cnn_step004_t2.py \\
      --configs Ref,F_wide,D_small_s --epochs 150

Output: results/cnn_step004_t2_seed{SEED}__{SLOT}.json
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from scripts.cnn_distiller.model_efficient_vgg import EfficientVGG, count_macs


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--device",   default="auto")
parser.add_argument("--epochs",   type=int, default=150)
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data_img", default="data/imagenette2-320")
parser.add_argument("--data_h5",  default="data/store.h5")
parser.add_argument("--configs",  default="Ref,F_wide,D_small_s")
parser.add_argument("--batch",    type=int, default=32)
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = args.batch
SEED   = args.seed
SLOT   = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"cnn_step004_t2_seed{SEED}__{SLOT}.json"

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
# Configs: (channels, dw_kernel, expansion, use_side_branch, use_crelu_block3, desc)
# ---------------------------------------------------------------------------

CONFIGS: dict[str, tuple] = {
    "D_small_s": ((32, 64,  128), 7, 2, True,  True, "small+side: 32/64/128 k=7"),
    "Ref":       ((64, 128, 256), 7, 2, True,  True, "ref: 64/128/256 k=7"),
    "F_wide":    ((128, 256, 256), 7, 2, True,  True, "wide: 128/256/256 k=7"),
}

# T1 results for Δ display — from cnn_step003_t1_seed42__mini_mps.json
T1_RESULTS = {
    "Ref":       0.7317,
    "F_wide":    0.7442,
    "D_small_s": 0.7096,
}


# ---------------------------------------------------------------------------
# Dataset — T2 uses FULL training set (no subset)
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
# Loss (same as step003)
# ---------------------------------------------------------------------------

def _std_logits(logits: torch.Tensor, T: float) -> torch.Tensor:
    mu  = logits.mean(-1, keepdim=True)
    sig = logits.std(-1,  keepdim=True).clamp(min=1e-6)
    return (logits - mu) / sig / T


def dkd_loss(logits, soft_labels, labels, T=4.0, alpha=1.0, beta=2.0):
    B, C = logits.shape
    p_s   = F.softmax(_std_logits(logits, T), dim=-1)
    t_log = torch.log(soft_labels.clamp(1e-8))
    t_mu  = t_log.mean(-1, keepdim=True)
    t_sig = t_log.std(-1,  keepdim=True).clamp(1e-6)
    p_t   = F.softmax((t_log - t_mu) / t_sig / T, dim=-1)

    mask  = F.one_hot(labels, C).float()
    p_s_y = (p_s * mask).sum(-1).clamp(1e-8, 1 - 1e-8)
    p_t_y = (p_t * mask).sum(-1).clamp(1e-8, 1 - 1e-8)

    tckd  = -(p_t_y * p_s_y.log() + (1 - p_t_y) * (1 - p_s_y).log()).mean()
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


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for image, _, _, label in loader:
        image, label = image.to(DEVICE), label.to(DEVICE)
        logits, _ = model(image)
        correct += (logits.argmax(1) == label).sum().item()
        total   += label.size(0)
    return correct / total


def make_model(cfg: str) -> nn.Module:
    torch.manual_seed(SEED)
    channels, dw_k, exp, side, crelu, _ = CONFIGS[cfg]
    return EfficientVGG(
        channels=channels, dw_kernel=dw_k, expansion=exp,
        use_side_branch=side, use_crelu_block3=crelu,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    img_root = ROOT / args.data_img
    h5_path  = ROOT / args.data_h5

    if not img_root.exists():
        print(f"ERROR: {img_root} not found"); sys.exit(1)
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found"); sys.exit(1)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # T2: full training set (no subset)
    train_ds = StudentDataset(str(img_root), str(h5_path), "train", TRAIN_TRANSFORM)
    val_ds   = StudentDataset(str(img_root), str(h5_path), "val",   VAL_TRANSFORM)

    pin = (DEVICE.type == "cuda")
    nw  = 2 if DEVICE.type == "cuda" else 0

    tr = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                    num_workers=nw, pin_memory=pin)
    va = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                    num_workers=nw, pin_memory=pin)

    keys = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    assert keys, f"No valid configs in: {args.configs}. Valid: {list(CONFIGS)}"

    # Print header
    print(f"\n{'='*72}")
    print(f"CNN step004 T2  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  train={len(train_ds)} (100% — T2 full dataset)  val={len(val_ds)}  batch={BATCH}")
    print(f"{'='*72}\n")
    print(f"  {'config':<12} {'params':>9}  {'MACs(M)':>8}  {'%VGG16conv':>10}  "
          f"{'T1_acc':>8}  desc")
    print(f"  {'-'*80}")
    for k in keys:
        m = make_model(k)
        np_ = sum(p.numel() for p in m.parameters())
        macs = count_macs(m) / 1e6
        _, _, _, _, _, desc = CONFIGS[k]
        t1 = T1_RESULTS.get(k, float("nan"))
        print(f"  {k:<12} {np_:>9,}  {macs:>8.1f}  {macs/14700*100:>9.3f}%  "
              f"{t1:>8.4f}  {desc}")
        del m
    print(f"  {'-'*80}")
    print(f"  {'VGG16-conv':<12} {'14,700,000':>9}  {'14700.0':>8}  {'100.000%':>10}  "
          f"{'N/A':>8}  (teacher reference)")

    results = {}
    ref_acc = None

    for key in keys:
        model = make_model(key)
        n_p   = sum(p.numel() for p in model.parameters())
        macs  = count_macs(model) / 1e6
        model = model.to(DEVICE)
        _, _, _, _, _, desc = CONFIGS[key]
        t1_acc = T1_RESULTS.get(key)
        t1_str = f"  T1={t1_acc:.4f}" if t1_acc else "  T1=pending"
        print(f"\n{'─'*60}")
        print(f"{key}: {desc}{t1_str}")
        print(f"  params={n_p:,}  MACs={macs:.1f}M  ({macs/14700*100:.3f}% of VGG16-conv)")

        opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

        best = 0.0; best_ep = 0
        t_start = time.time()

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
            if (ep + 1) % 10 == 0:
                print(f"  ep{ep+1:3d}  val={val_acc:.4f}  best={best:.4f}", flush=True)

        elapsed = time.time() - t_start
        if key == "Ref":
            ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else best)
        t1_delta = best - (t1_acc or best)
        print(f"  -> best={best:.4f} @ep{best_ep}  "
              f"Δ_vs_Ref={delta*100:+.2f}pp  "
              f"T1→T2={t1_delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "desc": desc, "n_params": n_p, "macs_M": round(macs, 1),
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4),
            "t1_acc": t1_acc,
            "t1_to_t2_delta": round(best - (t1_acc or best), 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*72}")
    print(f"CNN STEP004 T2 SUMMARY — (150ep, 100% data)")
    print(f"{'='*72}")
    print(f"  {'config':<12} {'params':>9}  {'MACs(M)':>8}  "
          f"{'T1_acc':>7}  {'T2_acc':>7}  {'Δ_Ref':>10}  {'T1→T2':>8}  efficiency")
    print(f"  {'-'*90}")
    for k, r in results.items():
        t1  = r["t1_acc"] or 0.0
        eff = r["best"] / r["macs_M"] if r["macs_M"] else 0
        print(f"  {k:<12} {r['n_params']:>9,}  {r['macs_M']:>8.1f}  "
              f"{t1:>7.4f}  {r['best']:>7.4f}  "
              f"{r['delta_vs_ref']*100:>+9.2f}pp  "
              f"{r['t1_to_t2_delta']*100:>+6.2f}pp  "
              f"acc/MACs={eff:.3f} %/M")

    if "Ref" in results and "D_small_s" in results:
        r_ref = results["Ref"]
        r_dsm = results["D_small_s"]
        mac_ratio = r_ref["macs_M"] / r_dsm["macs_M"]
        acc_gap   = (r_ref["best"] - r_dsm["best"]) * 100
        param_ratio = r_ref["n_params"] / r_dsm["n_params"]
        print(f"\n  Paper Pareto (D_small_s vs Ref):")
        print(f"    Accuracy gap:  {acc_gap:.2f}pp")
        print(f"    MACs ratio:    {mac_ratio:.1f}× fewer")
        print(f"    Params ratio:  {param_ratio:.1f}× fewer")
        print(f"    Advance basis: efficiency-Pareto (not accuracy threshold)")

    if "Ref" in results and "F_wide" in results:
        r_ref  = results["Ref"]
        r_fw   = results["F_wide"]
        acc_gain = (r_fw["best"] - r_ref["best"]) * 100
        mac_cost = r_fw["macs_M"] / r_ref["macs_M"]
        print(f"\n  Paper Pareto (F_wide vs Ref):")
        print(f"    Accuracy gain: +{acc_gain:.2f}pp")
        print(f"    MACs cost:     {mac_cost:.1f}× more")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
