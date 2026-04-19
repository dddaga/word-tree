"""CNN distillation step001 — T0 scout (20ep, 50% data).

Distills VGG16's conv+pool stage into EfficientVGG (~420K params vs VGG16 conv 14.7M).
Teacher features come from pre-extracted store.h5 (aligned with ImageFolder order).

Loss: 3-term distillation
  feat : cosine similarity between student [B,25088] and teacher [B,25088] features
  dkd  : Decoupled KD (CVPR 2022) — splits KL into TCKD (target-class binary KD)
         + NCKD (non-target inter-class KD) with independent weights α, β.
         Applied after logit z-score standardization (CVPR 2024): removes scale bias
         from student logits before temperature softmax.
  ce   : cross-entropy on hard labels

DKD rationale: standard KL treats target and non-target knowledge equally. DKD
allows β > α to up-weight inter-class structure knowledge (non-target) — the
component most useful for downstream feature matching.

Logit standardization: z-score normalize logits before softmax eliminates the
"logit amplitude" confound where student/teacher differ in raw output scale.
Equivalent to finding an optimal per-sample temperature (CVPR 2024 claim).

Configs
-------
  Ref      : Full EfficientVGG (side branch + CReLU Block3) + DKD + logit-std
  A_no_side: Side branch disabled → isolates channel attention contribution
  B_vanilla : No side branch, no CReLU → plain ConvNeXt baseline (floor)
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

from scripts.cnn_distiller.model_efficient_vgg import EfficientVGG


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data_img", default="data/imagenette2-320")
parser.add_argument("--data_h5",  default="data/store.h5")
parser.add_argument("--configs", default="Ref,A_no_side,B_vanilla")
parser.add_argument("--batch",   type=int, default=32)
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
OUT_PATH = ROOT / "results" / f"cnn_step001_t0_seed{SEED}__{SLOT}.json"

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
# Joint dataset: raw image + teacher features from store.h5
# ---------------------------------------------------------------------------

class StudentDataset(Dataset):
    """Returns (image, teacher_feat, soft_label, label) for CNN distillation.

    Images come from ImageFolder (same sorted order as store.h5 extraction).
    Teacher features + soft labels come from pre-extracted store.h5.
    """

    def __init__(self, img_root: str, h5_path: str, split: str,
                 transform=None) -> None:
        split_dir = Path(img_root) / split
        self._folder = ImageFolder(str(split_dir), transform=transform)
        # Build canonical label remap: ImageFolder alpha-idx → canonical 0-9
        self._remap = {
            v: IMAGENETTE_FOLDER_TO_IDX[k]
            for k, v in self._folder.class_to_idx.items()
        }
        with h5py.File(h5_path, "r") as f:
            self.teacher_feat = torch.from_numpy(f[f"{split}/features"][:])
            self.soft_labels  = torch.from_numpy(f[f"{split}/soft_labels"][:])

        assert len(self._folder) == len(self.teacher_feat), (
            f"Image/H5 count mismatch: {len(self._folder)} vs {len(self.teacher_feat)}"
        )

    def __len__(self) -> int:
        return len(self._folder)

    def __getitem__(self, idx: int):
        image, folder_label = self._folder[idx]
        label = self._remap[folder_label]
        return image, self.teacher_feat[idx], self.soft_labels[idx], label


# ---------------------------------------------------------------------------
# DKD + logit standardization (CVPR 2022 + CVPR 2024)
# ---------------------------------------------------------------------------

def _std_logits(logits: torch.Tensor, T: float = 4.0) -> torch.Tensor:
    """Z-score standardize logits then apply temperature."""
    mu  = logits.mean(-1, keepdim=True)
    sig = logits.std(-1, keepdim=True).clamp(min=1e-6)
    return (logits - mu) / sig / T


def dkd_loss(
    logits: torch.Tensor,
    soft_labels: torch.Tensor,   # teacher probs [B, C], T=1 softmax
    labels: torch.Tensor,         # hard labels [B]
    T: float = 4.0,
    alpha: float = 1.0,
    beta: float = 2.0,
) -> torch.Tensor:
    """Decoupled Knowledge Distillation with logit z-score standardization.

    Decomposes KL(teacher || student) into:
      TCKD: target-class binary KD (transfers per-sample difficulty)
      NCKD: non-target inter-class KD (transfers class structure)
    Allows independent weighting via alpha, beta.

    Teacher probs are temperature-sharpened from T=1 via log-rescale.
    Student logits are z-score standardized before temperature softmax.
    """
    B, C = logits.shape

    # Student: standardize then apply temperature
    p_s = F.softmax(_std_logits(logits, T), dim=-1)             # [B, C]

    # Teacher: recover logits via log, standardize, apply temperature
    t_log = torch.log(soft_labels.clamp(min=1e-8))
    t_mu  = t_log.mean(-1, keepdim=True)
    t_sig = t_log.std(-1, keepdim=True).clamp(min=1e-6)
    p_t   = F.softmax((t_log - t_mu) / t_sig / T, dim=-1)      # [B, C]

    target_mask = F.one_hot(labels, num_classes=C).float()       # [B, C]

    # TCKD: binary KL on target class confidence
    p_s_y  = (p_s * target_mask).sum(-1).clamp(1e-8, 1 - 1e-8)  # [B]
    p_t_y  = (p_t * target_mask).sum(-1).clamp(1e-8, 1 - 1e-8)  # [B]
    tckd = -(p_t_y * p_s_y.log() + (1 - p_t_y) * (1 - p_s_y).log()).mean()

    # NCKD: KL on renormalized non-target distribution
    p_s_nt = (p_s * (1 - target_mask)) / (1 - p_s_y.unsqueeze(-1) + 1e-8)
    p_t_nt = (p_t * (1 - target_mask)) / (1 - p_t_y.unsqueeze(-1) + 1e-8)
    nckd   = F.kl_div((p_s_nt + 1e-8).log(), p_t_nt, reduction="batchmean")

    return T * T * (alpha * tckd + beta * nckd)


def distil_loss(logits, student_feat, teacher_feat, soft_labels, labels,
                w_feat=0.50, w_dkd=0.30, w_ce=0.20,
                T=4.0, alpha=1.0, beta=2.0):
    feat_loss = (1.0 - F.cosine_similarity(student_feat, teacher_feat, dim=1)).mean()
    dkd       = dkd_loss(logits, soft_labels, labels, T=T, alpha=alpha, beta=beta)
    ce_loss   = F.cross_entropy(logits, labels)
    return w_feat * feat_loss + w_dkd * dkd + w_ce * ce_loss


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
    if cfg == "Ref":
        return EfficientVGG(use_side_branch=True,  use_crelu_block3=True)
    elif cfg == "A_no_side":
        return EfficientVGG(use_side_branch=False, use_crelu_block3=True)
    elif cfg == "B_vanilla":
        return EfficientVGG(use_side_branch=False, use_crelu_block3=False)
    else:
        raise ValueError(f"Unknown config: {cfg}")


CONFIGS = {
    "Ref":       "Full EfficientVGG (side branch + CReLU Block3)",
    "A_no_side": "No side-branch channel attention (CReLU Block3 kept)",
    "B_vanilla": "No side branch, no CReLU — plain ConvNeXt baseline",
}


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

    train_ds_full = StudentDataset(str(img_root), str(h5_path), "train", TRAIN_TRANSFORM)
    val_ds        = StudentDataset(str(img_root), str(h5_path), "val",   VAL_TRANSFORM)

    # 50% data subset
    n_full = len(train_ds_full)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    train_ds = Subset(train_ds_full, sub_idx.tolist())

    pin = (DEVICE.type == "cuda")
    nw  = 2 if DEVICE.type == "cuda" else 0

    tr = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                    num_workers=nw, pin_memory=pin)
    va = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                    num_workers=nw, pin_memory=pin)

    print(f"\n{'='*70}")
    print(f"CNN step001 T0  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  train={len(train_ds)}  val={len(val_ds)}  batch={BATCH}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}
    ref_acc = None

    for key in keys:
        model = make_model(key).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {CONFIGS[key]}  params={n_p:,}")

        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

        best = 0.0; best_ep = 0
        t0 = time.time()

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
        delta = best - (ref_acc or best)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "desc": CONFIGS[key], "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"CNN STEP001 T0 SUMMARY")
    print(f"{'='*70}")
    print(f"  {'config':<14} {'params':>8} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        print(f"  {k:<14} {r['n_params']:>8,} {r['best']:>7.4f} {r['delta_vs_ref']*100:>+9.2f}pp")

    print(f"\n  Teacher (VGG16 conv): ~14.7M params, 97.7% Imagenette")
    if "Ref" in results:
        n_ref = results["Ref"]["n_params"]
        print(f"  Student (Ref):         {n_ref:,} params ({100*n_ref/14_700_000:.2f}% of VGG16 conv)")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
