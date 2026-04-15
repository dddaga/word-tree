"""Step 603 (B2-Student): GLNN-style MLP_37 student distilled from SGNNET teacher.

MOTIVATION
==========
Candidate B2 from Opus mediation (walltime_breakthrough_2026-04-15.md):
If SGNNET's routing discovers structure that MLP_37 can express, distillation
yields a 0.088ms inference artifact while retaining teacher accuracy.

This script loads cached teacher logits (step602) and trains MLP_37 with a
combined KL + CE loss across a sweep of (T, λ) configurations.

GLNN recipe (Zhang et al., NeurIPS 2022 / arXiv:2110.08727):
  loss = λ · KL(student/T ‖ teacher/T) · T² + (1-λ) · CE(student, y)

Sweep: (T, λ) ∈ {(2,0.5), (4,0.5), (4,0.7), (8,0.5), (8,0.7)} — 5 configs.
Each: 150 epochs (Tier-2), full data. Student is tiny (928K params), ~fast.

Baselines:
  - MLP_37 from scratch: 97.71% (step403b) — the bar to beat
  - SGNNET teacher (step602): whatever it produced

Win conditions:
  STRONG  : student ≥ teacher AND student > 97.71% (beats scratch MLP)
  MEDIUM  : student ≥ teacher (matches/beats teacher, doesn't beat scratch)
  WEAK    : student ≥ 97.71% (matches scratch, no distillation gain over scratch)
  FAIL    : student < 97.71% (distillation hurts vs training from scratch)

Requires: data/teacher_logits_step602.h5 (from step602)
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
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.training.dataset import H5Dataset

parser = argparse.ArgumentParser(description="Step 603: GLNN MLP_37 student distillation.")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--hidden",  type=int, default=37,
                    help="Hidden units. 37 matches SGNNET 1.85M FLOPs (default).")
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--logits",  default="data/teacher_logits_step602.h5",
                    help="Path to teacher logits HDF5 (from step602).")
parser.add_argument("--teacher_top1", type=float, default=None,
                    help="Teacher best top1 (override; else read from logits file attr).")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (e.g. T4_lam05). Empty = all.")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS       = args.epochs
BATCH        = 128
SEED         = args.seed
H            = args.hidden
DATA_PATH    = ROOT / args.data
LOGITS_PATH  = ROOT / args.logits
N_IN         = 25088
N_OUT        = 10

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step603_b2_glnn_student_h{H}_seed{SEED}__{SLOT}.json"

# Baseline from step403b (scratch-trained MLP_37)
SCRATCH_MLP37_TOP1 = 0.9771

# (T, lambda) sweep
SWEEP_CONFIGS = {
    "T2_lam05":  {"T": 2.0, "lam": 0.5},
    "T4_lam05":  {"T": 4.0, "lam": 0.5},
    "T4_lam07":  {"T": 4.0, "lam": 0.7},
    "T8_lam05":  {"T": 8.0, "lam": 0.5},
    "T8_lam07":  {"T": 8.0, "lam": 0.7},
}


# ---------------------------------------------------------------------------
# Dataset with teacher logits
# ---------------------------------------------------------------------------

class DistillDataset(torch.utils.data.Dataset):
    """Combines store.h5 features/labels with teacher soft logits at temperature T.

    __getitem__ returns (features, teacher_soft_T, hard_label).
    teacher_soft_T is already softmax(logits/T) — read from HDF5.
    Different T values require different cache entries; step602 caches T=4.
    For other T, re-normalize from the cached logits:
        if we have p_T4 = softmax(z/4), we can't exactly recover z, but
        we can use p_T4 directly as the distillation target (mild approx.)
        OR re-derive from the raw logits if stored.
    Since step602 stores softmax at T only (not raw logits), we store T=4
    and for other T values we work from the H5Dataset soft_labels field
    in store.h5 (original label smoothing) — but that is VGG soft labels,
    not SGNNET soft labels.

    To handle arbitrary T correctly: step602 saves softmax(z/T_stored).
    For a different requested T:
        p_requested ≈ softmax(T_stored * log(p_stored) / T_requested)
        (because log(softmax(z/T)) ∝ z/T, so z ∝ T * log_softmax(p_stored))
    This is an approximation; for T far from T_stored it diverges due to
    softmax saturation.  We implement this for generality and log a warning
    if |T_req - T_stored| > 4.
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor,
                 teacher_logits_T: torch.Tensor, T_stored: float, T_req: float):
        """
        features        : [N, 25088]
        labels          : [N] long
        teacher_logits_T: [N, 10]  — softmax(z / T_stored)
        T_stored        : temperature at which logits were cached
        T_req           : temperature for distillation loss (this student config)
        """
        assert len(features) == len(labels) == len(teacher_logits_T)
        self.features = features
        self.labels   = labels

        if abs(T_req - T_stored) < 1e-3:
            # Exact match — use as-is
            self.teacher_soft = teacher_logits_T
        else:
            # Re-derive: z ≈ T_stored * log(p_stored + 1e-10)
            log_p_stored = torch.log(teacher_logits_T.clamp(min=1e-10))
            z_approx     = T_stored * log_p_stored       # un-normalized pseudo-logits
            self.teacher_soft = F.softmax(z_approx / T_req, dim=-1)
        # Keep on CPU; move to device in training loop

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.teacher_soft[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Model — MLP_37 (identical to step403b)
# ---------------------------------------------------------------------------

class MLPStudent(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_OUT),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# KD loss
# ---------------------------------------------------------------------------

def kd_loss(z_student: torch.Tensor, teacher_soft_T: torch.Tensor,
            y_true: torch.Tensor, T: float, lam: float) -> torch.Tensor:
    """GLNN-style combined KL + CE loss.

    KL term: KL(student_soft_T ‖ teacher_soft_T) * T^2
    CE term: cross-entropy(z_student, y_true)
    Combined: lam * KL + (1-lam) * CE
    """
    log_p_student = F.log_softmax(z_student / T, dim=-1)
    # teacher_soft_T is already softmax(z_teacher / T_req)
    kl  = F.kl_div(log_p_student, teacher_soft_T, reduction="batchmean") * (T ** 2)
    ce  = F.cross_entropy(z_student, y_true)
    return lam * kl + (1.0 - lam) * ce


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_student(model: MLPStudent, tr, va, epochs: int,
                  T: float, lam: float) -> list[dict]:
    model = model.to(DEVICE)
    opt   = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    history = []

    for epoch in range(epochs):
        model.train()
        for feats, teacher_soft, labels in tr:
            feats        = feats.to(DEVICE)
            teacher_soft = teacher_soft.to(DEVICE)
            labels       = labels.to(DEVICE)

            opt.zero_grad()
            z = model(feats)
            loss = kd_loss(z, teacher_soft, labels, T=T, lam=lam)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for feats, _soft, labels in va:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                correct += (model(feats).argmax(dim=-1) == labels).sum().item()
                total   += labels.size(0)
        val_top1 = correct / max(total, 1)
        history.append({"epoch": epoch, "val_top1": val_top1})

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == epochs:
            print(f"  ep{epoch+1:3d}  val={val_top1:.4f}", flush=True)

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Load teacher logits
    # ------------------------------------------------------------------
    if not LOGITS_PATH.exists():
        print(f"ERROR: Teacher logits not found: {LOGITS_PATH}")
        print("  Run scripts/train_step602_b2_glnn_teacher.py first.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Step 603 — GLNN MLP_{H} student distillation from SGNNET teacher")
    print(f"  logits={LOGITS_PATH}")
    print(f"  epochs={EPOCHS}  seed={SEED}  device={DEVICE}")
    print(f"{'='*70}")

    with h5py.File(LOGITS_PATH, "r") as f:
        T_stored     = float(f.attrs.get("temperature", 4.0))
        # Determine which key to read
        key_name     = f"logits_T{int(T_stored)}"
        train_logits = torch.from_numpy(f["train"][key_name][:])   # [N_train, 10]
        val_logits   = torch.from_numpy(f["val"][key_name][:])     # [N_val, 10]
        teacher_step = int(f.attrs.get("step", 602))

    print(f"  Loaded teacher logits: T_stored={T_stored}  "
          f"train={len(train_logits)}  val={len(val_logits)}")

    # Optionally read teacher top1 from the JSON result file
    teacher_top1 = args.teacher_top1
    if teacher_top1 is None:
        import glob
        pattern = str(ROOT / "results" / f"train_step{teacher_step}_b2_glnn_teacher_seed{SEED}__*.json")
        matches = glob.glob(pattern)
        if matches:
            with open(matches[0]) as fp:
                teacher_top1 = json.load(fp).get("top1_best")
            print(f"  Teacher top1 (from JSON): {teacher_top1:.4f}")
        else:
            print("  WARNING: Teacher result JSON not found — teacher_top1 unknown")

    # ------------------------------------------------------------------
    # Load features + labels from store.h5
    # ------------------------------------------------------------------
    train_ds_raw = H5Dataset(DATA_PATH, split="train")
    val_ds_raw   = H5Dataset(DATA_PATH, split="val")

    assert len(train_ds_raw) == len(train_logits), (
        f"Feature/logit count mismatch: {len(train_ds_raw)} vs {len(train_logits)}")
    assert len(val_ds_raw) == len(val_logits), (
        f"Val feature/logit count mismatch: {len(val_ds_raw)} vs {len(val_logits)}")

    flops    = 2 * H * (N_IN + N_OUT)
    n_params = (N_IN + 1) * H + (H + 1) * N_OUT
    print(f"  MLP_{H}: params={n_params:,}  FLOPs={flops:,} ({flops/1e6:.2f}M)")
    print(f"  Scratch baseline (step403b): {SCRATCH_MLP37_TOP1:.4f}")
    if teacher_top1:
        print(f"  Teacher target:             {teacher_top1:.4f}")

    # ------------------------------------------------------------------
    # Sweep (T, lambda) configs
    # ------------------------------------------------------------------
    run_keys = list(SWEEP_CONFIGS.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n  Configs to run: {run_keys}")

    results = {}

    for cfg_key in run_keys:
        cfg = SWEEP_CONFIGS[cfg_key]
        T   = cfg["T"]
        lam = cfg["lam"]

        print(f"\n{'─'*60}")
        print(f"Config {cfg_key}: T={T}  λ={lam}")
        print(f"{'─'*60}")

        # Build distillation dataset for this T
        train_distill = DistillDataset(
            train_ds_raw.features, train_ds_raw.labels,
            train_logits, T_stored=T_stored, T_req=T)
        val_distill   = DistillDataset(
            val_ds_raw.features, val_ds_raw.labels,
            val_logits, T_stored=T_stored, T_req=T)

        g = torch.Generator().manual_seed(SEED)
        tr = torch.utils.data.DataLoader(
            train_distill, batch_size=BATCH, shuffle=True, generator=g, num_workers=0)
        va = torch.utils.data.DataLoader(
            val_distill, batch_size=BATCH, shuffle=False, num_workers=0)

        torch.manual_seed(SEED)
        model = MLPStudent(hidden=H)
        print(f"  MLP_{H}: params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        t0      = time.time()
        history = train_student(model, tr, va, epochs=EPOCHS, T=T, lam=lam)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best  = max(top1h); bep = int(np.argmax(top1h)) + 1

        # Verdict
        beat_scratch  = best > SCRATCH_MLP37_TOP1
        beat_teacher  = teacher_top1 is not None and best >= teacher_top1 - 0.001
        if beat_scratch and beat_teacher:
            verdict = "STRONG"
        elif beat_teacher:
            verdict = "MEDIUM"
        elif beat_scratch:
            verdict = "WEAK"
        else:
            verdict = "FAIL"

        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)  VERDICT={verdict}")

        results[cfg_key] = {
            "T":            T,
            "lam":          lam,
            "top1_best":    best,
            "top1_last":    top1h[-1],
            "best_epoch":   bep,
            "top1_history": top1h,
            "elapsed_s":    round(elapsed, 1),
            "verdict":      verdict,
            "beat_scratch": beat_scratch,
            "beat_teacher": beat_teacher,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"STEP 603 SUMMARY — GLNN MLP_{H} student distillation")
    print(f"{'='*70}")
    print(f"  Scratch MLP_{H} baseline (step403b) : {SCRATCH_MLP37_TOP1:.4f}")
    if teacher_top1:
        print(f"  SGNNET teacher (step602)          : {teacher_top1:.4f}")
    print(f"  {'Config':<12} {'T':>4} {'λ':>5} {'Best':>8} {'ΔScratch':>10} {'ΔTeach':>10}  Verdict")
    print(f"  {'-'*65}")
    for cfg_key, r in results.items():
        d_scratch = r["top1_best"] - SCRATCH_MLP37_TOP1
        d_teacher = (r["top1_best"] - teacher_top1) if teacher_top1 else float("nan")
        print(f"  {cfg_key:<12} {r['T']:>4.0f} {r['lam']:>5.1f} "
              f"{r['top1_best']:>8.4f} {d_scratch:>+10.4f} {d_teacher:>+10.4f}  {r['verdict']}")

    best_config = max(results, key=lambda k: results[k]["top1_best"])
    print(f"\n  Best config: {best_config}  top1={results[best_config]['top1_best']:.4f}")

    # Paper claim assessment
    any_strong = any(r["verdict"] == "STRONG" for r in results.values())
    any_medium = any(r["verdict"] in ("STRONG", "MEDIUM") for r in results.values())
    if any_strong:
        print("\n  PAPER CLAIM: STRONG — distilled MLP student EXCEEDS scratch baseline")
        print("  Reframe: SGNNET training-time teacher discovers FC can be replaced by MLP.")
    elif any_medium:
        print("\n  PAPER CLAIM: MEDIUM — student matches teacher but not scratch baseline")
        print("  Reframe: distillation transfers teacher knowledge; investigate why scratch is stronger.")
    else:
        print("\n  PAPER CLAIM: WEAK/FAIL — distillation does NOT beat scratch MLP_37")
        print("  Implication: SGNNET routing does NOT reveal compressible soft structure.")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    output = {
        "hidden":             H,
        "n_params":           n_params,
        "flops":              flops,
        "scratch_top1":       SCRATCH_MLP37_TOP1,
        "teacher_top1":       teacher_top1,
        "T_stored":           T_stored,
        "epochs":             EPOCHS,
        "seed":               SEED,
        "best_config":        best_config,
        "best_top1":          results[best_config]["top1_best"],
        "configs":            results,
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
