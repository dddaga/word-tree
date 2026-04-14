"""Step 163: Progressive K_iter Distillation — intermediate state matching.

HYPOTHESIS
==========
step127 failed because:
1. Teacher and student had DIFFERENT conn_hh/conn_in (different seeds) —
   routing was over incomparable graphs, so Z_student[k] and Z_teacher[2k]
   are activations of neurons with different neighbor sets.
2. Output-only supervision: student got no signal about HOW to reach the
   representation, only what the destination looks like.

Fix:
1. All students share teacher's state_dict (load_state_dict) — conn_hh, conn_in,
   AND learned W_pos/theta/W_phase are transferred. K_iter is NOT in state_dict.
2. Dense intermediate supervision: Z_student[k] matches Z_teacher[t_k] where
   t_k = round(k / K_student * K_teacher) - 1 (0-indexed linear interpolation).
   For K_s=6, K_t=12: student step k → teacher step 2k (2:1 clean ratio).
   For K_s=8, K_t=12: linear interpolation.

KEY ABLATION
============
  Ref : K=12 teacher trained from scratch (establishes the ceiling)
  A   : K=6 scratch (SEED=42) — same topology as teacher, random init (step127 fair control)
  B   : K=6 teacher init, no distil — warm-start alone
  C   : K=6 teacher init + progressive distil α=0.3
  D   : K=6 teacher init + progressive distil α=0.5
  E   : K=8 teacher init + progressive distil α=0.3

A vs B: weight initialization effect (topology identical, init differs)
B vs C/D: progressive distillation effect on top of warm-start
C vs D: α sensitivity

Teacher checkpoint cached at results/train_step163_teacher.pt for restartability.

To reproduce:
    python -u scripts/train_step163_progressive_kd.py --device mps
    python -u scripts/train_step163_progressive_kd.py --device cpu
    python -u scripts/train_step163_progressive_kd.py --configs A,B,C --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.experiment_config   import trainer_kwargs, topology_kwargs
from src.training.dataset             import make_loaders

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--configs", default="",
                    help="Comma-sep config keys (e.g. Ref,A,B). Empty = all.")
parser.add_argument("--force-retrain-teacher", action="store_true",
                    help="Re-train teacher even if checkpoint exists.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
N      = 1024;  N_IN = 25088;  N_OUT = 10
D      = 16;    K_HH = 8;      K_IN  = 25
K_ITER_TEACHER = 12
ALPHA_REFLECT  = 0.5
ALPHA_TURING   = 0.0
ALPHA_AHEBB    = 1.0

TEACHER_PATH = ROOT / "results" / "train_step163_teacher.pt"
OUT_PATH     = ROOT / "results" / "train_step163_progressive_kd.json"


# ---------------------------------------------------------------------------
# Model with intermediate capture
# ---------------------------------------------------------------------------

class SGNNET_AH_WithIntermediates(SGNNET_AntiHebbian):
    """SGNNET_AntiHebbian extended with intermediate Z capture.

    forward() is identical to parent (Trainer-compatible).
    forward_with_intermediates() returns (logits, [Z_1, ..., Z_K]).

    state_dict() is identical to SGNNET_AntiHebbian — load_state_dict works
    across models that differ only in K_iter.
    """

    def forward_with_intermediates(self, x: torch.Tensor):
        """Run AH routing and collect Z at each step.

        Returns:
            logits       [B, N_out]
            intermediates list of K tensors [B, N, D] — one per routing step
        """
        base      = self.m.base
        Z         = base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = base.conn_hh
        N_h       = base.N_hidden

        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        Z_reflected   = torch.zeros_like(Z)
        intermediates = []

        for _ in range(base.K_iter):
            Z_fwd       = F.relu(Z - theta_pos)
            Z_nb        = Z_fwd[:, conn_hh, :]
            Z_struct    = (Z_nb * supp_w).sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z           = F.normalize(Z_new.clamp(-10, 10), dim=-1)
            intermediates.append(Z)

        logits = base._readout(Z)
        return logits, intermediates


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key:       str
    label:     str
    k_iter:    int
    warm_init: bool   # load teacher state_dict before training
    alpha_kd:  float  # distillation weight (0 = no distil)


CONFIGS = [
    Config("Ref", "K=12 teacher, scratch SEED=42 (ceiling)",              K_ITER_TEACHER, False, 0.0),
    Config("A",   "K=6 scratch SEED=42 (shared topology, random weights)", 6,              False, 0.0),
    Config("B",   "K=6 teacher init, no distil (warm-start alone)",       6,              True,  0.0),
    Config("C",   "K=6 teacher init + progressive distil α=0.3",          6,              True,  0.3),
    Config("D",   "K=6 teacher init + progressive distil α=0.5",          6,              True,  0.5),
    Config("E",   "K=8 teacher init + progressive distil α=0.3",          8,              True,  0.3),
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_loaders_cache = None

def get_loaders():
    global _loaders_cache
    if _loaders_cache is None:
        tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
        n   = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        sub = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr  = torch.utils.data.DataLoader(sub, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders_cache = (tr, va)
    return _loaders_cache


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(k_iter: int, seed: int = SEED) -> SGNNET_AH_WithIntermediates:
    torch.manual_seed(seed)
    K_random = max(1, K_HH // 4)
    K_local  = K_HH - K_random
    n_groups = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=k_iter,
        K_local=K_local, K_random=K_random, n_groups=n_groups,
        norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AH_WithIntermediates(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Teacher index mapping
# ---------------------------------------------------------------------------

def teacher_indices(K_s: int, K_t: int) -> List[int]:
    """Return 0-indexed teacher step for each 1-indexed student step k=1..K_s.

    For K_s=6, K_t=12: [1, 3, 5, 7, 9, 11]  (student k → teacher 2k, 0-indexed)
    For K_s=8, K_t=12: linear interpolation   [1, 2, 4, 5, 7, 8, 10, 11]
    """
    return [min(round(k / K_s * K_t) - 1, K_t - 1) for k in range(1, K_s + 1)]


# ---------------------------------------------------------------------------
# Training loop (unified for all configs)
# ---------------------------------------------------------------------------

def train_config(
    student:   SGNNET_AH_WithIntermediates,
    teacher:   Optional[SGNNET_AH_WithIntermediates],
    n_epochs:  int,
    alpha_kd:  float,
) -> List[dict]:
    """Train student. Teacher is frozen; used only when alpha_kd > 0.

    Optimizer mirrors Trainer: AdamW, W_pos only (no weight_decay), plateau sched.
    """
    tr, va = get_loaders()

    kw  = trainer_kwargs(N, n_epochs=n_epochs)
    opt = torch.optim.AdamW(
        [{"params": [student.W_pos], "lr": kw["lr_wpos"], "weight_decay": 0.0}]
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=kw["sched_factor"],
        patience=kw["sched_patience"], min_lr=kw["min_lr"],
    )

    # Determine teacher mapping once
    use_distil = alpha_kd > 0.0 and teacher is not None
    if use_distil:
        K_s = student.m.base.K_iter
        K_t = teacher.m.base.K_iter
        t_idx = teacher_indices(K_s, K_t)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    history = []
    for ep in range(1, n_epochs + 1):
        student.train()
        if hasattr(student, "tick_epoch"):
            student.tick_epoch()

        train_loss_sum = 0.0
        n_batches      = 0

        for features, soft_labels, _ in tr:
            features    = features.to(DEVICE)
            soft_labels = soft_labels.to(DEVICE)
            opt.zero_grad()

            if use_distil:
                # Teacher forward (no grad, collect intermediates)
                with torch.no_grad():
                    _, z_teacher = teacher.forward_with_intermediates(features)

                # Student forward (collect intermediates + logits)
                logits_s, z_student = student.forward_with_intermediates(features)

                task_loss = F.kl_div(
                    F.log_softmax(logits_s, dim=-1), soft_labels, reduction="batchmean"
                )

                # Progressive MSE: student step k ↔ teacher step t_idx[k]
                kd_terms = [
                    F.mse_loss(z_student[k], z_teacher[t_idx[k]].detach())
                    for k in range(len(z_student))
                ]
                kd_loss = sum(kd_terms) / len(kd_terms)
                loss = task_loss + alpha_kd * kd_loss

            else:
                # Standard task loss only
                logits_s = student(features)
                task_loss = F.kl_div(
                    F.log_softmax(logits_s, dim=-1), soft_labels, reduction="batchmean"
                )
                loss = task_loss

            loss.backward()
            opt.step()

            with torch.no_grad():
                student.W_pos.clamp_(0, 1.0)

            train_loss_sum += loss.item()
            n_batches      += 1

        # Validation
        student.eval()
        correct = total = 0
        with torch.no_grad():
            for features, _, labels in va:
                logits = student(features.to(DEVICE)).cpu()
                correct += (logits.argmax(1) == labels).sum().item()
                total   += len(labels)
        val_top1   = correct / total
        train_loss = train_loss_sum / max(n_batches, 1)
        sched.step(train_loss)

        if ep % 10 == 0 or ep == n_epochs:
            print(f"    ep{ep:3d}  loss={train_loss:.4f}  val={val_top1:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

        history.append({"train_loss": train_loss, "val_top1": round(val_top1, 4)})

    return history


# ---------------------------------------------------------------------------
# Teacher: train or load from cache
# ---------------------------------------------------------------------------

def get_or_train_teacher(force: bool = False) -> SGNNET_AH_WithIntermediates:
    teacher = make_model(K_ITER_TEACHER, seed=SEED).to(DEVICE)

    if not force and TEACHER_PATH.exists():
        print(f"  Loading cached teacher from {TEACHER_PATH}")
        state = torch.load(TEACHER_PATH, map_location=DEVICE)
        teacher.load_state_dict(state)
        return teacher, None

    print(f"\n{'─'*60}")
    print(f"Training teacher (Ref)  K_iter={K_ITER_TEACHER}  epochs={EPOCHS}")
    print(f"{'─'*60}")
    t0      = time.time()
    history = train_config(teacher, teacher=None, n_epochs=EPOCHS, alpha_kd=0.0)
    elapsed = time.time() - t0
    print(f"  Teacher done: top1={max(h['val_top1'] for h in history):.4f}  "
          f"elapsed={elapsed/60:.1f}min")
    TEACHER_PATH.parent.mkdir(exist_ok=True)
    torch.save(teacher.state_dict(), TEACHER_PATH)
    return teacher, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 163 — Progressive K_iter Distillation")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter teacher={K_ITER_TEACHER}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}")
    print("""
Fix over step127:
  [1] All students share teacher state_dict (conn_hh + conn_in + learned W_pos)
  [2] Dense intermediate supervision: Z_student[k] ↔ Z_teacher[t_k]
  [3] Warm-start isolation: Config B tests init alone (no distil)
""")

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active     = [c for c in CONFIGS if not cfg_filter or c.key in cfg_filter]

    # Always need teacher unless only running Ref
    need_teacher = any(c.warm_init or c.alpha_kd > 0 for c in active)
    run_ref      = any(c.key == "Ref" for c in active)

    results = {}
    OUT_PATH.parent.mkdir(exist_ok=True)
    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text())

    # ── Step 1: Teacher (Ref) ────────────────────────────────────────────────
    teacher, teacher_history = get_or_train_teacher(force=args.force_retrain_teacher)

    if run_ref and teacher_history is not None:
        top1_hist = [h["val_top1"] for h in teacher_history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1
        results["Ref"] = {
            "k_iter": K_ITER_TEACHER, "warm_init": False, "alpha_kd": 0.0,
            "top1_best": top1_best, "best_epoch": best_ep,
            "top1_history": top1_hist,
            "label": CONFIGS[0].label,
            "teacher_mapping": "N/A",
        }
        OUT_PATH.write_text(json.dumps(results, indent=2))
        print(f"  Ref: top1={top1_best:.4f}  best_ep={best_ep}")
    elif run_ref and "Ref" not in results:
        # Teacher was loaded from cache — evaluate it
        teacher.eval()
        _, va = get_loaders()
        correct = total = 0
        with torch.no_grad():
            for features, _, labels in va:
                logits = teacher(features.to(DEVICE)).cpu()
                correct += (logits.argmax(1) == labels).sum().item()
                total   += len(labels)
        top1_best = correct / total
        results["Ref"] = {
            "k_iter": K_ITER_TEACHER, "warm_init": False, "alpha_kd": 0.0,
            "top1_best": round(top1_best, 4), "best_epoch": -1,
            "top1_history": [], "label": CONFIGS[0].label,
            "teacher_mapping": "N/A", "note": "loaded_from_cache",
        }
        OUT_PATH.write_text(json.dumps(results, indent=2))
        print(f"  Ref (cached): top1={top1_best:.4f}")

    # ── Step 2: Student configs ──────────────────────────────────────────────
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student_configs = [c for c in active if c.key != "Ref"]
    for cfg in student_configs:
        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        t_map = teacher_indices(cfg.k_iter, K_ITER_TEACHER)
        print(f"  K_student={cfg.k_iter}  warm_init={cfg.warm_init}  α_kd={cfg.alpha_kd}")
        if cfg.alpha_kd > 0:
            print(f"  Teacher mapping (0-idx): {t_map}")

        student   = make_model(cfg.k_iter, seed=SEED).to(DEVICE)
        n_params  = count_params(student)

        if cfg.warm_init:
            student.load_state_dict(teacher.state_dict())
            print(f"  Initialized from teacher state_dict")

        t0      = time.time()
        history = train_config(
            student, teacher if cfg.alpha_kd > 0 else None,
            n_epochs=EPOCHS, alpha_kd=cfg.alpha_kd,
        )
        elapsed = time.time() - t0

        top1_hist = [h["val_top1"] for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1

        ref_best = results.get("Ref", {}).get("top1_best", 0.0)
        vs_ref   = top1_best - ref_best if ref_best > 0 else 0.0

        print(f"\n  top1={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"best_ep={best_ep}  elapsed={elapsed/60:.1f}min  params={n_params:,}")

        results[cfg.key] = {
            "k_iter":          cfg.k_iter,
            "warm_init":       cfg.warm_init,
            "alpha_kd":        cfg.alpha_kd,
            "top1_best":       top1_best,
            "best_epoch":      best_ep,
            "top1_history":    top1_hist,
            "n_params":        n_params,
            "elapsed_s":       round(elapsed, 1),
            "label":           cfg.label,
            "teacher_mapping": t_map,
        }
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"STEP 163 SUMMARY — Progressive K_iter Distillation")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0.0)
    print(f"{'Key':4s}  {'K':>4}  {'init':>8}  {'α_kd':>5}  {'top1':>7}  {'vs_Ref':>8}  label")
    print(f"{'─'*80}")
    for k, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"{k:4s}  {r['k_iter']:>4}  "
              f"{'teacher' if r['warm_init'] else 'scratch':>8}  "
              f"{r['alpha_kd']:>5.1f}  "
              f"{r['top1_best']:.4f}  {vs:>+.4f}  {r['label']}")

    print(f"""
Key comparisons:
  A vs B  → weight init effect (both K=6, topology shared)
  B vs C  → progressive distil on top of warm-start (α=0.3)
  C vs D  → α sensitivity (0.3 vs 0.5)
  B vs E  → K=8 vs K=6 with same warm-start + distil strategy
""")
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Results saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
