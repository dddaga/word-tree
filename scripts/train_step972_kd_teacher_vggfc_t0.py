"""Step 972: KD alpha sweep — how much does T=1 soft-label distillation vs pure hard CE matter?

MOTIVATION
==========
Current SGNNET training (all steps including step199 at 95.52%) already uses
KL-divergence against VGG FC soft labels at T=1 as the task loss. This IS
knowledge distillation, but at the default temperature.

This T0 scout sweeps two axes:
  1. alpha — interpolation between soft-label KD (alpha=0.0, current default)
             and hard CE (alpha=1.0, novel — no SGNNET run has used pure hard labels)
  2. T — distillation temperature (T=1 = current default; T=4 = standard KD)

With stored T=1 probabilities we approximate T=4 as: softmax(log(p) / T)
which recovers the correct scaled distribution. Per standard KD (Hinton 2015),
the KD term is scaled by T^2.

CONFIGS (N=2048, D=16, K_in=25, K_iter=5 — step199 base):
  Ref       alpha=0.0, T=1  — control, matches current step199 training
  A_a0T4    alpha=0.0, T=4  — softer labels, same formula scaled by T^2
  B_a03T4   alpha=0.3, T=4  — 30% hard CE + 70% soft KD@T4
  C_a05T4   alpha=0.5, T=4  — balanced
  D_a1      alpha=1.0, T=any — pure hard CE (novel for SGNNET)

T0: 20ep, 50% data, seed=42.

ADVANCE RULE: any config within 0.5pp of Ref → T1
              any config +0.5pp above Ref → fast-track to T1
"""
from __future__ import annotations
import argparse, json, sys, time, math
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld       import SGNNET_SmallWorld
from src.sgnnet.model_resonant         import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory  import SGNNET_AntiHebbian
from src.training.trainer              import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

# ── Constants ─────────────────────────────────────────────────────────────────
EPOCHS    = args.epochs
BATCH     = 128
SEED      = 42
DATA_FRAC = 0.5   # T0: 50% data
DATA_FILE = "data/store_aug.h5"

N      = 2048; N_IN = 25088; N_OUT = 10
D      = 16;   K_HH = 2;    K_IN  = 25;  K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

FLOPS = 3 * N * K_HH * D * K_ITER   # 983,040 ≈ 0.98M

OUT_PATH = ROOT / "results" / "train_step972_kd_alpha_sweep_t0_seed42.json"

# ── Config grid ───────────────────────────────────────────────────────────────
CONFIGS = [
    {"key": "Ref",    "alpha": 0.0, "T": 1, "label": "control (current default: pure KD T=1)"},
    {"key": "A_a0T4", "alpha": 0.0, "T": 4, "label": "pure KD T=4 (softer soft-labels, T^2 scaled)"},
    {"key": "B_a03T4","alpha": 0.3, "T": 4, "label": "30% CE + 70% KD T=4"},
    {"key": "C_a05T4","alpha": 0.5, "T": 4, "label": "50% CE + 50% KD T=4 (balanced)"},
    {"key": "D_a1",   "alpha": 1.0, "T": 1, "label": "pure hard CE (novel for SGNNET)"},
]


# ── KD Trainer subclass ───────────────────────────────────────────────────────

class KDTrainer(Trainer):
    """Trainer with configurable alpha blending between hard CE and soft KD.

    Loss = alpha * CE(logits, hard_labels)
         + (1 - alpha) * T^2 * KL(log_softmax(logits/T), soft_labels_T)

    where soft_labels_T = softmax(log(soft_labels_T1) / T)
    recovers temperature-scaled distribution from stored T=1 probs.

    Special case alpha=0.0, T=1: matches original Trainer exactly
    (pure KL against stored soft_labels, T^2 = 1).
    """

    def __init__(self, alpha: float, T: float, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.T = T

    def _compute_task_loss(
        self,
        scores: torch.Tensor,
        soft_labels: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute blended KD loss."""
        # ── Soft KD term ─────────────────────────────────────────────────────
        if self.alpha < 1.0:
            if self.T == 1:
                # Fast path: matches original Trainer exactly
                kd_loss = F.kl_div(
                    F.log_softmax(scores, dim=-1),
                    soft_labels,
                    reduction="batchmean",
                )
            else:
                # Temperature-scaled: recover log-probs from stored T=1 probs,
                # divide by T, then softmax to get teacher distribution at temp T.
                # Teacher: softmax(log(p_T1) / T)  [Hinton 2015 re-derivation]
                # log(p_T1) is the original logit up to a constant (softmax invariant)
                # Student: softmax(logits / T)
                log_p_teacher = torch.log(soft_labels.clamp(min=1e-10))
                soft_labels_T = F.softmax(log_p_teacher / self.T, dim=-1)
                kd_loss = (self.T ** 2) * F.kl_div(
                    F.log_softmax(scores / self.T, dim=-1),
                    soft_labels_T,
                    reduction="batchmean",
                )
        else:
            kd_loss = torch.tensor(0.0, device=scores.device)

        # ── Hard CE term ─────────────────────────────────────────────────────
        if self.alpha > 0.0:
            ce_loss = F.cross_entropy(scores, labels)
        else:
            ce_loss = torch.tensor(0.0, device=scores.device)

        return self.alpha * ce_loss + (1.0 - self.alpha) * kd_loss

    def train_epoch(self) -> dict:
        """Override to inject custom loss."""
        self.model.train()
        total_loss_sum = task_loss_sum = 0.0
        n_batches = 0

        for features, soft_labels, labels in self.train_loader:
            features    = features.to(self.device)
            soft_labels = soft_labels.to(self.device)
            labels      = labels.to(self.device)

            self.optimizer.zero_grad()

            _amp_ctx = (
                torch.autocast(str(self.device).split(":")[0], dtype=torch.float16)
                if self.use_amp
                else torch.autocast("cpu", enabled=False)
            )
            with _amp_ctx:
                scores = self.model(features)
                task_loss = self._compute_task_loss(scores, soft_labels, labels)
                loss = task_loss   # lambda_safety and lambda_lb are 0 in trainer_kwargs

            _opt_params = [p for g in self.optimizer.param_groups for p in g["params"]]

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(_opt_params, self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_opt_params, self.grad_clip_norm)
                self.optimizer.step()

            if hasattr(self.model, "tick_step"):
                self.model.tick_step()

            with torch.no_grad():
                self.model.W_pos.clamp_(0, self.box_size)

            total_loss_sum += loss.item()
            task_loss_sum  += task_loss.item()
            n_batches += 1

        n = max(n_batches, 1)
        return {
            "train_loss": total_loss_sum / n,
            "task_loss":  task_loss_sum / n,
            "safety_loss": 0.0,
            "lb_loss": 0.0,
        }

    def evaluate(self) -> dict:
        """Override: always evaluate with hard accuracy (device-agnostic)."""
        self.model.eval()
        all_scores = []
        all_labels = []
        val_loss_sum = 0.0
        n_batches = 0

        with torch.no_grad():
            for features, soft_labels, labels in self.val_loader:
                features    = features.to(self.device)
                soft_labels = soft_labels.to(self.device)
                labels      = labels.to(self.device)

                _amp_ctx = (
                    torch.autocast(str(self.device).split(":")[0], dtype=torch.float16)
                    if self.use_amp
                    else torch.autocast("cpu", enabled=False)
                )
                with _amp_ctx:
                    scores = self.model(features)
                    task_loss = self._compute_task_loss(scores, soft_labels, labels)

                val_loss_sum += task_loss.item()
                all_scores.append(scores.cpu())
                all_labels.append(labels.cpu())
                n_batches += 1

        n = max(n_batches, 1)
        all_scores_cat = torch.cat(all_scores, dim=0)
        all_labels_cat = torch.cat(all_labels, dim=0)
        preds = all_scores_cat.argmax(dim=-1)
        val_top1 = (preds == all_labels_cat).float().mean().item()
        return {"val_loss": val_loss_sum / n, "val_top1": val_top1}


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> nn.Module:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    model = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB,
                                variant="wpos").to(device)
    return model


# ── Data loader factory (with 50% data subsetting) ───────────────────────────

def make_loaders_frac(path: str, frac: float, batch: int, seed: int):
    """Return (train_loader, val_loader) with training set down-sampled to frac."""
    from src.training.dataset import H5Dataset
    import torch.utils.data as td

    full_train = H5Dataset(path, split="train")
    val_ds     = H5Dataset(path, split="val")

    n_sub = int(len(full_train) * frac)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(full_train), generator=g)[:n_sub]
    sub_train = td.Subset(full_train, idx.tolist())

    gb = (full_train.features.numel() + val_ds.features.numel()) * 4 / 1e9
    print(f"  Data: train_sub={n_sub}/{len(full_train)} ({frac*100:.0f}%)  "
          f"val={len(val_ds)}  ({gb:.2f} GB features in RAM)")

    g2 = torch.Generator().manual_seed(seed)
    tr_loader = td.DataLoader(sub_train, batch_size=batch, shuffle=True,
                               generator=g2, num_workers=0)
    va_loader = td.DataLoader(val_ds, batch_size=batch, shuffle=False,
                               num_workers=0)
    return tr_loader, va_loader


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print(f"Step 972: KD alpha sweep T0 — alpha ∈ {{0.0,0.3,0.5,1.0}}, T ∈ {{1,4}}")
    print(f"Base config: N={N} D={D} K_in={K_IN} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"KEY: alpha=0.0 T=1 = control (current step199 default).")
    print(f"     alpha=1.0 = pure hard CE — NOVEL for SGNNET.")
    print(f"Device: {DEVICE}")
    print(f"{'='*70}\n")

    tr_loader, va_loader = make_loaders_frac(
        str(ROOT / DATA_FILE), DATA_FRAC, BATCH, SEED
    )

    results = {}

    for cfg in CONFIGS:
        key   = cfg["key"]
        alpha = cfg["alpha"]
        T     = cfg["T"]
        label = cfg["label"]

        print(f"\n── {key}: {label} ──────────────────────────────")
        model = build_model(DEVICE)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}  alpha={alpha}  T={T}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = KDTrainer(
            alpha=alpha, T=T,
            model=model,
            train_loader=tr_loader,
            val_loader=va_loader,
            device=DEVICE,
            **kw,
        )

        t0 = time.time()

        def _log(m):
            if str(DEVICE) == "mps":
                torch.mps.empty_cache()
            ep = m["epoch"] + 1
            if ep % 5 == 0 or ep == 1:
                print(f"    ep{ep:3d}  val={m['val_top1']:.4f}  "
                      f"loss={m['train_loss']:.4f}", flush=True)

        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.0), 4) for h in history]
        best  = max(top1h)
        bep   = int(np.argmax(top1h)) + 1

        results[key] = {
            "key":       key,
            "alpha":     alpha,
            "T":         T,
            "label":     label,
            "n_params":  n_p,
            "flops":     FLOPS,
            "top1_best": best,
            "top1_last": top1h[-1],
            "best_epoch": bep,
            "epochs_run": len(history),
            "data_frac":  DATA_FRAC,
            "top1_history": top1h,
            "elapsed_s": round(elapsed, 1),
        }
        print(f"  ► {key}: best={best:.4f}@ep{bep}  elapsed={elapsed:.1f}s")

    # ── Save ─────────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*70}")
    print(f"Results → {OUT_PATH}")
    print(f"\nSummary (sorted by best_acc):")
    print(f"  {'Config':<12} {'alpha':>6} {'T':>3}  {'best_acc':>8}  {'vs_Ref':>8}  Label")
    print(f"  {'-'*70}")

    ref_best = results.get("Ref", {}).get("top1_best", 0.0)
    sorted_r = sorted(results.values(), key=lambda r: r["top1_best"], reverse=True)
    for r in sorted_r:
        delta = r["top1_best"] - ref_best
        mark  = " ▲" if delta > 0.005 else (" ▼" if delta < -0.005 else "  ")
        print(f"  {r['key']:<12} {r['alpha']:>6.1f} {r['T']:>3}  "
              f"{r['top1_best']:>8.4f}  {delta:>+8.4f}{mark}  {r['label']}")

    print(f"\nAdvance rule: within ±0.5pp of Ref → T1 | >+0.5pp → fast-track T1")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
