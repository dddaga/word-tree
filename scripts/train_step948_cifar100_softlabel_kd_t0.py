"""Step 948: CIFAR-100 soft-label KD T0 (20ep, 50% data).

Failure mode targeted: hard CE supervision signal may be too sparse for
CIFAR-100 (100 classes, many visually similar). CIFAR-100 HDF5 contains
VGG16 soft labels (100-class probability distributions from the same teacher
used for Imagenette KD). Training with KL divergence on soft labels instead
of hard one-hot targets gives richer inter-class similarity information.

IMPORTANT FRAMING NOTE:
  The standard Trainer hardcodes KL divergence on soft_labels (trainer.py:147).
  This means ALL prior CIFAR-100 SGNNET runs (e.g. step930) already train with
  A_kd_t1 (T=1 soft KD). Therefore:
    - Ref (hard CE) is the NEW control — testing whether removing soft labels hurts
    - A_kd_t1 reproduces the existing training regime — should match step930 T0 ≈ 0.42
    - B_kd_t2 and C_kd_mix_t1 are the genuine novelties

  A custom loop is used for ALL configs to ensure comparable training conditions
  (same AdamW, cosine LR, W_pos clamping, grad clipping as Trainer).

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, K_in=25)
  Ref          hard cross-entropy on hard labels (one-hot supervision)
  A_kd_t1      KL divergence on VGG soft labels, temperature T=1 (existing regime)
  B_kd_t2      KL divergence on VGG soft labels, temperature T=2 (softer peaks)
  C_kd_mix_t1  0.5 * hard_CE + 0.5 * KD_t1 (mixed)

Temperature scaling for soft KD:
  p_T = softmax(log(soft_labels) / T)  — sharpens (T<1) or softens (T>1) the distribution
  At T=1 (A_kd_t1): raw VGG probs used directly.
  At T=2 (B_kd_t2): softer probs — emphasizes similarity between near-miss classes.

ADVANCE: ≥+0.5pp vs Ref → T1.
Ref baseline: step930 K4I5 T0 best = 0.4201.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_cifar100.h5")
parser.add_argument("--configs", default="Ref,A_kd_t1,B_kd_t2,C_kd_mix_t1")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 100
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step948_cifar100_softlabel_kd_t0_seed{SEED}__{SLOT}.json"
STEP_REF = 0.4201   # step930 K4I5 T0 best (already trains with A_kd_t1)


# ---------------------------------------------------------------------------
# Config table: maps key → (loss_mode, temperature, mix_alpha)
#   loss_mode: "hard_ce" | "kd" | "mixed"
#   temperature: float (only used for kd / mixed)
#   mix_alpha: float weight on hard CE in mixed mode

CONFIG_SPEC = {
    "Ref":         ("hard_ce", 1.0, 0.0),
    "A_kd_t1":     ("kd",      1.0, 0.0),
    "B_kd_t2":     ("kd",      2.0, 0.0),
    "C_kd_mix_t1": ("mixed",   1.0, 0.5),
}


# ---------------------------------------------------------------------------
# ΔW-proj helpers

def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)

def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


class SGNNET_Ref(nn.Module):
    def __init__(self, resonant):
        super().__init__(); self.m = resonant

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x):
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj(self.m.W_pos, conn_hh)
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_agg = _dw_agg(Z_nb, dw)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def make_base():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )


def make_model(key):
    r = make_base()
    return SGNNET_Ref(r)


# ---------------------------------------------------------------------------
# Loss functions

def compute_loss(logits, soft_labels, labels, loss_mode: str, T: float, mix_alpha: float):
    """Compute loss for the given mode."""
    if loss_mode == "hard_ce":
        return F.cross_entropy(logits, labels)

    # Soft KD with temperature
    if T != 1.0:
        # Re-temperature: soft_labels are probs, convert to logits, re-scale, softmax
        log_soft = torch.log(soft_labels.clamp(min=1e-8))
        soft_T = F.softmax(log_soft / T, dim=-1)
    else:
        soft_T = soft_labels

    kd_loss = F.kl_div(F.log_softmax(logits, dim=-1), soft_T, reduction="batchmean")

    if loss_mode == "kd":
        return kd_loss

    # mixed: alpha * hard_CE + (1 - alpha) * KD
    hard_loss = F.cross_entropy(logits, labels)
    return mix_alpha * hard_loss + (1.0 - mix_alpha) * kd_loss


# ---------------------------------------------------------------------------
# Custom training loop — mirrors Trainer internals exactly

def _check_amp_support():
    """Return (use_amp, use_grad_scaler) flags for current device."""
    dev = DEVICE
    is_cpu = dev.type == "cpu"
    is_mps = dev.type == "mps"
    use_amp = not is_cpu
    parts = torch.__version__.split(".")[:2]
    major, minor = int(parts[0]), int(parts[1])
    scaler_ok = major > 2 or (major == 2 and minor >= 3)
    use_grad_scaler = use_amp and not is_mps and scaler_ok
    return use_amp, use_grad_scaler


def train_one_config(key, model, tr, va, loss_mode, T, mix_alpha):
    """Train for EPOCHS, return per-epoch history list of dicts."""
    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    lr_wpos = kw["lr_wpos"]
    grad_clip_norm = kw["grad_clip_norm"]
    min_lr = kw["min_lr"]
    early_stop_patience = kw["early_stop_patience"]
    early_stop_delta = kw["early_stop_delta"]

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(
        [{"params": [model.W_pos], "lr": lr_wpos, "weight_decay": 0.0}]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=min_lr
    )

    use_amp, use_grad_scaler = _check_amp_support()
    scaler = torch.amp.GradScaler(DEVICE.type) if use_grad_scaler else None

    def _amp_ctx():
        if use_amp:
            return torch.autocast(DEVICE.type.split(":")[0], dtype=torch.float16)
        return torch.autocast("cpu", enabled=False)

    # Early stopping (monitors train_loss)
    best_train_loss = float("inf")
    no_improve = 0

    history = []

    for epoch in range(EPOCHS):
        # ---- train ----
        model.train()
        train_loss_sum = 0.0; n_batches = 0
        for features, soft_labels, labels in tr:
            features   = features.to(DEVICE)
            soft_labels = soft_labels.to(DEVICE)
            labels      = labels.to(DEVICE)
            opt.zero_grad()
            with _amp_ctx():
                logits = model(features)
                loss = compute_loss(logits, soft_labels, labels, loss_mode, T, mix_alpha)
            _opt_params = [p for g in opt.param_groups for p in g["params"]]
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(_opt_params, grad_clip_norm)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_opt_params, grad_clip_norm)
                opt.step()
            with torch.no_grad():
                model.W_pos.clamp_(0.0, 1.0)
            train_loss_sum += loss.item(); n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)
        scheduler.step()

        # ---- eval ----
        model.eval()
        all_scores = []; all_labels = []
        with torch.no_grad():
            for features, soft_labels, labels in va:
                features = features.to(DEVICE)
                with _amp_ctx():
                    scores = model(features)
                all_scores.append(scores.cpu())
                all_labels.append(labels)
        all_scores_cat = torch.cat(all_scores, dim=0)
        all_labels_cat = torch.cat(all_labels, dim=0)
        preds = all_scores_cat.argmax(dim=-1)
        val_top1 = (preds == all_labels_cat).float().mean().item()

        row = {"epoch": epoch, "train_loss": round(train_loss, 6),
               "val_top1": round(val_top1, 4), "lr": opt.param_groups[0]["lr"]}
        history.append(row)
        print(f"  e{epoch+1:3d}  top1={val_top1:.4f}  loss={train_loss:.4f}  lr={opt.param_groups[0]['lr']:.2e}", flush=True)

        if math.isnan(train_loss):
            print(f"  NaN detected at epoch {epoch+1}"); break

        # Early stopping
        if train_loss < best_train_loss - early_stop_delta:
            best_train_loss = train_loss; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"  Early stop at epoch {epoch+1}"); break

    return history


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=(DEVICE.type == "cuda"))
    n_full = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=10,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"\n{'='*70}")
    print(f"step948 — CIFAR-100 soft-label KD T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}/K_in={K_IN}  N_OUT={N_OUT}")
    print(f"  NOTE: A_kd_t1 reproduces existing training regime (Trainer uses KL/T=1)")
    print(f"        Ref (hard CE) is the NEW control; B/C are genuine novelties")
    print(f"  STEP_REF (step930 K4I5 T0 = A_kd_t1 regime): {STEP_REF:.4f}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        loss_mode, T, mix_alpha = CONFIG_SPEC[key]
        model = make_model(key)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: loss={loss_mode}  T={T}  mix_alpha={mix_alpha}  params={n_p:,}")

        t0 = time.time()
        history = train_one_config(key, model, tr, va, loss_mode, T, mix_alpha)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP_REF)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {"loss_mode": loss_mode, "T": T, "mix_alpha": mix_alpha,
                        "n_params": n_p, "best": round(best, 4), "best_ep": best_ep,
                        "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 948 SUMMARY — CIFAR-100 soft-label KD T0")
    print(f"{'='*70}")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        v = "(baseline)" if k == "Ref" else ("ADVANCE→T1" if d >= 0.005 else ("NEUTRAL" if d >= -0.005 else "KILL"))
        print(f"  {k:<16} loss={r['loss_mode']:<9} T={r['T']}  best={r['best']:.4f}  Δ={d*100:+.2f}pp  {v}")
    print(f"\n  NOTE: A_kd_t1 should ≈ {STEP_REF:.4f} (existing training regime)")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
