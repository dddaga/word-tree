"""Step 401: Paper baselines — Pruned MLP vs SGNNET (random-proj ablation).

CLAIMS UNDER TEST
=================
1. "SGNNET efficiency claim" — do simple dense MLPs match SGNNET at the same
   parameter budget? If MLP_2/MLP_3 (≈50–75K params) match 95.52%, the SGNNET
   contribution is the dense projection, not the sparse routing geometry.

2. "Learned geometry is load-bearing" — SGNNET_RandProj freezes W_pos at
   random init and trains only theta + fc_out. If accuracy holds, the sparse
   routing structure drives performance regardless of learned positions.

CONFIGS (50% data, 20ep Tier-0 budget)
---------------------------------------
MLP baselines (plain CrossEntropy + Adam, no SGNNET Trainer):
  Lin_direct : Linear(25088, 10)                   — 250,890 params
  MLP_2      : Linear(25088, 2) + ReLU + Linear(2, 10)  — ~50K params
  MLP_3      : Linear(25088, 3) + ReLU + Linear(3, 10)  — ~75K params
  MLP_64     : Linear(25088, 64) + ReLU + Linear(64, 10) — ~1.6M params

SGNNET configs (N=2048, D=16, efficiency config — uses Trainer):
  SGNNET_Ref      : standard efficiency config (reference)
  SGNNET_RandProj : same config but W_pos frozen at random init

Reference: step199 efficiency config (95.52% best_ep=136 @ 0.98M FLOPs).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--configs", default="")
args   = parser.parse_args()

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"

# SGNNET efficiency config (step199 defaults)
N      = 2048
N_IN   = 25088
N_OUT  = 10
D      = 16
K_HH   = 2
K_IN   = 25
K_ITER = 5
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
ALPHA_AHEBB   = 1.0

OUT_PATH = ROOT / "results" / "train_step401_baselines.json"

# ---------------------------------------------------------------------------
# Configs to run
# ---------------------------------------------------------------------------

ALL_CONFIGS = ["Lin_direct", "MLP_2", "MLP_3", "MLP_64",
               "SGNNET_Ref", "SGNNET_RandProj"]

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

class LinDirect(nn.Module):
    """Single linear projection: 25088 → 10. 250,890 params."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(N_IN, N_OUT)

    def forward(self, x):
        return self.fc(x)


class MLPBaseline(nn.Module):
    """2-layer MLP: Linear(N_IN, h) → ReLU → Linear(h, N_OUT)."""
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_OUT),
        )

    def forward(self, x):
        return self.net(x)


def build_sgnnet(freeze_wpos: bool = False) -> nn.Module:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4)
    K_l = K_HH - K_r
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
    model = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")

    if freeze_wpos:
        # Freeze positional embedding — only theta and fc_out remain trainable.
        # model.m is SGNNET_Resonant; model.m.base is SGNNET_SmallWorld.
        model.m.base.W_pos.requires_grad_(False)

    return model


# ---------------------------------------------------------------------------
# Simple training loop for MLP baselines
# (Trainer directly accesses model.W_pos / model.W_phase — not suitable here)
# ---------------------------------------------------------------------------

def train_mlp(model: nn.Module, tr, va) -> list[dict]:
    """Minimal Adam + CosineAnnealingLR loop for plain nn.Module classifiers."""
    model = model.to(DEVICE)
    opt   = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-7)
    crit  = nn.CrossEntropyLoss()
    history = []

    for epoch in range(EPOCHS):
        # ── train ──
        model.train()
        for feats, _soft, labels in tr:
            feats  = feats.to(DEVICE)
            labels = labels.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(feats), labels)
            loss.backward()
            opt.step()
        sched.step()

        # ── validate ──
        model.eval()
        correct = total = 0
        val_loss_sum = n_batches = 0
        with torch.no_grad():
            for feats, _soft, labels in va:
                feats  = feats.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(feats)
                val_loss_sum += crit(logits, labels).item()
                preds   = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
                n_batches += 1

        val_top1 = correct / max(total, 1)
        history.append({"epoch": epoch, "val_top1": val_top1,
                         "val_loss": val_loss_sum / max(n_batches, 1)})

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  ep{epoch+1:3d}  val={val_top1:.4f}", flush=True)

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    run_keys = ALL_CONFIGS
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"Step 401 — Paper baselines: MLP vs SGNNET")
    print(f"  MLP baselines: Lin_direct, MLP_2, MLP_3, MLP_64")
    print(f"  SGNNET ablation: Ref + RandProj (frozen W_pos)")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    # ── data ──
    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(
        subset, batch_size=BATCH, shuffle=True, num_workers=0
    )

    results = {}

    for key in run_keys:
        print(f"\n{'─'*60}\nConfig {key}\n{'─'*60}")

        # ── build ──
        torch.manual_seed(SEED)
        if key == "Lin_direct":
            model = LinDirect()
        elif key == "MLP_2":
            model = MLPBaseline(hidden=2)
        elif key == "MLP_3":
            model = MLPBaseline(hidden=3)
        elif key == "MLP_64":
            model = MLPBaseline(hidden=64)
        elif key == "SGNNET_Ref":
            model = build_sgnnet(freeze_wpos=False)
        elif key == "SGNNET_RandProj":
            model = build_sgnnet(freeze_wpos=True)
        else:
            print(f"  Unknown config {key!r}, skipping.")
            continue

        n_p_total    = sum(p.numel() for p in model.parameters())
        n_p_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params total={n_p_total:,}  trainable={n_p_trainable:,}")

        t0 = time.time()

        if key in ("SGNNET_Ref", "SGNNET_RandProj"):
            # Use shared Trainer (handles W_pos clamping, AMP, etc.)
            kw = trainer_kwargs(N, n_epochs=EPOCHS)
            trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                              device=DEVICE, **kw)

            def _log(m):
                ep = m["epoch"] + 1
                if ep % 5 == 0 or ep == 1:
                    print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)

            history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        else:
            # Plain Adam loop (MLP models have no W_pos / W_phase)
            history = train_mlp(model, tr, va)

        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.0), 4) for h in history]
        best  = max(top1h)
        bep   = int(np.argmax(top1h)) + 1
        results[key] = {
            "top1_best":    best,
            "top1_last":    top1h[-1],
            "best_epoch":   bep,
            "top1_history": top1h,
            "elapsed_s":    round(elapsed, 1),
            "n_params_total":    n_p_total,
            "n_params_trainable": n_p_trainable,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    # ── summary ──
    sgnnet_ref = results.get("SGNNET_Ref", {}).get("top1_best", None)
    print(f"\n{'='*70}\nSTEP 401 SUMMARY\n{'='*70}")
    print(f"  {'Config':<18}  {'params':>10}  {'best':>8}  {'vs SGNNET_Ref':>14}")
    for key in run_keys:
        if key not in results:
            continue
        r = results[key]
        delta_str = ""
        if sgnnet_ref is not None and key != "SGNNET_Ref":
            delta_str = f"  {r['top1_best'] - sgnnet_ref:+.4f}"
        print(f"  {key:<18}  {r['n_params_trainable']:>10,}  "
              f"{r['top1_best']:>8.4f}{delta_str}")

    print(f"\nInterpretation guide:")
    print(f"  MLP_2/MLP_3 ≈ SGNNET_Ref   → sparse routing, not density, drives efficiency")
    print(f"  MLP_2/MLP_3 << SGNNET_Ref  → param budget insufficient for dense projection")
    print(f"  SGNNET_RandProj ≈ SGNNET_Ref → W_pos geometry not load-bearing")
    print(f"  SGNNET_RandProj << SGNNET_Ref → learned geometry is essential")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
