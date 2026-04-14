"""Step 155: Diagnostics baseline — health check on top-5 proven configurations.

PURPOSE
=======
Run diagnostics on the best-known configurations to establish baseline health
metrics. This tells us what "healthy" looks like for SGNNET training, and
provides reference values for evaluating future experiments.

CONFIGS (50%/20ep each — just enough to see training dynamics)
==============================================================
  A : N=4096, D=64, K_hh=4, K_iter=12, AH=1.0  (project best config)
  B : N=1024, D=16, K_hh=8, K_iter=8, AH=1.0   (efficiency baseline)
  C : N=1024, D=32, K_hh=4, K_iter=8, AH=1.0   (mid-range)
  D : N=256,  D=16, K_hh=4, K_iter=4, AH=1.0   (minimal config)
  E : N=1024, D=16, K_hh=8, K_iter=8, AH=0.0   (no AH — control)

To reproduce:
    python -u scripts/train_step155_diagnostics_baseline.py --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders
from src.training.diagnostics         import TrainingDiagnostics, format_diagnostics_summary

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--configs", default="")
args = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N_IN = 25088; N_OUT = 10


@dataclass
class Config:
    key: str
    label: str
    N: int
    D: int
    K_hh: int
    K_iter: int
    K_in: int
    alpha_ahebb: float


CONFIGS = [
    Config("A", "A  N=4096 D=64 K4 Ki12 (best)",  4096, 64, 4, 12, 50, 1.0),
    Config("B", "B  N=1024 D=16 K8 Ki8 (eff)",     1024, 16, 8, 8,  25, 1.0),
    Config("C", "C  N=1024 D=32 K4 Ki8 (mid)",     1024, 32, 4, 8,  50, 1.0),
    Config("D", "D  N=256  D=16 K4 Ki4 (minimal)",  256, 16, 4, 4,  25, 1.0),
    Config("E", "E  N=1024 D=16 K8 Ki8 AH=0 (ctl)", 1024, 16, 8, 8, 25, 0.0),
]


_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True,
                                          num_workers=0)
        _loaders = (tr, va)
    return _loaders


def make_model(cfg, seed_offset=0):
    torch.manual_seed(SEED + seed_offset)
    K_random = max(1, cfg.K_hh // 4)
    K_local = cfg.K_hh - K_random
    n_groups = max(8, cfg.N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=cfg.N, N_out=N_OUT, D=cfg.D, N_in=N_IN,
        K_in=cfg.K_in, K_iter=cfg.K_iter,
        K_local=K_local, K_random=K_random, n_groups=n_groups,
        norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=0.5,
        alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=cfg.alpha_ahebb, variant="wpos")


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    print(f"\n{'='*70}")
    print(f"Step 155 — Diagnostics Baseline (Health Check)")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    for c in CONFIGS:
        print(f"  {c.key}  {c.label}")
    print()

    get_loaders()
    results = {}
    out_path = ROOT / "results" / "train_step155_diagnostics_baseline.json"

    cfg_filter = ([k.strip() for k in args.configs.split(",") if k.strip()]
                  if args.configs else [])
    active = [(i, c) for i, c in enumerate(CONFIGS)
              if not cfg_filter or c.key in cfg_filter]

    for i, cfg in active:
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}  params={n_params:,}")
        print(f"{'─'*60}")

        t0 = time.time()
        kw = trainer_kwargs(cfg.N, n_epochs=EPOCHS)
        tr, va = get_loaders()

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=kw.get("lr", 1e-3),
            weight_decay=kw.get("weight_decay", 1e-4))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        # Initialize diagnostics — log every 5 epochs
        diag = TrainingDiagnostics(model, DEVICE, log_every=5)

        history = []
        for ep in range(1, EPOCHS + 1):
            model.train()
            if hasattr(model, 'tick_epoch'):
                model.tick_epoch()

            total_loss = 0.0
            for batch in tr:
                x = batch[0].to(DEVICE)
                soft_labels = batch[1].to(DEVICE)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, soft_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in va:
                    x = batch[0].to(DEVICE)
                    labels = batch[2].to(DEVICE)
                    out = model(x)
                    correct += (out.argmax(1) == labels).sum().item()
                    total += labels.size(0)

            val_top1 = correct / total
            avg_loss = total_loss / len(tr)
            scheduler.step(total_loss)
            history.append({"val_top1": val_top1, "train_loss": avg_loss})

            if ep % 5 == 0 or ep <= 2 or ep == EPOCHS:
                print(f"  e{ep:3d}  task={avg_loss:.4f}  top1={val_top1:.4f}"
                      f"  lr={optimizer.param_groups[0]['lr']:.2e}")

            # Run diagnostics at epoch boundaries (every log_every epochs)
            diag_metrics = diag.log_epoch(ep, model, va, optimizer)
            if diag_metrics:
                history[-1]["diagnostics"] = diag_metrics

        elapsed = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep = int(np.argmax(top1_hist)) + 1

        # Print diagnostics summary
        print(f"\n  top1_best={top1_best:.4f}  params={n_params:,}")
        print(format_diagnostics_summary(diag.history))

        results[cfg.key] = {
            "N": cfg.N, "D": cfg.D, "K_hh": cfg.K_hh,
            "K_iter": cfg.K_iter, "K_in": cfg.K_in,
            "alpha_ahebb": cfg.alpha_ahebb,
            "data_frac": 0.5,
            "top1_best": top1_best, "best_epoch": best_ep,
            "epochs_run": EPOCHS, "n_params": n_params,
            "top1_history": top1_hist,
            "diagnostics_history": diag.history,
            "elapsed_s": round(elapsed, 1),
            "label": cfg.label,
        }

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, default=str))

    # Final comparison
    print(f"\n{'='*70}")
    print(f"STEP 155 — DIAGNOSTICS BASELINE SUMMARY")
    print(f"{'='*70}")
    for k, r in results.items():
        dh = r.get("diagnostics_history", [])
        last_d = dh[-1] if dh else {}
        print(f"  {k}  top1={r['top1_best']:.4f}  "
              f"eff_rank={last_d.get('effective_rank', '?')}/{r['D']}  "
              f"neuron_util={last_d.get('neuron_util_pct', '?')}%  "
              f"wpos_cos={last_d.get('wpos_cos_mean', '?')}  "
              f"sep_ratio={last_d.get('separability_ratio', '?')}")

    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
