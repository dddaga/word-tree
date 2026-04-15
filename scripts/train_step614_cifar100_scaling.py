"""Step 614: CIFAR-100 N-scaling — find where SGNNET and MLP cross 95% (or their ceiling).

MOTIVATION
==========
step601 showed SGNNET_DeltaProj=37.81% and MLP_256=67.49% at 100 classes.
Neither crossed 95%. This script finds the scaling ceiling for both:
  - SGNNET: scale N from 2048 → 4096 → 8192 (then 16384 if CUDA available)
  - MLP:    scale hidden from 256 → 512 → 1024 → 2048 → 4096

Goal: characterise scaling efficiency on a hard task (100 classes).
      Report the accuracy-vs-params Pareto for both model families.
      Expected outcome: neither crosses 95% (VGG16 CIFAR-100 feature ceiling ≈70-75%),
      but the scaling behaviour is the paper finding.

CONFIGS (CIFAR-100 VGG16 features, N_IN=25088, N_OUT=100, 50ep full data)
  SGNNET_N2048  : N=2048  D=16 K_hh=2 K_iter=5 ΔW proj  (baseline = 37.81% from step601)
  SGNNET_N4096  : N=4096  D=16 K_hh=2 K_iter=5 ΔW proj
  SGNNET_N8192  : N=8192  D=16 K_hh=2 K_iter=5 ΔW proj
  MLP_256       : hidden=256   (baseline = 67.49% from step601)
  MLP_512       : hidden=512
  MLP_1024      : hidden=1024
  MLP_2048      : hidden=2048
  MLP_4096      : hidden=4096

To run:
    python -u scripts/train_step614_cifar100_scaling.py --device mps --epochs 50
    # For SGNNET only (faster):
    python -u scripts/train_step614_cifar100_scaling.py --device cuda --configs SGNNET_N4096,SGNNET_N8192
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
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=50)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_cifar100.h5")
parser.add_argument("--configs", default="SGNNET_N2048,SGNNET_N4096,SGNNET_N8192,MLP_256,MLP_512,MLP_1024,MLP_2048,MLP_4096")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N_IN = 25088; N_OUT = 100
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step614_cifar100_scaling_seed{SEED}__{SLOT}.json"


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class MLPBaseline(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN, hidden), nn.ReLU(),
            nn.Linear(hidden, N_OUT))
    def forward(self, x): return self.net(x)


class SGNNET_DeltaProj(nn.Module):
    """ΔW projection routing, variable N."""
    def __init__(self, N):
        super().__init__()
        torch.manual_seed(SEED)
        K_r = max(1, K_HH // 4); K_l = K_HH - K_r
        self.N = N
        self.base = SGNNET_SmallWorld(
            N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
            K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
            n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
        self.resonant = SGNNET_Resonant(
            self.base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0)
        self.alpha_reflect = ALPHA_REFLECT

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase
    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"): self.resonant.tick_epoch()

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh; N_h = self.base.N_hidden
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h = self.base.W_pos[:N_h]
        delta_w = W_h.unsqueeze(1) - W_h[conn_hh]
        dw_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)
        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            proj = (Z_nb * dw_norm).sum(-1, keepdim=True)
            Z_nb = Z_nb * proj.abs()
            Z_struct = Z_nb.sum(dim=2)
            Z_reflected = self.alpha_reflect * Z_reflected + (Z_fwd - Z)
            Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)
        return self.base._readout(Z)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_mlp(model, tr, va, epochs):
    model = model.to(DEVICE)
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    crit = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        for x, _, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, _, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(-1) == y).sum().item()
                total += y.numel()
        v = correct / max(total, 1)
        history.append({"epoch": epoch, "val_top1": v})
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  ep{epoch+1:3d}  val={v:.4f}", flush=True)
    return history


def train_sgnnet(model, tr, va, epochs):
    N = model.N
    model = model.to(DEVICE)
    kw = trainer_kwargs(N, n_epochs=epochs)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
    return trainer.train(n_epochs=epochs, log_fn=lambda m: (
        print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
        if (m['epoch']+1) % 10 == 0 else None))


# Config table
SGNNET_N_MAP = {
    "SGNNET_N2048": 2048,
    "SGNNET_N4096": 4096,
    "SGNNET_N8192": 8192,
}
MLP_H_MAP = {
    "MLP_256":  256,
    "MLP_512":  512,
    "MLP_1024": 1024,
    "MLP_2048": 2048,
    "MLP_4096": 4096,
}


def main():
    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run extract_cifar100_vgg16_features.py first.")
        sys.exit(1)

    print(f"Step 614 — CIFAR-100 N-scaling (SGNNET) + hidden-scaling (MLP)")
    print(f"  Goal: find scaling ceiling for both models at 100 classes")
    print(f"  data={args.data}  seed={SEED}  epochs={EPOCHS}  device={DEVICE}")

    tr, va = make_loaders(data_path, batch_size=BATCH, seed=SEED)
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}\n")

    results = {}
    for key in keys:
        if key in SGNNET_N_MAP:
            N = SGNNET_N_MAP[key]
            model = SGNNET_DeltaProj(N)
            is_sgnnet = True
            routing_macs = K_ITER * N * K_HH * D * 2
            flops = routing_macs
        elif key in MLP_H_MAP:
            h = MLP_H_MAP[key]
            model = MLPBaseline(h)
            is_sgnnet = False
            flops = 2 * h * (N_IN + N_OUT)
        else:
            print(f"  skip unknown config: {key}"); continue

        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'─'*60}")
        print(f"Config {key}  params={n_p:,}  flops≈{flops/1e6:.2f}M")
        print(f"{'─'*60}")

        t0 = time.time()
        history = train_sgnnet(model, tr, va, EPOCHS) if is_sgnnet else train_mlp(model, tr, va, EPOCHS)
        top1h = [round(h.get("val_top1", h.get("epoch", 0.0)), 4) if isinstance(h, dict) else 0.0
                 for h in history]
        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, bep = max(top1h), int(np.argmax(top1h)) + 1
        elapsed = time.time() - t0
        print(f"  → best={best:.4f} @ep{bep}  elapsed={elapsed:.0f}s")

        results[key] = {
            "config": key, "n_params": n_p, "flops_routing_macs": flops,
            "top1_best": best, "best_epoch": bep,
            "elapsed_s": round(elapsed, 1), "epochs": EPOCHS,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    print(f"STEP 614 SUMMARY — CIFAR-100 scaling (step601 baselines: SGNNET_N2048=37.81%, MLP_256=67.49%)")
    print(f"{'='*60}")
    print(f"  {'Config':<18} {'params':>10} {'FLOPs(M)':>10} {'best top1':>10}")
    sgnnet_rows = [(k, r) for k, r in results.items() if k.startswith("SGNNET")]
    mlp_rows    = [(k, r) for k, r in results.items() if k.startswith("MLP")]
    for k, r in sorted(sgnnet_rows, key=lambda x: x[1]['n_params']):
        print(f"  {k:<18} {r['n_params']:>10,} {r['flops_routing_macs']/1e6:>10.2f} {r['top1_best']:>10.4f}")
    print()
    for k, r in sorted(mlp_rows, key=lambda x: x[1]['n_params']):
        print(f"  {k:<18} {r['n_params']:>10,} {r['flops_routing_macs']/1e6:>10.2f} {r['top1_best']:>10.4f}")


if __name__ == "__main__":
    main()
