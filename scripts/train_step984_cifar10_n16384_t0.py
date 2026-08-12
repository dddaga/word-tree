"""Step 984: CIFAR-10 N=16384 T0 (20ep, 50% data).

MOTIVATION
==========
Extend CIFAR-10 scaling curve to 4th point. Existing: N=2048 (80.57%),
N=4096 (82.53%), N=8192 (83.55%). Power-law predicts ~84.5% at N=16384.
Linear probe ceiling = 86.24%.

K_in=15 (cross-dataset default for N>=4096, confirmed step920/923).
K_hh=2, D=16, K_iter=5 (canonical).

ADVANCE RULE
============
  T0 rejection filter only. If not clearly failing → T1 (step985).
  Expected: ~84-85% (power-law extrapolation).
  RAM: N=16384 D=16 → ~4.2M W_pos params. Needs ~6GB. Mini MPS OK.

Usage:
    python scripts/train_step984_cifar10_n16384_t0.py --device mps
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

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders, make_subset_loader

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_cifar10.h5")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 64
SEED   = args.seed
N = 16384; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 15; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step984_cifar10_n16384_t0_seed{SEED}__{SLOT}.json"


def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def make_model() -> nn.Module:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )

    class DeltaW(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = resonant

        @property
        def W_pos(self): return self.m.W_pos
        @property
        def W_phase(self): return getattr(self.m, "W_phase", None)
        def tick_epoch(self):
            if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

        def forward(self, x):
            Z         = self.m.base._seed(x)
            theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
            conn_hh   = self.m.base.conn_hh
            dw        = _dw_proj(self.m.W_pos, conn_hh)
            Z_ref     = torch.zeros_like(Z)
            for _ in range(K_ITER):
                Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
                Z_nb  = Z_fwd[:, conn_hh, :]
                Z_agg = _dw_agg(Z_nb, dw)
                Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
                Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
            return self.m.base._readout(Z)

    return DeltaW()


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    print(f"\n{'='*70}")
    print(f"step984 — CIFAR-10 N=16384 T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N} D={D} K_in={K_IN} K_hh={K_HH} K_iter={K_ITER}")
    print(f"  Scaling: N=2048→80.57%, N=4096→82.53%, N=8192→83.55%")
    print(f"  Expected: ~84-85% (power-law). Linear ceiling=86.24%")
    print(f"{'='*70}\n")

    tr = make_subset_loader(str(data_path), fraction=0.5, batch_size=BATCH,
                            seed=SEED, pin_memory=False)
    _, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                         pin_memory=False)

    model = make_model()
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params={n_p:,}  train_samples={len(tr.dataset)}")

    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    t0 = time.time()

    history = Trainer(
        model=model, train_loader=tr, val_loader=va,
        device=DEVICE, **kw,
    ).train(
        n_epochs=EPOCHS,
        log_fn=lambda m: print(
            f"  e{m['epoch']+1:3d}  loss={m['train_loss']:.4f}  "
            f"top1={m['val_top1']:.4f}  lr={m['lr']:.2e}", flush=True,
        ),
    )

    elapsed = time.time() - t0
    top1h   = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h)
               for h in history]
    best, best_ep = max(top1h), int(np.argmax(top1h)) + 1

    result = {
        "step": "step984",
        "config": f"N={N} D={D} K_in={K_IN} K_hh={K_HH} K_iter={K_ITER}",
        "n_params": n_p,
        "train_samples": len(tr.dataset),
        "best": round(best, 4),
        "best_ep": best_ep,
        "elapsed_s": round(elapsed, 1),
        "scaling_context": {
            "N2048": 80.57, "N4096": 82.53, "N8192": 83.55,
            "linear_ceiling": 86.24,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))

    print(f"\n{'='*70}")
    print(f"RESULT: best={best:.4f} @ep{best_ep}  elapsed={elapsed:.0f}s")
    print(f"Scaling: 2K=80.57  4K=82.53  8K=83.55  16K={best:.2f}")
    if best > 83.55:
        print(f"  → ADVANCE to T1 (+{best-83.55:.2f}pp vs N=8192)")
    else:
        print(f"  → WARNING: no gain vs N=8192 ({best-83.55:+.2f}pp)")
    print(f"→ {OUT_PATH}")


if __name__ == "__main__":
    main()
