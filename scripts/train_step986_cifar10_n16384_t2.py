"""Step 986: CIFAR-10 N=16384 T2 (150ep, 100% data).

T1 result (step986 T1): 82.85% @ep69 — non-monotonic vs N=8192 T2=83.55%.
T2 tests whether T1 underfit (75ep × 50% data) or ceiling confirmed.
Paper scaling section: N={2048,4096,8192} T2 + N=16384 T2 (this run).

Advance: N=16384 T2 >= 83.55% → scaling continues; else ceiling confirmed at N=8192.

Usage:
    python scripts/train_step986_cifar10_n16384_t2.py --device cuda
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
from src.sgnnet.model_resonant      import SGNNET_Resonant  # SGNNET_Resonant_CUDA
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
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
OUT_PATH = ROOT / "results" / f"train_step986_cifar10_n16384_t2_seed{SEED}__{SLOT}.json"


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
    print(f"step986 — CIFAR-10 N=16384 T2 (150ep, 100% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}  batch={BATCH}")
    print(f"  N={N} D={D} K_in={K_IN} K_hh={K_HH} K_iter={K_ITER}")
    print(f"  step986 T1: best=82.85% @ep69 (non-monotonic at T1 scale)")
    print(f"  N-scaling T2: N2048=80.57  N4096=82.53  N8192=83.55")
    print(f"  Advance: >= 83.55pp → scaling extends; else ceiling at N=8192")
    print(f"{'='*70}\n")

    pin = (DEVICE.type == "cuda")
    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                          pin_memory=pin)  # pin_memory=True, non_blocking=True for CUDA overlap

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
        "step": "step986",
        "tier": "T2",
        "config": f"N={N} D={D} K_in={K_IN} K_hh={K_HH} K_iter={K_ITER}",
        "epochs": EPOCHS,
        "n_params": n_p,
        "train_samples": len(tr.dataset),
        "best": round(best, 4),
        "best_ep": best_ep,
        "elapsed_s": round(elapsed, 1),
        "t1_ref": 0.8285,
        "scaling_context": {
            "N2048_T2": 80.57, "N4096_T2": 82.53, "N8192_T2": 83.55,
            "N16384_T1": 82.85, "linear_ceiling": 86.24,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))

    N8192 = 0.8355
    print(f"\n{'='*70}")
    print(f"RESULT: best={best:.4f} @ep{best_ep}  elapsed={elapsed:.0f}s")
    print(f"Scaling T2: 2K=80.57  4K=82.53  8K=83.55  16K(T2)={best*100:.2f}")
    if best >= N8192:
        print(f"  → SCALING EXTENDS: N=16384 >= N=8192 ({best*100:.2f}% >= 83.55%) → paper curve extended")
    else:
        print(f"  → CEILING CONFIRMED: N=16384={best*100:.2f}% < N=8192=83.55% → N=8192 is sweet spot")
    print(f"→ {OUT_PATH}")


if __name__ == "__main__":
    main()
