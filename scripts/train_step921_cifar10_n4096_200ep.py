"""Step 921: CIFAR-10 N=4096 Extended Training (200ep, 100% data).

MOTIVATION
==========
step909 ran N=4096 T2 (150ep, 100% data) and got 82.53% best @ep124.
Best was near the end — the model may not have converged.

CIFAR-10 has 50K training samples (5.2× Imagenette's 9,469).
At 150ep with N=4096, each parameter sees ~2.2M samples
(vs Imagenette's ~1.4M at 150ep). Additional 50ep → ~2.9M.

QUESTION: was ep124 in step909 a true plateau or was the model still
learning? If 200ep improves >1pp over 150ep: CIFAR-10 at N=4096 is
epoch-limited (underfitting), and additional training budget closes gap.

CONTEXT
=======
N-scaling curve so far:
  N=2048  (K_in=25, 150ep): 80.69% vs Linear 86.24% (Δ=−5.55pp) [step882]
  N=4096  (K_in=15, 150ep): 82.53% vs Linear 86.15% (Δ=−3.71pp) [step909]
  N=8192  (K_in=15, 150ep): still running [step914, ep130=83.58%]

ADVANCE RULE
============
  >+1.0pp vs step909: underfitting confirmed — add 200ep as default for CIFAR-10
  +0.5–1.0pp: mild benefit — mention as "additional training budget" in paper
  <+0.5pp: plateau already reached at 150ep; architectural gap, not underfitting
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
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=200)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_cifar10.h5")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = args.seed
N = 4096; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 15; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step921_cifar10_n4096_200ep_seed{SEED}__{SLOT}.json"

STEP909_REF = 0.8253   # N=4096, K_in=15, 150ep — step909 5060ti_cuda


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

    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                          pin_memory=(DEVICE.type == "cuda"))

    print(f"\n{'='*70}")
    print(f"step921 — CIFAR-10 N=4096 Extended Training (200ep, 100% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N} D={D} K_hh={K_HH} K_in={K_IN} K_iter={K_ITER} α={ALPHA_REFLECT}")
    print(f"  Reference: step909 N=4096 150ep → 82.53% best @ep124")
    print(f"  Question: was ep124 a plateau or still improving?")
    print(f"{'='*70}\n")

    model = make_model()
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_p:,}")

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
    delta_vs_909  = best - STEP909_REF

    print(f"\n{'='*70}")
    print(f"STEP 921 SUMMARY — CIFAR-10 N=4096 Extended 200ep")
    print(f"{'='*70}")
    print(f"  best={best:.4f} @ep{best_ep}  vs step909(150ep)={STEP909_REF:.4f}  Δ={delta_vs_909*100:+.2f}pp")
    if delta_vs_909 >= 0.01:
        v = "UNDERFIT_CONFIRMED" if delta_vs_909 >= 0.01 else "MILD_GAIN"
        print(f"  → {v}: 200ep improves CIFAR-10 N=4096")
    else:
        print(f"  → PLATEAU: 150ep already converged; gap is architectural, not epoch-limited")

    result = {
        "n_params": n_p,
        "best": round(best, 4), "best_ep": best_ep,
        "step909_ref_150ep": STEP909_REF,
        "delta_vs_step909": round(delta_vs_909, 4),
        "elapsed_s": round(elapsed),
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
