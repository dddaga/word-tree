"""Step 922: CIFAR-10 N=8192 Multi-Seed Validation T2.

MOTIVATION
==========
step914 established N=8192 → 83.58% (gap −2.66pp vs Linear), seed=42 only.
Paper requires mean±std for the N-scaling curve claim.

N-scaling curve so far (all seed=42):
  N=2048  (K_in=25, 150ep): 80.69%  gap=−5.55pp  [step882]
  N=4096  (K_in=15, 150ep): 82.53%  gap=−3.71pp  [step909]
  N=8192  (K_in=15, 150ep): 83.58%  gap=−2.66pp  [step914]

step887 showed Imagenette variance is ±0.18pp (3 seeds). Assuming similar
variance on CIFAR-10, N=8192 mean ≈ 83.3–83.8%.

CONFIGS (seeds 0 and 1 — seed 42 already done in step914)
  seed0: N=8192, K_in=15, 150ep, 100% data
  seed1: N=8192, K_in=15, 150ep, 100% data

PAPER CLAIM (expected)
  N=8192: mean≈83.3–83.6% ± ~0.2pp (gap ~−2.7pp vs Linear)
  "SGNNET scales with N: 80.69 → 82.53 → 83.6%, closing ~1.4pp per 2×N"
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
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--seeds",   default="0,1")
parser.add_argument("--data",    default="data/store_cifar10.h5")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
N = 8192; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 15; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step922_cifar10_n8192_multiseed__{SLOT}.json"

STEP914_SEED42 = 0.8358   # step914 reference
LINEAR_REF     = 0.8624   # step882 linear anchor


def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def make_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
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

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print(f"\n{'='*70}")
    print(f"step922 — CIFAR-10 N=8192 Multi-Seed T2 (150ep, 100% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seeds={seeds}")
    print(f"  N={N} D={D} K_hh={K_HH} K_in={K_IN} K_iter={K_ITER} α={ALPHA_REFLECT}")
    print(f"  Reference: step914 seed42=83.58% (gap −2.66pp vs Linear)")
    print(f"  Goal: mean±std for paper N-scaling table")
    print(f"{'='*70}\n")

    results = {}
    for seed in seeds:
        tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=seed,
                              pin_memory=(DEVICE.type == "cuda"))
        model = make_model(seed)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"seed={seed}  params={n_p:,}")

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
        print(f"  -> best={best:.4f} @ep{best_ep}  {elapsed:.0f}s")

        results[f"seed{seed}"] = {
            "seed": seed, "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 922 SUMMARY — CIFAR-10 N=8192 Multi-Seed")
    print(f"{'='*70}")
    accs = [r["best"] for r in results.values()]
    all_accs = accs + [STEP914_SEED42]
    mean_acc = float(np.mean(all_accs))
    std_acc  = float(np.std(all_accs))
    print(f"  seed42 (step914): {STEP914_SEED42:.4f}")
    for k, r in results.items():
        print(f"  {k}: {r['best']:.4f} @ep{r['best_ep']}")
    print(f"\n  N=8192 mean={mean_acc:.4f} ± {std_acc:.4f}pp (3 seeds)")
    print(f"  gap vs Linear ({LINEAR_REF:.4f}): {(mean_acc - LINEAR_REF)*100:+.2f}pp")
    print(f"\n  N-scaling: N2048=80.69% → N4096=82.53% → N8192={mean_acc:.2%}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
