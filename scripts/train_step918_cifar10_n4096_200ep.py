"""Step 918: CIFAR-10 N=4096 underfitting probe — 200ep vs 150ep.

MOTIVATION
==========
step909 B_N4096 achieved 82.53% at 150ep (gap = −3.71pp vs Linear 86.24%).
CIFAR-10 has 50K training images — 5.4× more than Imagenette. At 150ep,
relative epoch exposure is lower, raising the question: is the gap architectural
or simply due to underfitting?

This script extends B_N4096 to 200ep to test whether accuracy is still climbing
at ep150 or has plateaued. Result informs the paper's scaling analysis.

CONDITIONAL LAUNCH: after step914 (N=8192) completes.
  - step914 gap ≤1.9pp → capacity axis still closing; 200ep is supporting data.
  - step914 gap ≈3.71pp → architecture bottleneck; 200ep tests underfitting alt.

step914 result: N=8192 = 83.58% (gap = −2.66pp). Capacity axis IS helping.
200ep probe still relevant: if 200ep >> 82.53%, more epochs would be a cheaper
scaling axis than larger N.

CONFIGS
=======
  Ref_150ep   : N=4096, K_in=15, 150ep (sanity rerun, expected ≈82.53%)
  A_N4096_200ep: N=4096, K_in=15, 200ep (underfitting probe)
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
parser.add_argument("--device",   default="auto")
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data",     default="data/store_cifar10.h5")
parser.add_argument("--configs",  default="Ref_150ep,A_N4096_200ep")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

BATCH  = 128
SEED   = args.seed
N_IN   = 25088
N_OUT  = 10
N      = 4096
K_IN   = 15      # N≥4096 default (step926 confirmed NEUTRAL at N=4096)
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step918_cifar10_n4096_200ep_seed{SEED}__{SLOT}.json"

STEP909_N4096_150EP = 0.8253  # step909 B_N4096
LINEAR_BASELINE     = 0.8624  # step882 Linear


def make_model() -> nn.Module:
    torch.manual_seed(SEED)
    K_HH = 2; K_ITER = 5
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=16, N_in=N_IN,
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
        def W_pos(self):   return self.m.W_pos
        @property
        def W_phase(self): return getattr(self.m, "W_phase", None)

        def tick_epoch(self):
            if hasattr(self.m, "tick_epoch"):
                self.m.tick_epoch()

        def forward(self, x):
            Z         = self.m.base._seed(x)
            theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
            conn_hh   = self.m.base.conn_hh
            W_h       = self.m.W_pos[:self.m.base.N_hidden]
            dw = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
            Z_ref = torch.zeros_like(Z)
            for _ in range(self.m.base.K_iter):
                Z_fwd = F.relu(Z - theta_pos)
                Z_nb  = Z_fwd[:, conn_hh, :]
                proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
                Z_nb  = Z_nb * proj_coeff.abs()
                Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
                Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
            return self.m.base._readout(Z)

    return DeltaW()


CONFIGS = {
    "Ref_150ep":    150,
    "A_N4096_200ep": 200,
}


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    pin = (DEVICE.type == "cuda")
    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                          pin_memory=pin)

    print(f"\n{'='*70}")
    print(f"step918 — CIFAR-10 N=4096 underfitting probe  device={DEVICE}  seed={SEED}")
    print(f"  CIFAR-10: train={len(tr.dataset)}  val={len(va.dataset)}")
    print(f"  step909 B_N4096 ref: {STEP909_N4096_150EP:.4f} @ 150ep  (gap={STEP909_N4096_150EP-LINEAR_BASELINE:.4f})")
    print(f"  Question: still climbing at ep150, or plateau?")
    print(f"{'='*70}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}

    for key in keys:
        n_ep  = CONFIGS[key]
        model = make_model()
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'─'*60}")
        print(f"{key}: N={N} K_in={K_IN} epochs={n_ep}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=n_ep)
        t0 = time.time()

        history = Trainer(
            model=model, train_loader=tr, val_loader=va,
            device=DEVICE, **kw,
        ).train(
            n_epochs=n_ep,
            log_fn=lambda m: print(
                f"  e{m['epoch']+1:3d}  loss={m['train_loss']:.4f}  "
                f"top1={m['val_top1']:.4f}  lr={m['lr']:.2e}", flush=True,
            ) if (m['epoch'] + 1) % 10 == 0 else None,
        )

        elapsed = time.time() - t0
        top1h   = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h)
                   for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        gap_vs_linear = best - LINEAR_BASELINE
        delta_vs_150ep = best - STEP909_N4096_150EP

        print(f"  -> best={best:.4f} @ep{best_ep}  gap_vs_linear={gap_vs_linear*100:+.2f}pp  "
              f"delta_vs_150ep={delta_vs_150ep*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "n": N, "k_in": K_IN, "n_params": n_p, "max_epochs": n_ep,
            "best": round(best, 4), "best_ep": best_ep,
            "gap_vs_linear": round(gap_vs_linear, 4),
            "delta_vs_150ep": round(delta_vs_150ep, 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 918 SUMMARY — CIFAR-10 N=4096 underfitting probe")
    print(f"{'='*70}")
    print(f"  Linear baseline:   {LINEAR_BASELINE:.4f}")
    print(f"  step909 @ 150ep:   {STEP909_N4096_150EP:.4f}  (gap={STEP909_N4096_150EP-LINEAR_BASELINE:.4f})")
    for k, r in results.items():
        verdict = (
            "UNDERFIT confirmed — more epochs help" if r["delta_vs_150ep"] >= 0.01
            else "PLATEAU confirmed — gap is architectural" if r["delta_vs_150ep"] <= -0.001
            else "NEUTRAL — 150ep is sufficient"
        )
        print(f"  {k:<18} {r['best']:.4f} @ep{r['best_ep']}  "
              f"Δ150ep={r['delta_vs_150ep']*100:+.2f}pp  {verdict}")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
