"""Step 914: CIFAR-10 N=8192 capacity scaling T2 (150ep, 100% data).

MOTIVATION
==========
step909 T2 established the N-scaling curve on CIFAR-10 (VGG16 pool5 features):
  Ref_linear  (250,890p): 86.15% (step882 anchor: 86.24%)
  A_N2048_T3  ( 34,976p): 80.69% → gap −5.55pp (step882 at 150ep)
  B_N4096     ( 69,792p): 82.53% → gap −3.71pp, +1.84pp vs N2048 (step909 T2)

H1 status (step909): N=4096 closes gap +1.84pp — just under ≥2pp threshold.

This step extends the curve to N=8192:
  C_N8192     (~139,584p): target — closes gap further toward Linear.
  D_N4096_ref :           same as step909 B_N4096, for same-run cross-check.

3-point curve: N2048 → N4096 → N8192.
If N8192 gap < 2.5pp: capacity scaling is the dominant factor in CIFAR-10 gap.
If N8192 ≈ N4096 plateau: gap is fundamental (routing bottleneck, not capacity).

Paper contribution: empirical scaling curve for SGNNET on heterogeneous data.

ADVANCE RULE
============
  C_N8192 gap < 3.0pp (< −3.71pp is a no-go): scaling still helps → claim capacity
  C_N8192 gap 3.0–3.71pp: diminishing returns → paper note
  C_N8192 gap > 3.71pp: scaling reverses → architecture bottleneck confirmed

NOTE: 5060ti_cuda only. N=8192 needs ~18GB VRAM estimate. If OOM, fall back to
--device cpu (which the 5060ti also has 5060ti_cpu slot).
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
parser.add_argument("--configs", default="C_N8192")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = args.seed
N_IN   = 25088
N_OUT  = 10
ALPHA_REFLECT = 0.5
K_HH   = 2
K_ITER = 5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step914_cifar10_n8192_t2_seed{SEED}__{SLOT}.json"

# Anchor from step882/step909
STEP882_LINEAR  = 0.8624
STEP882_N2048   = 0.8069
STEP909_N4096   = 0.8253


def make_sgnnet(N: int, K_in: int) -> nn.Module:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4)
    K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=16, N_in=N_IN,
        K_in=K_in, K_iter=K_ITER, K_local=K_l, K_random=K_r,
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
            Z = self.m.base._seed(x)
            theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
            conn_hh   = self.m.base.conn_hh
            W_h       = self.m.W_pos[:self.m.base.N_hidden]
            dw        = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
            Z_ref     = torch.zeros_like(Z)
            for _ in range(self.m.base.K_iter):
                Z_fwd = F.relu(Z - theta_pos)
                Z_nb  = Z_fwd[:, conn_hh, :]
                proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
                Z_nb  = Z_nb * proj_coeff.abs()
                Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
                Z     = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
            return self.m.base._readout(Z)

    return DeltaW()


CONFIGS = {
    "C_N8192":    (8192, 15, "N=8192, K_in=15 — extend N-scaling curve"),
    "D_N4096_ref":(4096, 15, "N=4096, K_in=15 — same-run cross-check vs step909"),
}


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    pin = (DEVICE.type == "cuda")
    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                          pin_memory=pin)  # pin_memory=True, non_blocking=True for CUDA overlap

    print(f"\n{'='*70}")
    print(f"step914 — CIFAR-10 N=8192 scaling T2  device={DEVICE}  seed={SEED}")
    print(f"  N-scaling curve: N2048=80.69% → N4096=82.53% → N8192=?")
    print(f"  Linear anchor: 86.15% (step909) / 86.24% (step882)")
    print(f"  Success: gap < 3.0pp → capacity scaling dominant")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}

    for key in keys:
        N, K_in, desc = CONFIGS[key]
        model = make_sgnnet(N, K_in)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'─'*60}")
        print(f"{key}: {desc}")
        print(f"  params={n_p:,}  epochs={EPOCHS}  N={N}  K_in={K_in}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()
        history = Trainer(
            model=model, train_loader=tr, val_loader=va,
            device=DEVICE, **kw,
        ).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(
                f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}  "
                f"loss={m['train_loss']:.4f}", flush=True,
            ) if (m['epoch'] + 1) % 10 == 0 else None,
        )
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h)
                 for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1

        gap_vs_linear = best - STEP882_LINEAR
        gain_vs_n2048 = best - STEP882_N2048
        gain_vs_n4096 = best - STEP909_N4096

        print(f"  -> best={best:.4f} @ep{best_ep}  "
              f"gap_vs_linear={gap_vs_linear*100:+.2f}pp  "
              f"Δ_vs_N2048={gain_vs_n2048*100:+.2f}pp  "
              f"Δ_vs_N4096={gain_vs_n4096*100:+.2f}pp  "
              f"{elapsed:.0f}s")

        results[key] = {
            "desc": desc, "N": N, "K_in": K_in, "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "gap_vs_linear": round(gap_vs_linear, 4),
            "gain_vs_N2048": round(gain_vs_n2048, 4),
            "gain_vs_N4096": round(gain_vs_n4096, 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 914 SUMMARY — CIFAR-10 N=8192 T2")
    print(f"  N-scaling curve (step882/909/914):")
    print(f"  {'N':>6}  {'params':>8}  {'acc':>7}  {'gap_vs_linear':>14}  verdict")
    print(f"  {'2048':>6}  {'34,976':>8}  {'80.69%':>7}  {'−5.55pp':>14}  (step882 ref)")
    print(f"  {'4096':>6}  {'69,792':>8}  {'82.53%':>7}  {'−3.71pp':>14}  (step909)")
    for k, r in results.items():
        gap_str = f"{r['gap_vs_linear']*100:+.2f}pp"
        if r["gap_vs_linear"] > -0.030:
            v = "STRONG_SCALING"
        elif r["gap_vs_linear"] > -0.040:
            v = "CONTINUES"
        elif r["gap_vs_linear"] > STEP909_N4096 - STEP882_LINEAR:
            v = "MARGINAL"
        else:
            v = "PLATEAU"
        print(f"  {r['N']:>6}  {r['n_params']:>8,}  {r['best']:>7.4f}  {gap_str:>14}  {v}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
