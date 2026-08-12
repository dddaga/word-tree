"""Step 909: CIFAR-10 N-scaling — close the cross-dataset gap.

# CUDA-5060ti-validated

MOTIVATION
==========
Step882 T2 showed SGNNET (N=2048, 34,976 params) at 80.69% vs Linear 86.24%
on CIFAR-10 — a -5.55pp gap. On Imagenette, the same model achieves 95.95% vs
97.7% (−1.75pp). The cross-dataset gap is 3× larger.

HYPOTHESIS: gap is a capacity issue — N=2048 is a bottleneck when processing
50,000 CIFAR-10 training samples (5.4× more than Imagenette 9,296).
N=4096 doubles node count for same params, giving the routing graph more
representational capacity.

H1: N=4096 closes gap by ≥2pp vs N=2048 (capacity bottleneck)
H2: Extended training (200ep) closes gap by ≥1pp (SGNNET still learning at ep150)

Configs:
  Ref_linear   : nn.Linear(25088→10) — 250,890 params — upper bound
  A_N2048_T3   : canonical champion (N=2048, K_in=25), 200ep — extend step882
  B_N4096      : N=4096, K_in=15, 150ep — capacity scaling test

Success: gap <3.5pp (vs 5.55pp in step882) → N-scaling helps, paper claimable
Failure: gap ≥5pp → gap is fundamental to SGNNET's inductive bias on CIFAR-10

Dataset: CIFAR-10 VGG16 pool5 features (25088-dim), 50K train / 10K val.
Machine: 5060ti_cuda — 5060ti has store_cifar10.h5 confirmed.
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
parser.add_argument("--configs", default="Ref_linear,A_N2048_T3,B_N4096")
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

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step909_cifar10_n_scaling_seed{SEED}__{SLOT}.json"

# Step882 baselines (T2=150ep) for Δ tracking
STEP882 = {"Ref_linear": 0.8624, "A_sgnnet_N2048": 0.8069}


class LinearProbe(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.W_pos = self.fc.weight

    @property
    def W_phase(self): return None

    def tick_epoch(self): pass

    def forward(self, x):
        return self.fc(x)


def make_sgnnet(N: int, K_in: int) -> nn.Module:
    torch.manual_seed(SEED)
    K_HH = 2; K_ITER = 5
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=16, N_in=N_IN,
        K_in=K_in, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)

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
            conn_hh = self.m.base.conn_hh
            W_h = self.m.W_pos[:self.m.base.N_hidden]
            dw = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
            Z_ref = torch.zeros_like(Z)
            for _ in range(self.m.base.K_iter):
                Z_fwd = F.relu(Z - theta_pos)
                Z_nb = Z_fwd[:, conn_hh, :]
                proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
                Z_nb = Z_nb * proj_coeff.abs()
                Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
                Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
            return self.m.base._readout(Z)

    return DeltaW()


CONFIGS = {
    "Ref_linear": ("linear", None, None,
                   "nn.Linear(25088→10) — 250,890 params (upper bound)"),
    "A_N2048_T3": ("sgnnet", 2048, 25,
                   "N=2048, K_in=25, T3=200ep — extend step882 (was 150ep→80.69%)"),
    "B_N4096":    ("sgnnet", 4096, 15,
                   "N=4096, K_in=15 — capacity scaling (5.4× CIFAR vs Imagenette)"),
}


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    pin = (DEVICE.type == "cuda")
    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                          pin_memory=pin)

    print(f"\n{'='*70}")
    print(f"step909 — CIFAR-10 N-scaling T3  device={DEVICE}  seed={SEED}")
    print(f"  CIFAR-10: train={len(tr.dataset)}  val={len(va.dataset)}")
    print(f"  step882 baseline: linear=86.24%, SGNNET_N2048=80.69% (gap=-5.55pp)")
    print(f"{'='*70}\n")
    print(f"  {'config':<14} {'N':>5}  {'K_in':>5}  desc")
    print(f"  {'-'*65}")
    for k, (kind, N, K_in, desc) in CONFIGS.items():
        Nstr = str(N) if N else "25088"
        Kstr = str(K_in) if K_in else "  n/a"
        print(f"  {k:<14} {Nstr:>5}  {Kstr:>5}  {desc}")

    keys = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}
    ref_acc = STEP882["Ref_linear"]   # anchor vs step882 linear

    for key in keys:
        kind, N, K_in, desc = CONFIGS[key]
        if kind == "linear":
            torch.manual_seed(SEED)
            model = LinearProbe(N_IN, N_OUT)
        else:
            model = make_sgnnet(N, K_in)

        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_ep = 150 if key == "B_N4096" else EPOCHS   # N4096 capped at 150ep
        print(f"\n{'─'*60}\n{key}: {desc}")
        print(f"  params={n_p:,}  epochs={n_ep}")

        kw = trainer_kwargs(N if kind == "sgnnet" else N_IN, n_epochs=n_ep)
        t0 = time.time()
        history = Trainer(
            model=model, train_loader=tr, val_loader=va,
            device=DEVICE, **kw
        ).train(
            n_epochs=n_ep,
            log_fn=lambda m: print(
                f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True
            ) if (m['epoch'] + 1) % 25 == 0 else None,
        )
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h)
                 for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1

        delta_linear = best - ref_acc
        s882 = STEP882.get("A_sgnnet_N2048") if kind == "sgnnet" else None
        delta_s882 = (best - s882) if s882 else None

        s882_str = f"  Δ_vs_step882={delta_s882*100:+.2f}pp" if delta_s882 is not None else ""
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_linear={delta_linear*100:+.2f}pp{s882_str}  {elapsed:.0f}s")

        results[key] = {
            "kind": kind, "desc": desc, "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_linear_s882": round(delta_linear, 4),
            "delta_vs_sgnnet_s882": round(delta_s882, 4) if delta_s882 is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 909 SUMMARY — CIFAR-10 N-scaling")
    print(f"{'='*70}")
    print(f"  {'config':<14} {'params':>10}  {'best':>7}  {'Δ_linear':>10}  {'Δ_s882_N2048':>14}")
    for k, r in results.items():
        d_s = f"{r['delta_vs_sgnnet_s882']*100:>+12.2f}pp" if r["delta_vs_sgnnet_s882"] is not None else "            n/a"
        print(f"  {k:<14} {r['n_params']:>10,}  {r['best']:>7.4f}  "
              f"{r['delta_vs_linear_s882']*100:>+9.2f}pp  {d_s}")

    print(f"\n  step882 reference:  linear=86.24%, SGNNET_N2048=80.69%, gap=-5.55pp")
    sgnnet_keys = [k for k, r in results.items() if r["kind"] == "sgnnet"]
    if sgnnet_keys and "Ref_linear" in results:
        lin_best = results["Ref_linear"]["best"]
        for k in sgnnet_keys:
            best = results[k]["best"]
            gap = (best - lin_best) * 100
            s882_gap = results[k].get("delta_vs_sgnnet_s882") or 0
            if s882_gap * 100 >= 2.0:
                verdict = f"H1 CONFIRMED: N-scaling closes gap by {s882_gap*100:+.1f}pp"
            elif s882_gap * 100 >= 0.5:
                verdict = f"H1 PARTIAL: modest improvement {s882_gap*100:+.1f}pp"
            else:
                verdict = "H1 REJECTED: N-scaling does not close gap"
            print(f"\n  {k}: gap={gap:+.2f}pp vs linear → {verdict}")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
