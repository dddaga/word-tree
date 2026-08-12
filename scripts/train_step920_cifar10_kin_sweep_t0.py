"""Step 920: CIFAR-10 K_in Sweep T0 (20ep, 50% data).

MOTIVATION
==========
K_in=25 is the canonical SGNNET default for N=2048, tuned on Imagenette.
Imagenette sweet-spot is K_in=20 (step633 curve); K_in=25 is above the knee.

CIFAR-10 has harder, more fine-grained classes (10 similar objects vs
10 very distinct Imagenette categories). Hypothesis: denser input sampling
(K_in=50 or K_in=100) may help SGNNET nodes receive more discriminative
input features per class.

Prior constraints:
  - step631 Imagenette: K_in=10 fails (−1.17pp T1), K_in=15 is minimum viable
  - step632 Imagenette: K_in=15 T2=−0.33pp vs K_in=25 (confirmed)
  - N=2048 with K_in=100: each node sees 4× more inputs; memory ~fine

CONFIGS (T0, 20ep, 50% data, seed=42, CIFAR-10)
  A_k15     K_in=15  (Imagenette T2 sweet-spot — how does it do on CIFAR-10?)
  Ref_k25   K_in=25  (canonical CIFAR-10 default)
  B_k50     K_in=50  (2× density — tests if harder classes need more coverage)
  C_k100    K_in=100 (4× density — upper bound)

ADVANCE RULE
============
  Any config ≥+0.5pp vs Ref_k25 → advance to T1 (step921 or reuse slot)
  All neutral → K_in=25 is cross-dataset robust
  A_k15 advances → simplify: K_in=15 suffices on both datasets (efficiency win)

CONTEXT
=======
step882 (K_in=25, N=2048, T2): 80.69% vs Linear 86.24% (−5.55pp)
step909 (K_in=15, N=4096, T2): 82.53% vs Linear 86.15% (−3.71pp)
step920 question: can higher K_in close the gap at fixed N=2048?
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
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_cifar10.h5")
parser.add_argument("--configs", default="A_k15,Ref_k25,B_k50,C_k100")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step920_cifar10_kin_sweep_t0_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "A_k15":   15,
    "Ref_k25": 25,
    "B_k50":   50,
    "C_k100": 100,
}


def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def make_model(k_in: int) -> nn.Module:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=k_in, K_iter=K_ITER, K_local=K_l, K_random=K_r,
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

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=False)
    n_full  = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[: n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=10, pin_memory=False,
    )

    print(f"\n{'='*70}")
    print(f"step920 — CIFAR-10 K_in Sweep T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N} D={D} K_hh={K_HH} K_iter={K_ITER} α_reflect={ALPHA_REFLECT}")
    print(f"  Context: step882 K_in=25→80.69% | step633 Imagenette knee=K_in=20")
    print(f"  Question: does higher K_in help on harder CIFAR-10 classes?")
    print(f"{'='*70}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}
    ref_acc = None

    for key in keys:
        k_in  = CONFIGS[key]
        model = make_model(k_in)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: K_in={k_in}  params={n_p:,}")

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

        if key == "Ref_k25":
            ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else 0.0)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "k_in": k_in, "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 920 SUMMARY — CIFAR-10 K_in Sweep T0")
    print(f"{'='*70}")
    print(f"  Imagenette reference: K_in=15 T2=−0.33pp, K_in=20 knee (step633)")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "(baseline)"
        if k == "Ref_k25":
            v = "(baseline)"
        elif d is None:
            v = "—"
        elif d >= 0.005:
            v = "ADVANCE→T1"
        elif d >= -0.005:
            v = "NEUTRAL"
        elif d >= -0.05:
            v = "HURT(mild)"
        else:
            v = "KILL"
        print(f"  {k:<10} K_in={r['k_in']:>3}  {r['best']:>7.4f} {dstr:>10}  {v}")

    if ref_acc is not None:
        winners = [(k, r) for k, r in results.items()
                   if k != "Ref_k25" and r["delta_vs_ref"] is not None
                   and r["delta_vs_ref"] >= 0.005]
        if winners:
            bk, br = max(winners, key=lambda x: x[1]["delta_vs_ref"])
            print(f"\n  → ADVANCE: {bk} (K_in={br['k_in']}) → T1")
        else:
            print(f"\n  → K_in=25 is CIFAR-10 optimal (no benefit from denser input sampling)")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
