"""Step 919: Imagenette K_iter Sweep T0 (20ep, 50% data).

MOTIVATION
==========
step916 showed CIFAR-10 collapses catastrophically at K_iter>5:
  K_iter=10: −15.54pp, K_iter=15: −59.97pp (near random)

K_iter=5 was chosen as the Imagenette canonical WITHOUT testing higher values.
If Imagenette also collapses at K_iter>5: over-smoothing is a universal SGNNET property.
If Imagenette is robust: CIFAR-10 collapse is dataset-specific (feature separability).

Either result is paper-worthy:
  Universal: "K_iter=5 is the sweet-spot; over-smoothing confirmed on both datasets"
  Dataset-specific: "CIFAR-10 features less separable → MP collapse earlier"

CONFIGS (T0, 20ep, 50% data, seed=42, Imagenette)
  Ref_k5   K_iter=5  (canonical)
  A_k3     K_iter=3
  B_k8     K_iter=8  (moderate increase — tests onset of collapse)
  C_k10    K_iter=10 (CIFAR-10 showed −15.54pp here)
  D_k15    K_iter=15 (CIFAR-10 showed −59.97pp here)

ADVANCE RULE
============
  B_k8/C_k10 ≥+0.5pp → K_iter tuning helps Imagenette (advance to T1)
  B_k8/C_k10 neutral → K_iter=5 is Imagenette sweet-spot too
  B_k8/C_k10 collapse → universal over-smoothing (strong paper claim)

CONTEXT
=======
step199 (K_iter=5): 95.52% Imagenette T2
step916: K_iter sweep CIFAR-10 — K_iter=5 only survivor
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
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref_k5,A_k3,B_k8,C_k10,D_k15")
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
D = 16; K_HH = 2; K_IN = 25
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step919_imagenette_kiter_sweep_t0_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Ref_k5": 5,
    "A_k3":   3,
    "B_k8":   8,
    "C_k10": 10,
    "D_k15": 15,
}


def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def make_model(k_iter: int) -> nn.Module:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=k_iter, K_local=K_l, K_random=K_r,
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
            for _ in range(k_iter):
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
    print(f"step919 — Imagenette K_iter Sweep T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N} D={D} K_hh={K_HH} α_reflect={ALPHA_REFLECT}")
    print(f"  Context: step916 CIFAR-10: K_iter=10→−15.54pp, K_iter=15→−59.97pp")
    print(f"  Question: does Imagenette also over-smooth at K_iter>5?")
    print(f"{'='*70}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}
    ref_acc = None

    for key in keys:
        k_iter = CONFIGS[key]
        model  = make_model(k_iter)
        n_p    = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: K_iter={k_iter}  params={n_p:,}")

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

        if key == "Ref_k5":
            ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else 0.0)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "k_iter": k_iter, "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 919 SUMMARY — Imagenette K_iter Sweep T0")
    print(f"{'='*70}")
    print(f"  Compare with step916 CIFAR-10: k3=−2.12pp, k10=−15.54pp, k15=−59.97pp")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "(baseline)"
        if k == "Ref_k5":
            v = "(baseline)"
        elif d is None:
            v = "—"
        elif d >= 0.005:
            v = "ADVANCE→T1"
        elif d >= -0.005:
            v = "NEUTRAL"
        elif d >= -0.05:
            v = "HURT(mild)"
        elif d >= -0.15:
            v = "HURT(mod)"
        else:
            v = "COLLAPSE"
        print(f"  {k:<10} K_iter={r['k_iter']:>2}  {r['best']:>7.4f} {dstr:>10}  {v}")

    if ref_acc is not None:
        c10_map = {5: 0.0, 3: -2.12, 8: None, 10: -15.54, 15: -59.97}
        print(f"\n  Cross-dataset K_iter comparison:")
        for k, r in results.items():
            ki = r["k_iter"]
            c10 = c10_map.get(ki)
            c10_str = f"{c10:+.2f}pp" if c10 is not None else "  n/a"
            img_str = f"{r['delta_vs_ref']*100:+.2f}pp" if r['delta_vs_ref'] is not None else "  n/a"
            print(f"    K_iter={ki:>2}  Imagenette={img_str}  CIFAR-10={c10_str}")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
