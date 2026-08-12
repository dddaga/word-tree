"""Step 915: CIFAR-10 ΔW-proj Ablation T1 (75ep, 50% data).

MOTIVATION
==========
ΔW-proj is the confirmed routing mechanism on Imagenette:
  step706 T1: +1.49pp (confirmed)
  step886 T1: A_sign=−0.59pp, B_no_ref=−0.51pp (components load-bearing)
  step887 T2 multi-seed: 96.38% ± 0.18pp (canonical paper result)

But ALL CIFAR-10 experiments (step882/909/912/914) use ΔW-proj without a
no-ΔW baseline. The paper claims ΔW-proj as a general routing mechanism —
this is only justified if it also helps on CIFAR-10.

This T1 directly answers: does ΔW-proj generalize cross-dataset?

CONFIGS (T1, 75ep, 50% data, seed=42)
  Ref_dw    Standard ΔW-proj (matches step909/882 baseline).
  A_no_dw   Base SGNNET without ΔW-proj (uses _readout directly).
             Uses same W_pos/theta/conn_hh — only removes the projection gate.

ADVANCE RULE
============
  A_no_dw vs Ref_dw:
    Ref_dw wins ≥+0.5pp → ΔW-proj generalizes cross-dataset (paper claim strengthened)
    Ref_dw wins 0–+0.5pp → marginal generalization (paper: "primarily Imagenette")
    Ref_dw loses → ΔW-proj hurts on CIFAR-10 (paper: caveat needed)

CONTEXT
=======
step882 SGNNET (with ΔW-proj): 80.69% vs Linear 86.24% (−5.55pp)
step909 N=4096 (with ΔW-proj): 82.53% vs Linear 86.15% (−3.71pp)
ΔW-proj contribution on CIFAR-10: UNKNOWN (this step closes the gap)
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
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_cifar10.h5")
parser.add_argument("--configs", default="Ref_dw,A_no_dw")
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
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step915_cifar10_dwproj_ablation_t1_seed{SEED}__{SLOT}.json"

STEP882_REF = 0.8069   # step882 SGNNET T2 with ΔW-proj (150ep/100%)


def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def make_base() -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )


class SGNNET_DeltaW(nn.Module):
    """Standard ΔW-proj routing (matches step882/909 baseline)."""
    def __init__(self, resonant: SGNNET_Resonant):
        super().__init__(); self.m = resonant

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


class SGNNET_NoDeltaW(nn.Module):
    """SGNNET without ΔW-proj: plain mean-aggregation (no projection gate)."""
    def __init__(self, resonant: SGNNET_Resonant):
        super().__init__(); self.m = resonant

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
        Z_ref     = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_agg = Z_nb.sum(dim=2)           # plain sum, no projection gate
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


CONFIG_CLS = {
    "Ref_dw":  SGNNET_DeltaW,
    "A_no_dw": SGNNET_NoDeltaW,
}


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=(DEVICE.type == "cuda"))
    n_full  = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[: n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=10,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"\n{'='*70}")
    print(f"step915 — CIFAR-10 ΔW-proj Ablation T1 (75ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N} D={D} K_hh={K_HH} K_iter={K_ITER}")
    print(f"  Context: step882 SGNNET_dw=80.69% | step706 Imagenette +1.49pp")
    print(f"  Question: does ΔW-proj generalize to CIFAR-10?")
    print(f"{'='*70}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIG_CLS]
    results = {}
    ref_acc = None

    for key in keys:
        resonant = make_base()
        model    = CONFIG_CLS[key](resonant)
        n_p      = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: params={n_p:,}")

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

        if key == "Ref_dw":
            ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else 0.0)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "n_params": n_p, "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 915 SUMMARY — CIFAR-10 ΔW-proj Ablation T1")
    print(f"{'='*70}")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "(baseline)"
        if k == "Ref_dw":
            v = "(baseline)"
        elif d is None:
            v = "—"
        elif d >= 0.005:
            v = "DW_HURTS_CIFAR10"
        elif d >= -0.005:
            v = "NEUTRAL_CROSSDATASET"
        else:
            v = "DW_HELPS_CIFAR10"
        print(f"  {k:<12} {r['best']:>7.4f} {dstr:>10}  {v}")

    if ref_acc is not None and "A_no_dw" in results:
        gap = results["A_no_dw"]["delta_vs_ref"] * 100
        print(f"\n  KEY READ: A_no_dw vs Ref_dw = {gap:+.2f}pp")
        if gap < -0.5:
            print(f"  → ΔW-proj helps on CIFAR-10 ({gap:+.2f}pp). Cross-dataset claim STRENGTHENED.")
            print(f"    step882: SGNNET_dw=80.69%; without ΔW-proj would be ≈{STEP882_REF+gap/100:.4f}")
        elif gap < 0.5:
            print(f"  → ΔW-proj neutral on CIFAR-10 ({gap:+.2f}pp). Gain is Imagenette-specific.")
        else:
            print(f"  → ΔW-proj hurts CIFAR-10 ({gap:+.2f}pp). Paper: add cross-dataset caveat.")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
