"""Step 917: CIFAR-10 α_reflect Sweep T0 (20ep, 50% data).

MOTIVATION
==========
α_reflect=0.5 is the canonical SGNNET setting, tuned on Imagenette.
step915 confirmed ΔW-proj is essential cross-dataset (−62.41pp without it).

The residual reflection term Z_ref = α * Z_ref + (Z_fwd - Z) carries
"how much each node changed" across iterations. On Imagenette, α=0.5
was the best value. Is it also optimal on CIFAR-10?

If CIFAR-10 benefits from different α: paper should use per-dataset tuning
(or acknowledge α=0.5 may not be optimal cross-dataset).
If all configs neutral: α=0.5 is robust across datasets → clean paper claim.

CONFIGS (T0, 20ep, 50% data, seed=42)
  Ref_a05   α_reflect=0.5  (canonical — matches step882/909/914/915)
  A_a00     α_reflect=0.0  (no residual reflection — pure ΔW routing)
  B_a025    α_reflect=0.25 (weaker memory)
  C_a075    α_reflect=0.75 (stronger memory)
  D_a10     α_reflect=1.0  (full accumulation)

ADVANCE RULE
============
  Any config ≥+0.5pp vs Ref_a05 → advance to T1, retune CIFAR-10 runs
  All neutral → α=0.5 is cross-dataset robust (paper: parameter-free tuning)

CONTEXT
=======
step882 (α=0.5): 80.69% vs Linear 86.24% (−5.55pp)
step915: ΔW-proj essential (no-ΔW → 15.96% collapse)
This step: reflection memory angle on CIFAR-10 gap
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
parser.add_argument("--configs", default="Ref_a05,A_a00,B_a025,C_a075,D_a10")
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

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step917_cifar10_alpha_reflect_t0_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Ref_a05": 0.50,
    "A_a00":   0.00,
    "B_a025":  0.25,
    "C_a075":  0.75,
    "D_a10":   1.00,
}


def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def make_model(alpha_reflect: float) -> nn.Module:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=alpha_reflect, alpha_turing=0.0,
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
                Z_ref = alpha_reflect * Z_ref + (Z_fwd - Z)
                Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
            return self.m.base._readout(Z)

    return DeltaW()


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
    print(f"step917 — CIFAR-10 α_reflect Sweep T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N} D={D} K_hh={K_HH} K_iter={K_ITER}")
    print(f"  Context: step882 α=0.5 → 80.69% | step915 ΔW-proj essential")
    print(f"  Question: is α_reflect=0.5 optimal cross-dataset?")
    print(f"{'='*70}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}
    ref_acc = None

    for key in keys:
        alpha = CONFIGS[key]
        model = make_model(alpha)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: α_reflect={alpha}  params={n_p:,}")

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

        if key == "Ref_a05":
            ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else 0.0)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "alpha_reflect": alpha, "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 917 SUMMARY — CIFAR-10 α_reflect Sweep T0")
    print(f"{'='*70}")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "(baseline)"
        if k == "Ref_a05":
            v = "(baseline)"
        elif d is None:
            v = "—"
        elif d >= 0.005:
            v = "ADVANCE→T1(retune)"
        elif d >= -0.005:
            v = "NEUTRAL"
        else:
            v = "KILL"
        print(f"  {k:<10} α={r['alpha_reflect']:.2f}  {r['best']:>7.4f} {dstr:>10}  {v}")

    if ref_acc is not None:
        winners = [(k, r) for k, r in results.items()
                   if k != "Ref_a05" and r["delta_vs_ref"] is not None
                   and r["delta_vs_ref"] >= 0.005]
        if winners:
            best_k, best_r = max(winners, key=lambda x: x[1]["delta_vs_ref"])
            print(f"\n  → ADVANCE: {best_k} (α={best_r['alpha_reflect']}) — retune CIFAR-10 runs")
        else:
            print(f"\n  → α=0.5 is robust cross-dataset (paper: no per-dataset tuning needed)")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
