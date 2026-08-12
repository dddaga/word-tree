"""Step 926: CIFAR-10 K_in sweep T0 @ N=4096 (20ep, 50% data).

MOTIVATION
==========
step909 (N=4096, 82.53%) and step914 (N=8192, 83.58%) used K_in=15 by default,
following the Imagenette pattern where K_in=15 helps at N≥4096.

step923 showed K_in=15 costs −0.88pp vs K_in=25 on CIFAR-10 at N=2048.
This is WORSE than Imagenette (−0.33pp). The CIFAR-10 penalty may persist or
flip at N=4096 — unknown, since we never tested K_in=25 at N=4096 on CIFAR-10.

If K_in=25 wins at N=4096 too: step909 table is underperforming; need T2 re-run.
If K_in=15 wins or ties: N-scaling table is fine as-is.

Imagenette crossover is between N=2048 (−0.33pp) and N=4096 (+0.33pp T1, +0.05pp T2).
CIFAR-10 may have the crossover at a different N, or may not cross at all.

CONFIGS (T0, 20ep, 50% data, seed=42, N=4096, CIFAR-10)
  Ref_k25   K_in=25  (what step909 should have used as reference)
  A_k15     K_in=15  (what step909/914 actually used)

ADVANCE RULE
============
  K_in=25 ≥+0.5pp → step909/914 underperformed; queue T2 re-run with K_in=25
  K_in=15 ≥+0.5pp → Imagenette N-scaling pattern holds on CIFAR-10 too (surprising)
  Within ±0.5pp → current N-scaling table is valid; K_in choice neutral at N=4096
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
parser.add_argument("--configs", default="Ref_k25,A_k15")
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
D = 16; K_HH = 2; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step926_cifar10_kin_n4096_t0_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Ref_k25": 25,
    "A_k15":   15,
}

STEP909_K15_REF = 0.8253   # step909 B_N4096 with K_in=15 (T2 150ep)


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
    print(f"step926 — CIFAR-10 K_in Sweep T0 @ N=4096 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N} D={D} K_hh={K_HH} K_iter={K_ITER} α_reflect={ALPHA_REFLECT}")
    print(f"  Context: step923 K_in=15=−0.88pp @ N=2048. step909 used K_in=15 @ N=4096.")
    print(f"  Question: does K_in crossover exist on CIFAR-10 between N=2048 and N=4096?")
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
        delta = best - ref_acc if ref_acc is not None else None
        dstr  = f"{delta*100:+.2f}pp" if delta is not None else "(baseline)"
        print(f"  -> best={best:.4f} @ep{best_ep}  {dstr}  {elapsed:.0f}s")

        results[key] = {
            "k_in": k_in, "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if delta is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 924 SUMMARY — CIFAR-10 K_in @ N=4096 T0")
    print(f"{'='*70}")
    print(f"  N=2048 reference: K_in=15 Imagenette=−0.33pp, CIFAR-10=−0.88pp (step632/923)")
    print(f"  step909 T2 ref: N=4096 K_in=15 = {STEP909_K15_REF:.4f} (150ep)")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "(baseline)"
        if k == "Ref_k25":
            v = "(baseline)"
        elif d is None:
            v = "—"
        elif d >= 0.005:
            v = "K_in=15 WINS — crossover confirmed on CIFAR-10"
        elif d >= -0.005:
            v = "NEUTRAL — K_in choice doesn't matter at N=4096"
        else:
            v = f"K_in=25 WINS — step909 underperformed by ~{abs(d)*100:.1f}pp (queue T2 rerun)"
        print(f"  {k:<10} K_in={r['k_in']:>3}  {r['best']:>7.4f} {dstr:>10}  {v}")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
