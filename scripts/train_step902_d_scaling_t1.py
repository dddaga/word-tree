"""Step 902: D-scaling T1 (75ep, 50% data) — advance from step900.

CONTEXT
=======
step900 T0 tested D=24 and D=32 (with/without ΔW-proj). Advance rule:
A or B ≥+0.5pp vs Ref_d16 → T1 calibration (this script).

step863 D-ablation trend: D=8→91.44%, D=12→92.53%, D=16≈93.96%.
Monotone +1.53pp/doubling. Extrapolation: D=32→~95.49%.
step900 T1 confirms whether the trend holds at 75ep training.

KEY QUESTIONS AT T1
====================
1. Does D=24/32 gap vs D=16 widen, narrow, or hold at 75ep?
   Widening → D is genuinely load-bearing; T2 warranted.
   Narrowing → smaller D catches up with more training; D=16 is fine.
2. Does ΔW-proj gain persist at larger D?
   If C_d24_dw > A_d24 by ≥+0.5pp → ΔW scales with D (richer projection space).
   If gap shrinks → D subsumes the ΔW-proj benefit.
3. Pareto: D=24 adds ~1.5× FLOPs vs D=16. Worth it?

CONFIGS (T1, 75ep, 50% data, seed=42)
  Ref_d16    D=16, N=2048, K_hh=2, K_iter=5 (standard baseline)
  A_d24      D=24, no ΔW-proj
  B_d32      D=32, no ΔW-proj
  C_d24_dw   D=24 + ΔW-proj
  D_d32_dw   D=32 + ΔW-proj

ADVANCE RULE
============
  Winner ≥+1.0pp vs Ref_d16 at T1 → T2 (step904).
  ΔW-proj gain ≥+0.5pp at D=24/32 → ΔW scales with D; compound T2.
  Winner < +0.5pp at T1 → T0 was artifact; D=16 confirmed as ceiling.
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

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref_d16,A_d24,B_d32,C_d24_dw,D_d32_dw")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step902_d_scaling_t1_seed{SEED}__{SLOT}.json"

STEP_REF = 0.9536  # step878 Ref T1 (75ep, 50% data, D=16)

# Config spec: (D, use_dw_proj)
CONFIG_SPEC = {
    "Ref_d16":  (16, False),
    "A_d24":    (24, False),
    "B_d32":    (32, False),
    "C_d24_dw": (24, True),
    "D_d32_dw": (32, True),
}


def _dw_proj(W_pos: torch.Tensor, conn_hh: torch.Tensor, n: int) -> torch.Tensor:
    W_h = W_pos[:n]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


class SGNNET_DRef(nn.Module):
    """Plain resonant baseline at given D (no ΔW-proj)."""

    def __init__(self, resonant: SGNNET_Resonant):
        super().__init__()
        self.m = resonant

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_agg = Z_nb.sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_DDeltaW(nn.Module):
    """ΔW-proj at given D."""

    def __init__(self, resonant: SGNNET_Resonant, d: int):
        super().__init__()
        self.m = resonant
        self.d = d

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw        = _dw_proj(self.m.W_pos, conn_hh, N)
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_dw  = _dw_agg(Z_nb, dw)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_dw + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def make_base(d: int) -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=d, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )


def make_model(key: str) -> nn.Module:
    d, use_dw = CONFIG_SPEC[key]
    r = make_base(d)
    return SGNNET_DDeltaW(r, d) if use_dw else SGNNET_DRef(r)


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=(DEVICE.type == "cuda"))
    n_full = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=10,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"\n{'='*70}")
    print(f"step902 — D-scaling T1 (75ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  Ref context (step878 T1, D=16): {STEP_REF:.4f}")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        d, use_dw = CONFIG_SPEC[key]
        model  = make_model(key)
        n_p    = sum(p.numel() for p in model.parameters() if p.requires_grad)
        routing_macs = N * K_ITER * K_HH * d * 2
        print(f"{'─'*60}")
        print(f"{key}: D={d}  use_dw={use_dw}  params={n_p:,}  "
              f"routing_MACs={routing_macs/1e6:.2f}M")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(
                f"  e{m['epoch']+1:3d}  loss={m['train_loss']:.4f}  "
                f"task={m.get('task_loss', m['train_loss']):.4f}  "
                f"safety={m.get('safety_loss', 0.0):.4f}  "
                f"top1={m['val_top1']:.4f}  lr={m['lr']:.2e}",
                flush=True,
            ))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_d16":
            ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP_REF)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "D": d, "use_dw": use_dw,
            "n_params": n_p,
            "routing_macs": routing_macs,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 902 SUMMARY — D-scaling T1")
    print(f"{'='*70}")
    print(f"  {'config':<12} {'D':>3} {'dw':>4} {'params':>8} {'MACs(M)':>8} "
          f"{'best':>7} {'Δ_vs_Ref':>10}  verdict")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "  (ref)"
        v    = "ADVANCE→T2" if (d is not None and d >= 0.010) else \
               "CONFIRMED"  if (d is not None and d >= 0.005) else \
               "NEUTRAL"    if (d is not None and d >= -0.005) else \
               "MARGINAL"   if (d is not None and d >= -0.020) else "KILL"
        print(f"  {k:<12} {r['D']:>3} {str(r['use_dw']):>4} {r['n_params']:>8,} "
              f"{r['routing_macs']/1e6:>8.2f} {r['best']:>7.4f} {dstr:>10}  {v}")

    print(f"\n  Ref T1 context (step878/D=16): {STEP_REF}")
    print(f"  ADVANCE→T2: winner ≥+1.0pp → step904 D-scaling T2.")
    print(f"  KEY READ: B_d32 vs A_d24 → trend continues? C/D vs A/B → ΔW at larger D?")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
