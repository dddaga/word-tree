"""Step 900: D-scaling — does larger D break the 97% ceiling? (T0 scout).

MOTIVATION
==========
Why can't SGNNET reach 99%? Three hypotheses analyzed 2026-04-19:

1. N/N_in ratio — REJECTED. N-scaling plateaus at N=8192 (step205-209: N=4096→97.17%,
   N=8192→97.17%). More neurons don't help. The 97% ceiling is NOT a capacity issue.

2. Frozen VGG16 features — REAL, but fixed. Pool5 features cap any FC head at ~97-98%.
   We cannot improve beyond the feature quality. Paper context: step89 (D=64, N=4096)
   reached 97.86% — close to the ceiling.

3. D=16 representation ceiling — THE LEVER. step863 D-ablation: D=8→−1.86pp,
   D=12→−1.43pp vs D=16. Trend: accuracy RISES with D. Current D=16 is a budget choice.

KEY QUESTION: What accuracy does SGNNET reach at D=24 and D=32?
  - If D=24→+1pp over D=16: D is the primary bottleneck at current N=2048.
  - If D=32→+2pp: the trajectory toward 99% exists, just needs higher D.
  - If plateau at D=24: something else is the ceiling.

ARCHITECTURE IMPACT OF LARGER D
=================================
Parameter count:
  D=16: W_pos=[N, D]=32,768 values + W_θ=N values + W_out=[N_out×D]=160 + W_in=[N×K_in×D]=819,200
  → Roughly O(N×K_in×D) dominated by fan-in projection.
  D=24: W_in = N×K_in×24 = 1,228,800 (1.5× D=16)
  D=32: W_in = N×K_in×32 = 1,638,400 (2× D=16)
  Total params approximately: D=16≈34,976, D=24≈~52K, D=32≈~69K (estimates)

FLOPs:
  Routing MACs = N × K_iter × K_hh × D × 2:
  D=16: 0.98M, D=24: 1.47M, D=32: 1.97M
  At D=32: 2× routing FLOPs but still <2% of VGG FC (123M). Within efficiency budget.

WHAT THIS ANSWERS
==================
1. Does the D=16 Fourier encoding constrain routing diversity? (step155: eff_rank=4.6/16)
2. At D=24/32, does eff_rank saturate or rise above D=16?
3. What's the accuracy/FLOPs Pareto frontier as a function of D?
4. Paper claim: "D=16 is an efficient choice; D=32 reaches X% at Y× FLOPs"

RELATION TO step863 (D-ablation at N=2048, T0)
================================================
  step863: D=8→91.44%, D=12→92.53%, D=16 Ref≈93.96% (from step898)
  Trend: +1.53pp per doubling. Extrapolation: D=32→~95.49pp?
  step900 tests whether this trend continues or saturates.

CONFIGS (T0, 20ep, 50% data, seed=42)
  Ref_d16    D=16, N=2048, K_hh=2, K_iter=5 (standard baseline)
  A_d24      D=24, N=2048, K_hh=2, K_iter=5
  B_d32      D=32, N=2048, K_hh=2, K_iter=5
  C_d24_dw   D=24 + ΔW-proj (does ΔW gain persist at larger D?)
  D_d32_dw   D=32 + ΔW-proj

Note: ΔW-proj was confirmed +1.49pp at D=16. At larger D, it may help more
(more dimensions for relational projection) or become redundant (richer Z signals).

ADVANCE RULE
=============
  A or B ≥+0.5pp vs Ref_d16 → T1 D-scaling run (step902 or next).
  C or D ≥+0.3pp above their D baseline → ΔW-proj scales with D (T1 compound).
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
parser.add_argument("--epochs",  type=int, default=20)
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
OUT_PATH = ROOT / "results" / f"train_step900_d_scaling_t0_seed{SEED}__{SLOT}.json"

STEP_REF = 0.9396  # D=16 T0 reference (step898 Ref_dw)

# Config spec: (D, use_dw_proj)
CONFIG_SPEC = {
    "Ref_d16":  (16, False),
    "A_d24":    (24, False),
    "B_d32":    (32, False),
    "C_d24_dw": (24, True),
    "D_d32_dw": (32, True),
}


def _dw_proj(W_pos: torch.Tensor, conn_hh: torch.Tensor, n: int) -> torch.Tensor:
    """Precompute ΔW-proj direction vectors. [1, N, K_hh, D]."""
    W_h = W_pos[:n]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
    """ΔW-proj aggregation. Returns Z_dw [B, N, D]."""
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


class SGNNET_DRef(nn.Module):
    """Plain AH/resonant baseline at given D (no ΔW-proj)."""

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
    if use_dw:
        return SGNNET_DDeltaW(r, d)
    else:
        return SGNNET_DRef(r)


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
    print(f"step900 — D-scaling (D=24, D=32) T0 scout")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  Hypothesis: D=16 Fourier ceiling; larger D → higher accuracy")
    print(f"  Baseline: step863 trend D=8→91.44%, D=12→92.53%, D=16~93.96%")
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
    print(f"STEP 900 SUMMARY — D-scaling T0")
    print(f"{'='*70}")
    print(f"  {'config':<12} {'D':>3} {'dw':>4} {'params':>8} {'MACs(M)':>8} "
          f"{'best':>7} {'Δ_vs_Ref':>10}  verdict")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "  (ref)"
        v    = "ADVANCE→T1" if (d is not None and d >= 0.005) else \
               "NEUTRAL"   if (d is not None and d >= -0.005) else \
               "MARGINAL"  if (d is not None and d >= -0.020) else "KILL"
        print(f"  {k:<12} {r['D']:>3} {str(r['use_dw']):>4} {r['n_params']:>8,} "
              f"{r['routing_macs']/1e6:>8.2f} {r['best']:>7.4f} {dstr:>10}  {v}")

    print(f"\n  Ref T0 context (step898/863 D=16): {STEP_REF}")
    print(f"  ADVANCE: A or B ≥+0.5pp → T1 D-scaling (step902).")
    print(f"  KEY READ: B vs A → does D=32 continue the trend from step863?")
    print(f"           D vs C → ΔW-proj gain at larger D.")
    print(f"           C vs A, D vs B → ΔW benefit at each D.")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
