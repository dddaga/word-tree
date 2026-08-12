"""Step 955: W_edge channel rotation ON TOP of ΔW-proj T0 (20ep, 50% data).

Step 950 failed because it removed ΔW-proj and tested W_edge alone.
Without ΔW-proj, gather-sum causes Z collapse (loss=13.19 at ep1).
W_edge is useless without a non-collapsing base signal.

This experiment corrects that design: W_edge is applied AFTER ΔW-proj aggregation.
ΔW-proj does its geometric scalar gating → W_edge mixes the resulting D-dim signal.

MECHANISM:
  Z_nb  = gather neighbors                       [B, N, K_hh, D]
  c_ij  = (Z_nb · dw)                           scalar projection
  Z_wt  = Z_nb * |c_ij|                         ΔW-proj weighted
  Z_rot = einsum(Z_wt, W_edge)                  channel rotation (NEW)
  Z_agg = Z_rot.sum(dim=2)

Step 950 insight: W_edge alone = structural failure. W_edge + ΔW-proj = orthogonal combination.
ΔW-proj operates on SCALARS (geometric gating). W_edge operates on CHANNELS (learned mixing).
Different signal paths → compounding is safe per compounding rule.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, ΔW-proj, 20ep/50%)
  Ref:      standard ΔW-proj (no W_edge) — same as step887 base
  A_shared: W_edge [K_hh, D, D] — 512 extra params  → total: 35,488
  B_symm:   W_edge [K_hh, D, D] + symmetry constraint (W = (W + W^T)/2)
             Symmetry preserves channel norms, limits expressivity
  C_residual: W_edge applied as residual: Z_rot = Z_wt + einsum(Z_wt, W_edge)
             W_edge init to zero → starts exactly at Ref. Learns a correction.
             Extra params: K_hh * D * D = 512 — same as A_shared

Init: identity (A_shared, B_symm) or zeros (C_residual) → all start at Ref behavior.

ADVANCE: ≥+0.5pp vs Ref → T1.
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
parser.add_argument("--configs", default="Ref,A_shared,B_symm,C_residual")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 25
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step955_wedge_dwproj_t0_seed{SEED}__{SLOT}.json"

DW_REF = 0.9638  # step887 canonical


def _dw_proj_vec(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)  # [1,N,K_hh,D]


class SGNNET_Ref(nn.Module):
    def __init__(self, resonant):
        super().__init__(); self.m = resonant

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x):
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj_vec(self.m.W_pos, conn_hh)
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]                       # [B,N,K_hh,D]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_WedgeDW(nn.Module):
    """ΔW-proj scalar gating + W_edge channel rotation.

    Pipeline per step:
      1. Gather neighbors → Z_nb [B,N,K_hh,D]
      2. ΔW-proj: weight by |Z_nb · dw| → Z_wt
      3. W_edge: channel-mix Z_wt → Z_rot
      4. Aggregate: sum over K_hh → Z_agg
    """

    def __init__(self, resonant, mode: str):
        super().__init__()
        self.m    = resonant
        self.mode = mode

        if mode == "shared":
            W = torch.eye(D).unsqueeze(0).expand(K_HH, -1, -1).clone()
            self.W_edge = nn.Parameter(W)                  # [K_hh, D, D]

        elif mode == "symm":
            W = torch.eye(D).unsqueeze(0).expand(K_HH, -1, -1).clone()
            self.W_raw = nn.Parameter(W)                   # symmetrised at forward

        elif mode == "residual":
            # W_edge init to zero → at init Z_rot = Z_wt (exact Ref behavior)
            # Learns a residual correction on top of ΔW-proj signal
            self.W_edge = nn.Parameter(torch.zeros(K_HH, D, D))  # [K_hh, D, D]

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def _get_W_edge(self):
        if self.mode in ("shared", "residual"):
            return self.W_edge                             # [K_hh, D, D]
        else:  # symm
            return (self.W_raw + self.W_raw.transpose(-1, -2)) / 2.0

    def forward(self, x):
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj_vec(self.m.W_pos, conn_hh)
        W_edge = self._get_W_edge()                        # [K_hh, D, D]
        Z_ref = torch.zeros_like(Z)

        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]                  # [B,N,K_hh,D]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True) # [B,N,K_hh,1]
            Z_wt  = Z_nb * c_ij.abs()                     # [B,N,K_hh,D] ΔW-proj
            # W_edge channel rotation: Z_rot[b,n,k,e] = sum_d Z_wt[b,n,k,d] * W[k,d,e]
            Z_mix = torch.einsum('bnkd,kde->bnke', Z_wt, W_edge)
            # residual mode: Z_rot = Z_wt + correction; others: Z_rot = full rotation
            if self.mode == "residual":
                Z_rot = Z_wt + Z_mix
            else:
                Z_rot = Z_mix
            Z_agg = Z_rot.sum(dim=2)                       # [B,N,D]
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


CONFIG_SPEC = {
    "Ref":        "ref",
    "A_shared":   "shared",
    "B_symm":     "symm",
    "C_residual": "residual",
}


def _param_count(key):
    mode = CONFIG_SPEC[key]
    base = (N + N_OUT) * D + N
    extra = {
        "ref":      0,
        "shared":   K_HH * D * D,
        "symm":     K_HH * D * D,
        "residual": K_HH * D * D,
    }[mode]
    return base, extra, base + extra


def make_base():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )


def make_model(key):
    mode = CONFIG_SPEC[key]
    r = make_base()
    if mode == "ref":
        return SGNNET_Ref(r)
    return SGNNET_WedgeDW(r, mode=mode)


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

    print(f"\n{'='*72}")
    print(f"step955 — W_edge channel rotation ON TOP of ΔW-proj T0 (20ep, 50%)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  step950 failure: W_edge alone → Z collapse. This fixes it.")
    print(f"  ΔW-proj (geometric scalar) + W_edge (channel rotation) = orthogonal paths.")
    print(f"  Context: step887 canonical = {DW_REF:.4f}")
    print()
    print(f"  {'Config':<12} {'mode':<10} {'extra_params':>12}  {'total':>8}")
    for k in CONFIG_SPEC:
        b, ex, tot = _param_count(k)
        print(f"  {k:<12} {CONFIG_SPEC[k]:<10} {ex:>12,}  {tot:>8,}")
    print(f"{'='*72}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        model = make_model(key).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _, _, exp_total = _param_count(key)
        print(f"{'─'*60}")
        print(f"{key}: mode={CONFIG_SPEC[key]}  params={n_p:,}  (expected={exp_total:,})")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()

        def log_fn(m):
            print(f"  e{m['epoch']+1:3d}  top1={m['val_top1']:.4f}  lr={m['lr']:.2e}", flush=True)

        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(n_epochs=EPOCHS, log_fn=log_fn)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else DW_REF)
        verdict = ("(baseline)" if key == "Ref"
                   else "ADVANCE→T1" if delta >= 0.005
                   else ("NEUTRAL" if delta >= -0.005 else "KILL"))
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {verdict}")

        results[key] = {
            "mode": CONFIG_SPEC[key], "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*72}")
    print(f"STEP 955 SUMMARY — W_edge + ΔW-proj T0")
    print(f"{'='*72}")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        v = ("(ref)" if k == "Ref"
             else ("ADV→T1" if d >= 0.005 else ("NEU" if d >= -0.005 else "KILL")))
        print(f"  {k:<12} mode={r['mode']:<10} params={r['n_params']:>8,}  "
              f"best={r['best']:.4f}  Δ={d*100:+.2f}pp  {v}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
