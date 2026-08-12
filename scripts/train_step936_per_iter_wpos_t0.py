"""Step 936: Per-iteration W_pos transform T0 (20ep, 50% data).

Gemma 4 PLE analogue: give each routing iteration its OWN view of the position
space via a small learned transform on top of shared W_pos. Each round "looks"
at different angular slices of the node embedding space.

MOTIVATION:
  Standard ΔW-proj uses ONE fixed dw = normalize(W_pos[i] - W_pos[j]) across all
  K_iter routing steps. This means every step uses the same geometric signal.
  Per-iter transform lets each step attend to different directions in pos-space,
  potentially encoding a curriculum: coarse→fine or rotating basis per step.

MECHANISM:
  Shared W_pos [N, D] + per-iter transform → pos_k [N, D] → dw_k = normalize(pos_k[i] - pos_k[j])
  At iteration k, dw_k replaces the static dw.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, ΔW-proj, 20ep/50%)
  Ref:        shared W_pos, static dw — same as step887
  A_proj:     per-iter A_k [K_iter, D, D], identity init
              pos_k = W_h @ A_k[k]  — 5×16×16 = 1,280 extra params
  B_scale:    per-iter gamma_k [K_iter, D], ones init
              pos_k = W_h * gamma_k[k]  — 5×16 = 80 extra params
  C_residual: per-iter delta_k [K_iter, D, D], zero init
              pos_k = W_h + W_h @ delta_k[k]  — starts exactly at Ref
              5×16×16 = 1,280 extra params

Init: identity (A_proj) / ones (B_scale) / zeros (C_residual) → all start at Ref.
B_scale is fewest params; C_residual safest init for Ref parity.

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
parser.add_argument("--configs", default="Ref,A_proj,B_scale,C_residual")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 25
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step936_per_iter_wpos_t0_seed{SEED}__{SLOT}.json"

DW_REF = 0.9638  # step887 canonical


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
        W_h = self.m.W_pos[:N]                                    # [N, D]
        dw  = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)  # [1,N,K_hh,D]
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]                          # [B,N,K_hh,D]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_PerIterWpos(nn.Module):
    """Per-iteration W_pos transform: each routing step uses a different pos_k.

    mode="proj":      pos_k = W_h @ A_k          — full D×D rotation per iter
    mode="scale":     pos_k = W_h * gamma_k       — elementwise scale per iter
    mode="residual":  pos_k = W_h + W_h @ delta_k — additive residual per iter
    """

    def __init__(self, resonant, mode: str):
        super().__init__()
        self.m    = resonant
        self.mode = mode

        if mode == "proj":
            # identity init: each A_k starts as identity → pos_k = W_h at init
            A_init = torch.eye(D).unsqueeze(0).expand(K_ITER, -1, -1).clone()
            self.A_list = nn.ParameterList(
                [nn.Parameter(A_init[k].clone()) for k in range(K_ITER)]
            )

        elif mode == "scale":
            # ones init: gamma_k = 1 → pos_k = W_h at init
            self.gamma_list = nn.ParameterList(
                [nn.Parameter(torch.ones(D)) for _ in range(K_ITER)]
            )

        elif mode == "residual":
            # zeros init: delta_k = 0 → pos_k = W_h at init (exact Ref)
            self.delta_list = nn.ParameterList(
                [nn.Parameter(torch.zeros(D, D)) for _ in range(K_ITER)]
            )

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x):
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h = self.m.W_pos[:N]                                    # [N, D]
        Z_ref = torch.zeros_like(Z)

        for k in range(K_ITER):
            # compute per-iter pos transform
            if self.mode == "proj":
                pos_k = W_h @ self.A_list[k]                      # [N, D]
            elif self.mode == "scale":
                pos_k = W_h * self.gamma_list[k].unsqueeze(0)     # [N, D]
            elif self.mode == "residual":
                pos_k = W_h + W_h @ self.delta_list[k]            # [N, D]

            dw = F.normalize(pos_k.unsqueeze(1) - pos_k[conn_hh], dim=-1).unsqueeze(0)  # [1,N,K_hh,D]

            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]                          # [B,N,K_hh,D]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# mode string → (mode_key, extra_params)
CONFIG_SPEC = {
    "Ref":        ("ref",      0),
    "A_proj":     ("proj",     K_ITER * D * D),   # 5*16*16 = 1280
    "B_scale":    ("scale",    K_ITER * D),        # 5*16   =   80
    "C_residual": ("residual", K_ITER * D * D),   # 5*16*16 = 1280
}


def _param_count(key):
    _, extra = CONFIG_SPEC[key]
    base = (N + N_OUT) * D + N
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
    mode, _ = CONFIG_SPEC[key]
    r = make_base()
    if mode == "ref":
        return SGNNET_Ref(r)
    return SGNNET_PerIterWpos(r, mode=mode)


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
    print(f"step936 — Per-iteration W_pos transform T0 (20ep, 50%)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  Each routing step uses a different view of position space.")
    print(f"  Analogue: Gemma 4 Per-Layer Embedding — per-step pos basis.")
    print(f"  Context: step887 canonical = {DW_REF:.4f}")
    print()
    print(f"  {'Config':<12} {'mode':<10} {'extra_params':>12}  {'total':>8}")
    for k in CONFIG_SPEC:
        b, ex, tot = _param_count(k)
        print(f"  {k:<12} {CONFIG_SPEC[k][0]:<10} {ex:>12,}  {tot:>8,}")
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
        print(f"{key}: mode={CONFIG_SPEC[key][0]}  params={n_p:,}  (expected={exp_total:,})")

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
            "mode": CONFIG_SPEC[key][0], "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*72}")
    print(f"STEP 936 SUMMARY — Per-iteration W_pos transform T0")
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
