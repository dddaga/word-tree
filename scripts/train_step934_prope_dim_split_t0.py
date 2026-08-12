"""Step 934: p-RoPE dim-split T0 (20ep, 50% data).

Gemma 4 p-RoPE insight: apply rotation to only p fraction of dims, leave
rest as "content" dims.

MECHANISM: Split W_pos into two channel groups:
  Positional channels: first p_D = int(p * D) dims — used for ΔW direction
  Content channels: remaining (1-p)*D dims — raw content signal, not routed

Routing coefficient uses only positional dims:
  c_ij = (Z_nb[:,:,:,:p_D] * dw[:,:,:,:p_D]).sum(-1, keepdim=True)
Full Z_nb still aggregated (both channel groups weighted by same c_ij scalar).

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 20ep/50%)
  Ref:    standard ΔW-proj (p_D=16, all dims used for routing) — same as step887
  A_p025: p=0.25 (4 dims for routing direction)
  B_p050: p=0.50 (8 dims for routing direction)
  C_p075: p=0.75 (12 dims for routing direction)

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
parser.add_argument("--configs", default="Ref,A_p025,B_p050,C_p075")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 25
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step934_prope_dim_split_t0_seed{SEED}__{SLOT}.json"

DW_REF = 0.9638  # step887 canonical


def _dw_proj_vec(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)  # [1,N,K_hh,D]


# p value per config
CONFIG_SPEC = {
    "Ref":    1.00,
    "A_p025": 0.25,
    "B_p050": 0.50,
    "C_p075": 0.75,
}


class SGNNET_PropeDimSplit(nn.Module):
    """ΔW-proj routing using only first p_D positional dims; full Z_nb aggregated.

    Ref (p=1.0): identical to step887 standard ΔW-proj.
    p < 1.0: routing coefficient from p_D dims only; remaining dims are
             "content" channels that flow through but don't steer routing.
    """

    def __init__(self, resonant, p: float):
        super().__init__()
        self.m   = resonant
        self.p   = p
        self.p_D = max(1, int(p * D))

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x):
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h = self.m.W_pos[:N]
        # ΔW direction using only first p_D dims
        dw_full = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)  # [1,N,K_hh,D]
        dw = dw_full[:, :, :, :self.p_D]  # [1,N,K_hh,p_D]
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]                          # [B,N,K_hh,D]
            # routing coefficient from positional dims only
            c_ij  = (Z_nb[:, :, :, :self.p_D] * dw).sum(dim=-1, keepdim=True)  # [B,N,K_hh,1]
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)                # still weights FULL Z_nb
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


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
    p = CONFIG_SPEC[key]
    r = make_base()
    return SGNNET_PropeDimSplit(r, p=p)


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
    print(f"step934 — p-RoPE dim-split T0 (20ep, 50%)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  Insight: route via first p_D positional dims only (Gemma 4 p-RoPE).")
    print(f"  Full Z_nb aggregated; only routing coefficient restricted to p_D dims.")
    print(f"  Context: step887 canonical = {DW_REF:.4f}")
    print()
    print(f"  {'Config':<12} {'p':>5}  {'p_D':>5}")
    for k, p in CONFIG_SPEC.items():
        p_D = max(1, int(p * D))
        print(f"  {k:<12} {p:>5.2f}  {p_D:>5d}")
    print(f"{'='*72}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        model = make_model(key).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        p_val = CONFIG_SPEC[key]
        p_D   = max(1, int(p_val * D))
        print(f"{'─'*60}")
        print(f"{key}: p={p_val:.2f}  p_D={p_D}  params={n_p:,}")

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
            "p": p_val, "p_D": p_D, "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*72}")
    print(f"STEP 934 SUMMARY — p-RoPE dim-split T0")
    print(f"{'='*72}")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        v = ("(ref)" if k == "Ref"
             else ("ADV→T1" if d >= 0.005 else ("NEU" if d >= -0.005 else "KILL")))
        print(f"  {k:<12} p={r['p']:.2f}  p_D={r['p_D']:>2d}  params={r['n_params']:>8,}  "
              f"best={r['best']:.4f}  Δ={d*100:+.2f}pp  {v}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
