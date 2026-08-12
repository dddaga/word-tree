"""Step 956: Refractory neurons T1 (75ep, 50% data).

T0 result (step938):
  Ref=93.89%. A_β07_αr2=−1.94pp KILL (too strong). B_β05_αr1=−0.03pp NEUTRAL.
  C_β09_αr1=−0.28pp NEUTRAL. B+C advance per rejection-filter protocol.

Mechanism: Z_t -= α_r * max(0, β * |Z_{t-1}|) — temporal diversity.
Additive suppression — gate-death theorem does not apply.
Encourages different neurons to fire across routing steps.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, ΔW-proj, 75ep/50%)
  Ref:       standard ΔW-proj (no refractory)
  B_β05_αr1: β=0.5, α_r=1.0 — mild suppression (−0.03pp T0)
  C_β09_αr1: β=0.9, α_r=1.0 — high threshold, mild suppression (−0.28pp T0)

ADVANCE: ≥+0.5pp vs Ref → update defaults.
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
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,B_β05_αr1,C_β09_αr1")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step956_refractory_t1_seed{SEED}__{SLOT}.json"

T0_REF   = 0.9389  # step938 Ref T0
DW_REF   = 0.9638  # step887 canonical


def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)

def _dw_agg(Z_nb, dw):
    return (Z_nb * (Z_nb * dw).sum(dim=-1, keepdim=True).abs()).sum(dim=2)


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
        dw = _dw_proj(self.m.W_pos, conn_hh)
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_agg = _dw_agg(Z_nb, dw)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_Refractory(nn.Module):
    def __init__(self, resonant, beta: float, alpha_r: float):
        super().__init__()
        self.m = resonant
        self.beta = beta
        self.alpha_r = alpha_r

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x):
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj(self.m.W_pos, conn_hh)
        Z_ref  = torch.zeros_like(Z)
        Z_prev = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z      = Z - self.alpha_r * torch.clamp(self.beta * Z_prev.abs(), min=0)
            Z_fwd  = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb   = Z_fwd[:, conn_hh, :]
            Z_agg  = _dw_agg(Z_nb, dw)
            Z_ref  = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z_prev = Z_fwd.detach()
            Z      = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


CONFIG_SPEC = {
    "Ref":       (None,  None),
    "B_β05_αr1": (0.5,   1.0),
    "C_β09_αr1": (0.9,   1.0),
}


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
    beta, alpha_r = CONFIG_SPEC[key]
    r = make_base()
    if beta is None:
        return SGNNET_Ref(r)
    return SGNNET_Refractory(r, beta=beta, alpha_r=alpha_r)


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
    print(f"step956 — Refractory neurons T1 (75ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  T0 context: B=−0.03pp NEUTRAL, C=−0.28pp NEUTRAL. Both advance.")
    print(f"  Canonical ref (step887): {DW_REF:.4f}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        model = make_model(key).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        beta, alpha_r = CONFIG_SPEC[key]
        print(f"{'─'*60}")
        print(f"{key}: β={beta}  α_r={alpha_r}  params={n_p:,}")

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
                   else "ADVANCE→defaults" if delta >= 0.005
                   else ("NEUTRAL" if delta >= -0.005 else "KILL"))
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {verdict}")

        results[key] = {"beta": beta, "alpha_r": alpha_r, "n_params": n_p,
                        "best": round(best, 4), "best_ep": best_ep,
                        "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 956 SUMMARY — Refractory neurons T1")
    print(f"{'='*70}")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        v = ("(ref)" if k == "Ref"
             else ("ADV→defaults" if d >= 0.005 else ("NEU" if d >= -0.005 else "KILL")))
        print(f"  {k:<14} β={r['beta']}  α_r={r['alpha_r']}  "
              f"best={r['best']:.4f}  Δ={d*100:+.2f}pp  {v}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
