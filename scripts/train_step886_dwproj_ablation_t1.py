"""Step 886: ΔW-proj ablation T1 — confirm T0 findings for paper table.

MOTIVATION
==========
step883 T0 results (20ep, 50% data):
  A_sign   (clamp(0))  : -1.83pp — magnitude gating is load-bearing
  B_no_ref (α_r=0)     : -1.32pp — reflection signal is load-bearing
  C_no_theta (θ=0)     : -0.71pp — theta gate is moderate (may flip neutral at T1)
  D_rand_dir           : -76.56pp — geometric direction is ESSENTIAL (skip T1, obvious)

T0 ρ=0.80 → C_no_theta (-0.71pp) has ~20% chance of flipping neutral at T1.
T1 confirms which components earn a "load-bearing" row in the paper ablation table.

CONFIGS (T1: 75ep, 50% data, seed=42)
  Ref_dw     : standard ΔW-proj — all components
  A_sign     : proj_coeff.clamp(0) — directional-only
  B_no_ref   : alpha_reflect=0 — no reflection
  C_no_theta : theta=0 — no per-neuron threshold

Skip D_rand_dir — -76.56pp is unambiguous, T1 unnecessary.

SUCCESS: B_no_ref and C_no_theta both < -0.5pp → all three components are load-bearing
PARTIAL: C_no_theta goes neutral → theta is not load-bearing (update paper table)
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
parser.add_argument("--configs", default="Ref_dw,A_sign,B_no_ref,C_no_theta")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step886_dwproj_ablation_t1_seed{SEED}__{SLOT}.json"

STEP883_REF  = 0.9396
STEP883_SIGN = 0.9213   # -1.83pp
STEP883_NREF = 0.9264   # -1.32pp
STEP883_NTHT = 0.9325   # -0.71pp


class SGNNET_DeltaW_Ablation(nn.Module):
    def __init__(self, resonant, mode: str = "standard"):
        super().__init__()
        self.m = resonant
        self.mode = mode

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x):
        Z = self.m.base._seed(x)
        conn_hh = self.m.base.conn_hh
        W_h = self.m.W_pos[:self.m.base.N_hidden]
        dw = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)

        if self.mode == "C_no_theta":
            theta_pos = torch.zeros(1, 1, 1, device=Z.device)
        else:
            theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)

        alpha_ref = 0.0 if self.mode == "B_no_ref" else ALPHA_REFLECT

        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
            if self.mode == "A_sign":
                Z_nb = Z_nb * proj_coeff.clamp(0)
            else:
                Z_nb = Z_nb * proj_coeff.abs()
            Z_ref = alpha_ref * Z_ref + (Z_fwd - Z)
            Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


CONFIGS = {
    "Ref_dw":    ("standard",   "all components active"),
    "A_sign":    ("A_sign",     "proj_coeff.clamp(0) — directional-only"),
    "B_no_ref":  ("B_no_ref",   "alpha_reflect=0 — no reflection"),
    "C_no_theta":("C_no_theta", "theta=0 — no per-neuron threshold"),
}


def make_model(mode: str) -> nn.Module:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_DeltaW_Ablation(resonant, mode=mode)


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=False)
    n_full = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=False,
    )

    print(f"\n{'='*70}")
    print(f"step886 — ΔW-proj ablation T1 (75ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  T0 context: sign={STEP883_SIGN:.4f}(-1.83pp), no_ref={STEP883_NREF:.4f}(-1.32pp), "
          f"no_theta={STEP883_NTHT:.4f}(-0.71pp)")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        mode, desc = CONFIGS[key]
        model = make_model(mode)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}",
                                   flush=True) if (m['epoch']+1) % 25 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_dw": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP883_REF)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "mode": mode, "label": desc, "n_params": n_p,
            "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "t0_delta": {"A_sign": -0.0183, "B_no_ref": -0.0132, "C_no_theta": -0.0071}.get(key),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 886 SUMMARY — ΔW-proj ablation T1")
    print(f"{'='*70}")
    print(f"  {'config':<12} {'T0_Δ':>8} {'T1_Δ':>8}  verdict")
    for k, r in results.items():
        t0d = f"{r['t0_delta']*100:+.2f}pp" if r.get('t0_delta') else "  (ref)"
        t1d = f"{r['delta_vs_ref']*100:+.2f}pp" if r['delta_vs_ref'] is not None else "  (ref)"
        v = "LOAD-BEARING" if (r['delta_vs_ref'] is not None and r['delta_vs_ref'] < -0.005) else "NEUTRAL"
        print(f"  {k:<12} {t0d:>8} {t1d:>8}  {v}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
