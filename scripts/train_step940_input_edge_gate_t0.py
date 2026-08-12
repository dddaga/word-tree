"""Step 940: Input-conditioned edge reweighting T0 (20ep, 50% data).

Port of step36 (AH era, +2.34pp at D=64 without AntiHebb) to current ΔW-proj arch.
Edge weights modulated by raw input feature similarity (not Z state — no gate-death).
  w_ij = sigmoid(x_sum_i · x_sum_j / tau)  [fixed per sample, not updated by K_iter]
  Z_agg = Σ Z_nb × |proj_coeff| × w_ij

KEY DISTINCTION from killed gates:
  - Z-state gates: g^{K_iter} → 0 (gate-death). KILLED in steps 895-898.
  - Input-based: x is FIXED per sample, no iterative compounding. Different family.
  - Step898 tested edge gating with Z-based bias → neutral. Input-based untested.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, ΔW-proj + α_AH=1.0, 20ep/50%)
  Ref          standard ΔW-proj (no edge gate)
  A_tau1       τ=1.0 (default sharpness)
  B_tau3       τ=3.0 (softer modulation)
  C_tau_lr     τ learned (initialized=1.0, scalar param)

ADVANCE: any config ≥+0.5pp vs Ref → T1.
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
parser.add_argument("--configs", default="Ref,A_tau1,B_tau3,C_tau_lr")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step940_input_edge_gate_t0_seed{SEED}__{SLOT}.json"
STEP_REF = 0.9396


# ---------------------------------------------------------------------------
def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)

def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)

def _dw_agg_weighted(Z_nb, dw, edge_w):
    """edge_w: [B, N, K_hh] — per-edge scalar weight."""
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs() * edge_w.unsqueeze(-1)).sum(dim=2)


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


class SGNNET_InputEdgeGate(nn.Module):
    """Input-conditioned edge reweighting: w_ij = sigmoid(x_i·x_j / tau)."""

    def __init__(self, resonant, tau: float, learn_tau: bool = False):
        super().__init__()
        self.m = resonant
        if learn_tau:
            self.log_tau = nn.Parameter(torch.tensor(float(tau)).log())
        else:
            self.register_buffer("log_tau", torch.tensor(float(tau)).log())
        self.learn_tau = learn_tau

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x):
        # x: [B, N_in]; scatter to [B, N, D] via seed, but we need node-level features
        Z = self.m.base._seed(x)           # [B, N, D]
        conn_hh   = self.m.base.conn_hh    # [N, K_hh]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj(self.m.W_pos, conn_hh)

        # Compute input edge weights: use Z at init (post-seed) as proxy for node input features
        # Z is [B, N, D]; compute similarity between node i and its neighbors conn_hh[i]
        tau = self.log_tau.exp()
        Z0_i = Z                                    # [B, N, D]
        Z0_j = Z[:, conn_hh, :]                     # [B, N, K_hh, D]
        # dot product: [B, N, K_hh]
        sim = (Z0_i.unsqueeze(2) * Z0_j).sum(-1) / tau
        edge_w = torch.sigmoid(sim)                 # [B, N, K_hh] — fixed per sample

        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_agg = _dw_agg_weighted(Z_nb, dw, edge_w)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


CONFIG_SPEC = {
    "Ref":       ("ref",      1.0,  False),
    "A_tau1":    ("gate",     1.0,  False),
    "B_tau3":    ("gate",     3.0,  False),
    "C_tau_lr":  ("gate",     1.0,  True),
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
    spec, tau, learn_tau = CONFIG_SPEC[key]
    r = make_base()
    if spec == "ref":
        return SGNNET_Ref(r)
    return SGNNET_InputEdgeGate(r, tau=tau, learn_tau=learn_tau)


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
    print(f"step940 — Input edge reweighting T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  AH era ref (step36): +2.34pp at D=64 w/o AntiHebb")
    print(f"  Ref context (step907/Ref_dw): {STEP_REF:.4f}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        model = make_model(key).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        spec, tau, learn_tau = CONFIG_SPEC[key]
        print(f"{'─'*60}")
        print(f"{key}: spec={spec}  tau={tau}  learn_tau={learn_tau}  params={n_p:,}")

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
        delta = best - (ref_acc if ref_acc is not None else STEP_REF)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {"spec": spec, "tau": tau, "learn_tau": learn_tau,
                        "n_params": n_p, "best": round(best, 4), "best_ep": best_ep,
                        "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 940 SUMMARY — Input edge reweighting T0")
    print(f"{'='*70}")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        v = "(baseline)" if k == "Ref" else ("ADVANCE→T1" if d >= 0.005 else ("NEUTRAL" if d >= -0.005 else "KILL"))
        print(f"  {k:<14} params={r['n_params']:>6,}  best={r['best']:.4f}  Δ={d*100:+.2f}pp  {v}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
