"""Step 896: Biased softmax routing — learnable neighbor-slot bias T0 scout.

MOTIVATION
==========
step859 (KILLED): soft routing used W_pos distance score → T0 artifact + ΔW collapse.
step855 (KILLED): sparse BFS → non-differentiable top-k, capacity starvation.

New hypothesis: softmax redistribution over K_hh neighbors with a SINGLE
learnable bias vector b ∈ ℝ^{K_hh} (2 params for K_hh=2). The bias breaks
symmetry between neighbor slot 0 (local Watts-Strogatz edge) and slot 1
(long-range edge), learning a global asymmetric preference from data.

AH logit provides the structural anchor — prevents the Z-dot noise collapse
(Failure Mode 3, D=16). No Z-dot in the score — purely AH + learned bias.

SIGNAL PATH: b_j is applied to routing weights, NOT to Z propagation directly.
Completely different signal path from ΔW-proj (which modulates Z magnitude via
projection). → Safe to compound (tested in E_compound after isolation).

NOTE: All configs use LeakyReLU (negative_slope=0.01) instead of ReLU for the
threshold activation. Leaky allows gradient flow through sub-threshold neurons,
giving the bias term more opportunity to adjust routing before neurons die.

CONFIGS (T0, 20ep, 50% data, seed=42)
  Ref_dw          : standard ΔW-proj baseline (~93.96%)
  A_bias_khh      : softmax(AH_logit_j + b_j) — b_j ∈ ℝ^{K_hh}, 2 params
  B_bias_step     : per-routing-step bias b_t ∈ ℝ^{K_iter}, 5 params
  C_bias_full     : b_{j,t} ∈ ℝ^{K_hh × K_iter}, 10 params. Full schedule.
  D_temp_only     : learned global temperature γ ∈ ℝ (1 param): AH/γ
  E_compound      : A_bias_khh + ΔW-proj additive (orthogonal path test)
  F_bias_per_node : b_i ∈ ℝ^{N × K_hh}, 4096 params. Each neuron learns its own
                    sociability: how strongly it prefers local vs. long-range
                    neighbor. "Some neurons are more social than others."

SUCCESS: Any config ≥+0.5pp over Ref_dw → advance to T1 (step899)
COMPOUND E: tests whether bias routing + ΔW-proj are truly orthogonal
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
parser.add_argument("--configs", default="Ref_dw,A_bias_khh,B_bias_step,C_bias_full,D_temp_only,E_compound,F_bias_per_node")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step896_biased_soft_routing_t0_seed{SEED}__{SLOT}.json"

STEP_REF = 0.9396


def _ah_logit(resonant: SGNNET_Resonant, conn_hh: torch.Tensor) -> torch.Tensor:
    """W_pos cosine similarity as static AH structural anchor. [1, N, K_hh]."""
    W_h   = resonant.W_pos[:resonant.base.N_hidden]
    W_nb  = W_h[conn_hh]
    W_h_n = F.normalize(W_h, dim=-1)
    W_nb_n= F.normalize(W_nb, dim=-1)
    return (W_h_n.unsqueeze(1) * W_nb_n).sum(-1).unsqueeze(0)  # [1, N, K_hh]


class SGNNET_BiasKhh(nn.Module):
    """Softmax(AH_logit_j + b_j) — b_j ∈ ℝ^{K_hh}. 2 new params."""

    def __init__(self, resonant: SGNNET_Resonant, tau: float = 1.0):
        super().__init__()
        self.m   = resonant
        self.tau = tau
        self.b   = nn.Parameter(torch.zeros(K_HH))           # [K_hh]

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        ah        = _ah_logit(self.m, conn_hh)                # [1, N, K_hh]
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            score = (ah + self.b) / self.tau                   # [1, N, K_hh] → broadcast
            wt    = torch.softmax(score.expand(Z.shape[0], -1, -1), dim=-1).unsqueeze(-1)
            Z_agg = (wt * Z_nb).sum(2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_BiasStep(nn.Module):
    """Per-routing-step scalar bias b_t ∈ ℝ^{K_iter}. 5 new params.

    At step t: sharpness of routing = exp(b_t) * AH_logit.
    b_t > 0 → sharper routing (more focused); b_t < 0 → flatter (more diffuse).
    """

    def __init__(self, resonant: SGNNET_Resonant, tau: float = 1.0):
        super().__init__()
        self.m   = resonant
        self.tau = tau
        self.b   = nn.Parameter(torch.zeros(K_ITER))          # [K_iter]

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        ah        = _ah_logit(self.m, conn_hh)                # [1, N, K_hh]
        Z_ref = torch.zeros_like(Z)
        for t in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            # b_t scales the entire AH logit at step t
            tau_t = self.tau / (self.b[t].exp() + 1e-8)       # learned sharpness per step
            score = ah / tau_t                                 # [1, N, K_hh]
            wt    = torch.softmax(score.expand(Z.shape[0], -1, -1), dim=-1).unsqueeze(-1)
            Z_agg = (wt * Z_nb).sum(2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_BiasFull(nn.Module):
    """b_{j,t} ∈ ℝ^{K_hh × K_iter}. 10 new params. Full routing schedule."""

    def __init__(self, resonant: SGNNET_Resonant, tau: float = 1.0):
        super().__init__()
        self.m   = resonant
        self.tau = tau
        self.b   = nn.Parameter(torch.zeros(K_ITER, K_HH))   # [K_iter, K_hh]

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        ah        = _ah_logit(self.m, conn_hh)                # [1, N, K_hh]
        Z_ref = torch.zeros_like(Z)
        for t in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            score = (ah + self.b[t]) / self.tau               # [1, N, K_hh]
            wt    = torch.softmax(score.expand(Z.shape[0], -1, -1), dim=-1).unsqueeze(-1)
            Z_agg = (wt * Z_nb).sum(2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_TempOnly(nn.Module):
    """Global learned temperature γ ∈ ℝ (1 param). softmax(AH/γ). """

    def __init__(self, resonant: SGNNET_Resonant):
        super().__init__()
        self.m     = resonant
        self.log_g = nn.Parameter(torch.zeros(1))             # log(γ), init γ=1

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        ah        = _ah_logit(self.m, conn_hh)
        tau       = self.log_g.exp().clamp(0.05, 10.0)        # γ ∈ [0.05, 10]
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            score = ah / tau
            wt    = torch.softmax(score.expand(Z.shape[0], -1, -1), dim=-1).unsqueeze(-1)
            Z_agg = (wt * Z_nb).sum(2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_Compound(nn.Module):
    """A_bias_khh + ΔW-proj additive. Tests orthogonality of signal paths.

    Routing weight from bias-AH softmax; ΔW-proj correction added additively.
    Both act: bias routing selects neighbor preference; ΔW gates signal magnitude.
    If orthogonal → compound ≥ A_bias_khh. If same path → cancellation.
    """

    def __init__(self, resonant: SGNNET_Resonant, tau: float = 1.0):
        super().__init__()
        self.m   = resonant
        self.tau = tau
        self.b   = nn.Parameter(torch.zeros(K_HH))

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        ah        = _ah_logit(self.m, conn_hh)

        W_h  = self.m.W_pos[:self.m.base.N_hidden]
        dw   = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)

        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            # Bias-AH routing weight
            score = (ah + self.b) / self.tau
            wt    = torch.softmax(score.expand(Z.shape[0], -1, -1), dim=-1).unsqueeze(-1)
            Z_agg = (wt * Z_nb).sum(2)                        # weighted aggregation
            # ΔW-proj additive correction (same as Ref_dw but on top of Z_agg)
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
            dw_correction = (Z_nb * proj_coeff.abs()).sum(2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + dw_correction + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_BiasPerNode(nn.Module):
    """b_i ∈ ℝ^{N × K_hh}. Each neuron learns its own routing preference.

    "Some neurons are more social than others" — social neurons route strongly
    toward long-range edges; introverted neurons stick to local topology.
    Score = (AH_logit + b_i) / tau, where b_i is neuron-specific.
    Total params: N * K_hh = 2048 * 2 = 4096.
    """

    def __init__(self, resonant: SGNNET_Resonant, tau: float = 1.0):
        super().__init__()
        self.m   = resonant
        self.tau = tau
        self.b   = nn.Parameter(torch.zeros(N, K_HH))        # [N, K_hh]

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        ah        = _ah_logit(self.m, conn_hh)                # [1, N, K_hh]
        # Per-node bias: different sociability per neuron
        score_base = ah + self.b.unsqueeze(0)                 # [1, N, K_hh]
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            score = score_base / self.tau
            wt    = torch.softmax(score.expand(Z.shape[0], -1, -1), dim=-1).unsqueeze(-1)
            Z_agg = (wt * Z_nb).sum(2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_DeltaW_Ref(nn.Module):
    def __init__(self, resonant):
        super().__init__()
        self.m = resonant

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x):
        Z = self.m.base._seed(x)
        conn_hh = self.m.base.conn_hh
        W_h = self.m.W_pos[:self.m.base.N_hidden]
        dw  = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_nb  = Z_nb * proj_coeff.abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


EXTRA_PARAMS = {
    "Ref_dw": 0, "A_bias_khh": K_HH, "B_bias_step": K_ITER,
    "C_bias_full": K_HH * K_ITER, "D_temp_only": 1, "E_compound": K_HH,
    "F_bias_per_node": N * K_HH,
}


def make_base() -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )


def make_model(key: str) -> nn.Module:
    r = make_base()
    if key == "Ref_dw":          return SGNNET_DeltaW_Ref(r)
    if key == "A_bias_khh":      return SGNNET_BiasKhh(r)
    if key == "B_bias_step":     return SGNNET_BiasStep(r)
    if key == "C_bias_full":     return SGNNET_BiasFull(r)
    if key == "D_temp_only":     return SGNNET_TempOnly(r)
    if key == "E_compound":      return SGNNET_Compound(r)
    if key == "F_bias_per_node": return SGNNET_BiasPerNode(r)
    raise ValueError(f"Unknown config: {key}")


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
        batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"\n{'='*70}")
    print(f"step896 — biased softmax routing T0 scout")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  Mechanism: softmax(AH_logit + b) — learnable bias, no Z-dot")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in EXTRA_PARAMS:
            print(f"  skip unknown: {key}"); continue
        model = make_model(key)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        extra = EXTRA_PARAMS[key]
        print(f"{'─'*60}\n{key}: total_params={n_p:,}  extra_routing_params={extra}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}",
                                   flush=True) if (m['epoch']+1) % 5 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_dw":
            ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP_REF)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "extra_params": extra, "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 896 SUMMARY — biased softmax routing T0")
    print(f"{'='*70}")
    print(f"  {'config':<14} {'extra_p':>8} {'best':>7} {'Δ_vs_Ref':>10}  verdict")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "  (ref)"
        v    = "ADVANCE→T1" if (d is not None and d >= 0.005) else \
               "NEUTRAL"   if (d is not None and d >= -0.005) else \
               "KILL"      if (d is not None and d < -0.005) else "(ref)"
        print(f"  {k:<14} {r['extra_params']:>8} {r['best']:>7.4f} {dstr:>10}  {v}")
    print(f"\n  Ref T0 context (step883): {STEP_REF:.4f}")
    print(f"  SUCCESS: A/B/C/D/F ≥+0.5pp → advance to T1 (step899)")
    print(f"  E_compound: tests orthogonality with ΔW-proj")
    print(f"  F_bias_per_node: 4096 params, per-neuron sociability")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
