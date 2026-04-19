"""Step 895: Parameter-free and micro-param routing — T0 scout.

MOTIVATION
==========
step894 tests Z-dot+AH (0 new params). This companion script tests routing
mechanisms that avoid Z-dot entirely (avoiding D=16 noise risk):

- Norm-weighted (0 params): route proportionally to neighbor activation norm
- Shared query W_q (16 params): single routing query vector shared across all N neurons
- Factored attention (128 params): key-query attention at d_k=4 (0.37% of SGNNET)

Also tests input-modulated temperature from step75 (N×D params) to isolate
whether the mechanism works at D=16 at all, before constraining param count.

CONFIGS (T0, 20ep, 50% data, seed=42)
  Ref_dw         : standard ΔW-proj baseline (~93.96%)
  A_norm_weighted: weight_j = softmax(||Z_j||/τ=1.0). 0 new params.
  B_shared_query : score_j = W_q · Z_j / τ=1.0. W_q ∈ ℝ^D = 16 params.
  C_factored_attn: Q=Z_h@W_q^T, K_j=Z_j@W_k^T, d_k=4. 128 params total.
  D_learned_temp : τ_h = τ_0*(1+sigmoid(W_temp_h·Z_h)). W_temp=[N,D]=32768 params.
                   Tests if mechanism works at all (not param-efficient).

SUCCESS: Any config ≥+0.5pp over Ref_dw → advance to T1 (micro-param variant if D)
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
parser.add_argument("--configs", default="Ref_dw,A_norm_weighted,B_shared_query,C_factored_attn,D_learned_temp")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5
D_K = 4  # factored attention key dim

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step895_paramfree_routing_t0_seed{SEED}__{SLOT}.json"

STEP_REF = 0.9396


class SGNNET_NormWeighted(nn.Module):
    """Route by neighbor activation norm — 0 new params."""

    def __init__(self, resonant: SGNNET_Resonant, tau: float = 1.0):
        super().__init__()
        self.m = resonant; self.tau = tau

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)
        conn_hh = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]                   # [B, N, K_hh, D]
            norms = Z_nb.norm(dim=-1) / self.tau            # [B, N, K_hh]
            wt    = torch.softmax(norms, dim=-1).unsqueeze(-1)
            Z_agg = (wt * Z_nb).sum(2)                      # [B, N, D]
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_SharedQuery(nn.Module):
    """Route by shared query W_q · Z_j — 16 new params."""

    def __init__(self, resonant: SGNNET_Resonant, tau: float = 1.0):
        super().__init__()
        self.m   = resonant; self.tau = tau
        self.W_q = nn.Parameter(torch.randn(D) / D**0.5)   # [D]

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)
        conn_hh = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]                   # [B, N, K_hh, D]
            # score_j = W_q · Z_j
            score = (Z_nb * self.W_q).sum(-1) / self.tau    # [B, N, K_hh]
            wt    = torch.softmax(score, dim=-1).unsqueeze(-1)
            Z_agg = (wt * Z_nb).sum(2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_FactoredAttn(nn.Module):
    """Key-query attention at d_k=4 — 128 new params."""

    def __init__(self, resonant: SGNNET_Resonant, d_k: int = D_K):
        super().__init__()
        self.m   = resonant
        self.d_k = d_k
        self.W_q = nn.Parameter(torch.randn(d_k, D) / D**0.5)  # [d_k, D]
        self.W_k = nn.Parameter(torch.randn(d_k, D) / D**0.5)  # [d_k, D]

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)
        conn_hh = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]                   # [B, N, K_hh, D]
            # Q = Z_h @ W_q^T → [B, N, d_k]
            Q = Z_fwd @ self.W_q.T                          # [B, N, d_k]
            # K = Z_nb @ W_k^T → [B, N, K_hh, d_k]
            K = Z_nb @ self.W_k.T                           # [B, N, K_hh, d_k]
            # score = Q · K / sqrt(d_k)
            score = (Q.unsqueeze(2) * K).sum(-1) / self.d_k**0.5  # [B, N, K_hh]
            wt    = torch.softmax(score, dim=-1).unsqueeze(-1)
            Z_agg = (wt * Z_nb).sum(2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_LearnedTemp(nn.Module):
    """Input-modulated temperature (step75 translated to current arch). N×D params."""

    def __init__(self, resonant: SGNNET_Resonant, tau_0: float = 1.0):
        super().__init__()
        self.m     = resonant
        self.tau_0 = tau_0
        # Per-neuron temperature weights — same scale as Z
        self.W_temp = nn.Parameter(torch.randn(N, D) / D**0.5)  # [N, D]

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)
        conn_hh = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]
            # Input-modulated temperature per neuron
            tau_h = self.tau_0 * (1 + torch.sigmoid(
                (Z_fwd * self.W_temp.unsqueeze(0)).sum(-1)))  # [B, N]
            # Use W_pos cosine as routing score, modulated by per-neuron tau
            W_h   = self.m.W_pos[:self.m.base.N_hidden]
            W_nb  = W_h[conn_hh]
            W_h_n = F.normalize(W_h, dim=-1)
            W_nb_n= F.normalize(W_nb, dim=-1)
            score = (W_h_n.unsqueeze(1) * W_nb_n).sum(-1).unsqueeze(0)  # [1, N, K_hh]
            score = score / tau_h.unsqueeze(-1)                          # [B, N, K_hh]
            wt    = torch.softmax(score, dim=-1).unsqueeze(-1)
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
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_nb  = Z_nb * proj_coeff.abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


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
    if key == "A_norm_weighted":  return SGNNET_NormWeighted(r, tau=1.0)
    if key == "B_shared_query":   return SGNNET_SharedQuery(r, tau=1.0)
    if key == "C_factored_attn":  return SGNNET_FactoredAttn(r, d_k=D_K)
    if key == "D_learned_temp":   return SGNNET_LearnedTemp(r, tau_0=1.0)
    raise ValueError(f"Unknown config: {key}")


EXTRA_PARAMS = {
    "Ref_dw": 0, "A_norm_weighted": 0, "B_shared_query": D,
    "C_factored_attn": 2 * D_K * D, "D_learned_temp": N * D,
}


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
    print(f"step895 — parameter-free and micro-param routing T0 scout")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}")
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
        print(f"{'─'*60}\n{key}: params={n_p:,}  extra_routing_params={extra}")

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
    print(f"STEP 895 SUMMARY — parameter-free routing T0")
    print(f"{'='*70}")
    print(f"  {'config':<18} {'extra_p':>8} {'best':>7} {'Δ_vs_Ref':>10}  verdict")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "  (ref)"
        v    = "ADVANCE→T1" if (d is not None and d >= 0.005) else \
               "NEUTRAL"   if (d is not None and d >= -0.005) else \
               "KILL"      if (d is not None and d < -0.005) else "(ref)"
        print(f"  {k:<18} {r['extra_params']:>8} {r['best']:>7.4f} {dstr:>10}  {v}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
