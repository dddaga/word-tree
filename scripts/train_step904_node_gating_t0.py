"""Step 904: Node-Level Input-Conditioned Gating T0 (20ep, 50% data).

MOTIVATION
==========
All prior dynamic routing attempts gated EDGES (which neighbor to aggregate
from). Every one collapsed via gradient noise. This script gates NODES instead
— analogous to channel attention (SE-Net, CBAM, FGSEGNet v2).

The key insight: SGNNET's _seed(x) computes x_sum[i] = Σ_k x[conn_in[i,k]],
the raw scalar input activation at node i. This varies per input and captures
which nodes are "lit up" by the current image. A z-score gate over x_sum:

    gate[b,i] = sigmoid(τ · (x_sum[b,i] - μ[b]) / σ[b])

amplifies above-average nodes and suppresses below-average ones. This is:
  1. ZERO extra parameters in the base variant
  2. Applied ONCE after seeding (not per K_iter step — no recurrent noise)
  3. Directly analogous to SE-Net channel attention applied after spatial pooling
  4. Input-conditioned: different images → different node subsets active

Gate is applied to Z after _seed, before K_iter routing. ΔW-proj is retained.

DIVERSITY METRIC
================
Per-batch gate entropy H = E[-p·log(p) - (1-p)·log(1-p)] logged each eval
epoch. Low H → gate is static (near 0 or 1 always). High H → gate is diverse
and input-conditioned. This distinguishes "dynamic selection" from mere scaling.

CONFIGS (T0, 20ep, 50% data, seed=42)
  Ref_dw          no gate (standard ΔW-proj baseline)
  A_tau1          z-score gate τ=1.0 (soft)
  B_tau3          z-score gate τ=3.0 (sharper — top ~16% active)
  C_tau_learnt    τ as nn.Parameter (starts 1.0, adapts)
  D_se_bn         SE-Net bottleneck: Z_seed.mean → W_down[D,4] → W_up[4,N]
  E_wpos_geo      geometric gate: cosine(W_pos[i], Z_seed.mean(dim=1))
  F_per_iter      z-score gate τ=1.0 recomputed each K_iter (recurrent, risky)

ADVANCE RULE
============
  Any config ≥+0.5pp vs Ref_dw → T1 (step906, 75ep/50%).
  KEY READ: gate_entropy in logs — does the gate actually vary across inputs?
  If gate_entropy < 0.1 nats → static routing, mechanism is degenerate.
  If gate_entropy > 0.5 nats AND accuracy ≥ Ref → genuine input-conditioned routing.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
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
parser.add_argument("--configs", default="Ref_dw,A_tau1,B_tau3,C_tau_learnt,D_se_bn,E_wpos_geo,F_per_iter")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step904_node_gating_t0_seed{SEED}__{SLOT}.json"

STEP_REF = 0.9396  # step898 Ref_dw T0 (D=16+ΔW, 20ep/50%)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _dw_proj(W_pos: torch.Tensor, conn_hh: torch.Tensor) -> torch.Tensor:
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def _gate_entropy(gate: torch.Tensor) -> float:
    """Mean binary entropy of gate values. Max = ln(2) ≈ 0.693 nats."""
    p = gate.detach().clamp(1e-6, 1 - 1e-6)
    return (-(p * p.log() + (1 - p) * (1 - p).log())).mean().item()


def _zscore_gate(x_sum: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
    """Per-sample z-score gate over node input activations.

    x_sum: [B, N] raw input activations.
    Returns gate [B, N] in (0, 1).
    """
    mu  = x_sum.mean(dim=1, keepdim=True)
    std = x_sum.std(dim=1, keepdim=True) + 1e-6
    return torch.sigmoid(tau * (x_sum - mu) / std)


# ── Model wrappers ────────────────────────────────────────────────────────────

class SGNNET_Ref(nn.Module):
    """Standard ΔW-proj baseline — no gate."""

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
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            dw    = _dw_proj(self.m.W_pos, conn_hh)
            Z_agg = _dw_agg(Z_nb, dw)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_ZscoreGate(nn.Module):
    """Z-score gate on node input activations, applied once after seeding."""

    def __init__(self, resonant: SGNNET_Resonant, tau: float = 1.0,
                 learnable_tau: bool = False, per_iter: bool = False):
        super().__init__()
        self.m        = resonant
        self.per_iter = per_iter
        if learnable_tau:
            self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        else:
            self.register_buffer("log_tau", torch.tensor(math.log(tau)))
        self.last_gate_entropy = 0.0

    @property
    def tau(self): return self.log_tau.exp()

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Recompute x_sum from raw input — same computation as _seed's first step."""
        x_sum = x[:, self.m.base.conn_in].sum(dim=2)  # [B, N]
        gate  = _zscore_gate(x_sum, self.tau)
        self.last_gate_entropy = _gate_entropy(gate)
        return gate  # [B, N]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)

        if not self.per_iter:
            gate = self._compute_gate(x).unsqueeze(-1)  # [B, N, 1]
            Z = Z * gate

        Z_ref = torch.zeros_like(Z)
        for step in range(K_ITER):
            if self.per_iter and step == 0:
                gate = self._compute_gate(x).unsqueeze(-1)
            elif self.per_iter:
                # Recompute gate from current Z (recurrent)
                z_mean = Z.mean(dim=1)                       # [B, D]
                z_mean_n = F.normalize(z_mean, dim=-1)
                W_h  = F.normalize(self.m.W_pos[:N].detach(), dim=-1)
                gate = torch.sigmoid(self.tau * (W_h @ z_mean_n.T).T
                                     ).unsqueeze(-1)         # [B, N, 1]
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            if self.per_iter:
                Z_fwd = Z_fwd * gate
            Z_nb  = Z_fwd[:, conn_hh, :]
            dw    = _dw_proj(self.m.W_pos, conn_hh)
            Z_agg = _dw_agg(Z_nb, dw)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_SEGate(nn.Module):
    """SE-Net bottleneck: Z_seed global mean → W_down → ReLU → W_up → sigmoid."""

    def __init__(self, resonant: SGNNET_Resonant, r: int = 4):
        super().__init__()
        self.m      = resonant
        self.W_down = nn.Linear(D, r, bias=False)
        self.W_up   = nn.Linear(r, N, bias=False)
        self.last_gate_entropy = 0.0

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

        z_avg = Z.mean(dim=1)                                   # [B, D]
        gate  = torch.sigmoid(self.W_up(F.relu(self.W_down(z_avg))))  # [B, N]
        self.last_gate_entropy = _gate_entropy(gate)
        Z = Z * gate.unsqueeze(-1)                             # [B, N, D]

        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            dw    = _dw_proj(self.m.W_pos, conn_hh)
            Z_agg = _dw_agg(Z_nb, dw)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_WposGeoGate(nn.Module):
    """Geometric gate: sigmoid(cosine(W_pos[i], Z_seed_mean)). Zero new params."""

    def __init__(self, resonant: SGNNET_Resonant):
        super().__init__()
        self.m = resonant
        self.last_gate_entropy = 0.0

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

        z_mean = F.normalize(Z.mean(dim=1), dim=-1)            # [B, D]
        W_h    = F.normalize(self.m.W_pos[:N], dim=-1)         # [N, D]
        gate   = torch.sigmoid((W_h @ z_mean.T).T).unsqueeze(-1)  # [B, N, 1]
        self.last_gate_entropy = _gate_entropy(gate.squeeze(-1))
        Z = Z * gate

        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            dw    = _dw_proj(self.m.W_pos, conn_hh)
            Z_agg = _dw_agg(Z_nb, dw)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


# ── Construction ──────────────────────────────────────────────────────────────

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
    if key == "Ref_dw":       return SGNNET_Ref(r)
    if key == "A_tau1":       return SGNNET_ZscoreGate(r, tau=1.0)
    if key == "B_tau3":       return SGNNET_ZscoreGate(r, tau=3.0)
    if key == "C_tau_learnt": return SGNNET_ZscoreGate(r, tau=1.0, learnable_tau=True)
    if key == "D_se_bn":      return SGNNET_SEGate(r, r=4)
    if key == "E_wpos_geo":   return SGNNET_WposGeoGate(r)
    if key == "F_per_iter":   return SGNNET_ZscoreGate(r, tau=1.0, per_iter=True)
    raise ValueError(key)


# ── Main ──────────────────────────────────────────────────────────────────────

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
    print(f"step904 — Node-Level Input-Conditioned Gating T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Gate signal: x_sum = x[:, conn_in].sum(dim=2)  [B, N]")
    print(f"  Applied: ONCE after seeding, before K_iter routing")
    print(f"  Logging: gate_entropy per eval epoch (0=static, ln2≈0.693=maximally diverse)")
    print(f"  Ref context (step898): {STEP_REF:.4f}")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        model = make_model(key)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: params={n_p:,}")

        # Track gate entropy during validation
        gate_entropies: list[float] = []
        _orig_eval = model.eval

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()

        def log_fn(m: dict):
            ge = getattr(model, "last_gate_entropy", None)
            ge_str = f"  gate_H={ge:.3f}" if ge is not None else ""
            print(f"  e{m['epoch']+1:3d}  loss={m['train_loss']:.4f}  "
                  f"top1={m['val_top1']:.4f}  lr={m['lr']:.2e}{ge_str}",
                  flush=True)
            if ge is not None:
                gate_entropies.append(ge)

        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(n_epochs=EPOCHS, log_fn=log_fn)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h)
                 for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_dw":
            ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP_REF)
        mean_ge = float(np.mean(gate_entropies)) if gate_entropies else None
        ge_str = f"{mean_ge:.3f}" if mean_ge is not None else "n/a"
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  "
              f"gate_H_mean={ge_str}  {elapsed:.0f}s")

        results[key] = {
            "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "gate_entropy_mean": round(mean_ge, 4) if mean_ge is not None else None,
            "gate_entropy_curve": [round(v, 4) for v in gate_entropies],
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 904 SUMMARY — Node-Level Input-Conditioned Gating T0")
    print(f"{'='*70}")
    print(f"  {'config':<14} {'params':>8} {'best':>7} {'Δ':>8}  {'gate_H':>7}  verdict")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "  (ref)"
        ge   = r["gate_entropy_mean"]
        ge_s = f"{ge:.3f}" if ge is not None else "  n/a"
        v    = "ADVANCE→T1" if (d is not None and d >= 0.005) else \
               "NEUTRAL"    if (d is not None and d >= -0.005) else \
               "MARGINAL"   if (d is not None and d >= -0.020) else "KILL"
        dyn  = "" if ge is None else (" [DYNAMIC]" if ge > 0.4 else " [static]")
        print(f"  {k:<14} {r['n_params']:>8,} {r['best']:>7.4f} {dstr:>8}  {ge_s:>7}  {v}{dyn}")

    print(f"\n  gate_H interpretation: <0.1=degenerate, 0.1-0.4=partial, >0.4=input-conditioned")
    print(f"  ln(2)={math.log(2):.3f} nats = maximum possible entropy (uniform gate)")
    print(f"  ADVANCE: ≥+0.5pp AND gate_H>0.3 → genuine input-conditioned routing (step906)")
    print(f"  PARTIAL: ≥+0.5pp BUT gate_H<0.1 → accuracy gain without real dynamic routing")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
