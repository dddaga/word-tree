"""Step 906: Top-K Activation Sparsity T0 (20ep, 50% data).

HYPOTHESIS
==========
SGNNET currently activates ALL N=2048 nodes for every input. FGSEGNet v2 /
channel attention work shows that input-conditioned selection of WHICH nodes
(channels) fire can improve both efficiency and accuracy by reducing noise from
irrelevant nodes.

This script tests hard top-K and soft top-K activation gating via the x_sum
signal — the ONLY input-dependent component per node in _seed:

  x_sum[b, i] = sum(x[b, conn_in[i]])    [B, N] scalar per node per sample

High |x_sum| → node i has strong input activation from THIS sample.
Low |x_sum|  → node i is in the "background" for this sample.

Gate applied ONCE after seeding (same as step904 soft gate, but different signal):
  hard: Z[b, i, :] = 0 if |x_sum[b,i]| < threshold_k   (binary mask)
  soft: Z[b, i, :] = Z[b, i, :] * sigmoid(τ · |x_sum|)  (smooth)

EFFICIENCY CASE
===============
If top-50% sparsity holds accuracy: real 2× compute reduction on message-
passing steps (inactive nodes skip aggregation). If top-25%: 4× compute reduction.
Combined with K=1 distil student (step605): could yield sub-0.1M FLOPs at accuracy.

CONFIGS (T0, 20ep, 50% data, seed=42)
  Ref_dw      No gating (standard ΔW-proj baseline)
  A_top75     Keep top 75% nodes (1536/2048) by |x_sum|
  B_top50     Keep top 50% nodes (1024/2048) — 2× message-passing speedup
  C_top25     Keep top 25% nodes (512/2048)  — 4× message-passing speedup
  D_top10     Keep top 10% nodes (205/2048)  — 10× message-passing speedup
  E_soft_tau1 Soft sigmoid gate, τ=1.0 on z-scored |x_sum|
  F_soft_tau3 Soft sigmoid gate, τ=3.0 on z-scored |x_sum|

METRICS TRACKED
===============
  - val_top1: standard accuracy
  - sparsity: mean fraction of nodes zeroed per batch
  - gate_entropy: diversity measure (H>0.3 → genuinely input-conditioned)

ADVANCE RULE
============
  Any config ≥−0.5pp vs Ref_dw AND sparsity ≥20% → accuracy holds under sparsity.
  Any config ≥+0.0pp (neutral) → T1 (step909, 75ep/50%).
  KEY READ: B_top50 and C_top25 — where does accuracy cliff start?
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
parser.add_argument("--device",   default="auto")
parser.add_argument("--epochs",   type=int, default=20)
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data",     default="data/store.h5")
parser.add_argument("--configs",  default="Ref_dw,A_top75,B_top50,C_top25,D_top10,E_soft_tau1,F_soft_tau3")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step906_topk_activation_t0_seed{SEED}__{SLOT}.json"

STEP_REF = 0.9396  # step898 Ref_dw T0 (D=16+ΔW, 20ep/50%)

# Config spec: (gate_type, param)
# gate_type: "none", "topk", "soft"
# param: topk → fraction kept (0.0–1.0); soft → tau value
CONFIG_SPEC = {
    "Ref_dw":      ("none",  0.0),
    "A_top75":     ("topk",  0.75),
    "B_top50":     ("topk",  0.50),
    "C_top25":     ("topk",  0.25),
    "D_top10":     ("topk",  0.10),
    "E_soft_tau1": ("soft",  1.0),
    "F_soft_tau3": ("soft",  3.0),
}


def _dw_proj(W_pos: torch.Tensor, conn_hh: torch.Tensor) -> torch.Tensor:
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def _gate_entropy(gate: torch.Tensor) -> float:
    p = gate.detach().clamp(1e-6, 1 - 1e-6)
    return (-(p * p.log() + (1 - p) * (1 - p).log())).mean().item()


def _topk_gate(x_sum_abs: torch.Tensor, keep_frac: float) -> torch.Tensor:
    """Binary mask: top (keep_frac*N) nodes by |x_sum|, rest zeroed.

    x_sum_abs: [B, N] non-negative.
    Returns: [B, N] float in {0, 1}.
    """
    k = max(1, int(N * keep_frac))
    # kth-largest value per sample
    threshold = x_sum_abs.topk(k, dim=1).values[:, -1:]  # [B, 1]
    return (x_sum_abs >= threshold).float()               # [B, N]


def _soft_gate(x_sum_abs: torch.Tensor, tau: float) -> torch.Tensor:
    """Sigmoid gate on z-scored |x_sum|.

    x_sum_abs: [B, N].
    Returns: [B, N] float in (0, 1).
    """
    mu  = x_sum_abs.mean(dim=1, keepdim=True)
    std = x_sum_abs.std(dim=1, keepdim=True) + 1e-6
    return torch.sigmoid(tau * (x_sum_abs - mu) / std)


class SGNNET_Ref(nn.Module):
    """Standard ΔW-proj baseline — no gating."""

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
        dw        = _dw_proj(self.m.W_pos, conn_hh)
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_agg = _dw_agg(Z_nb, dw)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_TopKGate(nn.Module):
    """Top-K hard gating by |x_sum| applied once after seeding."""

    def __init__(self, resonant: SGNNET_Resonant, gate_type: str, param: float):
        super().__init__()
        self.m         = resonant
        self.gate_type = gate_type
        self.param     = param
        self.last_sparsity    = 0.0
        self.last_gate_entropy = 0.0

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)   # [B, N, D]

        # Recompute x_sum from raw input (same as _seed internals)
        x_sum_abs = x[:, self.m.base.conn_in].sum(dim=2).abs()  # [B, N]

        if self.gate_type == "topk":
            gate = _topk_gate(x_sum_abs, self.param)              # [B, N] {0,1}
        else:  # soft
            gate = _soft_gate(x_sum_abs, self.param)              # [B, N] (0,1)

        # Track diagnostics
        with torch.no_grad():
            self.last_sparsity     = (gate < 0.5).float().mean().item()
            self.last_gate_entropy = _gate_entropy(gate)

        Z = Z * gate.unsqueeze(-1)   # [B, N, D] — apply gate once before K_iter

        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw        = _dw_proj(self.m.W_pos, conn_hh)
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_agg = _dw_agg(Z_nb, dw)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
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
    gate_type, param = CONFIG_SPEC[key]
    r = make_base()
    if gate_type == "none":
        return SGNNET_Ref(r)
    return SGNNET_TopKGate(r, gate_type=gate_type, param=param)


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
    print(f"step906 — Top-K Activation Sparsity T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  Ref context (step898 Ref_dw T0): {STEP_REF:.4f}")
    print(f"  Gate signal: |x_sum[b,i]| = |sum(x[b, conn_in[i]])|")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        gate_type, param = CONFIG_SPEC[key]
        model = make_model(key)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: gate={gate_type}  param={param}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()

        gate_sparsities = []
        gate_entropies  = []

        def log_fn(m: dict) -> None:
            # Collect gate stats from model after each eval epoch
            if hasattr(model, "last_sparsity"):
                gate_sparsities.append(model.last_sparsity)
                gate_entropies.append(model.last_gate_entropy)
            ep_str = (f"  e{m['epoch']+1:3d}  loss={m['train_loss']:.4f}  "
                      f"top1={m['val_top1']:.4f}  lr={m['lr']:.2e}")
            if hasattr(model, "last_sparsity"):
                ep_str += (f"  spar={model.last_sparsity:.2f}"
                           f"  gH={model.last_gate_entropy:.3f}")
            print(ep_str, flush=True)

        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS, log_fn=log_fn)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h)
                 for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_dw":
            ref_acc = best
        delta    = best - (ref_acc if ref_acc is not None else STEP_REF)
        mean_spar = float(np.mean(gate_sparsities)) if gate_sparsities else 0.0
        mean_ge   = float(np.mean(gate_entropies))  if gate_entropies  else None
        ge_str    = f"{mean_ge:.3f}" if mean_ge is not None else "n/a"
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  "
              f"sparsity={mean_spar:.2f}  gate_H={ge_str}  {elapsed:.0f}s")

        results[key] = {
            "gate_type": gate_type, "param": param,
            "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "mean_sparsity": round(mean_spar, 4),
            "gate_entropy_mean": round(mean_ge, 4) if mean_ge is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 906 SUMMARY — Top-K Activation Sparsity T0")
    print(f"{'='*70}")
    print(f"  {'config':<14} {'gate':>6} {'param':>5} {'params':>8} {'best':>7} "
          f"{'Δ_vs_Ref':>10} {'spar':>5} {'gH':>6}  verdict")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "  (ref)"
        ge   = r["gate_entropy_mean"]
        ge_s = f"{ge:.3f}" if ge is not None else "  n/a"
        spar = r["mean_sparsity"]
        if k == "Ref_dw":
            v = "(baseline)"
        elif d is None:
            v = "—"
        elif d >= -0.005 and spar >= 0.20:
            v = "ADVANCE→T1" if d >= 0.000 else "HOLD_EFF"
        elif d >= -0.020:
            v = "MARGINAL"
        else:
            v = "KILL"
        print(f"  {k:<14} {r['gate_type']:>6} {r['param']:>5.2f} "
              f"{r['n_params']:>8,} {r['best']:>7.4f} {dstr:>10} "
              f"{spar:>5.2f} {ge_s:>6}  {v}")

    print(f"\n  Ref T0 context (step898/Ref_dw): {STEP_REF}")
    print(f"  ADVANCE→T1: Δ≥0pp + sparsity≥20% → step909 (75ep/50%).")
    print(f"  HOLD_EFF:   Δ≥−0.5pp + sparsity≥20% → efficiency win even at slight acc loss.")
    print(f"  KEY READS: B_top50 cliff? C_top25 cliff? E/F soft vs hard?")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
