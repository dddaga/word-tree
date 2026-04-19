"""Step 898: K_hh cosine gate — Z-dependent reweighting of static K_hh=2 edges (T0 scout).

# CUDA-5060ti-validated

MOTIVATION
==========
step897 (N×N full dynamic gate) failed: −54pp collapse at ep1.
Root cause: FM8 — dense N×N aggregation creates O(N)=O(2048) gradient on the shared
recurrent Z, vs O(K_hh=2) for ΔW-proj. Gate dominates 1024×, Z dynamics hijacked.

FIX: limit dynamic gate to the SAME K_hh=2 static topology as ΔW-proj.
Both mechanisms then have O(K_hh) gradient on Z → balanced, no hijacking.

MECHANISM
=========
Dynamic gate uses SAME conn_hh as ΔW-proj, but DIFFERENT scoring:
  ΔW-proj: gate = |Z_nb · dw|  (geometric relational direction, static)
  cosine gate: gate = leaky_relu(Z_fwd[i] · W_key_n[j] + b)  (learned, Z-dependent)

  Z_nb   = Z_fwd[:, conn_hh, :]              # [B, N, K_hh, D] — same K_hh neighbors
  W_nb   = W_key_n[conn_hh]                  # [N, K_hh, D] — learned key per neighbor
  scores = (Z_fwd.unsqueeze(2) * W_nb).sum(-1)  # [B, N, K_hh] cosine similarities
  gate   = leaky_relu(scores + b, 0.01)      # sparse threshold, not softmax
  Z_dyn  = sum_k gate_k * Z_nb_k / gate_sum # weighted mean [B, N, D]
  Z      = normalize(Z_dw + Z_dyn + Z_ref)  # ADDITIVE to ΔW-proj

WHY THIS AVOIDS ALL FAILURE MODES
===================================
FM7 (softmax K_hh=2 fixed point): NOT softmax. LeakyReLU has non-symmetric init
  because scores = Z_fwd[i] · W_key_n[j] vary per input. No fixed point.
FM8 (O(N) gradient dominance): K_hh=2 topology → O(K_hh) gradient on Z. Balanced.
FM5 (same-path co-adaptation): Z_dw and Z_dyn both use conn_hh but DIFFERENT parameters
  (W_pos vs W_key) and DIFFERENT aggregation logic (projection vs cosine threshold).
  They're additive, not competitive. C_gate_only ablation tests standalone viability.
FM1 (gate-death): additive (not multiplicative), no g^K decay.

KEY DISTINCTION FROM step896 (biased softmax over K_hh=2):
  step896: score = AH_logit + bias → softmax(K_hh=2) → FM7 fixed point
  step898: score = Z_fwd · W_key_n → leaky_relu → no fixed point (Z varies per input)
  The Z-dependence breaks the symmetry that killed step896.

INFERENCE PATH
==============
At inference, replace W_key_n[conn_hh] with HNSW approximation:
  - Same static topology (conn_hh) — no change needed
  - W_key vectors replace W_pos in the HNSW index (optional, for cross-query routing)
  - Gate is just K_hh=2 cosine lookups — O(K_hh) per neuron per step
No approximation needed — gate operates on existing K_hh=2 edges.

CONFIGS (T0, 20ep, 50% data, seed=42)
  Ref_dw        : standard ΔW-proj baseline (~93.96%)
  A_khh_gate    : ΔW-proj + K_hh cosine gate (W_key, b scalar 1p)
  B_per_neuron_b: ΔW-proj + K_hh cosine gate (W_key, b ∈ ℝ^N per neuron 2048p)
  C_gate_only   : K_hh cosine gate ONLY, no ΔW-proj (isolation test)
  D_per_edge_b  : ΔW-proj + K_hh cosine gate (W_key, b ∈ ℝ^{N×K_hh} per-edge 4096p)

SUCCESS: A or B ≥+0.5pp → T1 (step900)
ISOLATION: C tells if cosine gate works standalone
EXPRESSIVITY: D tests if per-edge bias adds beyond global threshold
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
parser.add_argument("--configs", default="Ref_dw,A_khh_gate,B_per_neuron_b,C_gate_only,D_per_edge_b")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step898_khh_cosine_gate_t0_seed{SEED}__{SLOT}.json"

STEP_REF = 0.9396


def _dw_proj(W_pos: torch.Tensor, conn_hh: torch.Tensor) -> torch.Tensor:
    """Precompute ΔW-proj direction vectors. [1, N, K_hh, D]."""
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
    """ΔW-proj aggregation. Returns Z_dw [B, N, D]."""
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def _khh_cosine_agg(
    Z_fwd: torch.Tensor,
    W_key_n: torch.Tensor,
    conn_hh: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """K_hh cosine gate — Z-dependent sparse gate over static K_hh=2 neighbors.

    Gradient on Z_fwd: O(K_hh=2). Matches ΔW-proj scale. No gradient hijacking.

    Z_fwd:  [B, N, D]     — current activations (unit sphere)
    W_key_n:[N, D]        — normalized key vectors (unit sphere)
    conn_hh:[N, K_hh]     — static neighbor indices
    b:      broadcastable — threshold (scalar, per-neuron, or per-edge)

    NOTE: NO division by gate_sum. leaky_relu returns NEGATIVE values for negative
    inputs (slope=0.01). With K_hh=2 and b=0, ~25% of neurons have both gates
    negative → gate_sum < 0 → clamp(1e-6) creates 4000× amplification → explosion.
    Use weighted SUM (like _dw_agg) — magnitude scales naturally with gate strength.
    """
    Z_nb   = Z_fwd[:, conn_hh, :]                                   # [B, N, K_hh, D]
    W_nb   = W_key_n[conn_hh]                                       # [N, K_hh, D]
    scores = (Z_fwd.unsqueeze(2) * W_nb.unsqueeze(0)).sum(-1)       # [B, N, K_hh]
    gate   = F.leaky_relu(scores + b, negative_slope=0.01)          # [B, N, K_hh]
    return (gate.unsqueeze(-1) * Z_nb).sum(2)                       # [B, N, D]


class SGNNET_DeltaW_Ref(nn.Module):
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
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_dw  = _dw_agg(Z_nb, dw)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_dw + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_KhhGate(nn.Module):
    """ΔW-proj + K_hh cosine gate. W_key separate from W_pos. b scalar (1 param).

    Scoring: Z_fwd[i] · W_key_n[conn_hh[i,k]] — Z-dependent cosine per edge.
    Gate: leaky_relu (no softmax fixed-point). ADDITIVE to ΔW-proj.
    Gradient on Z: O(K_hh=2). Same scale as ΔW-proj — no hijacking.
    """

    def __init__(self, resonant: SGNNET_Resonant):
        super().__init__()
        self.m     = resonant
        self.W_key = nn.Parameter(torch.empty(N, D))
        nn.init.xavier_uniform_(self.W_key)
        self.b     = nn.Parameter(torch.zeros(1))

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
        W_key_n   = F.normalize(self.W_key, dim=-1)
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_dw  = _dw_agg(Z_nb, dw)
            Z_dyn = _khh_cosine_agg(Z_fwd, W_key_n, conn_hh, self.b)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_dw + Z_dyn + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_KhhGatePerNeuronB(nn.Module):
    """ΔW-proj + K_hh cosine gate. W_key separate. b ∈ ℝ^N per-neuron (2048 params).

    Each neuron learns its own selectivity threshold.
    b[i] > 0 → neuron i is selective (prefers strongly-aligning neighbors).
    b[i] < 0 → neuron i accepts weakly-aligning neighbors.
    """

    def __init__(self, resonant: SGNNET_Resonant):
        super().__init__()
        self.m     = resonant
        self.W_key = nn.Parameter(torch.empty(N, D))
        nn.init.xavier_uniform_(self.W_key)
        self.b     = nn.Parameter(torch.zeros(N))   # [N]

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
        W_key_n   = F.normalize(self.W_key, dim=-1)
        b_q       = self.b.unsqueeze(-1)             # [N, 1] → broadcasts over K_hh
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_dw  = _dw_agg(Z_nb, dw)
            Z_dyn = _khh_cosine_agg(Z_fwd, W_key_n, conn_hh, b_q)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_dw + Z_dyn + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_GateOnly(nn.Module):
    """K_hh cosine gate WITHOUT ΔW-proj. Isolation ablation. b scalar.

    If this matches Ref_dw → cosine gate is a viable standalone mechanism.
    If this kills → cosine gate needs ΔW-proj foundation.
    """

    def __init__(self, resonant: SGNNET_Resonant):
        super().__init__()
        self.m     = resonant
        self.W_key = nn.Parameter(torch.empty(N, D))
        nn.init.xavier_uniform_(self.W_key)
        self.b     = nn.Parameter(torch.zeros(1))

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
        W_key_n   = F.normalize(self.W_key, dim=-1)
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_dyn = _khh_cosine_agg(Z_fwd, W_key_n, conn_hh, self.b)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_dyn + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_KhhGatePerEdgeB(nn.Module):
    """ΔW-proj + K_hh cosine gate. W_key separate. b ∈ ℝ^{N×K_hh} per-edge (4096 params).

    Most expressive per-edge threshold: each directed edge (i→j) has its own threshold.
    Tests if per-edge selectivity beyond per-neuron adds signal.
    """

    def __init__(self, resonant: SGNNET_Resonant):
        super().__init__()
        self.m     = resonant
        self.W_key = nn.Parameter(torch.empty(N, D))
        nn.init.xavier_uniform_(self.W_key)
        self.b     = nn.Parameter(torch.zeros(N, K_HH))   # [N, K_hh]

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
        W_key_n   = F.normalize(self.W_key, dim=-1)
        b_e       = self.b.unsqueeze(0)              # [1, N, K_hh]
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_dw  = _dw_agg(Z_nb, dw)
            Z_dyn = _khh_cosine_agg(Z_fwd, W_key_n, conn_hh, b_e)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_dw + Z_dyn + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


EXTRA_PARAMS = {
    "Ref_dw":         0,
    "A_khh_gate":     N * D + 1,        # W_key + b_scalar
    "B_per_neuron_b": N * D + N,        # W_key + b_per_neuron
    "C_gate_only":    N * D + 1,        # W_key + b (no dw)
    "D_per_edge_b":   N * D + N * K_HH, # W_key + b_per_edge
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
    if key == "Ref_dw":         return SGNNET_DeltaW_Ref(r)
    if key == "A_khh_gate":     return SGNNET_KhhGate(r)
    if key == "B_per_neuron_b": return SGNNET_KhhGatePerNeuronB(r)
    if key == "C_gate_only":    return SGNNET_GateOnly(r)
    if key == "D_per_edge_b":   return SGNNET_KhhGatePerEdgeB(r)
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
    print(f"step898 — K_hh cosine gate (Z-dependent, O(K_hh) gradient) T0 scout")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  FIX: K_hh=2 topology → O(K_hh) gradient on Z. Matches ΔW-proj scale.")
    print(f"  KEY vs step896: leaky_relu (not softmax) + Z-dependent score (not AH)")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in EXTRA_PARAMS:
            print(f"  skip unknown: {key}"); continue
        model  = make_model(key)
        n_p    = sum(p.numel() for p in model.parameters() if p.requires_grad)
        extra  = EXTRA_PARAMS[key]
        print(f"{'─'*60}\n{key}: total_params={n_p:,}  extra_routing_params={extra}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(
                f"  e{m['epoch']+1:3d}  loss={m['train_loss']:.4f}  "
                f"task={m.get('task_loss', m['train_loss']):.4f}  "
                f"safety={m.get('safety_loss', 0.0):.4f}  "
                f"top1={m['val_top1']:.4f}  lr={m['lr']:.2e}",
                flush=True,
            ))
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
    print(f"STEP 898 SUMMARY — K_hh cosine gate T0")
    print(f"{'='*70}")
    print(f"  {'config':<16} {'extra_p':>8} {'best':>7} {'Δ_vs_Ref':>10}  verdict")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "  (ref)"
        v    = "ADVANCE→T1" if (d is not None and d >= 0.005) else \
               "NEUTRAL"   if (d is not None and d >= -0.005) else \
               "KILL"      if (d is not None and d < -0.005) else "(ref)"
        print(f"  {k:<16} {r['extra_params']:>8} {r['best']:>7.4f} {dstr:>10}  {v}")
    print(f"\n  Ref T0 context (step896): {STEP_REF:.4f}")
    print(f"  SUCCESS: A or B ≥+0.5pp → T1 (step900).")
    print(f"  ISOLATION: C confirms standalone viability of cosine gate.")
    print(f"  EXPRESSIVITY: D tests per-edge vs global threshold.")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
