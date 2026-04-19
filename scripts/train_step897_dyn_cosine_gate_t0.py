"""Step 897: Dynamic cosine gate — activation-to-key-weight routing T0 scout.

# CUDA-5060ti-validated

MOTIVATION
==========
All softmax-over-K_hh routing directions killed (step894-896). Root cause: Failure
Mode 7 — softmax [K_hh=2] symmetric fixed point at init, parameter-count-independent.

NEW DIRECTION: dynamic cosine gate over ALL N neurons using a SEPARATE key weight
W_key ∈ ℝ^{N×D} (decoupled from W_pos). W_pos is reserved for ΔW-proj geometry;
W_key learns which neurons to "tune into" dynamically.

MECHANISM
=========
  W_key_n   = normalize(W_key, dim=-1)             # unit key vectors [N, D]
  scores    = Z_fwd @ W_key_n.T                    # cosine similarity [B, N, N]
                                                    #   ∈ [-1, 1] since Z_fwd is l2-normed
  gate      = leaky_relu(scores + b, slope=0.01)   # sparse selection, b is threshold
  Z_dyn     = gate @ Z_fwd / gate.sum(-1, kp=1)   # weighted-mean aggregation [B, N, D]
  Z_out     = normalize((Z_dw + Z_dyn + Z_ref))    # ADDITIVE to ΔW-proj

WHY SEPARATE W_key (not W_pos)
================================
v1 of step897 used normalize(W_pos) for the gate. Collapsed −61pp in ep1.
Root cause: gradient of Z_dyn w.r.t. W_pos has magnitude O(N)=O(2048),
while gradient of Z_dw w.r.t. W_pos has magnitude O(K_hh)=O(2).
Gate gradient dominates by 1024×, hijacking W_pos geometry → ΔW-proj collapses.

Fix: dedicated W_key parameter. W_pos ← owned by ΔW-proj. W_key ← owned by gate.
At inference: HNSW index built on W_key. Query Z_h against W_key to find dynamic
neighbors. W_pos-based static topology (conn_hh) unchanged.

WHY NO SIGMOID (answered by question on normalization)
=======================================================
Z_fwd is l2-normalized (unit sphere). W_key_n is l2-normalized. Scores ∈ [-1, 1].
Sigmoid over [-1,1] → [0.27, 0.73] — compressed, no added selectivity.
LeakyReLU + bias: clean geometric threshold. b=0 selects cos>0 (angle < 90°).
E_raw_wh config tests raw (unnormalized) W_key — magnitude as diversity signal.

CONFIGS (T0, 20ep, 50% data, seed=42)
  Ref_dw       : standard ΔW-proj baseline (~93.96%)
  A_global_b   : ΔW-proj + dyn gate, normalized W_key, b scalar (1 param)
  B_perneuron_b: ΔW-proj + dyn gate, normalized W_key, b ∈ ℝ^N (2048 params)
  C_gate_only  : dyn gate WITHOUT ΔW-proj, b scalar — isolation ablation
  D_topk_eval  : A_global_b at train, W_key topk K=64 at eval — HNSW proxy test
  E_raw_wh     : A_global_b but raw (unnormalized) W_key — magnitude as signal

SUCCESS: A or B ≥+0.5pp → advance to T1 (step900)
ISOLATION: C standalone tests if gate works without ΔW-proj dependency
INFERENCE: D within 1pp of A → W_key topk is viable HNSW approximation
NORM TEST: compare A vs E → does W_key magnitude add or subtract diversity?

Memory: full N×N [B=128, N=2048] = 2.1 GB fp32. Fine for 5060ti (16 GB VRAM)
and mini (64 GB unified memory).
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
parser.add_argument("--configs", default="Ref_dw,A_global_b,B_perneuron_b,C_gate_only,D_topk_eval,E_raw_wh")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5
K_DYN = 64  # topk candidates for D_topk_eval HNSW proxy

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step897_dyn_cosine_gate_t0_seed{SEED}__{SLOT}.json"

STEP_REF = 0.9396


def _dw_proj(W_pos: torch.Tensor, conn_hh: torch.Tensor) -> torch.Tensor:
    """Precompute ΔW-proj direction vectors. [1, N, K_hh, D]."""
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
    """ΔW-proj aggregation. Returns Z_dw [B, N, D]."""
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def _dyn_agg_full(Z_fwd: torch.Tensor, W_key: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Dynamic gate over all N neurons (training path).

    W_key: [N, D] — may be normalized or raw depending on config.
    scores [B, N, N] = Z_fwd @ W_key.T
    gate   [B, N, N] = leaky_relu(scores + b)
    Z_dyn  [B, N, D] = weighted mean of Z_fwd by gate rows
    """
    B = Z_fwd.shape[0]
    scores   = torch.bmm(Z_fwd, W_key.T.unsqueeze(0).expand(B, -1, -1))
    gate     = F.leaky_relu(scores + b, negative_slope=0.01)
    gate_sum = gate.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.bmm(gate, Z_fwd) / gate_sum


def _dyn_agg_topk(
    Z_fwd: torch.Tensor, W_key: torch.Tensor, topk_idx: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Dynamic gate over K_DYN W_key-nearest candidates (HNSW proxy, eval only).

    topk_idx: [N, K_DYN] precomputed W_key-space nearest neighbors.
    """
    Z_cand   = Z_fwd[:, topk_idx, :]                               # [B, N, K_DYN, D]
    W_cand   = W_key[topk_idx]                                     # [N, K_DYN, D]
    scores   = (Z_fwd.unsqueeze(2) * W_cand.unsqueeze(0)).sum(-1)  # [B, N, K_DYN]
    gate     = F.leaky_relu(scores + b, negative_slope=0.01)
    gate_sum = gate.sum(-1, keepdim=True).clamp(min=1e-6)
    return (gate.unsqueeze(-1) * Z_cand).sum(2) / gate_sum         # [B, N, D]


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


class SGNNET_DynGlobalB(nn.Module):
    """ΔW-proj + dyn gate. W_key separate from W_pos. b scalar ∈ ℝ (1 param).

    W_key init: xavier_uniform → unit-scale at D=16. Normalized before scores.
    b = 0 at init → selects ~50% of neurons (cos>0). Network learns threshold.
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
            Z_dyn = _dyn_agg_full(Z_fwd, W_key_n, self.b)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_dw + Z_dyn + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_DynPerNeuronB(nn.Module):
    """ΔW-proj + dyn gate. W_key separate. b ∈ ℝ^N (2048 params).

    Per-neuron query-side threshold. Each neuron learns its own selectivity.
    'Some neurons tune in broadly; others narrowly.'
    """

    def __init__(self, resonant: SGNNET_Resonant):
        super().__init__()
        self.m     = resonant
        self.W_key = nn.Parameter(torch.empty(N, D))
        nn.init.xavier_uniform_(self.W_key)
        self.b     = nn.Parameter(torch.zeros(N))   # [N] query threshold

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
        b_q       = self.b.unsqueeze(0).unsqueeze(-1)              # [1, N, 1]
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_dw  = _dw_agg(Z_nb, dw)
            Z_dyn = _dyn_agg_full(Z_fwd, W_key_n, b_q)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_dw + Z_dyn + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_GateOnly(nn.Module):
    """Dyn gate WITHOUT ΔW-proj. Isolation ablation. b scalar, 1 param.

    If this matches Ref_dw → gate is a full replacement.
    If this kills → gate alone is insufficient; needs ΔW-proj foundation.
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
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_key_n   = F.normalize(self.W_key, dim=-1)
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_dyn = _dyn_agg_full(Z_fwd, W_key_n, self.b)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_dyn + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_TopkEval(nn.Module):
    """A_global_b at train; W_key topk K_DYN=64 at eval (HNSW proxy).

    At eval: precompute W_key cosine similarity matrix, retrieve top-K_DYN
    candidates per neuron, apply same gate. Cached per tick_epoch (W_key changes).
    Gap between A and D = approximation cost of W_key-space topk vs full N×N.
    """

    def __init__(self, resonant: SGNNET_Resonant):
        super().__init__()
        self.m     = resonant
        self.W_key = nn.Parameter(torch.empty(N, D))
        nn.init.xavier_uniform_(self.W_key)
        self.b     = nn.Parameter(torch.zeros(1))
        self._topk_idx: torch.Tensor | None = None

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()
        self._topk_idx = None   # W_key changed → recompute at next eval

    def _get_topk_idx(self, W_key_n: torch.Tensor) -> torch.Tensor:
        if self._topk_idx is None or self._topk_idx.device != W_key_n.device:
            with torch.no_grad():
                w_sim = W_key_n @ W_key_n.T                        # [N, N]
                _, idx = w_sim.topk(K_DYN + 1, dim=-1)
                self._topk_idx = idx[:, 1:].contiguous()           # [N, K_DYN]
        return self._topk_idx

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
            if self.training:
                Z_dyn = _dyn_agg_full(Z_fwd, W_key_n, self.b)
            else:
                topk_idx = self._get_topk_idx(W_key_n)
                Z_dyn = _dyn_agg_topk(Z_fwd, W_key_n, topk_idx, self.b)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_dw + Z_dyn + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_DynRawKey(nn.Module):
    """ΔW-proj + dyn gate with RAW (unnormalized) W_key. b scalar (1 param).

    scores = Z_fwd @ W_key.T — ‖W_key‖ scales the dot product.
    High-magnitude key neurons are easier to select — magnitude as 'broadcasting power'.
    Contrast A_global_b (normalized → pure cosine geometry).
    Scores unbounded; b calibrates the threshold scale.
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
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_dw  = _dw_agg(Z_nb, dw)
            Z_dyn = _dyn_agg_full(Z_fwd, self.W_key, self.b)      # raw W_key
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_dw + Z_dyn + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


EXTRA_PARAMS = {
    "Ref_dw":        0,
    "A_global_b":    N * D + 1,        # W_key + b
    "B_perneuron_b": N * D + N,        # W_key + b_per_neuron
    "C_gate_only":   N * D + 1,        # W_key + b (no dw)
    "D_topk_eval":   N * D + 1,        # W_key + b
    "E_raw_wh":      N * D + 1,        # W_key (raw) + b
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
    if key == "A_global_b":     return SGNNET_DynGlobalB(r)
    if key == "B_perneuron_b":  return SGNNET_DynPerNeuronB(r)
    if key == "C_gate_only":    return SGNNET_GateOnly(r)
    if key == "D_topk_eval":    return SGNNET_TopkEval(r)
    if key == "E_raw_wh":       return SGNNET_DynRawKey(r)
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
    print(f"step897v2 — dynamic cosine gate (W_key decoupled from W_pos) T0 scout")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}  K_dyn(eval)={K_DYN}")
    print(f"  FIX: separate W_key ∈ ℝ^{{N×D}} — prevents W_pos gradient hijacking")
    print(f"  Full N×N training: [{BATCH},{N},{N}] = {BATCH*N*N*4/1e9:.2f} GB fp32")
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
    print(f"STEP 897 SUMMARY — dynamic cosine gate T0")
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
    print(f"  SUCCESS: A or B ≥+0.5pp → T1. C standalone → routing direction confirmed.")
    print(f"  INFERENCE: D within 1pp of A → HNSW viable. NORM: compare A vs E.")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
