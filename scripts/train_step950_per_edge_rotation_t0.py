"""Step 950: Per-edge channel rotation T0 (20ep, 50% data).

Tests whether a learned per-edge D×D matrix W_edge provides useful channel
mixing during routing, independent of ΔW-proj.

Ref has NO ΔW-proj — clean isolation of W_edge contribution.
Question: can W_edge alone (without geometric scalar weighting) learn
selective signal propagation?

CONFIGS (N=2048, K_hh=2, K_iter=5, NO ΔW-proj, 20ep/50%)
  Ref           D=16, simple gather-sum, no ΔW-proj, no W_edge
                Param count: 34,976 (same as canonical SGNNET)
  A_shared      D=16, W_edge SHARED [K_hh, D, D] = [2, 16, 16]
                Extra params: 512  → total: 35,488
  B_peredge     D=16, W_edge PER NODE [N, K_hh, D, D] = [2048, 2, 16, 16]
                Extra params: 1,048,576  → total: 1,083,552
  C_small_D8    D=8, W_edge PER NODE [N, K_hh, D, D] = [2048, 2, 8, 8]
                Extra params: 262,144  → total: ~280,656

NOTE: step887 canonical ΔW-proj Ref = 96.38%. step937 Ref (no ΔW-proj) = ~94.19%.
This baseline difference is expected and load-bearing: W_edge needs to close
or exceed the ~2pp gap that ΔW-proj provides.

ADVANCE: ≥+0.5pp vs THIS experiment's Ref → T1.
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
parser.add_argument("--configs", default="Ref,A_shared,B_peredge,C_small_D8")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step950_per_edge_rotation_t0_seed{SEED}__{SLOT}.json"

# step887 canonical ΔW-proj result — context only, not the comparison baseline
DW_PROJ_REF = 0.9638


# ---------------------------------------------------------------------------
class SGNNET_GatherSum(nn.Module):
    """Simple gather-sum baseline — no ΔW-proj, no W_edge.

    This is the Ref for step950. Intentionally weaker than canonical ΔW-proj
    to measure what W_edge adds on its own.
    """

    D = 16

    def __init__(self, resonant):
        super().__init__()
        self.m = resonant

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x):
        Z = self.m.base._seed(x)                           # [B, N, D]
        conn_hh   = self.m.base.conn_hh                    # [N, K_hh]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        Z_ref = torch.zeros_like(Z)

        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]                   # [B, N, K_hh, D]
            Z_agg = Z_nb.sum(dim=2)                        # [B, N, D]
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ---------------------------------------------------------------------------
class SGNNET_PerEdgeRot(nn.Module):
    """Per-edge D×D matrix transforms neighbor activations before aggregation.

    No scalar weighting from ΔW-proj. W_edge learns its own notion of
    "which directions matter" through channel rotation.

    W_edge modes:
      'shared'   — [K_hh, D, D]       shared across all nodes
      'per_node' — [N, K_hh, D, D]    specialized per node-edge pair
    Init: identity matrix → model starts at gather-sum baseline.
    """

    def __init__(self, resonant, D: int, mode: str):
        super().__init__()
        self.m    = resonant
        self.D    = D
        self.mode = mode

        if mode == "shared":
            W = torch.eye(D).unsqueeze(0).expand(K_HH, -1, -1).clone()
            self.W_edge = nn.Parameter(W)                  # [K_hh, D, D]
        elif mode == "per_node":
            W = torch.eye(D).unsqueeze(0).unsqueeze(0).expand(N, K_HH, -1, -1).clone()
            self.W_edge = nn.Parameter(W)                  # [N, K_hh, D, D]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x):
        Z = self.m.base._seed(x)                           # [B, N, D]
        conn_hh   = self.m.base.conn_hh                    # [N, K_hh]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        Z_ref = torch.zeros_like(Z)

        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]                   # [B, N, K_hh, D]

            if self.mode == "shared":
                # W_edge: [K_hh, D, D]
                Z_rot = torch.einsum('bnkd,kde->bnke', Z_nb, self.W_edge)
            else:
                # W_edge: [N, K_hh, D, D]
                Z_rot = torch.einsum('bnkd,nkde->bnke', Z_nb, self.W_edge)

            Z_agg = Z_rot.sum(dim=2)                       # [B, N, D]
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ---------------------------------------------------------------------------
CONFIG_SPEC = {
    # (D, mode)
    "Ref":         (16, "ref"),
    "A_shared":    (16, "shared"),
    "B_peredge":   (16, "per_node"),
    "C_small_D8":  (8,  "per_node"),
}


def _param_count_expected(key):
    D, mode = CONFIG_SPEC[key]
    base = (N + N_OUT) * D + N   # W_pos + theta
    if mode == "ref":
        w_edge = 0
    elif mode == "shared":
        w_edge = K_HH * D * D
    else:  # per_node
        w_edge = N * K_HH * D * D
    return base, w_edge, base + w_edge


def make_base(D: int = 16):
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
    D, mode = CONFIG_SPEC[key]
    r = make_base(D)
    if mode == "ref":
        return SGNNET_GatherSum(r)
    return SGNNET_PerEdgeRot(r, D=D, mode=mode)


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
    print(f"step950 — Per-edge channel rotation T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/K_hh={K_HH}/K_iter={K_ITER}/K_in={K_IN}")
    print(f"  NO ΔW-proj — isolating W_edge contribution only")
    print(f"  W_edge init: identity (starts at gather-sum baseline)")
    print(f"  Context: step887 ΔW-proj canonical = {DW_PROJ_REF:.4f}")
    print(f"")
    print(f"  Expected param counts:")
    for k in CONFIG_SPEC:
        base, w_edge, total = _param_count_expected(k)
        D, mode = CONFIG_SPEC[k]
        print(f"    {k:<18} D={D}  base={base:>6,}  W_edge={w_edge:>9,}  total={total:>10,}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        model = make_model(key).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        D, mode = CONFIG_SPEC[key]
        _, _, total_exp = _param_count_expected(key)
        print(f"{'─'*60}")
        print(f"{key}: D={D}  mode={mode}  params={n_p:,}  (expected={total_exp:,})")

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
        delta = best - (ref_acc if ref_acc is not None else 0.9419)
        verdict = ("(baseline)" if key == "Ref"
                   else "ADVANCE→T1" if delta >= 0.005
                   else ("NEUTRAL" if delta >= -0.005 else "KILL"))
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp vs gather-sum Ref  {verdict}")

        results[key] = {
            "D": D, "mode": mode, "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 950 SUMMARY — Per-edge channel rotation T0")
    print(f"  (Ref = gather-sum WITHOUT ΔW-proj)")
    print(f"  (Context: ΔW-proj canonical = {DW_PROJ_REF:.4f})")
    print(f"{'='*70}")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        v = ("(baseline)" if k == "Ref"
             else ("ADVANCE→T1" if d >= 0.005 else ("NEUTRAL" if d >= -0.005 else "KILL")))
        print(f"  {k:<18} D={r['D']}  params={r['n_params']:>10,}  best={r['best']:.4f}  Δ={d*100:+.2f}pp  {v}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
