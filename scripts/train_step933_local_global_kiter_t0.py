"""Step 933: Local/Global alternating K_iter rounds T0 (20ep, 50% data).

The canonical step887 base aggregates from fixed K_hh=2 neighbors every round.
This experiment alternates between:
  - Local round:  aggregate from conn_hh (K_hh=2 spatial neighbors)
  - Global round: aggregate from K_global randomly-sampled nodes across all N

Pattern controlled by `global_every` — the last round in each window is global.
  global_every=2: [local, global, local, global, local]  (2 global in 5)
  global_every=3: [local, local, global, local, local]   (1 global in 5)
  global_every=5: [local, local, local, local, global]   (1 global in 5, last only)

Both local and global rounds use ΔW-proj direction (geometric scalar gating).
Global round samples fresh random conn each forward pass — no fixed structure.
Aggregation is SUM in both cases → no gate-death risk.
Z_ref (reflection buffer) applies uniformly to all rounds.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, ΔW-proj, 20ep/50%)
  Ref:       standard ΔW-proj (all local, K_hh=2) — step887 canonical
  A_ge2_kg4: global_every=2, K_global=4
  B_ge3_kg4: global_every=3, K_global=4
  C_ge2_kg8: global_every=2, K_global=8
  D_ge5_kg4: global_every=5 (1 global in 5), K_global=4

ADVANCE: ≥+0.5pp vs Ref → T1.
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
parser.add_argument("--configs", default="Ref,A_ge2_kg4,B_ge3_kg4,C_ge2_kg8,D_ge5_kg4")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 25
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step933_local_global_kiter_t0_seed{SEED}__{SLOT}.json"

DW_REF = 0.9638  # step887 canonical


# CONFIG_SPEC: (global_every, K_global) — None means pure local (Ref)
CONFIG_SPEC = {
    "Ref":       None,
    "A_ge2_kg4": (2, 4),
    "B_ge3_kg4": (3, 4),
    "C_ge2_kg8": (2, 8),
    "D_ge5_kg4": (5, 4),
}


def _dw_proj_vec(W_pos, conn):
    """ΔW-proj direction vectors for arbitrary conn [N, K].

    Returns [1, N, K, D] normalized difference vectors.
    """
    W_h = W_pos[:N]                                           # [N, D]
    # W_h[conn] → [N, K, D];  unsqueeze → [1, N, K, D]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn], dim=-1).unsqueeze(0)


class SGNNET_Ref(nn.Module):
    """Standard ΔW-proj, all-local (step887 canonical)."""

    def __init__(self, resonant):
        super().__init__()
        self.m = resonant

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x):
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj_vec(self.m.W_pos, conn_hh)             # [1,N,K_hh,D]
        Z_ref = torch.zeros_like(Z)

        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]                     # [B,N,K_hh,D]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


class SGNNET_LocalGlobal(nn.Module):
    """Alternating local/global K_iter rounds with ΔW-proj.

    Local rounds use conn_hh (fixed spatial K_hh neighbors).
    Global rounds sample K_global random indices per node per step.
    Both use ΔW-proj geometric scalar gating.
    """

    def __init__(self, resonant, global_every: int, K_global: int):
        super().__init__()
        self.m            = resonant
        self.global_every = global_every
        self.K_global     = K_global

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x):
        B = x.shape[0]
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh                      # [N, K_hh]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw_local  = _dw_proj_vec(self.m.W_pos, conn_hh)      # [1,N,K_hh,D] — precomputed

        Z_ref = torch.zeros_like(Z)

        for k in range(K_ITER):
            # Last round in each window of size global_every is a global round
            is_global = (k % self.global_every) == (self.global_every - 1)

            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)

            if is_global:
                # Sample K_global random node indices per node — fresh each step
                g_conn = torch.randint(
                    0, N, (N, self.K_global), device=Z.device
                )                                             # [N, K_global]
                dw = _dw_proj_vec(self.m.W_pos, g_conn)      # [1,N,K_global,D]
                # Gather: Z_fwd[:, g_conn, :] via flatten trick
                Z_nb = Z_fwd[:, g_conn.view(-1), :].view(B, N, self.K_global, D)
            else:
                dw   = dw_local                              # [1,N,K_hh,D]
                Z_nb = Z_fwd[:, conn_hh, :]                  # [B,N,K_hh,D]

            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)   # [B,N,K,1]
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)           # [B,N,D]
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


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
    spec = CONFIG_SPEC[key]
    r = make_base()
    if spec is None:
        return SGNNET_Ref(r)
    global_every, K_global = spec
    return SGNNET_LocalGlobal(r, global_every=global_every, K_global=K_global)


def _desc(key):
    spec = CONFIG_SPEC[key]
    if spec is None:
        return "all-local (K_hh=2)"
    ge, kg = spec
    return f"global_every={ge}, K_global={kg}"


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

    print(f"\n{'='*72}")
    print(f"step933 — Local/Global alternating K_iter rounds T0 (20ep, 50%)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}/K_in={K_IN}")
    print(f"  Mechanism: alternate local (conn_hh) / global (random) rounds")
    print(f"  Both rounds use ΔW-proj scalar gating — SUM aggregation, no gate-death.")
    print(f"  Context: step887 canonical = {DW_REF:.4f}")
    print()
    print(f"  {'Config':<12} {'description'}")
    for k in CONFIG_SPEC:
        print(f"  {k:<12} {_desc(k)}")
    print(f"{'='*72}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue

        model = make_model(key).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"{'─'*60}")
        print(f"{key}: {_desc(key)}  params={n_p:,}")

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
                   else "ADVANCE→T1" if delta >= 0.005
                   else ("NEUTRAL" if delta >= -0.005 else "KILL"))
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {verdict}")

        results[key] = {
            "config": _desc(key),
            "n_params": n_p,
            "best": round(best, 4),
            "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*72}")
    print(f"STEP 933 SUMMARY — Local/Global K_iter alternation T0")
    print(f"{'='*72}")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        v = ("(ref)" if k == "Ref"
             else ("ADV→T1" if d >= 0.005 else ("NEU" if d >= -0.005 else "KILL")))
        print(f"  {k:<12} params={r['n_params']:>8,}  "
              f"best={r['best']:.4f}  Δ={d*100:+.2f}pp  {v}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
