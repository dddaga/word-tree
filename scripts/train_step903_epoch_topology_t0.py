"""Step 903: Epoch-Topology Dynamic Slot T0 (20ep, 50% data).

HYPOTHESIS
==========
Static small-world graph hard-wires topology at init. The K_random slot (one
long-range shortcut per node) is frozen despite W_pos drifting throughout
training. Epoch-level KNN rebuild of that single slot lets the graph re-wire
to match the *learned* geometry without paying per-step gradient noise.

Key distinction from step852 (epoch topology rebuild):
  step852 rebuilt the FULL K_hh matrix every N epochs → −1.91pp best.
  This script keeps K_local=1 ring fixed and updates ONLY the K_random=1
  dynamic slot — less disruption, lower noise, still topology-adaptive.

LR reduction variants test whether slowing W_pos drift makes rebuilds more
stable (edges flip less per epoch → cleaner KNN signal).

CONFIGS (T0, 20ep, 50% data, seed=42)
  Ref_static      K_hh=2, no dynamic rebuild  (standard ΔW-proj baseline)
  A_std_lr        dynamic slot, LR × 1.0
  B_half_lr       dynamic slot, LR × 0.5
  C_tenth_lr      dynamic slot, LR × 0.1
  D_warmup10      dynamic slot, rebuild only after epoch 10

FORWARD PASS (all dynamic configs)
  conn = [ring_conn [N,1], dynamic_conn [N,1]]  → shape [N, 2]
  ΔW-proj on both edges (recomputed at tick_epoch for dynamic slot).
  K_iter=5 steps with reflect buffer (same as step898 Ref_dw).

DYNAMIC SLOT REBUILD (tick_epoch)
  W_h = normalize(W_pos[:N])       [N, D]
  sim = W_h @ W_h.T               [N, N]  — cosine similarity
  Mask self and ring neighbor.
  dynamic_conn[i] = argmax_j sim[i, j]   — closest node not in ring.

ADVANCE RULE
============
  Any config ≥+0.5pp vs Ref_static → T1 (step905, 75ep/50%).
  All configs < Ref_static → static graph wins; direction closed.
  KEY READ: B/C vs A — does LR reduction help stability?
            D vs A — does delayed rebuild matter?
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
parser.add_argument("--configs",  default="Ref_static,A_std_lr,B_half_lr,C_tenth_lr,D_warmup10")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step903_epoch_topology_t0_seed{SEED}__{SLOT}.json"

STEP_REF = 0.9396  # step898 Ref_dw T0 (D=16+ΔW, 20ep/50%)

# Config spec: (lr_scale, warmup_epochs)
# lr_scale: multiplier on default lr_wpos and base lr
# warmup_epochs: epochs before dynamic rebuild begins (0 = from start)
CONFIG_SPEC = {
    "Ref_static":  (1.0, -1),   # no dynamic rebuild
    "A_std_lr":    (1.0,  0),
    "B_half_lr":   (0.5,  0),
    "C_tenth_lr":  (0.1,  0),
    "D_warmup10":  (1.0, 10),
}


def _dw_proj(W_pos: torch.Tensor, conn_hh: torch.Tensor) -> torch.Tensor:
    """conn_hh: [N, K] → returns [1, N, K, D]."""
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
    """Z_nb: [B, N, K, D], dw: [1, N, K, D] → [B, N, D]."""
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def _rebuild_dynamic(W_pos: torch.Tensor, ring_conn: torch.Tensor) -> torch.Tensor:
    """Rebuild dynamic slot via top-1 cosine KNN, excluding self + ring neighbor.

    ring_conn: [N] int64 on same device as W_pos.
    Returns: [N] int64 on same device.
    """
    W_h = F.normalize(W_pos[:N].detach(), dim=-1)  # [N, D]
    sim = W_h @ W_h.T                               # [N, N]
    idx = torch.arange(N, device=W_h.device)
    sim[idx, idx]         = -2.0                    # mask self
    sim[idx, ring_conn]   = -2.0                    # mask ring neighbor
    return sim.argmax(dim=-1)                        # [N]


class SGNNET_StaticRef(nn.Module):
    """Standard ΔW-proj baseline — ring + random edges, both static."""

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


class SGNNET_EpochTopology(nn.Module):
    """Static ring + one epoch-updated dynamic slot, both with ΔW-proj."""

    def __init__(self, resonant: SGNNET_Resonant, warmup_epochs: int = 0):
        super().__init__()
        self.m              = resonant
        self.warmup_epochs  = warmup_epochs
        self._epoch         = 0

        # ring_conn: [N] — first column of static conn_hh (K_local=1 ring)
        self.register_buffer("ring_conn",
            resonant.base.conn_hh[:, 0].clone())
        # dynamic_conn: [N] — initialised to static K_random slot
        self.register_buffer("dynamic_conn",
            resonant.base.conn_hh[:, 1].clone())

        # build initial conn_cache (int64, no grad)
        self._rebuild_conn()

    def _rebuild_conn(self):
        self._conn_cache = torch.stack(
            [self.ring_conn, self.dynamic_conn], dim=-1)  # [N, 2] int64

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()
        self._epoch += 1
        if self._epoch > self.warmup_epochs:
            self.dynamic_conn = _rebuild_dynamic(
                self.m.W_pos, self.ring_conn)
        self._rebuild_conn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        conn_hh   = self._conn_cache
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw        = _dw_proj(self.m.W_pos, conn_hh)   # fresh each forward — W_pos has grad
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
    r = make_base()
    lr_scale, warmup = CONFIG_SPEC[key]
    if warmup == -1:
        return SGNNET_StaticRef(r)
    return SGNNET_EpochTopology(r, warmup_epochs=warmup)


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
    print(f"step903 — Epoch-Topology Dynamic Slot T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  Ref context (step898 Ref_dw T0): {STEP_REF:.4f}")
    print(f"  K_local=1 ring fixed; K_random=1 slot rebuilt via W_pos KNN per epoch")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        lr_scale, warmup = CONFIG_SPEC[key]
        model = make_model(key)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: lr_scale={lr_scale}  warmup_epochs={warmup}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS, lr_wpos=2.364e-3 * lr_scale)

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
        if key == "Ref_static":
            ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP_REF)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "lr_scale": lr_scale, "warmup_epochs": warmup,
            "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 903 SUMMARY — Epoch-Topology Dynamic Slot T0")
    print(f"{'='*70}")
    print(f"  {'config':<14} {'lr_sc':>5} {'warm':>4} {'params':>8} {'best':>7} "
          f"{'Δ_vs_Ref':>10}  verdict")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "  (ref)"
        v    = "ADVANCE→T1" if (d is not None and d >= 0.005) else \
               "NEUTRAL"    if (d is not None and d >= -0.005) else \
               "MARGINAL"   if (d is not None and d >= -0.020) else "KILL"
        print(f"  {k:<14} {r['lr_scale']:>5} {r['warmup_epochs']:>4} "
              f"{r['n_params']:>8,} {r['best']:>7.4f} {dstr:>10}  {v}")

    print(f"\n  Ref T0 context (step898/Ref_dw): {STEP_REF}")
    print(f"  ADVANCE→T1: any winner ≥+0.5pp → step905 (75ep/50%).")
    print(f"  KEY READS: B/C vs A (LR stability); D vs A (delayed rebuild).")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
