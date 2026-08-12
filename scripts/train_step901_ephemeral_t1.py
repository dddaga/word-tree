"""Step 901: Ephemeral Teleportation T1 (75ep, 50% data) — advance from step899.

CONTEXT
=======
step899 T0 scout tested 8 configs for ephemeral random long-range connections
per K_iter step. Advance rule: any config ≥+0.5pp vs Ref_dw → T1.

This script runs the T1 calibration tier (75ep, 50% data) on the winning configs
from step899. Default configs are the K_local=2 + K_ep=1 variants (keep static
edges, add ephemeral) — most likely survivors per the topology diameter hypothesis.
Override with --configs for exact step899 winners.

HYPOTHESIS (from step899)
==========================
K_hh=2 × K_iter=5 = max 5 hops < avg_path≈6 in N=2048 small-world.
Ephemeral re-sampling collapses effective diameter to ≤2 hops per forward pass.
If T0 gap holds at T1 → topology diameter is a real constraint.

CONFIGS (default: K_local=2 + ep=1 variants + ref)
  Ref_dw          K_local=2, K_ep=0, ΔW_local (standard baseline)
  E_ep1_l2_nodw   K_local=2, K_ep=1, no ΔW   (baseline + raw random)
  F_ep1_l2_ldw    K_local=2, K_ep=1, ΔW_local (baseline + random, ΔW on static only)
  G_ep1_l2_bothdw K_local=2, K_ep=1, ΔW_both  (baseline + random, ΔW on both)

Full config set available via --configs (same keys as step899):
  A_local1_dw, B_ep1_l1_nodw, C_ep1_l1_ldw, D_ep1_l1_bothdw also available.

ADVANCE RULE
============
  Winner ≥+1.0pp vs Ref_dw at T1 → T2 (step903).
  Winner ≥+0.5pp → CONFIRMED: topology diameter is load-bearing; T1 is publishable.
  Winner < Ref_dw → T0 gain was artifact; kill.
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
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref_dw,E_ep1_l2_nodw,F_ep1_l2_ldw,G_ep1_l2_bothdw")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step901_ephemeral_t1_seed{SEED}__{SLOT}.json"

STEP_REF = 0.9536  # step878 Ref T1 baseline (75ep, 50% data)

# Config spec: (k_local, k_ep, dw_local, dw_ep)
CONFIG_SPEC = {
    "Ref_dw":           (2, 0, True,  False),
    "A_local1_dw":      (1, 0, True,  False),
    "B_ep1_l1_nodw":    (1, 1, False, False),
    "C_ep1_l1_ldw":     (1, 1, True,  False),
    "D_ep1_l1_bothdw":  (1, 1, True,  True),
    "E_ep1_l2_nodw":    (2, 1, False, False),
    "F_ep1_l2_ldw":     (2, 1, True,  False),
    "G_ep1_l2_bothdw":  (2, 1, True,  True),
}


def _dw_proj_static(W_pos: torch.Tensor, conn: torch.Tensor) -> torch.Tensor:
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn], dim=-1).unsqueeze(0)


def _dw_proj_dynamic(W_pos: torch.Tensor, conn_ep: torch.Tensor) -> torch.Tensor:
    W_h = W_pos[:N]
    return F.normalize(W_h - W_h[conn_ep], dim=-1).unsqueeze(0).unsqueeze(2)


def _dw_agg(Z_nb: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


class SGNNET_EphemeralTeleport(nn.Module):
    def __init__(self, resonant: SGNNET_Resonant, k_local: int, k_ep: int,
                 dw_local: bool, dw_ep: bool):
        super().__init__()
        self.m        = resonant
        self.k_local  = k_local
        self.k_ep     = k_ep
        self.dw_local = dw_local
        self.dw_ep    = dw_ep

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        conn_loc  = self.m.base.conn_hh[:, :self.k_local]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw_loc    = _dw_proj_static(self.m.W_pos, conn_loc) if self.dw_local else None

        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb_loc = Z_fwd[:, conn_loc, :]
            Z_agg    = _dw_agg(Z_nb_loc, dw_loc) if self.dw_local else Z_nb_loc.sum(dim=2)

            if self.k_ep > 0:
                conn_ep  = torch.randint(0, N, (N,), device=Z.device)
                Z_nb_ep  = Z_fwd[:, conn_ep, :].unsqueeze(2)
                if self.dw_ep:
                    dw_e  = _dw_proj_dynamic(self.m.W_pos, conn_ep)
                    Z_agg = Z_agg + _dw_agg(Z_nb_ep, dw_e)
                else:
                    Z_agg = Z_agg + Z_nb_ep.sum(dim=2)

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
    k_local, k_ep, dw_local, dw_ep = CONFIG_SPEC[key]
    return SGNNET_EphemeralTeleport(r, k_local=k_local, k_ep=k_ep,
                                    dw_local=dw_local, dw_ep=dw_ep)


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
    print(f"step901 — Ephemeral Teleportation T1 (75ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  Ref context (step878 T1): {STEP_REF:.4f}")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        k_local, k_ep, dw_local, dw_ep = CONFIG_SPEC[key]
        model = make_model(key)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: k_local={k_local} k_ep={k_ep} dw_local={dw_local} "
              f"dw_ep={dw_ep}  params={n_p:,}")

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
            "k_local": k_local, "k_ep": k_ep,
            "dw_local": dw_local, "dw_ep": dw_ep,
            "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 901 SUMMARY — Ephemeral Teleportation T1")
    print(f"{'='*70}")
    print(f"  {'config':<18} {'kl':>2} {'ke':>2} {'dl':>4} {'de':>4} "
          f"{'params':>8} {'best':>7} {'Δ_vs_Ref':>10}  verdict")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "  (ref)"
        v    = "ADVANCE→T2" if (d is not None and d >= 0.010) else \
               "CONFIRMED"  if (d is not None and d >= 0.005) else \
               "NEUTRAL"    if (d is not None and d >= -0.005) else \
               "MARGINAL"   if (d is not None and d >= -0.020) else "KILL"
        print(f"  {k:<18} {r['k_local']:>2} {r['k_ep']:>2} {str(r['dw_local']):>4} "
              f"{str(r['dw_ep']):>4} {r['n_params']:>8,} {r['best']:>7.4f} "
              f"{dstr:>10}  {v}")

    print(f"\n  Ref T1 context (step878): {STEP_REF}")
    print(f"  ADVANCE→T2: winner ≥+1.0pp → step903.")
    print(f"  CONFIRMED: winner ≥+0.5pp → topology diameter is load-bearing.")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
