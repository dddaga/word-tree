"""Step 707 Tier-2: ΔW projection at N=1024, 150ep, 100% data (paper-ready).

MOTIVATION
==========
step707 Tier-1 (75ep, 50% data) CONFIRMED routing headroom hypothesis:
  Ref=88.61%  A_proj=93.48%  Δ=+4.87pp

This is LARGER than N=2048 Tier-2 gain (+1.56pp step706) — confirming that ΔW
projection gains scale inversely with distance to the D=16 ceiling.

If A_proj Tier-2 reaches ~95%, that's ≥95% accuracy at 0.49M FLOPs — half the
FLOPs of the current efficiency record (0.98M, step199/step706).

N=1024 Tier-1 → Tier-2 extrapolation (+1-2pp typical): expected ~94.5-95.5%.

CONFIGS (N=1024, D=16, K_hh=2, K_iter=5, 100% data, 150ep — Tier-2)
  Ref    : AH α=1.0 (N=1024 baseline — paper N-scaling row)
  A_proj : ΔW projection, no AH (step707 Tier-1 winner, +4.87pp)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser(description="Step 707 Tier-2: ΔW proj N=1024 paper-ready")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (e.g. Ref,A_proj). Empty = all.")
args   = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

FLOPS    = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step707_tier2_delta_n1024.json"


# ── ΔW projection model ─────────────────────────────────────────────────────────

class SGNNET_DeltaAH(nn.Module):
    def __init__(self, base: SGNNET_Resonant, alpha_ahebb: float = 0.0):
        super().__init__()
        self.m           = base
        self.alpha_ahebb = alpha_ahebb

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x):
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden

        W_h = self.m.W_pos[:N_h]
        W_n = F.normalize(W_h, dim=-1)

        supp_w = None
        if self.alpha_ahebb > 0:
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
            supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                       ).unsqueeze(0).unsqueeze(-1)

        delta_w      = W_h.unsqueeze(1) - W_h[conn_hh]
        delta_w_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]

            if supp_w is not None:
                Z_nb = Z_nb * supp_w

            proj_coeff = (Z_nb * delta_w_norm).sum(dim=-1, keepdim=True)
            Z_nb       = Z_nb * proj_coeff.abs()

            Z_struct    = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = ALPHA_REFLECT * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z           = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ── Builders ─────────────────────────────────────────────────────────────────────

def _base_resonant():
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, mode="dynamic_z_geo",
    )


def build_ref():
    torch.manual_seed(SEED)
    return SGNNET_AntiHebbian(_base_resonant(), alpha_ahebb=1.0, variant="wpos")


def build_delta_proj():
    torch.manual_seed(SEED)
    return SGNNET_DeltaAH(_base_resonant(), alpha_ahebb=0.0)


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    import math
    all_keys = ["Ref", "A_proj"]
    run_keys = ([k.strip() for k in args.configs.split(",")]
                if args.configs else all_keys)

    labels = {
        "Ref":    "Ref: AH α=1.0 (N=1024 baseline)",
        "A_proj": "A_proj: ΔW projection, no AH (step707 T1 winner +4.87pp)",
    }

    print(f"\n{'='*70}")
    print(f"Step 707 Tier-2 — ΔW projection at N=1024 (paper-ready, 150ep, 100% data)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {EPOCHS}ep 100% data")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.3f}M) = 50% of N=2048 record (0.98M)")
    print(f"Device={DEVICE}")
    print(f"Tier-1 (step707): Ref=88.61%  A_proj=93.48%  Δ=+4.87pp")
    print(f"Tier-2 Ref known: ~89% (step198 Tier-2 N=1024)")
    print(f"Tier-2 expected: ~94.5-95.5%  (if yes: new efficiency record at 0.49M)")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    print(f"Train={len(tr.dataset)}  Val={len(va.dataset)}")

    results = {}
    for key in run_keys:
        print(f"\n{'─'*60}")
        print(f"Config {labels.get(key, key)}")
        print(f"{'─'*60}")

        model = build_ref() if key == "Ref" else build_delta_proj()
        model = model.to(DEVICE)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch'] + 1) % 10 == 0 else None
        ))
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best  = max(top1h)
        bep   = int(np.argmax(top1h)) + 1
        results[key] = {
            "label":        labels.get(key, key),
            "top1_best":    best,
            "top1_last":    top1h[-1],
            "best_epoch":   bep,
            "epochs_run":   len(history),
            "top1_history": top1h,
            "elapsed_s":    round(elapsed, 1),
            "n_params":     n_p,
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "flops": FLOPS,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    ref_best  = results.get("Ref",    {}).get("top1_best", 0.)
    proj_best = results.get("A_proj", {}).get("top1_best", 0.)
    delta     = proj_best - ref_best if "A_proj" in results and "Ref" in results else float("nan")

    print(f"\n{'='*70}")
    print("STEP 707 TIER-2 SUMMARY — ΔW projection at N=1024 (paper-ready)")
    print(f"{'='*70}")
    print(f"  Tier-1 (step707): Ref=88.61%  A_proj=93.48%  Δ=+4.87pp")
    print(f"  N=2048 Tier-2 record (step706): A_proj=96.87% @ 0.98M FLOPs")
    print(f"  N=1024 Tier-2 (this run):")
    for key in run_keys:
        r     = results[key]
        d_str = f"  Δ={r['top1_best'] - ref_best:+.4f}" if key != "Ref" and "Ref" in results else ""
        print(f"    {key}: {r['top1_best']:.4f}{d_str}")

    if not math.isnan(delta):
        if proj_best >= 0.950:
            verdict = f"NEW EFFICIENCY RECORD — {proj_best:.4f} @ {FLOPS/1e6:.3f}M FLOPs (50% of N=2048 record)"
        elif proj_best >= 0.940:
            verdict = f"Near record — {proj_best:.4f} @ {FLOPS/1e6:.3f}M FLOPs (+{delta:.4f} over Ref)"
        else:
            verdict = f"Below record threshold — {proj_best:.4f} (Δ={delta:+.4f})"
        print(f"\n  Verdict: {verdict}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
