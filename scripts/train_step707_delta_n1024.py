"""Step 707: ΔW projection N-scaling test at N=1024 (routing headroom hypothesis).

MOTIVATION
==========
step706 confirmed ΔW proj beats AH at N=2048 Tier-2 (96.87% vs 95.31%, +1.56pp).
step704 confirmed ΔW proj KILLS at N=4096 (−0.74pp Tier-1).

Routing headroom hypothesis: ΔW proj gains increase with distance from D=16 ceiling
(97.17%). N=2048 headroom ≈ 2-3pp before ΔW. N=1024 headroom ≈ 9pp.
Prediction: ΔW proj at N=1024 gives >+1.61pp gain.

Alternative: the gain is N=2048-specific, not headroom-driven.
N=1024 gives similar or smaller gain.

CONFIGS (N=1024, D=16, K_hh=2, K_iter=5, 50% data, 75ep — Tier-1)
  Ref    : AH α=1.0 (efficiency config baseline at N=1024)
  A_proj : ΔW projection, no AH (step706 winner at N=2048)

N=1024 Tier-1 Ref: ~88-89% (step198: 88.92% @ 0.59M, step402a: 87.95% @0.49M Tier-0)
Expected if headroom theory: >+1.61pp (>90.5%)
Expected if N-specific: ≤+1pp
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

parser = argparse.ArgumentParser(description="Step 707: ΔW proj at N=1024")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
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
OUT_PATH = ROOT / "results" / "train_step707_delta_n1024.json"


# ── ΔW projection model ─────────────────────────────────────────────────────────

class SGNNET_DeltaAH(nn.Module):
    """ΔW projection with optional AH suppression.

    Projects Z_nb onto ΔW = W_pos[receiver] - W_pos[sender].
    alpha_ahebb=0: pure ΔW projection (step403/step706 winner).
    """
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

        delta_w      = W_h.unsqueeze(1) - W_h[conn_hh]        # [N, K_hh, D]
        delta_w_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)  # [1, N, K_hh, D]

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]                       # [B, N, K_hh, D]

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
    all_keys = ["Ref", "A_proj"]
    run_keys = ([k.strip() for k in args.configs.split(",")]
                if args.configs else all_keys)

    labels = {
        "Ref":    "Ref: AH α=1.0 (efficiency config at N=1024)",
        "A_proj": "A_proj: ΔW projection, no AH (step706 winner at N=2048)",
    }

    print(f"\n{'='*70}")
    print(f"Step 707 — ΔW projection at N=1024 (routing headroom hypothesis)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)  Device={DEVICE}")
    print(f"N=2048 refs: Ref=95.31%  A_proj=96.87% (+1.56pp) — step706 Tier-2")
    print(f"N=4096 refs: ΔW proj KILLS (−0.74pp Tier-1) — step704")
    print(f"Headroom theory: N=1024 gap~9pp → predicts >+1.61pp gain")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total,
                         generator=torch.Generator().manual_seed(SEED))[:n_total // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True,
                                     num_workers=0,
                                     generator=torch.Generator().manual_seed(SEED))
    print(f"Train={len(subset)}  Val={len(va.dataset)}")

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
    print("STEP 707 SUMMARY — ΔW projection at N=1024 (headroom hypothesis)")
    print(f"{'='*70}")
    print(f"  N=2048 (step706 Tier-2): Ref=95.31%  A_proj=96.87%  Δ=+1.56pp")
    print(f"  N=4096 (step704 Tier-1): ΔW proj KILLS (−0.74pp)")
    print(f"  N=1024 (this run):")
    for key in run_keys:
        r     = results[key]
        d_str = f"  Δ={r['top1_best'] - ref_best:+.4f}" if key != "Ref" and "Ref" in results else ""
        print(f"    {key}: {r['top1_best']:.4f}{d_str}")

    import math
    if not math.isnan(delta):
        if delta > 0.020:
            verdict = f"HEADROOM CONFIRMED — ΔW proj gains MORE at N=1024 (+{delta:.4f} > +1.56pp N=2048)"
        elif delta >= 0.010:
            verdict = f"HEADROOM PLAUSIBLE — ΔW proj gains similarly at N=1024 (+{delta:.4f})"
        elif delta >= 0.000:
            verdict = f"N-SPECIFIC LIKELY — smaller gain at N=1024 (+{delta:.4f} < +1.56pp N=2048)"
        else:
            verdict = f"N-SPECIFIC CONFIRMED — ΔW proj hurts at N=1024 (Δ={delta:.4f})"
        print(f"\n  Verdict: {verdict}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
