"""Step 403: ΔW projection Tier-1 validation + DeltaPolar combo.

MOTIVATION
==========
step234 (Tier-0, 20ep): ΔW proj (no AH) = 95.44% (+3.77pp over step199 baseline).
This Tier-1 run (75ep, 50% data) confirms whether the gain holds and tests whether
combining ΔW projection with polarizer amplification gives further gains.

step301 (Tier-0) tests ΔW + weak AH. This script handles the Tier-1 validation
of the 4 key configs that establish the paper claim for ΔW projection.

SGNNET_DeltaPolar: ΔW projection THEN polarizer amplification.
  1. Project Z_nb onto ΔW direction (routing quality signal)
  2. Polarize result onto W_pos[receiver] (amplify alignment with receiver)
  Combined: selects signals that are BOTH moving in a relevant direction
            AND aligned with the receiver's positional encoding.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 75ep — Tier-1)
  Ref         : AH α=1.0 (step199 baseline)
  A_proj      : ΔW projection, no AH (step234 winner)
  A_proj_pa15 : ΔW projection + polarizer α=1.5 (DeltaPolar — two mechanisms combined)
  A_proj_a025 : ΔW projection + AH α=0.25 (weak AH as edge gate; step301 context)
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

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys to run, e.g. 'Ref,A_proj'")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

OUT_PATH = ROOT / "results" / "train_step403_delta_tier1.json"


# ── Model wrappers ─────────────────────────────────────────────────────────────

class SGNNET_DeltaAH(nn.Module):
    """ΔW projection with optional AH suppression (from step301).

    Projects Z_nb onto the ΔW = W_pos[receiver] - W_pos[sender] direction.
    Captures the 'relational axis' between connected neurons.
    alpha_ahebb=0: pure ΔW projection, no suppression.
    """
    def __init__(self, base: SGNNET_Resonant, alpha_ahebb=0.0):
        super().__init__()
        self.m = base
        self.alpha_ahebb = alpha_ahebb

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x):
        Z = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden

        W_h = self.m.W_pos[:N_h]
        W_n = F.normalize(W_h, dim=-1)

        # AH suppression weights (computed once — static topology)
        supp_w = None
        if self.alpha_ahebb > 0:
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
            supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                       ).unsqueeze(0).unsqueeze(-1)

        # ΔW direction: [N, K_hh, D] → normalized → [1, N, K, D]
        delta_w      = W_h.unsqueeze(1) - W_h[conn_hh]
        delta_w_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]  # [B, N, K_hh, D]

            if supp_w is not None:
                Z_nb = Z_nb * supp_w

            # Project onto ΔW, scale by |projection|
            proj_coeff = (Z_nb * delta_w_norm).sum(dim=-1, keepdim=True)
            Z_nb       = Z_nb * proj_coeff.abs()

            Z_struct    = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z           = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


class SGNNET_DeltaPolar(nn.Module):
    """ΔW projection → polarizer amplification (two mechanisms chained).

    Step 1 (ΔW projection): weight each neighbour signal by |Z_nb · ΔW_unit|
      → retains signals that align with the receiver-sender relational axis
    Step 2 (polarizer): amplify the component of the result that aligns with
      W_pos[receiver] by factor polar_alpha
      → further sharpens routing toward receiver's positional encoding

    No AH (ΔW projection already handles routing quality; AH would be redundant).
    """
    def __init__(self, base: SGNNET_Resonant, polar_alpha=1.5):
        super().__init__()
        self.m           = base
        self.polar_alpha = polar_alpha

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x):
        Z = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden

        W_h = self.m.W_pos[:N_h]
        W_n = F.normalize(W_h, dim=-1)  # [N, D]

        # ΔW direction: [1, N, K, D]
        delta_w      = W_h.unsqueeze(1) - W_h[conn_hh]
        delta_w_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)

        # Receiver W_pos unit vector: [1, N, 1, D]
        W_recv = W_n.unsqueeze(0).unsqueeze(2)

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]  # [B, N, K_hh, D]

            # Step 1: ΔW projection (scale by |alignment with relational axis|)
            proj_delta = (Z_nb * delta_w_norm).sum(dim=-1, keepdim=True)  # [B, N, K, 1]
            Z_nb       = Z_nb * proj_delta.abs()

            # Step 2: polarizer (amplify component along W_pos[receiver])
            proj_recv  = (Z_nb * W_recv).sum(dim=-1, keepdim=True)        # [B, N, K, 1]
            z_parallel = proj_recv * W_recv
            z_perp     = Z_nb - z_parallel
            Z_nb       = self.polar_alpha * z_parallel + z_perp

            Z_struct    = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z           = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ── Builders ───────────────────────────────────────────────────────────────────

def _base_resonant():
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )


def build_ref():
    torch.manual_seed(SEED)
    return SGNNET_AntiHebbian(_base_resonant(), alpha_ahebb=1.0, variant="wpos")


def build_delta_ah(alpha_ahebb):
    torch.manual_seed(SEED)
    return SGNNET_DeltaAH(_base_resonant(), alpha_ahebb=alpha_ahebb)


def build_delta_polar(polar_alpha):
    torch.manual_seed(SEED)
    return SGNNET_DeltaPolar(_base_resonant(), polar_alpha=polar_alpha)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # (kind, param): "ref"=baseline, "delta_ah"=ΔW+AH, "delta_polar"=ΔW+polarizer
    configs = {
        "Ref":          ("ref",         None),
        "A_proj":       ("delta_ah",    0.00),
        "A_proj_pa15":  ("delta_polar", 1.50),
        "A_proj_a025":  ("delta_ah",    0.25),
    }
    run_keys = list(configs.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"Step 403 — ΔW projection Tier-1 validation (75ep, 50% data)")
    print(f"Testing ΔW alone + DeltaPolar combo vs AH baseline")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))[:n_total // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        kind, param = configs[key]
        print(f"\n{'─'*60}\nConfig {key}: kind={kind}, param={param}\n{'─'*60}")

        if kind == "ref":
            model = build_ref()
        elif kind == "delta_ah":
            model = build_delta_ah(param)
        else:  # delta_polar
            model = build_delta_polar(param)
        model = model.to(DEVICE)

        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        kw      = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)
        t0 = time.time()
        def _log(m):
            ep = m["epoch"] + 1
            if ep % 10 == 0 or ep == 1 or ep == EPOCHS:
                print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)
        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best  = max(top1h); bep = int(np.argmax(top1h)) + 1
        results[key] = {
            "top1_best":    best,
            "top1_last":    top1h[-1],
            "best_epoch":   bep,
            "top1_history": top1h,
            "elapsed_s":    round(elapsed, 1),
            "n_params":     n_p,
            "kind":         kind,
            "param":        param,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best    = results.get("Ref",    {}).get("top1_best", 0.)
    a_proj_best = results.get("A_proj", {}).get("top1_best", 0.)

    print(f"\n{'='*70}\nSTEP 403 SUMMARY — ΔW projection Tier-1\n{'='*70}")
    for key in run_keys:
        r     = results[key]
        d_ref = f"  Δref={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        d_proj = (f"  Δproj={r['top1_best']-a_proj_best:+.4f}"
                  if key not in ("Ref", "A_proj") and "A_proj" in results else "")
        desc  = f"kind={r['kind']}, param={r['param']}"
        print(f"  {key:14s} ({desc}): {r['top1_best']:.4f}{d_ref}{d_proj}")

    print(f"\nInterpretation:")
    print(f"  A_proj > Ref → ΔW projection confirmed as winner (paper claim)")
    print(f"  A_proj_pa15 > A_proj → DeltaPolar combo is additive (novel mechanism)")
    print(f"  A_proj_a025 > A_proj → weak AH is complementary to ΔW (edge gating role)")
    print(f"\nstep234 Tier-0 reference: A_proj = 95.44% (+3.77pp)")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
