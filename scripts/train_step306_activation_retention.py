"""Step 306: Activation retention mechanism — what stays on the sender node?

USER HYPOTHESIS (2026-04-13)
============================
Currently, when a neuron is active above threshold, its activation
hops ENTIRELY through the gather pattern to neighbors. Nothing is
retained on the sender.

Proposal: when neuron i fires, retain a portion on i and propagate
the rest. This creates persistent activation reservoirs — firing
patterns don't lose their source identity.

Three variants tested:
  A_static_p10  : retain p=0.1 constant across K_iter
  A_static_p30  : retain p=0.3 constant
  A_static_p50  : retain p=0.5 constant
  B_decay       : retain starts p=0.5 at iter 0, decays to 0.1 at final iter
  C_norm_cons   : norm-conserving: if projection magnitude |p|=α, retain sqrt(1-α²)
                  (energy split so |Z_propagated|² + |Z_retained|² = |Z_fwd|²)
  D_reinject    : full propagation (retain p=0) + re-inject propagated signal back

LOW-N FIRST (per user: test at low N first)
  N=1024, D=16, K_hh=2, K_iter=5, 50% data, 20ep Tier-0

Reference: standard AH (step199-equivalent at N=1024 base ~89%).

Note: This is a NEW mechanism — retention is orthogonal to AH/ΔW.
Test with AH active (closest to baseline) to isolate retention effect.
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

from src.sgnnet.model_smallworld     import SGNNET_SmallWorld
from src.sgnnet.model_resonant       import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer            import Trainer
from src.training.experiment_config  import trainer_kwargs
from src.training.dataset            import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--N", type=int, default=1024,
                    help="Hidden neurons (default 1024 per user: low-N first)")
parser.add_argument("--configs", default="")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = args.N; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

OUT_PATH = ROOT / "results" / f"train_step306_activation_retention_N{N}.json"


class SGNNET_Retention(nn.Module):
    """SGNNET with activation retention: portion of Z_fwd stays on sender.

    mode:
      'static'   : constant retention fraction `p` across K_iter
      'decay'    : retention decays linearly from p_start → p_end across K_iter
      'norm_cons': norm-conserving split based on projection magnitude
                   Z_retained has magnitude sqrt(1 - mean(proj²))
      'reinject' : p=0 but propagated signal injected back (identity-like residual)
    """
    def __init__(self, base: SGNNET_Resonant, alpha_ahebb=1.0,
                 mode="static", p=0.3, p_end=0.1):
        super().__init__()
        self.m = base
        self.alpha_ahebb = alpha_ahebb
        self.mode = mode
        self.p = p
        self.p_end = p_end

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _retention_fraction(self, iter_idx, K):
        """Return retention fraction p at iteration iter_idx of K total."""
        if self.mode == "static" or self.mode == "norm_cons" or self.mode == "reinject":
            return self.p
        elif self.mode == "decay":
            # Linear interpolation from p_start=self.p at iter 0 → p_end at iter K-1
            frac = iter_idx / max(1, K - 1)
            return self.p * (1 - frac) + self.p_end * frac

    def forward(self, x):
        Z = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh = self.m.base.conn_hh
        N_h = self.m.base.N_hidden

        W_n = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)
        K = self.m.base.K_iter

        for k in range(K):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            Z_struct_raw = (Z_nb * supp_w).sum(dim=2)  # [B, N, D]

            # Apply retention mechanism
            if self.mode == "norm_cons":
                # Energy-conserving split: propagate p fraction of sender magnitude
                # to structure; retain sqrt(1-p²) on sender.
                p = self._retention_fraction(k, K)
                retain_coeff = (1.0 - p * p) ** 0.5
                Z_struct = p * Z_struct_raw
                Z_retained = retain_coeff * Z_fwd
            elif self.mode == "reinject":
                # Full propagation, but re-inject Z_fwd onto sender (identity residual)
                Z_struct = Z_struct_raw
                Z_retained = self.p * Z_fwd  # self.p is re-inject strength
            else:
                # Static or decay: (1-p) propagates, p stays
                p = self._retention_fraction(k, K)
                Z_struct = (1.0 - p) * Z_struct_raw
                Z_retained = p * Z_fwd

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_retained + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


def build_ref():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def build_retention(mode, p, p_end=0.1):
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_Retention(resonant, alpha_ahebb=ALPHA_AHEBB,
                            mode=mode, p=p, p_end=p_end)


def main():
    configs = {
        "Ref":          ("ref",       None,  None),
        "A_static_p10": ("static",    0.1,   None),
        "A_static_p30": ("static",    0.3,   None),
        "A_static_p50": ("static",    0.5,   None),
        "B_decay":      ("decay",     0.5,   0.1),   # decays 0.5 → 0.1
        "C_norm_cons":  ("norm_cons", 0.7,   None),  # 70% propagated, sqrt(0.51)~71% retained
        "D_reinject":   ("reinject",  0.3,   None),  # 30% reinjected
    }
    run_keys = list(configs.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"Step 306 — Activation retention at N={N} D=16 (Tier-0, 20ep, 50% data)")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        mode, p, p_end = configs[key]
        print(f"\n{'─'*60}\nConfig {key}: mode={mode}, p={p}, p_end={p_end}\n{'─'*60}")
        model = (build_ref() if mode == "ref"
                 else build_retention(mode, p, p_end or 0.1)).to(DEVICE)
        n_p = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
        print(f"  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)
        t0 = time.time()
        def _log(m):
            ep = m["epoch"] + 1
            if ep % 5 == 0 or ep == 1:
                print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)
        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1
        results[key] = {
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "top1_history": top1h, "elapsed_s": round(elapsed, 1),
            "n_params": n_p, "mode": mode, "p": p, "p_end": p_end,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"\n{'='*70}\nSTEP 306 SUMMARY — Activation retention at N={N}\n{'='*70}")
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        p_str = f"p={r['p']}" if r['p'] is not None else "-"
        print(f"  {key:14s} ({r['mode']:10s}, {p_str}): {r['top1_best']:.4f}{delta}")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
