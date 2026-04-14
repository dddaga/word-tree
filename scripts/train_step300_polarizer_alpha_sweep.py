"""Step 300: Polarizer α Tier-1 sweep at N=2048 D=16.

MOTIVATION
==========
step217b: polarizer α=1.5 = 95.92% (+1.91pp over step199) at Tier-1.
GA v2 (2026-04-13) independently converged on pa=1.5 for ΔW projection,
suggesting α=1.5 optimum may hold for polarizer too.

This run calibrates α={0.5, 1.0, 1.5, 2.0, 2.5} at 75ep/50% data
(Tier-1) to confirm monotonic trend OR find a plateau/turning point.

POLARIZER MECHANISM (step217b)
==============================
Project Z_nb onto W_pos[receiver] direction before aggregation:
  proj = (Z_nb · W_pos[i]) / ||W_pos[i]||
  Z_polarized = α * proj * W_pos[i]_unit + (Z_nb - proj * W_pos[i]_unit)
  # over-polarizer (α>1): amplifies the aligned component
  # pure polarizer (α=1): pass-through

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 75ep Tier-1)
  Ref   : Standard AH (step199, no polarizer)
  A050  : AH + polarizer α=0.5
  A100  : AH + polarizer α=1.0
  A150  : AH + polarizer α=1.5   (step217b winner)
  A200  : AH + polarizer α=2.0
  A250  : AH + polarizer α=2.5
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
parser.add_argument("--epochs", type=int, default=75)
parser.add_argument("--configs", default="")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

OUT_PATH = ROOT / "results" / "train_step300_polarizer_alpha_sweep.json"


class SGNNET_Polarizer(nn.Module):
    """AH + polarizer routing: project Z_nb onto W_pos[receiver] direction."""

    def __init__(self, base: SGNNET_Resonant, alpha_ahebb=1.0, polar_alpha=1.5):
        super().__init__()
        self.m = base
        self.alpha_ahebb = alpha_ahebb
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
        conn_hh = self.m.base.conn_hh
        N_h = self.m.base.N_hidden

        W_n = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        # Pre-compute receiver W_pos unit vector — [N, D]
        W_recv = W_n  # already normalized

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]  # [B, N, K_hh, D]

            # Project Z_nb onto W_pos[receiver]
            # W_recv: [N, D] → [1, N, 1, D]
            W_recv_expanded = W_recv.unsqueeze(0).unsqueeze(2)  # [1, N, 1, D]
            proj_coeff = (Z_nb * W_recv_expanded).sum(dim=-1, keepdim=True)  # [B, N, K, 1]
            z_parallel = proj_coeff * W_recv_expanded  # component along W_pos
            z_perp = Z_nb - z_parallel                  # perpendicular component

            # Over-polarizer: amplify parallel component by (alpha - 1)
            # α=1: identity. α>1: boost alignment. α<1: attenuate.
            Z_polar = self.polar_alpha * z_parallel + z_perp

            # Apply AH suppression
            Z_polar = Z_polar * supp_w
            Z_struct = Z_polar.sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
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


def build_polar(alpha):
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_Polarizer(resonant, alpha_ahebb=ALPHA_AHEBB, polar_alpha=alpha)


def main():
    configs = {
        "Ref":   ("ref",   None),
        "A050":  ("polar", 0.5),
        "A100":  ("polar", 1.0),
        "A150":  ("polar", 1.5),
        "A200":  ("polar", 2.0),
        "A250":  ("polar", 2.5),
    }
    run_keys = list(configs.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"Step 300 — Polarizer α sweep at N=2048 D=16 (Tier-1, 75ep, 50% data)")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        kind, alpha = configs[key]
        print(f"\n{'─'*60}\nConfig {key}: kind={kind}, alpha={alpha}\n{'─'*60}")
        model = (build_ref() if kind == "ref" else build_polar(alpha)).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
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
        best = max(top1h); bep = int(np.argmax(top1h)) + 1
        results[key] = {
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "top1_history": top1h, "elapsed_s": round(elapsed, 1),
            "n_params": n_p, "polar_alpha": alpha,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"\n{'='*70}\nSTEP 300 SUMMARY — polarizer α sweep\n{'='*70}")
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        a = f"α={r['polar_alpha']}" if r.get('polar_alpha') is not None else "no polar"
        print(f"  {key} ({a}): {r['top1_best']:.4f}{delta}")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
