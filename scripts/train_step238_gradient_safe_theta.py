"""Step 238: Gradient-safe θ-edge parameterizations.

MOTIVATION
==========
Standard cos(θ) has zero gradient at θ=0 (w=1) and θ=π (w=0) — stationary
points where edge weights get stuck. Step233 tests raw cos(θ); this step
tests parameterizations that AVOID stationary points.

The key insight: we want bounded [0,1] edge weights with non-zero gradient
everywhere, so the network is always in a region where it can learn.

PARAMETERIZATIONS:
  cos(θ)           — baseline, has dead zones at 0 and π
  cos(θ + π/4)     — phase-shifted, stationary points at -π/4 and 3π/4
  sigmoid(sin(θ))  — smooth, gradient everywhere, different dead zones
  roots_of_unity   — θ quantized to midpoints between stationary points
  phase_delta      — θ_eff = θ + ε·sin(2θ), pushes away from stationary points
  sawtooth         — linear ramp mod 2π, gradient=const (no dead zones at all)

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep)
  Ref : Standard AH (reference)
  A   : cos(θ) — step233 baseline (for comparison)
  B   : cos(θ + π/4) — phase-shifted
  C   : sigmoid(sin(θ)) — different gradient landscape
  D   : phase_delta: θ + 0.1·sin(2θ) — anti-stationary perturbation
  E   : triangle wave — linear in θ, gradient never zero (except at folds)
  F   : cos(θ) with roots-of-unity init — init at max-gradient points
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
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys. Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

OUT_PATH = ROOT / "results" / "train_step238_gradient_safe_theta.json"


class SGNNET_GradSafeTheta(nn.Module):
    """θ-edge with gradient-safe parameterizations.

    All variants map θ → w ∈ [0, 1] but with different gradient landscapes.
    Uses the EXACT forward pass structure as SGNNET_AntiHebbian.
    """

    def __init__(self, N_hidden, N_out, D, N_in, K_in, K_iter,
                 K_local, K_random, n_groups, alpha_reflect,
                 param_mode="cos", seed=42):
        super().__init__()
        torch.manual_seed(seed)

        self.base_sw = SGNNET_SmallWorld(
            N_hidden=N_hidden, N_out=N_out, D=D, N_in=N_in,
            K_in=K_in, K_iter=K_iter, K_local=K_local, K_random=K_random,
            n_groups=n_groups, norm_mode="l2", encoding_mode="fourier")

        self.resonant = SGNNET_Resonant(
            self.base_sw, K_phase=8, alpha_reflect=alpha_reflect,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0)

        self.alpha_reflect = alpha_reflect
        self.param_mode = param_mode
        K_hh = K_local + K_random

        # θ initialization depends on mode
        if param_mode == "roots_init":
            # Init at 8th roots of unity midpoints: π/8, 3π/8, 5π/8, ...
            # These are max-gradient points of cos(θ)
            rng = torch.Generator().manual_seed(seed)
            midpoints = torch.tensor([np.pi/8, 3*np.pi/8, 5*np.pi/8, 7*np.pi/8,
                                       9*np.pi/8, 11*np.pi/8, 13*np.pi/8, 15*np.pi/8])
            idx = torch.randint(0, 8, (N_hidden, K_hh), generator=rng)
            self.theta_edge = nn.Parameter(midpoints[idx])
        else:
            # Default init: π/3 (moderate flow, away from stationary points)
            self.theta_edge = nn.Parameter(torch.full((N_hidden, K_hh), np.pi / 3))

        self._n_hidden = N_hidden

    @property
    def W_pos(self):
        return self.base_sw.W_pos

    @property
    def W_phase(self):
        return self.resonant.W_phase

    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()

    def _compute_supp_w(self):
        """Compute edge weights with gradient-safe parameterization."""
        theta = self.theta_edge

        if self.param_mode == "cos":
            # Standard: w = (1 + cos(θ)) / 2
            w = (1.0 + torch.cos(theta)) / 2.0

        elif self.param_mode == "cos_shifted":
            # Phase-shifted: stationary points moved to -π/4 and 3π/4
            w = (1.0 + torch.cos(theta + np.pi / 4)) / 2.0

        elif self.param_mode == "sigmoid_sin":
            # sigmoid(sin(θ)): smooth, gradient everywhere
            # sin(θ) ∈ [-1, 1], sigmoid maps to (0.27, 0.73) — narrower range
            # Scale: sigmoid(3·sin(θ)) gives (0.05, 0.95) — wider useful range
            w = torch.sigmoid(3.0 * torch.sin(theta))

        elif self.param_mode == "phase_delta":
            # Anti-stationary: θ_eff = θ + ε·sin(2θ)
            # At θ=0: sin(0)=0, but d/dθ[ε·sin(2θ)] = 2ε·cos(2θ) ≠ 0
            # This creates a restoring force pushing θ away from stationary points
            epsilon = 0.15
            theta_eff = theta + epsilon * torch.sin(2.0 * theta)
            w = (1.0 + torch.cos(theta_eff)) / 2.0

        elif self.param_mode == "triangle":
            # Triangle wave: linear in θ, gradient magnitude = const
            # Map θ to [0, 1] via triangle wave with period 2π
            # acos(cos(θ))/π gives a triangle wave
            w = torch.acos(torch.cos(theta).clamp(-1+1e-6, 1-1e-6)) / np.pi
            # This gives w=0 at θ=0 (and 2π), w=1 at θ=π
            # Invert so w=1 at θ=0: w = 1 - triangle
            w = 1.0 - w

        elif self.param_mode == "roots_init":
            # Same cos(θ) but initialized at roots-of-unity midpoints
            w = (1.0 + torch.cos(theta)) / 2.0

        else:
            raise ValueError(f"Unknown param_mode: {self.param_mode}")

        return w.unsqueeze(0).unsqueeze(-1)  # [1, N, K_hh, 1]

    def forward(self, x):
        Z = self.base_sw._seed(x)
        conn_hh = self.base_sw.conn_hh
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)

        supp_w = self._compute_supp_w()

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base_sw.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.base_sw._readout(Z)


def build_ref():
    """Standard SGNNET_AntiHebbian reference."""
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def build_gradsafe(param_mode):
    """Gradient-safe θ-edge model."""
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    return SGNNET_GradSafeTheta(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, alpha_reflect=ALPHA_REFLECT,
        param_mode=param_mode, seed=SEED)


def main():
    configs = {
        "Ref": ("ah", None),
        "A":   ("theta", "cos"),
        "B":   ("theta", "cos_shifted"),
        "C":   ("theta", "sigmoid_sin"),
        "D":   ("theta", "phase_delta"),
        "E":   ("theta", "triangle"),
        "F":   ("theta", "roots_init"),
    }

    run_keys = list(configs.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"Step 238 — Gradient-Safe θ-Edge Parameterizations")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        kind, param_mode = configs[key]
        print(f"\n{'─'*60}")
        if kind == "ah":
            print(f"Config {key}: Standard AH (reference)")
            model = build_ref()
        else:
            print(f"Config {key}: θ-edge, param_mode={param_mode}")
            model = build_gradsafe(param_mode)
        print(f"{'─'*60}")

        model = model.to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        theta_params = model.theta_edge.numel() if hasattr(model, 'theta_edge') else 0
        print(f"  total params={n_p:,}  θ-edge params={theta_params:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

        # Inject theta_edge into optimizer
        if hasattr(model, 'theta_edge'):
            trainer.optimizer.add_param_group({
                "params": [model.theta_edge],
                "lr": kw.get("lr_wpos", 2.36e-3),
                "weight_decay": 0.0,
            })
            print(f"  [injected theta_edge into optimizer]")

        t0 = time.time()
        def _log(m):
            ep = m["epoch"] + 1
            if ep % 5 == 0 or ep == 1:
                print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)
        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1

        # θ statistics
        theta_stats = {}
        if hasattr(model, 'theta_edge'):
            t = model.theta_edge.detach().cpu()
            w = model._compute_supp_w().squeeze().detach().cpu()
            theta_stats = {
                "theta_mean": round(t.mean().item(), 4),
                "theta_std": round(t.std().item(), 4),
                "w_mean": round(w.mean().item(), 4),
                "w_std": round(w.std().item(), 4),
                "w_min": round(w.min().item(), 4),
                "w_max": round(w.max().item(), 4),
                "frac_suppressed": round((w < 0.3).float().mean().item(), 4),
                "frac_open": round((w > 0.7).float().mean().item(), 4),
            }
            print(f"  θ: mean={theta_stats['theta_mean']:.3f} std={theta_stats['theta_std']:.3f}")
            print(f"  w: mean={theta_stats['w_mean']:.3f} std={theta_stats['w_std']:.3f} "
                  f"suppressed={theta_stats['frac_suppressed']:.1%} "
                  f"open={theta_stats['frac_open']:.1%}")

        results[key] = {
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "top1_history": top1h, "elapsed_s": round(elapsed, 1),
            "n_params": n_p, "theta_params": theta_params,
            "theta_stats": theta_stats,
            "param_mode": param_mode or "ah",
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"\n{'='*70}")
    print("STEP 238 SUMMARY — Gradient-Safe θ-Edge")
    print(f"{'='*70}")
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        std_info = ""
        if r.get("theta_stats", {}).get("w_std", 0) > 0:
            std_info = f"  w_std={r['theta_stats']['w_std']:.3f}"
        print(f"  {key} ({r['param_mode']:>12s}): {r['top1_best']:.4f}{delta}{std_info}")

    # Key diagnostic: did any variant learn diverse edge weights?
    print(f"\n  KEY DIAGNOSTIC: w_std > 0 means θ actually learned (not stuck)")
    for key in run_keys:
        r = results[key]
        ts = r.get("theta_stats", {})
        if ts.get("w_std", 0) > 0.01:
            print(f"  ✓ {key}: LEARNED (w_std={ts['w_std']:.3f})")
        elif ts.get("w_std", -1) >= 0:
            print(f"  ✗ {key}: STUCK (w_std={ts['w_std']:.4f})")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
