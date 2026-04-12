"""Step 233: Sinusoidal θ-edge weights — precision ablation replacing AH.

MOTIVATION
==========
Step232 tried scalar edge weights but broke the forward pass (gradients
didn't flow — all configs collapsed to ~18%). This experiment fixes that by
surgically replacing ONLY the supp_w computation inside the working
SGNNET_AntiHebbian forward pass.

Each edge (i,j) in conn_hh gets a learnable θ ∈ [0, 2π]. The edge weight is:
    w_edge = (1 + cos(θ)) / 2    ∈ [0, 1]

This is a smooth, periodic, bounded parameterization. At θ=0 → w=1 (full flow),
at θ=π → w=0 (full suppression). The sinusoidal form gives smooth gradients
everywhere unlike sigmoid which saturates.

KEY QUESTION: Does a single angle per edge (N×K_hh params at varying precision)
match the 32K hidden W_pos + cosine AH computation?

PRECISION ABLATION: θ is stored in different dtypes to measure information
requirements. If fp16 ≈ fp32, we can halve memory. If int8 works, we need
only ~1 byte per edge.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep)
  Ref : Standard AH (32K hidden W_pos, cosine suppression)
  A   : θ-edge fp32 (4K params, full precision)
  B   : θ-edge fp16 (4K params, half precision — cast for forward, fp32 grad)
  C   : θ-edge bf16 (4K params, bfloat16 — wider range, less mantissa)
  D   : θ-edge int8-quantized (4K params, quantize after each step to 256 levels)
  E   : θ-edge fp64 (4K params, double precision — upper bound on "more precision helps")

If A ≈ Ref: AH IS just edge weights; previous step232 was broken.
If A << Ref: hidden W_pos genuinely carries info beyond edge control.
If B/C ≈ A: half precision sufficient — 2 bytes per edge.
If D ≈ A: 1 byte per edge is enough — extreme compression.
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
                    help="Comma-separated config keys (e.g. Ref,A). Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

OUT_PATH = ROOT / "results" / "train_step233_theta_edge_precision.json"


class SGNNET_ThetaEdge(nn.Module):
    """Drop-in AH replacement: sinusoidal θ per edge.

    Uses the EXACT same forward pass as SGNNET_AntiHebbian, only replacing
    the supp_w computation. This avoids step232's forward-pass divergence bug.

    w_edge = (1 + cos(θ)) / 2   ∈ [0, 1]
    θ ∈ [0, 2π], one per edge in conn_hh.
    """

    def __init__(self, N_hidden, N_out, D, N_in, K_in, K_iter,
                 K_local, K_random, n_groups, alpha_reflect,
                 precision="fp32", seed=42):
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
        self.precision = precision
        K_hh = K_local + K_random

        # θ per edge: init at π/3 → w = (1+cos(π/3))/2 = 0.75 (moderate flow)
        self.theta_edge = nn.Parameter(torch.full((N_hidden, K_hh), np.pi / 3))

        self._n_hidden = N_hidden
        self._quantize_levels = 256  # for int8 mode

    @property
    def W_pos(self):
        return self.base_sw.W_pos

    @property
    def W_phase(self):
        return self.resonant.W_phase

    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()
        # Int8 quantization: snap θ to nearest of 256 levels after each epoch
        if self.precision == "int8":
            with torch.no_grad():
                # Map θ to [0, 2π], quantize to 256 levels, map back
                t = self.theta_edge.data % (2 * np.pi)
                t_q = torch.round(t / (2 * np.pi) * 255) / 255 * (2 * np.pi)
                self.theta_edge.data.copy_(t_q)

    def _compute_supp_w(self):
        """Compute edge weights from θ, with precision casting."""
        theta = self.theta_edge  # [N, K_hh]

        if self.precision == "fp16":
            theta_cast = theta.half()
            w = (1.0 + torch.cos(theta_cast.float())) / 2.0
        elif self.precision == "bf16":
            theta_cast = theta.bfloat16()
            w = (1.0 + torch.cos(theta_cast.float())) / 2.0
        elif self.precision == "fp64":
            # MPS doesn't support float64 — compute on CPU, move back
            theta_cast = theta.detach().cpu().double()
            w = ((1.0 + torch.cos(theta_cast)) / 2.0).float().to(theta.device)
            # STE: use cpu-computed value in forward, real gradient in backward
            w_ste = (1.0 + torch.cos(theta)) / 2.0  # fp32 for gradient
            w = w_ste + (w - w_ste).detach()  # forward uses fp64 value, backward uses fp32 grad
        elif self.precision == "int8":
            # Forward: quantize to 256 levels, compute cos
            with torch.no_grad():
                t_q = torch.round((theta % (2 * np.pi)) / (2 * np.pi) * 255) / 255 * (2 * np.pi)
            # STE: use quantized value in forward, real gradient in backward
            theta_ste = theta + (t_q - theta).detach()
            w = (1.0 + torch.cos(theta_ste)) / 2.0
        else:  # fp32
            w = (1.0 + torch.cos(theta)) / 2.0

        return w.unsqueeze(0).unsqueeze(-1)  # [1, N, K_hh, 1]

    def forward(self, x):
        # EXACT same structure as SGNNET_AntiHebbian.forward
        Z = self.base_sw._seed(x)
        conn_hh = self.base_sw.conn_hh
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)

        # θ-edge suppression weights (replaces AH cosine computation)
        supp_w = self._compute_supp_w()

        # Reflection accumulator
        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base_sw.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]  # [B, N, K_hh, D]

            # Apply edge weights (same as AH line 157)
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            # Reflection
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


def build_theta(precision):
    """θ-edge model with specified precision."""
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    return SGNNET_ThetaEdge(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, alpha_reflect=ALPHA_REFLECT,
        precision=precision, seed=SEED)


def main():
    configs = {
        "Ref": ("ah", None),
        "A":   ("theta", "fp32"),
        "B":   ("theta", "fp16"),
        "C":   ("theta", "bf16"),
        "D":   ("theta", "int8"),
        "E":   ("theta", "fp64"),
    }

    run_keys = list(configs.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"Step 233 — θ-Edge Precision: sinusoidal edge weights replacing AH")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        kind, precision = configs[key]
        print(f"\n{'─'*60}")
        if kind == "ah":
            print(f"Config {key}: Standard AH (reference)")
            model = build_ref()
        else:
            print(f"Config {key}: θ-edge, precision={precision}")
            model = build_theta(precision)
        print(f"{'─'*60}")

        model = model.to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Count θ params
        theta_params = 0
        if hasattr(model, 'theta_edge'):
            theta_params = model.theta_edge.numel()

        # Count hidden W_pos params (AH uses these, θ-edge doesn't)
        hidden_wpos_params = 0
        if kind == "ah":
            hidden_wpos_params = N * D  # 2048 * 16 = 32768

        print(f"  total params={n_p:,}  θ-edge params={theta_params:,}  "
              f"hidden W_pos={hidden_wpos_params:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

        # CRITICAL: Trainer only optimizes W_pos + W_phase. Inject theta_edge
        # into the optimizer so it actually receives gradient updates.
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
            w = ((1 + torch.cos(t)) / 2)
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
            print(f"  edge weights: mean={theta_stats['w_mean']}, "
                  f"std={theta_stats['w_std']}, "
                  f"suppressed(<0.3)={theta_stats['frac_suppressed']:.1%}, "
                  f"open(>0.7)={theta_stats['frac_open']:.1%}")

        results[key] = {
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "top1_history": top1h, "elapsed_s": round(elapsed, 1),
            "n_params": n_p, "theta_params": theta_params,
            "hidden_wpos_params": hidden_wpos_params,
            "theta_stats": theta_stats,
            "precision": precision or "ah",
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"\n{'='*70}")
    print("STEP 233 SUMMARY — θ-Edge Precision Ablation")
    print(f"{'='*70}")
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        p_note = f"  ({r['theta_params']} θ-params)" if r['theta_params'] > 0 else f"  ({r['hidden_wpos_params']} W_pos params)"
        print(f"  {key} ({r['precision']:>5s}): {r['top1_best']:.4f}{delta}{p_note}")

    if "A" in results and "Ref" in results:
        gap = results["Ref"]["top1_best"] - results["A"]["top1_best"]
        if gap < 0.01:
            print(f"\n  ✓ A ≈ Ref (gap={gap:.4f}): AH IS just edge weights!")
            print(f"    → Hidden W_pos can be removed. θ-edge saves {N*D - N*K_HH} params.")
        elif gap < 0.03:
            print(f"\n  ~ A ≈ Ref (gap={gap:.4f}): Marginal — advance to Tier-1.")
        else:
            print(f"\n  ✗ A << Ref (gap={gap:.4f}): W_pos carries info beyond edge control.")

    # Precision comparison
    precision_keys = [k for k in run_keys if k in ("A", "B", "C", "D", "E")]
    if len(precision_keys) > 1:
        print(f"\n  Precision ranking:")
        ranked = sorted(precision_keys, key=lambda k: results[k]["top1_best"], reverse=True)
        for k in ranked:
            print(f"    {k} ({results[k]['precision']:>5s}): {results[k]['top1_best']:.4f}")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
