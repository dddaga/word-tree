"""Step 232: Scalar edge weights — can we replace AH + hidden W_pos?

MOTIVATION
==========
AH computes supp_w = 1 - α * max(0, cos(W_pos[i], W_pos[j])) per edge.
This uses 2048×16 = 32K hidden W_pos params to produce 2048×2 = 4K edge weights.
If AH is fundamentally just "learned edge weights", a direct scalar per edge
should match performance at 8× fewer params and no cosine compute.

KEY QUESTION: Is W_pos an expensive edge-weight controller, or does it carry
information beyond edge modulation (e.g., through interaction with readout)?

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep)
  Ref : Standard SGNNET_AntiHebbian (32K hidden W_pos, AH suppression)
  A   : Sigmoid scalar weights (4K params, constrained [0,1], init ~0.88)
  B   : Unconstrained scalar weights (4K params, init 1.0, can go negative)
  C   : Random FIXED weights (uniform [0.3, 1.0], NOT learned — baseline)
  D   : No modulation (no AH, no edge weights — should collapse ~18%)

If A/B ≈ Ref: AH is just edge weights. Simplify architecture.
If A/B << Ref: W_pos carries more information than edge weights alone.
If C ≈ Ref: ANY non-uniform modulation works, learning isn't even needed.
If D collapses: confirms AH/modulation is prerequisite (step218 replication).
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

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders

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

OUT_PATH = ROOT / "results" / "train_step232_scalar_edge_weights.json"


class SGNNET_ScalarEdge(nn.Module):
    """SGNNET with scalar learned edge weights replacing AH + hidden W_pos.

    Each edge (i, j) in conn_hh gets a single scalar weight.
    Output W_pos is kept for readout (10 × D = 160 params).
    Hidden W_pos is NOT used — routing doesn't need it.
    """

    def __init__(self, N_hidden, N_out, D, N_in, K_in, K_iter,
                 K_local, K_random, n_groups, alpha_reflect,
                 mode="sigmoid", seed=42):
        super().__init__()
        torch.manual_seed(seed)

        self.base = SGNNET_SmallWorld(
            N_hidden=N_hidden, N_out=N_out, D=D, N_in=N_in,
            K_in=K_in, K_iter=K_iter, K_local=K_local, K_random=K_random,
            n_groups=n_groups, norm_mode="l2", encoding_mode="fourier")

        self.alpha_reflect = alpha_reflect
        self.mode = mode  # "sigmoid", "unconstrained", "fixed", "none"
        K_hh = K_local + K_random

        # Scalar edge weights: [N_hidden, K_hh]
        if mode == "sigmoid":
            # Init logit=2.0 → sigmoid(2.0) ≈ 0.88 (similar to AH with slightly non-orthogonal init)
            self.edge_logits = nn.Parameter(torch.full((N_hidden, K_hh), 2.0))
        elif mode == "unconstrained":
            # Init at 1.0 (no suppression, like AH at orthogonal init)
            self.edge_weights_param = nn.Parameter(torch.ones(N_hidden, K_hh))
        elif mode == "fixed":
            # Random fixed weights in [0.3, 1.0] — not learned
            rng = torch.Generator().manual_seed(seed)
            fixed = 0.3 + 0.7 * torch.rand(N_hidden, K_hh, generator=rng)
            self.register_buffer("fixed_weights", fixed)
        # mode == "none": no modulation at all

        # Output W_pos: still needed for readout (10 class directions)
        # We'll use the base's W_pos[N_hidden:] which is already a parameter
        # But we need to zero gradient on hidden W_pos since we're not using it
        # Actually, the base.W_pos is [N_hidden + N_out, D]. We need it for readout.
        # To keep things clean: freeze hidden portion, let output portion learn.
        self._setup_wpos_grad_mask(N_hidden)

        # Threshold (from resonant-style routing)
        self.theta = nn.Parameter(torch.full((N_hidden,), 0.01))

    def _setup_wpos_grad_mask(self, N_hidden):
        """Mask gradient so hidden W_pos doesn't update (only output W_pos learns)."""
        self._n_hidden = N_hidden
        self.base.W_pos.register_hook(self._mask_hidden_grad)

    def _mask_hidden_grad(self, grad):
        mask = grad.clone()
        mask[:self._n_hidden] = 0
        return mask

    @property
    def W_pos(self):
        return self.base.W_pos

    @property
    def W_phase(self):
        # Dummy for Trainer compatibility
        return self.theta  # just needs to be a parameter

    def tick_epoch(self):
        pass

    def _get_edge_weights(self):
        """Return [N_hidden, K_hh] edge weights based on mode."""
        if self.mode == "sigmoid":
            return torch.sigmoid(self.edge_logits)     # [0, 1]
        elif self.mode == "unconstrained":
            return self.edge_weights_param              # can be anything
        elif self.mode == "fixed":
            return self.fixed_weights
        else:  # "none"
            return None

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh
        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)

        edge_w = self._get_edge_weights()
        if edge_w is not None:
            edge_w = edge_w.unsqueeze(0).unsqueeze(-1)  # [1, N, K_hh, 1]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]  # [B, N, K_hh, D]

            if edge_w is not None:
                Z_struct = (Z_nb * edge_w).sum(dim=2)
            else:
                Z_struct = Z_nb.sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.base._readout(Z)


def build_ref():
    """Standard SGNNET_AntiHebbian for reference."""
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def build_scalar(mode):
    """Scalar edge weight model."""
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    return SGNNET_ScalarEdge(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, alpha_reflect=ALPHA_REFLECT,
        mode=mode, seed=SEED)


def main():
    configs = {
        "Ref": ("ah", None),
        "A": ("scalar", "sigmoid"),
        "B": ("scalar", "unconstrained"),
        "C": ("scalar", "fixed"),
        "D": ("scalar", "none"),
    }

    run_keys = list(configs.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"Step 232 — Scalar Edge Weights: Is AH just learned edge weights?")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        kind, mode = configs[key]
        print(f"\n{'─'*60}")
        if kind == "ah":
            print(f"Config {key}: Standard AH (reference)")
            model = build_ref()
        else:
            print(f"Config {key}: Scalar edge weights, mode={mode}")
            model = build_scalar(mode)
        print(f"{'─'*60}")

        model = model.to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Count edge weight params specifically
        edge_params = 0
        if hasattr(model, 'edge_logits'):
            edge_params = model.edge_logits.numel()
        elif hasattr(model, 'edge_weights_param'):
            edge_params = model.edge_weights_param.numel()

        print(f"  total params={n_p:,}  edge params={edge_params:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

        t0 = time.time()
        def _log(m):
            ep = m["epoch"] + 1
            if ep % 5 == 0 or ep == 1:
                print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)
        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1

        # Log edge weight statistics for learned configs
        edge_stats = {}
        if hasattr(model, 'edge_logits'):
            w = torch.sigmoid(model.edge_logits).detach().cpu()
            edge_stats = {"mean": round(w.mean().item(), 4),
                          "std": round(w.std().item(), 4),
                          "min": round(w.min().item(), 4),
                          "max": round(w.max().item(), 4)}
            print(f"  edge weights: mean={edge_stats['mean']}, std={edge_stats['std']}, "
                  f"range=[{edge_stats['min']}, {edge_stats['max']}]")
        elif hasattr(model, 'edge_weights_param'):
            w = model.edge_weights_param.detach().cpu()
            edge_stats = {"mean": round(w.mean().item(), 4),
                          "std": round(w.std().item(), 4),
                          "min": round(w.min().item(), 4),
                          "max": round(w.max().item(), 4)}
            print(f"  edge weights: mean={edge_stats['mean']}, std={edge_stats['std']}, "
                  f"range=[{edge_stats['min']}, {edge_stats['max']}]")

        results[key] = {
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "top1_history": top1h, "elapsed_s": round(elapsed, 1),
            "n_params": n_p, "edge_params": edge_params,
            "edge_stats": edge_stats, "mode": mode or "ah",
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"\n{'='*70}")
    print("STEP 232 SUMMARY — Scalar Edge Weights vs AH")
    print(f"{'='*70}")
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        params_note = f"  ({r['edge_params']} edge params)" if r['edge_params'] > 0 else ""
        print(f"  {key} ({r['mode']}): {r['top1_best']:.4f}{delta}{params_note}")

    print(f"\nInterpretation guide:")
    print(f"  A/B ≈ Ref → AH is just edge weights. Architecture can be simplified.")
    print(f"  A/B << Ref → W_pos carries info beyond edge control (readout interaction?).")
    print(f"  C ≈ Ref   → ANY non-uniform modulation works, learning not needed.")
    print(f"  D collapses → modulation is prerequisite (confirms step218).")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
