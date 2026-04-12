"""Step 237: Connectivity sweep — start dense, prune 1 edge/neuron/epoch.

MOTIVATION
==========
User-directed experiment: Start with every neuron connected to many others,
then uniformly remove 1 connection per neuron per epoch. Track accuracy at
each connectivity level to find the optimal K_hh.

This is different from step236 (AH-guided pruning): here the pruning is
UNIFORM and based on AH suppression weights (weakest edge goes first).
Every neuron loses exactly 1 edge per epoch.

DESIGN:
- Start with K_hh=20 (or max feasible)
- Epoch 1: train with K=20, measure val accuracy
- After epoch 1: remove weakest AH edge from each neuron → K=19
- Epoch 2: train with K=19, measure val accuracy
- ...continue until K=1
- Plot: accuracy vs K_hh

This gives us the optimal connectivity curve and identifies:
1. At what K does accuracy plateau? (above this, extra edges are wasted)
2. At what K does accuracy collapse? (minimum viable connectivity)
3. Is AH-guided pruning better than random pruning?

CONFIGS (N=2048, D=16, K_iter=5, 50% data)
  A : AH-guided pruning (weakest supp_w edge removed per neuron per epoch)
  B : Random pruning (random edge removed per neuron per epoch)
  C : Reverse: start sparse (K=1), ADD 1 edge/neuron/epoch to K=20
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
parser.add_argument("--k-start", type=int, default=16,
                    help="Starting connectivity (edges per neuron)")
parser.add_argument("--k-end", type=int, default=1,
                    help="Final connectivity after pruning")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys. Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

K_START = args.k_start
K_END = args.k_end
# One epoch per connectivity level: total epochs = K_START - K_END + 1
EPOCHS = K_START - K_END + 1

OUT_PATH = ROOT / "results" / "train_step237_connectivity_sweep.json"


class SGNNET_ConnectivitySweep(nn.Module):
    """SGNNET with progressive per-neuron edge pruning.

    Each epoch, remove exactly 1 edge per neuron (the weakest by AH weight
    or randomly). Track accuracy at each connectivity level.
    """

    def __init__(self, N_hidden, N_out, D, N_in, K_in, K_iter,
                 K_hh_start, n_groups, alpha_reflect,
                 alpha_ahebb=1.0, prune_mode="ah", seed=42):
        super().__init__()
        torch.manual_seed(seed)

        K_r = max(1, K_hh_start // 4); K_l = K_hh_start - K_r
        self.base = SGNNET_SmallWorld(
            N_hidden=N_hidden, N_out=N_out, D=D, N_in=N_in,
            K_in=K_in, K_iter=K_iter, K_local=K_l, K_random=K_r,
            n_groups=n_groups, norm_mode="l2", encoding_mode="fourier")

        self.resonant = SGNNET_Resonant(
            self.base, K_phase=8, alpha_reflect=alpha_reflect,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0)

        self.alpha_ahebb = alpha_ahebb
        self.alpha_reflect = alpha_reflect
        self.prune_mode = prune_mode
        self._n_hidden = N_hidden
        self._k_hh = K_hh_start

        # Edge mask: [N, K_hh_start], True=active
        self.register_buffer("edge_mask",
                             torch.ones(N_hidden, K_hh_start, dtype=torch.bool))

        self._epoch_count = 0
        self._rng = np.random.default_rng(seed)

    @property
    def W_pos(self):
        return self.base.W_pos

    @property
    def W_phase(self):
        return self.resonant.W_phase

    @property
    def current_k(self):
        return self.edge_mask.float().sum(dim=1).mean().item()

    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()
        self._epoch_count += 1
        self._prune_one_per_neuron()

    def _prune_one_per_neuron(self):
        """Remove exactly 1 active edge per neuron."""
        N_h = self._n_hidden

        if self.prune_mode == "ah":
            # AH-guided: remove the edge with lowest supp_w per neuron
            W_n = F.normalize(self.W_pos[:N_h], dim=-1)
            conn_hh = self.base.conn_hh
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
            supp_w = 1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)  # [N, K]

            # Set pruned edges to infinity so they're not re-selected
            supp_w_masked = supp_w.clone()
            supp_w_masked[~self.edge_mask] = float('inf')

            # Per-neuron: find the active edge with lowest weight
            min_vals, min_idx = supp_w_masked.min(dim=1)  # [N]

            # Only prune if there's more than 1 active edge
            active_count = self.edge_mask.sum(dim=1)  # [N]
            can_prune = active_count > 1

            # Prune
            for i in range(N_h):
                if can_prune[i] and min_vals[i] < float('inf'):
                    self.edge_mask[i, min_idx[i]] = False

        elif self.prune_mode == "random":
            # Random: remove a random active edge per neuron
            for i in range(N_h):
                active = self.edge_mask[i].nonzero(as_tuple=True)[0]
                if len(active) > 1:
                    idx = active[self._rng.integers(len(active))]
                    self.edge_mask[i, idx] = False

        avg_k = self.current_k
        return avg_k

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh
        N_h = self._n_hidden
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)

        W_n = F.normalize(self.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0))

        # Apply edge mask
        supp_w = supp_w * self.edge_mask.float()
        supp_w = supp_w.unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.base._readout(Z)


def build_sweep(prune_mode):
    """Build connectivity sweep model."""
    ng = max(8, N // 8)
    return SGNNET_ConnectivitySweep(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_hh_start=K_START,
        n_groups=ng, alpha_reflect=ALPHA_REFLECT,
        alpha_ahebb=ALPHA_AHEBB, prune_mode=prune_mode, seed=SEED)


def main():
    configs = {
        "A": "ah",       # AH-guided pruning (weakest edge removed)
        "B": "random",   # Random pruning
    }

    run_keys = list(configs.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"Step 237 — Connectivity Sweep: K={K_START}→{K_END}, "
          f"1 edge/neuron/epoch")
    print(f"Total epochs: {EPOCHS} (one per connectivity level)")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        prune_mode = configs[key]
        print(f"\n{'─'*60}")
        print(f"Config {key}: {prune_mode} pruning, K={K_START}→{K_END}")
        print(f"{'─'*60}")

        model = build_sweep(prune_mode).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}  initial K={K_START}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

        t0 = time.time()
        k_history = []

        def _log(m):
            ep = m["epoch"] + 1
            avg_k = model.current_k
            k_history.append(round(avg_k, 1))
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}  avg_k={avg_k:.1f}", flush=True)

        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1
        best_k = k_history[bep - 1] if bep <= len(k_history) else K_START

        results[key] = {
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "best_k": best_k,
            "top1_history": top1h, "k_history": k_history,
            "elapsed_s": round(elapsed, 1),
            "n_params": n_p, "prune_mode": prune_mode,
        }
        print(f"  → best={best:.4f} @ ep{bep} (K≈{best_k})  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 237 SUMMARY — Connectivity Sweep K={K_START}→{K_END}")
    print(f"{'='*70}")
    for key in run_keys:
        r = results[key]
        print(f"\n  {key} ({r['prune_mode']} pruning):")
        print(f"    best={r['top1_best']:.4f} @ K≈{r['best_k']}")
        print(f"    Accuracy curve (K → acc):")
        # Print every other point for readability
        for i in range(0, len(r['top1_history']), max(1, len(r['top1_history'])//15)):
            k = r['k_history'][i] if i < len(r['k_history']) else "?"
            print(f"      K={k:>5}  acc={r['top1_history'][i]:.4f}")

    print(f"\n  OPTIMAL K: look for the knee in accuracy vs connectivity.")
    print(f"  If best_k ≈ 2: current K_hh=2 is already optimal.")
    print(f"  If best_k > 2: we're under-connected (should use more edges).")
    print(f"  If accuracy plateaus early: most edges are redundant.")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
