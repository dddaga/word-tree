"""Step 236: AH weight histogram + adaptive graph pruning.

PHASE 1: DIAGNOSTIC
====================
Train standard AH model, then extract the full histogram of supp_w values.
supp_w = 1 - alpha * max(0, cos(W_pos[i], W_pos[j])) for every edge (i,j).

This tells us:
- How many edges does AH effectively silence (supp_w < 0.1)?
- How many are fully open (supp_w > 0.9)?
- Is the distribution bimodal (binary on/off) or continuous?
- Do dead-end neurons (out_degree=0) correlate with suppressed edges?

PHASE 2: ADAPTIVE PRUNING EXPERIMENTS
======================================
Based on the histogram, test progressive edge deletion:

  Ref   : Standard AH (reference)
  A     : Dense start (K_hh=8) → prune to K_hh=2 based on AH weights at epoch boundaries
  B     : Dense start (all-to-all in groups) → prune to K_hh=2 based on AH weights
  C     : Like A, but reintroduce edges randomly where pruned (stochastic regrowth)
  D     : Like A, but reintroduce edges based on co-activation patterns

The pruning threshold decreases each epoch until only ~K_hh=2 edges remain per neuron.

CONFIGS (N=2048, D=16, K_iter=5, 50% data, 20ep)
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
parser.add_argument("--phase", default="all",
                    help="'diagnostic' for histogram only, 'pruning' for experiments, 'all' for both")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

OUT_PATH = ROOT / "results" / "train_step236_ah_weight_histogram.json"


def compute_ah_histogram(model, device):
    """Extract the full AH suppression weight distribution from a trained model.

    Returns a dict with histogram data and statistics.
    """
    N_h = model.m.base.N_hidden
    W_n = F.normalize(model.m.W_pos[:N_h].to(device), dim=-1)
    conn_hh = model.m.base.conn_hh  # [N, K_hh]

    # Compute cosine similarity for every edge
    pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)  # [N, K_hh]
    supp_w = (1.0 - ALPHA_AHEBB * pos_sim.clamp(min=0)).detach().cpu().numpy()

    # Flatten
    w_flat = supp_w.flatten()

    # Statistics
    stats = {
        "mean": float(np.mean(w_flat)),
        "std": float(np.std(w_flat)),
        "median": float(np.median(w_flat)),
        "min": float(np.min(w_flat)),
        "max": float(np.max(w_flat)),
        "frac_dead": float(np.mean(w_flat < 0.1)),        # effectively silenced
        "frac_weak": float(np.mean(w_flat < 0.3)),        # weak flow
        "frac_moderate": float(np.mean((w_flat >= 0.3) & (w_flat <= 0.7))),
        "frac_strong": float(np.mean(w_flat > 0.7)),      # strong flow
        "frac_full": float(np.mean(w_flat > 0.9)),        # fully open
    }

    # Histogram bins (0 to 1, 20 bins)
    hist, bin_edges = np.histogram(w_flat, bins=20, range=(0, 1))
    stats["histogram_counts"] = hist.tolist()
    stats["histogram_bins"] = bin_edges.tolist()

    # Per-neuron statistics: which neurons have mostly dead edges?
    neuron_mean_w = supp_w.mean(axis=1)  # [N]
    stats["neurons_mostly_dead"] = int(np.sum(neuron_mean_w < 0.2))
    stats["neurons_mostly_open"] = int(np.sum(neuron_mean_w > 0.8))
    stats["neuron_mean_w_std"] = float(np.std(neuron_mean_w))

    # Out-degree analysis: do high-outdegree neurons have different AH patterns?
    conn_hh_np = conn_hh.cpu().numpy()
    out_degree = np.bincount(conn_hh_np.flatten(), minlength=N_h)
    hub_mask = out_degree >= 4
    terminal_mask = out_degree == 0
    if hub_mask.any():
        stats["hub_mean_w"] = float(np.mean(supp_w[hub_mask]))
    if terminal_mask.any():
        stats["terminal_mean_w"] = float(np.mean(supp_w[terminal_mask]))
    stats["n_hubs"] = int(hub_mask.sum())
    stats["n_terminals"] = int(terminal_mask.sum())

    return stats


class SGNNET_AdaptivePrune(nn.Module):
    """SGNNET that starts dense and progressively prunes based on AH weights.

    Strategy:
    - Start with K_hh_start neighbors per neuron
    - At each epoch boundary, compute AH supp_w for all edges
    - Delete edges with lowest supp_w (AH says "silence these")
    - Stop when K_hh_target edges per neuron remain
    - Optional: regrow edges (random or co-activation based)
    """

    def __init__(self, N_hidden, N_out, D, N_in, K_in, K_iter,
                 K_hh_start, K_hh_target, n_groups, alpha_reflect,
                 alpha_ahebb=1.0, regrow_mode="none", seed=42):
        super().__init__()
        torch.manual_seed(seed)

        # Build with K_hh_start (dense initial graph)
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
        self.K_hh_start = K_hh_start
        self.K_hh_target = K_hh_target
        self.regrow_mode = regrow_mode
        self._n_hidden = N_hidden

        # Track which edges are active [N, K_hh_start]: 1=active, 0=pruned
        self.register_buffer("edge_mask",
                             torch.ones(N_hidden, K_hh_start, dtype=torch.bool))
        self._current_k = K_hh_start
        self._prune_count = 0

    @property
    def W_pos(self):
        return self.base.W_pos

    @property
    def W_phase(self):
        return self.resonant.W_phase

    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()
        # Progressive pruning: delete weakest edges this epoch
        self._prune_step()

    def _prune_step(self):
        """Delete edges with lowest AH weights until K_hh_target reached."""
        if self._current_k <= self.K_hh_target:
            return  # already at target

        N_h = self._n_hidden
        W_n = F.normalize(self.W_pos[:N_h], dim=-1)
        conn_hh = self.base.conn_hh

        # Compute AH weights for all edges
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)  # [N, K_hh]
        supp_w = 1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)

        # Mask already-pruned edges with infinity so they're not re-selected
        supp_w_masked = supp_w.clone()
        supp_w_masked[~self.edge_mask] = float('inf')

        # How many edges to prune this step
        # Linear schedule: prune evenly across epochs
        total_to_prune = (self.K_hh_start - self.K_hh_target) * N_h
        total_pruned = (~self.edge_mask).sum().item()
        remaining = total_to_prune - total_pruned
        if remaining <= 0:
            return

        # Prune ~10% of remaining per epoch (smooth schedule)
        n_prune = max(1, int(remaining * 0.15))

        # Find the n_prune weakest active edges
        flat_w = supp_w_masked.flatten()
        _, indices = flat_w.topk(n_prune, largest=False)

        # Convert flat indices to (neuron, edge) pairs
        neuron_idx = indices // self.K_hh_start
        edge_idx = indices % self.K_hh_start

        # Prune
        self.edge_mask[neuron_idx, edge_idx] = False
        self._prune_count += n_prune

        # Regrow (if enabled)
        if self.regrow_mode == "random" and n_prune > 0:
            self._regrow_random(n_prune)

        active = self.edge_mask.sum().item()
        avg_k = active / N_h
        self._current_k = avg_k

    def _regrow_random(self, n_regrow):
        """Regrow n_regrow random edges where edges were pruned."""
        # Find pruned positions
        pruned = ~self.edge_mask
        pruned_flat = pruned.flatten()
        pruned_indices = pruned_flat.nonzero(as_tuple=True)[0]
        if len(pruned_indices) == 0:
            return

        # Randomly reactivate n_regrow of them
        perm = torch.randperm(len(pruned_indices))[:n_regrow]
        reactivate = pruned_indices[perm]

        neuron_idx = reactivate // self.K_hh_start
        edge_idx = reactivate % self.K_hh_start

        # Rewire: assign new random targets
        new_targets = torch.randint(0, self._n_hidden, (n_regrow,),
                                    device=self.base.conn_hh.device)
        self.base.conn_hh[neuron_idx, edge_idx] = new_targets
        self.edge_mask[neuron_idx, edge_idx] = True

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh
        N_h = self._n_hidden
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)

        # AH suppression weights
        W_n = F.normalize(self.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0))

        # Apply edge mask: pruned edges get weight 0
        supp_w = supp_w * self.edge_mask.float()
        supp_w = supp_w.unsqueeze(0).unsqueeze(-1)  # [1, N, K_hh, 1]

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


def build_ref():
    """Standard SGNNET_AntiHebbian reference (K_hh=2)."""
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def build_prune(K_start, regrow):
    """Adaptive pruning model."""
    ng = max(8, N // 8)
    return SGNNET_AdaptivePrune(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER,
        K_hh_start=K_start, K_hh_target=K_HH,
        n_groups=ng, alpha_reflect=ALPHA_REFLECT,
        alpha_ahebb=ALPHA_AHEBB, regrow_mode=regrow, seed=SEED)


def run_diagnostic(tr, va):
    """Phase 1: Train ref model, extract AH weight histogram."""
    print(f"\n{'='*70}")
    print("PHASE 1: AH Weight Histogram Diagnostic")
    print(f"{'='*70}")

    model = build_ref().to(DEVICE)
    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

    def _log(m):
        ep = m["epoch"] + 1
        if ep % 5 == 0 or ep == 1 or ep == EPOCHS:
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)
    history = trainer.train(n_epochs=EPOCHS, log_fn=_log)

    top1h = [round(h.get("val_top1", 0.), 4) for h in history]
    best = max(top1h)
    print(f"  → Ref best={best:.4f}")

    # Extract histogram
    print(f"\nExtracting AH weight distribution...")
    stats = compute_ah_histogram(model, DEVICE)

    print(f"\n  AH Suppression Weight Distribution (N={N}, K_hh={K_HH}, {EPOCHS}ep):")
    print(f"  ──────────────────────────────────────────")
    print(f"  mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
          f"median={stats['median']:.4f}")
    print(f"  range=[{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  dead (<0.1):     {stats['frac_dead']:.1%}")
    print(f"  weak (<0.3):     {stats['frac_weak']:.1%}")
    print(f"  moderate:        {stats['frac_moderate']:.1%}")
    print(f"  strong (>0.7):   {stats['frac_strong']:.1%}")
    print(f"  fully open (>0.9): {stats['frac_full']:.1%}")
    print(f"  neurons mostly dead: {stats['neurons_mostly_dead']}/{N}")
    print(f"  neurons mostly open: {stats['neurons_mostly_open']}/{N}")
    if 'hub_mean_w' in stats:
        print(f"  hub neurons (deg>=4): n={stats['n_hubs']}, mean_w={stats['hub_mean_w']:.4f}")
    if 'terminal_mean_w' in stats:
        print(f"  terminal neurons (deg=0): n={stats['n_terminals']}, "
              f"mean_w={stats['terminal_mean_w']:.4f}")

    # Print histogram
    print(f"\n  Histogram (20 bins, 0 to 1):")
    bins = stats['histogram_bins']
    counts = stats['histogram_counts']
    total = sum(counts)
    for i, c in enumerate(counts):
        bar = '█' * int(c / total * 60)
        print(f"  [{bins[i]:.2f}-{bins[i+1]:.2f}] {c:5d} ({c/total:5.1%}) {bar}")

    return {"ref_best": best, "ah_stats": stats}


def run_pruning(tr, va, diagnostic_results):
    """Phase 2: Adaptive pruning experiments."""
    print(f"\n{'='*70}")
    print("PHASE 2: Adaptive Graph Pruning")
    print(f"{'='*70}")

    configs = {
        "Ref": ("ref", None, None),
        "A":   ("prune", 8, "none"),      # K=8 → K=2, no regrow
        "B":   ("prune", 4, "none"),      # K=4 → K=2, no regrow
        "C":   ("prune", 8, "random"),    # K=8 → K=2, random regrow
    }

    run_keys = list(configs.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    results = {}
    for key in run_keys:
        kind, K_start, regrow = configs[key]
        print(f"\n{'─'*60}")
        if kind == "ref":
            print(f"Config {key}: Standard AH K_hh={K_HH} (reference)")
            model = build_ref()
        else:
            print(f"Config {key}: Dense K={K_start} → prune to K={K_HH}, regrow={regrow}")
            model = build_prune(K_start, regrow)
        print(f"{'─'*60}")

        model = model.to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

        t0 = time.time()
        def _log(m):
            ep = m["epoch"] + 1
            if ep % 5 == 0 or ep == 1:
                suffix = ""
                if hasattr(model, '_current_k'):
                    suffix = f"  avg_k={model._current_k:.1f}"
                print(f"  ep{ep:3d}  val={m['val_top1']:.4f}{suffix}", flush=True)
        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1

        # Final pruning stats
        prune_stats = {}
        if hasattr(model, 'edge_mask'):
            active = model.edge_mask.sum().item()
            total = model.edge_mask.numel()
            prune_stats = {
                "active_edges": int(active),
                "total_edges": int(total),
                "sparsity": round(1.0 - active / total, 4),
                "avg_k": round(active / N, 2),
            }
            print(f"  final: {active}/{total} edges active "
                  f"(sparsity={prune_stats['sparsity']:.1%}, avg_k={prune_stats['avg_k']})")

        results[key] = {
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "top1_history": top1h, "elapsed_s": round(elapsed, 1),
            "n_params": n_p, "K_start": K_start, "regrow": regrow,
            "prune_stats": prune_stats,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    return results


def main():
    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    all_results = {}

    if args.phase in ("diagnostic", "all"):
        diag = run_diagnostic(tr, va)
        all_results["diagnostic"] = diag

    if args.phase in ("pruning", "all"):
        pruning = run_pruning(tr, va, all_results.get("diagnostic"))
        all_results["pruning"] = pruning

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(all_results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print("STEP 236 SUMMARY — AH Weight Histogram + Adaptive Pruning")
    print(f"{'='*70}")

    if "diagnostic" in all_results:
        s = all_results["diagnostic"]["ah_stats"]
        print(f"\n  AH histogram: {s['frac_dead']:.1%} dead, {s['frac_strong']:.1%} strong")
        if s['frac_dead'] > 0.2:
            print(f"  → Significant silencing: {s['frac_dead']:.0%} edges effectively pruned by AH")
        if s['frac_full'] > 0.5:
            print(f"  → Most edges fully open: AH is gentle at this config")

    if "pruning" in all_results:
        ref_best = all_results["pruning"].get("Ref", {}).get("top1_best", 0)
        for key, r in all_results["pruning"].items():
            delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
            k_info = f"  K={r['K_start']}→{K_HH}" if r.get('K_start') else ""
            print(f"  {key}: {r['top1_best']:.4f}{delta}{k_info}")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
