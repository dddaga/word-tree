"""Step 220: Heterogeneous K_hh — neurons with different connectivity levels.

MOTIVATION
==========
Current: every neuron has K_hh=2 (uniform). But the natural topology already
creates heterogeneous roles (hubs vs dead-ends). What if we make this explicit?

Neurons are grouped by connectivity level {K=1, K=2, K=3, K=4}.
Group sizes sampled to keep total edges ≈ N × K_hh_avg = 4096.
Neurons with same K are contiguous → efficient vectorized gather per group.

Anti-preferential sampling within each group ensures uniform out-degree.
Hub neurons (K=4) have 4× information flow; leaf neurons (K=1) are cheap relays.

Three distributions tested (all with total_edges ≈ 4096, anti-pref + coverage):
  Ref : Uniform K_hh=2 everywhere (step199 baseline, small-world)
  A   : Hub-heavy: 25% K=1, 25% K=2, 25% K=3, 25% K=4 (avg K=2.5, ~5120 edges)
  B   : Leaf-heavy: 50% K=1, 25% K=2, 15% K=3, 10% K=4 (avg K=1.85, ~3789 edges)
  C   : Budget-matched: 40% K=1, 35% K=2, 20% K=3, 5% K=4 (avg K=1.9, ~3891 edges)
  D   : Extreme hubs: 60% K=1, 10% K=2, 10% K=3, 20% K=4 (avg K=2.1, ~4301 edges)

FLOPs per group: 3 × N_group × K_group × D × K_iter
Total FLOPs varies by config but stays within ±25% of baseline.

CONFIGS (N=2048, D=16, K_iter=5, 50% data, 20ep — Tier-0)
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

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld, _build_smallworld_conn
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (e.g. A,D). Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

OUT_PATH = ROOT / "results" / "train_step220_heterogeneous_khh.json"

# K_hh distributions: {K_level: fraction_of_neurons}
DISTRIBUTIONS = {
    "Ref": None,  # uniform K_hh=2
    "A": {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25},   # hub-heavy (avg 2.5)
    "B": {1: 0.50, 2: 0.25, 3: 0.15, 4: 0.10},   # leaf-heavy (avg 1.85)
    "C": {1: 0.40, 2: 0.35, 3: 0.20, 4: 0.05},   # budget-matched (avg 1.9)
    "D": {1: 0.60, 2: 0.10, 3: 0.10, 4: 0.20},   # extreme hubs (avg 2.1)
}


# ---------------------------------------------------------------------------
# Heterogeneous connectivity builder
# ---------------------------------------------------------------------------

def build_hetero_conn(N, k_dist, seed=42):
    """Build heterogeneous connectivity with grouped neurons.

    Args:
        N: total neurons
        k_dist: dict {K_level: fraction} e.g. {1: 0.5, 2: 0.25, 3: 0.15, 4: 0.10}

    Returns:
        groups: list of (start, end, K_level) tuples
        conn_padded: [N, max_K] padded connection table
        conn_mask: [N, max_K] bool mask (True = valid edge)
    """
    rng = np.random.default_rng(seed)

    # Assign neurons to groups
    groups = []
    offset = 0
    max_K = max(k_dist.keys())
    total_edges = 0

    for K_level in sorted(k_dist.keys()):
        frac = k_dist[K_level]
        n_neurons = int(N * frac)
        if K_level == max(k_dist.keys()):
            n_neurons = N - offset  # absorb rounding
        groups.append((offset, offset + n_neurons, K_level))
        total_edges += n_neurons * K_level
        offset += n_neurons

    # Build connectivity with anti-preferential sampling + coverage guarantee
    conn_padded = np.zeros((N, max_K), dtype=np.int64)
    conn_mask = np.zeros((N, max_K), dtype=bool)
    out_degree = np.zeros(N, dtype=np.int64)

    # Phase 1: guarantee every neuron appears as source at least once
    uncovered = list(range(N))
    rng.shuffle(uncovered)

    # Build map: neuron -> its K_level
    neuron_k = np.zeros(N, dtype=int)
    filled = np.zeros(N, dtype=int)
    for start, end, K_level in groups:
        neuron_k[start:end] = K_level

    for src in uncovered:
        # Pick a gatherer with empty slots
        candidates = np.where(filled < neuron_k)[0]
        candidates = candidates[candidates != src]
        if len(candidates) == 0:
            continue
        gatherer = rng.choice(candidates)
        slot = filled[gatherer]
        conn_padded[gatherer, slot] = src
        conn_mask[gatherer, slot] = True
        filled[gatherer] += 1
        out_degree[src] += 1

    # Phase 2: fill remaining slots with anti-preferential sampling
    for i in range(N):
        K_level = neuron_k[i]
        for k in range(filled[i], K_level):
            weights = 1.0 / (1.0 + out_degree.astype(np.float64))
            weights[i] = 0.0
            weights /= weights.sum()
            src = rng.choice(N, p=weights)
            conn_padded[i, k] = src
            conn_mask[i, k] = True
            out_degree[src] += 1

    return groups, torch.tensor(conn_padded, dtype=torch.long), \
           torch.tensor(conn_mask, dtype=torch.bool), total_edges


class SGNNET_HeteroKhh(nn.Module):
    """SGNNET with heterogeneous K_hh per neuron group.

    Neurons are grouped by connectivity level. Each group processes
    independently with its own K_hh, then results are concatenated.
    """

    def __init__(self, base, resonant, alpha_ahebb=1.0,
                 groups=None, conn_padded=None, conn_mask=None):
        super().__init__()
        self.base = base
        self.resonant = resonant
        self.alpha_ahebb = alpha_ahebb
        self.groups = groups  # list of (start, end, K_level)

        if groups is not None:
            self.register_buffer("conn_padded", conn_padded)
            self.register_buffer("conn_mask", conn_mask)

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase

    def tick_epoch(self):
        self.resonant.tick_epoch()

    def forward(self, x):
        Z = self.base._seed(x)
        N_h = self.base.N_hidden
        conn_hh = self.base.conn_hh

        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)

        # AH suppression
        W_n = F.normalize(self.W_pos[:N_h], dim=-1)

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)

            if self.groups is not None:
                # Heterogeneous routing: process each group separately
                Z_struct = torch.zeros_like(Z)

                for start, end, K_level in self.groups:
                    if K_level == 0:
                        continue
                    # Get this group's connections: [group_size, K_level]
                    group_conn = self.conn_padded[start:end, :K_level]

                    # Gather neighbors for this group
                    Z_nb = Z_fwd[:, group_conn, :]  # [B, group_size, K_level, D]

                    # AH suppression for this group
                    w_group = W_n[start:end]  # [group_size, D]
                    w_nb = W_n[group_conn]    # [group_size, K_level, D]
                    pos_sim = (w_group.unsqueeze(1) * w_nb).sum(-1)  # [group_size, K_level]
                    supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                              ).unsqueeze(0).unsqueeze(-1)  # [1, group_size, K_level, 1]

                    Z_nb = Z_nb * supp_w
                    Z_struct[:, start:end, :] = Z_nb.sum(dim=2)
            else:
                # Standard uniform K_hh routing with AH
                pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
                supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                          ).unsqueeze(0).unsqueeze(-1)
                Z_nb = Z_fwd[:, conn_hh, :]
                Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.resonant.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.base._readout(Z)


def analyze_hetero_topology(groups, conn_padded, conn_mask, N):
    """Print topology stats for heterogeneous connectivity."""
    conn_np = conn_padded.numpy()
    mask_np = conn_mask.numpy()

    total_edges = mask_np.sum()
    avg_k = total_edges / N
    flops = 0

    print(f"  Groups:")
    for start, end, K_level in groups:
        n = end - start
        f = 3 * n * K_level * D * K_ITER
        flops += f
        print(f"    K={K_level}: neurons [{start}:{end}] ({n} neurons, "
              f"{n*K_level} edges, {f/1e6:.2f}M FLOPs)")

    # Out-degree
    out_degree = np.zeros(N, dtype=int)
    for i in range(N):
        for k in range(conn_np.shape[1]):
            if mask_np[i, k]:
                out_degree[conn_np[i, k]] += 1

    dead = (out_degree == 0).sum()
    print(f"  Total edges: {total_edges}  avg_K: {avg_k:.2f}")
    print(f"  Total FLOPs: {flops/1e6:.2f}M (vs baseline {3*N*2*D*K_ITER/1e6:.2f}M)")
    print(f"  Out-degree: mean={out_degree.mean():.2f} std={out_degree.std():.2f} "
          f"min={out_degree.min()} max={out_degree.max()}")
    print(f"  Dead-ends: {dead} ({dead/N*100:.1f}%)")

    return {"total_edges": int(total_edges), "avg_k": round(avg_k, 2),
            "flops": flops, "dead_ends": int(dead),
            "out_std": round(float(out_degree.std()), 3)}


def build_model(config_key):
    torch.manual_seed(SEED)
    K_r = max(1, 2 // 4); K_l = 2 - K_r; ng = max(8, N // 8)

    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)

    dist = DISTRIBUTIONS[config_key]
    if dist is None:
        return SGNNET_HeteroKhh(base, resonant, alpha_ahebb=ALPHA_AHEBB)
    else:
        groups, conn_padded, conn_mask, total_edges = build_hetero_conn(N, dist, seed=SEED)
        return SGNNET_HeteroKhh(base, resonant, alpha_ahebb=ALPHA_AHEBB,
                                groups=groups, conn_padded=conn_padded,
                                conn_mask=conn_mask)


def main():
    all_keys = ["Ref", "A", "B", "C", "D"]
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]
    else:
        run_keys = all_keys

    labels = {
        "Ref": "Ref: uniform K_hh=2 (step199 baseline)",
        "A":   "A: hub-heavy {1:25%, 2:25%, 3:25%, 4:25%} avg_K=2.5",
        "B":   "B: leaf-heavy {1:50%, 2:25%, 3:15%, 4:10%} avg_K=1.85",
        "C":   "C: budget-matched {1:40%, 2:35%, 3:20%, 4:5%} avg_K=1.9",
        "D":   "D: extreme hubs {1:60%, 2:10%, 3:10%, 4:20%} avg_K=2.1",
    }

    print(f"\n{'='*70}")
    print(f"Step 220 — Heterogeneous K_hh (Tier-0 scouts)")
    print(f"N={N} D={D} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"Question: does heterogeneous connectivity beat uniform K_hh=2?")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    # Topology analysis
    print(f"\n{'─'*60}")
    print("TOPOLOGY ANALYSIS")
    print(f"{'─'*60}")
    topo_stats = {}

    for key in all_keys:
        dist = DISTRIBUTIONS[key]
        if dist is None:
            print(f"\n  Config {key}: uniform K_hh=2")
            baseline_flops = 3 * N * 2 * D * K_ITER
            print(f"  Total edges: {N*2}  avg_K: 2.00")
            print(f"  Total FLOPs: {baseline_flops/1e6:.2f}M")
            topo_stats[key] = {"total_edges": N*2, "avg_k": 2.0, "flops": baseline_flops}
        else:
            print(f"\n  Config {key}: {labels[key]}")
            groups, conn_p, conn_m, te = build_hetero_conn(N, dist, seed=SEED)
            topo_stats[key] = analyze_hetero_topology(groups, conn_p, conn_m, N)

    # Data
    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        print(f"\n{'─'*60}")
        print(f"Config {labels.get(key, key)}")
        print(f"{'─'*60}")

        model = build_model(key).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: None)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1

        results[key] = {
            "label": labels.get(key, key),
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "epochs_run": len(history), "top1_history": top1h,
            "elapsed_s": round(elapsed, 1), "n_params": n_p,
            "topology_stats": topo_stats.get(key, {}),
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print("STEP 220 SUMMARY — Heterogeneous K_hh")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for key in run_keys:
        r = results[key]
        ts = r.get("topology_stats", {})
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        flops_str = f"  FLOPs={ts.get('flops', 0)/1e6:.2f}M" if 'flops' in ts else ""
        print(f"  {key}: {r['top1_best']:.4f}{delta}{flops_str}")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
