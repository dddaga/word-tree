"""Step 219: Anti-preferential topology — uniform out-degree via biased sampling.

MOTIVATION
==========
At K_hh=2, the current small-world graph has:
  - 12.5% dead-end neurons (zero out-degree, signal never propagates)
  - Signal reach after K_iter=5: only 1-5% of network
  - Out-degree std=1.36 (Poisson-like, highly variable)

Proposal (Dhiraj): Build connectivity using anti-preferential sampling.
  1. Ensure all inputs covered (conn_in unchanged)
  2. For conn_hh: sample targets with weight = 1/(1 + existing_out_degree)
     → neurons already heavily connected become LESS likely targets
  3. Guarantee every neuron has ≥1 out-degree (no dead-ends)
  4. Same total edges (N × K_hh = 4096) → same FLOPs

This creates a tree-like branching structure where signal spreads far
and distribution is more uniform. Each neuron's signal actually reaches
other neurons during routing.

Three topology variants tested:
  Ref : Standard small-world (current, K_local=1 K_random=1)
  A   : Anti-preferential (weight = 1/(1+out_degree), same total edges)
  B   : Anti-pref + guaranteed coverage (every neuron out_degree ≥ 1)
  C   : Pure uniform random (no group structure, baseline comparison)

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep — Tier-0)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import Counter

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
                    help="Comma-separated config keys (e.g. A,C). Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step219_antipreferential_topology.json"


# ---------------------------------------------------------------------------
# Topology builders
# ---------------------------------------------------------------------------

def build_antipreferential_conn(N, K_hh, seed=42, guarantee_coverage=False):
    """Build conn_hh using anti-preferential sampling.

    For each neuron i, sample K_hh targets with probability:
        P(j) ∝ 1 / (1 + out_degree(j))
    where out_degree counts how many neurons already gather from j.

    This ensures neurons that are already heavily referenced become
    less likely to be picked, spreading the graph uniformly.

    If guarantee_coverage=True, first pass ensures every neuron appears
    as a source at least once (no dead-ends), then fills remaining edges
    with anti-preferential sampling.
    """
    rng = np.random.default_rng(seed)
    conn = np.zeros((N, K_hh), dtype=np.int64)

    # Track out-degree: how many times each neuron appears as a source
    out_degree = np.zeros(N, dtype=np.int64)

    if guarantee_coverage:
        # Phase 1: ensure every neuron is referenced at least once.
        # Assign one guaranteed edge per uncovered neuron.
        # Shuffle neurons; for each, assign it as a source to a random gatherer.
        uncovered = list(range(N))
        rng.shuffle(uncovered)

        # Each neuron needs K_hh sources. Track how many we've filled.
        filled = np.zeros(N, dtype=np.int64)

        for src in uncovered:
            # Pick a random gatherer that still has empty slots
            candidates = np.where(filled < K_hh)[0]
            # Exclude self
            candidates = candidates[candidates != src]
            if len(candidates) == 0:
                continue
            gatherer = rng.choice(candidates)
            slot = filled[gatherer]
            conn[gatherer, slot] = src
            filled[gatherer] += 1
            out_degree[src] += 1

        # Phase 2: fill remaining empty slots with anti-preferential sampling
        for i in range(N):
            for k in range(filled[i], K_hh):
                # Anti-preferential weights
                weights = 1.0 / (1.0 + out_degree.astype(np.float64))
                weights[i] = 0.0  # no self-loops
                weights /= weights.sum()
                src = rng.choice(N, p=weights)
                conn[i, k] = src
                out_degree[src] += 1

    else:
        # Pure anti-preferential: for each neuron, sample K_hh sources
        for i in range(N):
            for k in range(K_hh):
                weights = 1.0 / (1.0 + out_degree.astype(np.float64))
                weights[i] = 0.0  # no self-loops
                weights /= weights.sum()
                src = rng.choice(N, p=weights)
                conn[i, k] = src
                out_degree[src] += 1

    return torch.tensor(conn, dtype=torch.long)


def build_uniform_random_conn(N, K_hh, seed=42):
    """Pure uniform random connectivity — no group structure."""
    rng = np.random.default_rng(seed)
    conn = np.zeros((N, K_hh), dtype=np.int64)
    for i in range(N):
        candidates = np.concatenate([np.arange(0, i), np.arange(i+1, N)])
        conn[i] = rng.choice(candidates, size=K_hh, replace=False)
    return torch.tensor(conn, dtype=torch.long)


def analyze_topology(conn, name):
    """Print topology statistics."""
    N, K_hh = conn.shape
    conn_np = conn.numpy()

    # Out-degree
    out_degree = np.zeros(N, dtype=int)
    for i in range(N):
        for k in range(K_hh):
            out_degree[conn_np[i, k]] += 1

    dead_ends = (out_degree == 0).sum()
    print(f"  Topology '{name}':")
    print(f"    Out-degree: mean={out_degree.mean():.2f} std={out_degree.std():.2f} "
          f"min={out_degree.min()} max={out_degree.max()}")
    print(f"    Dead-ends (out_deg=0): {dead_ends} ({dead_ends/N*100:.1f}%)")

    # Signal reach after K_iter=5 (BFS from neuron 0)
    # Build reverse adjacency (who does each neuron's signal reach?)
    reverse_adj = [[] for _ in range(N)]
    for j in range(N):
        for k in range(K_hh):
            src = conn_np[j, k]
            reverse_adj[src].append(j)

    from collections import deque
    visited = {0: 0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for nb in reverse_adj[node]:
            if nb not in visited:
                visited[nb] = visited[node] + 1
                queue.append(nb)

    reach_5 = sum(1 for d in visited.values() if d <= 5)
    reach_all = len(visited)
    max_dist = max(visited.values()) if len(visited) > 1 else 0
    print(f"    Neuron 0 reach @ K_iter=5: {reach_5}/{N} ({reach_5/N*100:.1f}%)")
    print(f"    Neuron 0 total reach: {reach_all}/{N} ({reach_all/N*100:.1f}%)")
    print(f"    Graph diameter from neuron 0: {max_dist} hops")

    return {"dead_ends": int(dead_ends), "dead_pct": round(dead_ends/N*100, 1),
            "out_std": round(float(out_degree.std()), 3),
            "reach_5": reach_5, "reach_pct": round(reach_5/N*100, 1),
            "diameter": max_dist}


def build_model(config_key):
    """Build model with specified topology."""
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)

    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")

    # Override conn_hh based on config
    if config_key == "Ref":
        pass  # keep default small-world
    elif config_key == "A":
        base.conn_hh = build_antipreferential_conn(N, K_HH, seed=SEED,
                                                     guarantee_coverage=False).to(base.conn_hh.device)
    elif config_key == "B":
        base.conn_hh = build_antipreferential_conn(N, K_HH, seed=SEED,
                                                     guarantee_coverage=True).to(base.conn_hh.device)
    elif config_key == "C":
        base.conn_hh = build_uniform_random_conn(N, K_HH, seed=SEED).to(base.conn_hh.device)

    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    model = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return model


def main():
    all_keys = ["Ref", "A", "B", "C"]
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]
    else:
        run_keys = all_keys

    labels = {
        "Ref": "Ref: small-world (K_local=1, K_random=1)",
        "A":   "A: anti-preferential (no coverage guarantee)",
        "B":   "B: anti-preferential + guaranteed coverage (no dead-ends)",
        "C":   "C: pure uniform random (no group structure)",
    }

    print(f"\n{'='*70}")
    print(f"Step 219 — Anti-Preferential Topology (Tier-0 scouts)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M) — same for all configs")
    print(f"Question: does uniform out-degree beat small-world at K_hh=2?")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    # Analyze topologies first
    print(f"\n{'─'*60}")
    print("TOPOLOGY ANALYSIS")
    print(f"{'─'*60}")
    topo_stats = {}

    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)

    from src.sgnnet.model_smallworld import _build_smallworld_conn
    conn_ref = _build_smallworld_conn(N, K_l, K_r, ng, seed=0)
    topo_stats["Ref"] = analyze_topology(conn_ref, "small-world")

    conn_a = build_antipreferential_conn(N, K_HH, seed=SEED, guarantee_coverage=False)
    topo_stats["A"] = analyze_topology(conn_a, "anti-preferential")

    conn_b = build_antipreferential_conn(N, K_HH, seed=SEED, guarantee_coverage=True)
    topo_stats["B"] = analyze_topology(conn_b, "anti-pref+coverage")

    conn_c = build_uniform_random_conn(N, K_HH, seed=SEED)
    topo_stats["C"] = analyze_topology(conn_c, "uniform-random")

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
    print("STEP 219 SUMMARY — Topology comparison")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for key in run_keys:
        r = results[key]
        ts = r.get("topology_stats", {})
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        dead = f"  dead={ts.get('dead_pct', '?')}%" if ts else ""
        reach = f"  reach@5={ts.get('reach_pct', '?')}%" if ts else ""
        print(f"  {key}: {r['top1_best']:.4f}{delta}{dead}{reach}")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
