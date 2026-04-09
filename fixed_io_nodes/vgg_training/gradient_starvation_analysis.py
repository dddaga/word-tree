#!/usr/bin/env python3
"""
Gradient starvation analysis for NativeNeurographLayer.

Loads the run10 checkpoint and measures how gradient signal flows to intermediate
nodes. Reports by node type and by hop distance from output nodes (i.e. how many
message-passing steps separate each intermediate node from the loss signal).

Usage (from vgg_training/):
    /Volumes/T9/IndraAstra/sudarshan/.venv/bin/python gradient_starvation_analysis.py
"""
import copy
import sys
from collections import deque, defaultdict, Counter
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]  # word-tree/ (contains fixed_io_nodes/)
sys.path.insert(0, str(ROOT))

from fixed_io_nodes.native import NativeNeurographLayer, NativeGNNOptimizer  # noqa: E402
from fixed_io_nodes.native.checkpoint import load_full_model  # noqa: E402
from fixed_io_nodes.main import load_config  # noqa: E402

VGG_DIR = Path(__file__).resolve().parent
CFG_PATH = VGG_DIR / "training_runs/run10/config.yaml"
WEIGHTS_PATH = VGG_DIR / "training_runs/run10/run10_weights.pt"
DATA_PATH = VGG_DIR / "data/imagenette_val_data.pt"
BATCH = 16          # small enough to run on CPU quickly
ZERO_THRESH = 1e-6  # grad norm below this = "starved"


# ---------------------------------------------------------------------------
# Model definition (mirrors training_run10.py CustomHybridModel)
# ---------------------------------------------------------------------------
class CustomHybridModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_nodes = cfg["graph"]["input_nodes"]
        self.vector_dim = cfg["model"]["vector_dim"]
        self.output_nodes = cfg["graph"]["output_nodes"]
        self.gnn = NativeNeurographLayer(cfg)

    def forward(self, x):
        h = x.view(x.size(0), self.input_nodes, self.vector_dim)
        return self.gnn(h)


# ---------------------------------------------------------------------------
# BFS: hop distance from output nodes following reverse edges
# ---------------------------------------------------------------------------
def compute_hop_distances(edge_indices: torch.Tensor, output_ids: list, N: int) -> list:
    """
    For each node, compute the minimum number of reverse-edge hops to reach an output.
    Gradient flows backwards along edges, so hop k = gradient travels k edges.
    Returns list of length N; -1 means unreachable within the graph.
    """
    src_arr = edge_indices[0].cpu().tolist()
    dst_arr = edge_indices[1].cpu().tolist()

    # Reverse adjacency: pred[n] = list of nodes that send an edge TO n
    pred = [[] for _ in range(N)]
    for s, d in zip(src_arr, dst_arr):
        if s != d:          # skip self-loops
            pred[d].append(s)

    hop = [-1] * N
    q = deque()
    for o in output_ids:
        hop[o] = 0
        q.append(o)

    while q:
        node = q.popleft()
        for p in pred[node]:
            if hop[p] == -1:
                hop[p] = hop[node] + 1
                q.append(p)

    return hop


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("GRADIENT STARVATION ANALYSIS — run10 checkpoint")
    print("=" * 65)

    # Load config and force CPU (checkpoint was saved on MPS)
    cfg = load_config(str(CFG_PATH))
    cfg = copy.deepcopy(cfg)
    cfg["system"]["device"] = "cpu"
    DEVICE = "cpu"

    model = CustomHybridModel(cfg).to(DEVICE)
    load_full_model(model, str(WEIGHTS_PATH), map_location=DEVICE)
    print(f"\nLoaded: {WEIGHTS_PATH}")

    ns = model.gnn._node_store
    N = ns.total_nodes
    n_in = ns.input_nodes
    n_out = ns.output_nodes
    n_int = N - n_in - n_out
    input_ids = sorted(ns.input_nodeids)
    output_ids = sorted(ns.output_nodeids)
    inter_ids = sorted(ns.intermediate_nodeids)
    V = ns.vector_dim
    iters = model.gnn._iterations
    card = ns.cardinality

    print(f"  Nodes : {N} total = {n_in} input + {n_int} intermediate + {n_out} output")
    print(f"  Config: cardinality={card}, vector_dim={V}, iterations={iters}")

    # Load data
    val = torch.load(str(DATA_PATH), map_location=DEVICE, weights_only=False)
    x = val["data"][:BATCH].to(DEVICE)
    y = val["label"][:BATCH].to(DEVICE)
    print(f"  Batch : {BATCH} val samples\n")

    # -----------------------------------------------------------------------
    # SECTION 1: Gradient norms after backward pass
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("SECTION 1 — Gradient norms by node type")
    print("=" * 65)

    model.train()
    model.zero_grad()
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    print(f"  Loss = {loss.item():.4f}  |  "
          f"Preds: {(logits.argmax(1) == y.argmax(1)).sum().item()}/{BATCH} correct")
    loss.backward()

    pg = ns.phase_weight.grad   # (N, V)
    mg = ns.mag_weight.grad     # (N, V) or None

    if pg is None:
        print("ERROR: phase_weight.grad is None — check backward path")
        return

    phase_norms = pg.norm(dim=1)          # (N,)
    mag_norms = mg.norm(dim=1) if mg is not None else torch.zeros(N)
    grad_norms = (phase_norms + mag_norms) / 2  # combined per-node gradient magnitude

    def report_group(label, ids):
        g = grad_norms[ids]
        pct_zero = (g < ZERO_THRESH).float().mean().item() * 100
        print(f"  {label:<18}  n={len(ids):>6}  "
              f"mean={g.mean():.3e}  median={g.median():.3e}  "
              f"p95={g.quantile(0.95):.3e}  starved={pct_zero:.1f}%")

    report_group("Input",        input_ids)
    report_group("Intermediate", inter_ids)
    report_group("Output",       output_ids)

    # -----------------------------------------------------------------------
    # SECTION 2: Gradient by hop distance from output nodes
    # -----------------------------------------------------------------------
    print()
    print("=" * 65)
    print("SECTION 2 — Gradient norms by reverse-hop distance from outputs")
    print("(hop 0 = output node; hop k = gradient travels k edges back)")
    print("=" * 65)

    print("  Computing BFS hop distances ... ", end="", flush=True)
    hop = compute_hop_distances(ns.edge_indices, output_ids, N)
    print("done")

    dist_groups = defaultdict(list)
    for i, d in enumerate(hop):
        dist_groups[d].append(i)

    print(f"\n  {'Hop':>12} {'#nodes':>8} {'mean grad':>11} {'median':>11} {'% starved':>11}")
    print("  " + "-" * 57)
    for d in sorted(k for k in dist_groups if k >= 0):
        ids = dist_groups[d]
        g = grad_norms[ids]
        pct_zero = (g < ZERO_THRESH).float().mean().item() * 100
        tag = " (outputs)" if d == 0 else ""
        print(f"  {'hop ' + str(d) + tag:>20}  {len(ids):>6}  "
              f"{g.mean():.3e}  {g.median():.3e}  {pct_zero:>9.1f}%")
    if -1 in dist_groups:
        ids = dist_groups[-1]
        g = grad_norms[ids]
        pct_zero = (g < ZERO_THRESH).float().mean().item() * 100
        print(f"  {'UNREACHABLE':>20}  {len(ids):>6}  "
              f"{g.mean():.3e}  {g.median():.3e}  {pct_zero:>9.1f}%")

    # Distribution of hop distances for intermediate nodes only
    int_hops = Counter(hop[i] for i in inter_ids)
    print("\n  Intermediate nodes by hop distance:")
    for d in sorted(int_hops):
        label = f"hop {d}" if d >= 0 else "unreachable"
        pct = 100 * int_hops[d] / n_int
        bar = "█" * int(pct / 2)
        print(f"    {label:>12}: {int_hops[d]:>5} ({pct:5.1f}%)  {bar}")

    # -----------------------------------------------------------------------
    # SECTION 3: Activation strength distribution across iterations
    # -----------------------------------------------------------------------
    print()
    print("=" * 65)
    print("SECTION 3 — Activation strength per iteration (eval, no_grad)")
    print("=" * 65)

    iter_act = {}

    def _hook(iter_idx, act_detached, B, _N):
        # act_detached: (B*N,) after this iteration
        as_2d = act_detached.view(B, _N).mean(0)  # (N,) batch-averaged
        iter_act[iter_idx] = as_2d.cpu()

    model.gnn._iter_stats_hook = _hook
    model.eval()
    with torch.no_grad():
        _ = model(x)
    model.gnn._iter_stats_hook = None

    print(f"\n  {'iter':>6} {'mean':>10} {'std':>10} {'|max|':>10} {'% near-0':>12}")
    print("  " + "-" * 53)
    for it in sorted(iter_act):
        a = iter_act[it]
        near0 = (a.abs() < 0.01).float().mean().item() * 100
        print(f"  {it:>6} {a.mean():>10.4f} {a.std():>10.4f} "
              f"{a.abs().max():>10.4f} {near0:>11.1f}%")

    if iter_act:
        last = max(iter_act)
        a = iter_act[last]
        print(f"\n  Final iteration breakdown (iter {last}):")
        for label, ids in [("inputs", input_ids), ("intermediate", inter_ids), ("outputs", output_ids)]:
            sub = a[ids]
            print(f"    {label:<14}  mean={sub.mean():.4f}  std={sub.std():.4f}  "
                  f"% near-0={(sub.abs() < 0.01).float().mean() * 100:.1f}%")

    # -----------------------------------------------------------------------
    # SECTION 4: Gradient concentration
    # -----------------------------------------------------------------------
    print()
    print("=" * 65)
    print("SECTION 4 — Gradient concentration (how many nodes carry the signal)")
    print("=" * 65)

    sorted_norms, _ = grad_norms.sort(descending=True)
    total_grad = sorted_norms.sum().item()
    if total_grad > 0:
        cum = sorted_norms.cumsum(0)
        n_for_50 = (cum < 0.5 * total_grad).sum().item()
        n_for_90 = (cum < 0.9 * total_grad).sum().item()
        pct_50 = 100 * n_for_50 / N
        pct_90 = 100 * n_for_90 / N
        pct_zero = (grad_norms < ZERO_THRESH).float().mean().item() * 100
        print(f"\n  {pct_50:.1f}% of nodes carry 50% of the total gradient signal")
        print(f"  {pct_90:.1f}% of nodes carry 90% of the total gradient signal")
        print(f"  {pct_zero:.1f}% of all nodes are starved (grad < {ZERO_THRESH:.0e})")
        print(f"  → Intermediate node starvation: "
              f"{(grad_norms[inter_ids] < ZERO_THRESH).float().mean() * 100:.1f}%")

    # -----------------------------------------------------------------------
    # SECTION 5: Interpretation and proposals
    # -----------------------------------------------------------------------
    print()
    print("=" * 65)
    print("SECTION 5 — Root causes and experiments to try")
    print("=" * 65)

    # Estimate reachable intermediates under current config
    reachable_now = sum(1 for i in inter_ids if 0 <= hop[i] <= iters)
    pct_reachable = 100 * reachable_now / n_int

    # Estimate extra reach with more iterations
    reachable_8 = sum(1 for i in inter_ids if 0 <= hop[i] <= 8)
    pct_reach_8 = 100 * reachable_8 / n_int

    print(f"""
  ROOT CAUSES
  -----------
  1. OUTPUT BOTTLENECK  ({n_out} outputs × ~{card} in-edges = ~{n_out * card:,} gradient paths)
     With {iters} iterations, only intermediates within hop ≤ {iters} receive gradient.
     That is {reachable_now:,} / {n_int:,} intermediates = {pct_reachable:.1f}%.
     The rest are unreachable in the backward pass for this run.

  2. SOFTMAX ROUTING CONCENTRATION  (in update_activations)
     Routing weights = softmax(act_strength over incoming sources).
     If one source dominates (high act_strength), it gets weight ≈ 1 and others ≈ 0.
     Those others receive essentially zero gradient → they can't learn.

  3. GRADIENT DILUTION THROUGH DEPTH
     4 chained grad_checkpoint calls multiply vanishing: gradient at hop-k
     passes through k softmax + scatter_add + complex-multiply operations.
     Even reachable nodes at hop 3-4 get much smaller gradient than hop 1-2.

  EXPERIMENTS TO TRY (ordered by expected impact)
  -------------------------------------------------
  A. run12: More iterations  (5 → 8 or 10)
     With 8 iters: {reachable_8:,} / {n_int:,} intermediates reachable ({pct_reach_8:.1f}%).
     Cost: ~1.6× slower per step. Easy config change.

  B. run13: Higher cardinality  (200 → 400)
     Each output node gets 2× more direct predecessors → 2× gradient surface.
     Also reduces hop distances: more shortcuts in the graph.

  C. run14: Auxiliary intermediate loss
     Sample 512 random intermediate nodes per batch. Add a small auxiliary
     CrossEntropy loss (weight=0.1) computed from their act_strength projected
     to 10 classes via a frozen random projection (no learnable params needed).
     Forces direct gradient to deep intermediates without adding classification power.

  D. run15: Soft routing (temperature scaling)
     Replace softmax routing in update_activations with temperature-scaled version:
       routing = softmax(act_strength / T)  with T starting at 2.0 decaying to 1.0
     Higher T → more uniform routing → more nodes receive gradient early in training.

  E. run16: More output nodes (10 → 64) for gradient surface
     output_nodes=64 gives 6× more gradient paths (64×200=12,800 direct predecessors).
     The 64 act_strength values are still used as logits via a *learnable 1D conv kernel*
     (groups=1, kernel=64→10, no bias) — this is a single dot product, not an MLP.
     The GNN still does all the representation learning; the kernel just mixes 64→10.
""")


if __name__ == "__main__":
    main()
