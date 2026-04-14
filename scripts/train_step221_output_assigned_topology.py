"""Step 221: Output-assigned topology — every neuron has K_hh=2 OUTPUTS (not inputs).

MOTIVATION
==========
Current conn_hh: each neuron gathers from exactly K_hh=2 sources.
  → in-degree = 2 (fixed), out-degree varies (12.5% dead-ends)

Proposal (Dhiraj): flip the assignment. Each neuron SENDS to exactly 2 targets.
  → out-degree = 2 (fixed), in-degree varies (Poisson-like)
  → ZERO dead-ends: every neuron's signal always propagates
  → Some neurons become aggregation hubs (in-degree 4+)
  → Some neurons receive 0 routing inputs (only seed) — but still send

Implementation: build output table [N, 2], invert to gather table with variable
in-degree, pad to max_in_degree and mask during sum.

Three variants:
  Ref : Standard (each neuron has 2 inputs, small-world)
  A   : Output-assigned (each neuron has 2 outputs, random targets)
  B   : Output-assigned + anti-preferential (targets sampled with 1/(1+in_degree))
  C   : Output-assigned with group-local bias (50% local, 50% random — small-world-like)

Same total edges (4096) in all configs.

CONFIGS (N=2048, D=16, K_iter=5, 50% data, 20ep — Tier-0)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
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

FLOPS_BASELINE = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step221_output_assigned_topology.json"


# ---------------------------------------------------------------------------
# Output-assigned topology builders
# ---------------------------------------------------------------------------

def build_output_assigned_conn(N, K_out=2, seed=42, mode="random"):
    """Build connectivity where each neuron has exactly K_out output targets.

    Returns conn_hh in GATHER format [N, max_in_degree] + mask [N, max_in_degree].

    mode:
      "random"     — uniform random target selection
      "antipref"   — targets sampled with weight 1/(1+in_degree)
      "local"      — 50% local (same group), 50% random (small-world analog)
    """
    rng = np.random.default_rng(seed)
    n_groups = max(8, N // 8)
    group_size = N // n_groups

    # Step 1: build output table — each neuron sends to K_out targets
    # out_table[i] = [target1, target2, ...] where neuron i's signal goes
    out_table = np.zeros((N, K_out), dtype=np.int64)
    in_degree = np.zeros(N, dtype=np.int64)

    if mode == "random":
        for i in range(N):
            targets = rng.choice([j for j in range(N) if j != i],
                                  size=K_out, replace=False)
            out_table[i] = targets
            for t in targets:
                in_degree[t] += 1

    elif mode == "antipref":
        for i in range(N):
            for k in range(K_out):
                weights = 1.0 / (1.0 + in_degree.astype(np.float64))
                weights[i] = 0.0  # no self-loops
                weights /= weights.sum()
                target = rng.choice(N, p=weights)
                out_table[i, k] = target
                in_degree[target] += 1

    elif mode == "local":
        for i in range(N):
            g = i // group_size
            g_start = g * group_size
            g_end = min(g_start + group_size, N)
            local_pool = [j for j in range(g_start, g_end) if j != i]

            targets = []
            # First half: local targets
            n_local = K_out // 2
            n_random = K_out - n_local
            if n_local > 0 and local_pool:
                local_targets = rng.choice(local_pool,
                                            size=min(n_local, len(local_pool)),
                                            replace=False).tolist()
                targets.extend(local_targets)
            # Remaining: random targets
            global_pool = [j for j in range(N) if j != i and j not in targets]
            random_targets = rng.choice(global_pool,
                                         size=K_out - len(targets),
                                         replace=False).tolist()
            targets.extend(random_targets)

            out_table[i] = targets[:K_out]
            for t in targets[:K_out]:
                in_degree[t] += 1

    # Step 2: invert to gather format
    # For each neuron j, collect all neurons that send to j
    sources = defaultdict(list)
    for i in range(N):
        for k in range(K_out):
            target = out_table[i, k]
            sources[target].append(i)

    max_in = max(len(sources.get(j, [])) for j in range(N))
    # Neurons with 0 sources: they still have their seed, just no routing input
    # We'll pad with self-index (self-loop) and mask it out
    conn_gather = np.zeros((N, max(max_in, 1)), dtype=np.int64)
    conn_mask = np.zeros((N, max(max_in, 1)), dtype=bool)

    for j in range(N):
        srcs = sources.get(j, [])
        for k, src in enumerate(srcs):
            conn_gather[j, k] = src
            conn_mask[j, k] = True
        # Leave remaining slots as 0 with mask=False

    return (torch.tensor(conn_gather, dtype=torch.long),
            torch.tensor(conn_mask, dtype=torch.bool),
            in_degree, max_in)


def analyze_output_topology(conn_gather, conn_mask, in_degree, N, name):
    """Print topology stats."""
    max_in = conn_gather.shape[1]
    dead_in = (in_degree == 0).sum()
    total_edges = conn_mask.sum().item()

    print(f"  Topology '{name}':")
    print(f"    Out-degree: 2 (fixed for all neurons)")
    print(f"    In-degree: mean={in_degree.mean():.2f} std={in_degree.std():.2f} "
          f"min={in_degree.min()} max={in_degree.max()}")
    print(f"    Max in-degree (pad size): {max_in}")
    print(f"    Zero-input neurons: {dead_in} ({dead_in/N*100:.1f}%) — seed only, but still SEND")
    print(f"    Dead-end neurons: 0 (0.0%) — guaranteed by construction")
    print(f"    Total edges: {total_edges}")

    # Effective FLOPs: sum over neurons of K_in_actual × D × 3
    effective_flops = sum(in_degree) * D * 3 * K_ITER
    padded_flops = N * max_in * D * 3 * K_ITER
    print(f"    Effective FLOPs: {effective_flops/1e6:.2f}M  "
          f"(padded: {padded_flops/1e6:.2f}M)")

    return {"dead_ends": 0, "zero_input": int(dead_in),
            "in_std": round(float(in_degree.std()), 3),
            "max_in": int(max_in), "total_edges": int(total_edges),
            "effective_flops": effective_flops, "padded_flops": padded_flops}


class SGNNET_OutputAssigned(nn.Module):
    """SGNNET with output-assigned topology (variable in-degree, fixed out-degree).

    Uses masked gather+sum to handle variable-length source lists.
    """

    def __init__(self, base, resonant, alpha_ahebb=1.0,
                 conn_gather=None, conn_mask=None):
        super().__init__()
        self.base = base
        self.resonant = resonant
        self.alpha_ahebb = alpha_ahebb
        self.use_variable_in = conn_gather is not None

        if conn_gather is not None:
            self.register_buffer("conn_gather", conn_gather)
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

        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_n = F.normalize(self.W_pos[:N_h], dim=-1)

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)

            if self.use_variable_in:
                # Variable in-degree: masked gather+sum
                conn = self.conn_gather      # [N, max_in]
                mask = self.conn_mask.float() # [N, max_in]

                Z_nb = Z_fwd[:, conn, :]     # [B, N, max_in, D]

                # AH suppression
                w_nb = W_n[conn]             # [N, max_in, D]
                pos_sim = (W_n.unsqueeze(1) * w_nb).sum(-1)  # [N, max_in]
                supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0))  # [N, max_in]

                # Apply mask + suppression
                weight = (mask * supp_w).unsqueeze(0).unsqueeze(-1)  # [1, N, max_in, 1]
                Z_struct = (Z_nb * weight).sum(dim=2)  # [B, N, D]
            else:
                # Standard uniform K_hh=2
                conn_hh = self.base.conn_hh
                Z_nb = Z_fwd[:, conn_hh, :]
                pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
                supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                          ).unsqueeze(0).unsqueeze(-1)
                Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.resonant.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.base._readout(Z)


def build_model(config_key):
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)

    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)

    if config_key == "Ref":
        return SGNNET_OutputAssigned(base, resonant, alpha_ahebb=ALPHA_AHEBB)

    modes = {"A": "random", "B": "antipref", "C": "local"}
    conn_g, conn_m, in_deg, max_in = build_output_assigned_conn(
        N, K_out=K_HH, seed=SEED, mode=modes[config_key])

    return SGNNET_OutputAssigned(base, resonant, alpha_ahebb=ALPHA_AHEBB,
                                 conn_gather=conn_g, conn_mask=conn_m)


def main():
    all_keys = ["Ref", "A", "B", "C"]
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]
    else:
        run_keys = all_keys

    labels = {
        "Ref": "Ref: standard input-assigned K_hh=2 (small-world)",
        "A":   "A: output-assigned K_out=2 (random targets)",
        "B":   "B: output-assigned K_out=2 (anti-pref targets, uniform in-degree)",
        "C":   "C: output-assigned K_out=2 (local+random, small-world analog)",
    }

    print(f"\n{'='*70}")
    print(f"Step 221 — Output-Assigned Topology (Tier-0 scouts)")
    print(f"N={N} D={D} K_hh/K_out={K_HH} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"Baseline FLOPs={FLOPS_BASELINE:,} (~{FLOPS_BASELINE/1e6:.2f}M)")
    print(f"Question: does fixed out-degree (no dead-ends) beat fixed in-degree?")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    # Topology analysis
    print(f"\n{'─'*60}")
    print("TOPOLOGY ANALYSIS")
    print(f"{'─'*60}")
    topo_stats = {}

    for key in ["A", "B", "C"]:
        modes = {"A": "random", "B": "antipref", "C": "local"}
        conn_g, conn_m, in_deg, max_in = build_output_assigned_conn(
            N, K_out=K_HH, seed=SEED, mode=modes[key])
        topo_stats[key] = analyze_output_topology(conn_g, conn_m, in_deg, N, labels[key])

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
    print("STEP 221 SUMMARY — Output-assigned topology")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        print(f"  {key}: {r['top1_best']:.4f}{delta}")

    print(f"\nIf A/B/C > Ref: eliminating dead-ends matters more than uniform in-degree.")
    print(f"If Ref > A/B/C: guaranteed input aggregation (in-degree=2) matters more.")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
