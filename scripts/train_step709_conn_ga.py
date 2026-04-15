"""Step 709: Evolutionary Connectivity GA — population-based topology search.

MOTIVATION
==========
SGNNET's conn_hh (hidden→hidden connectivity) is currently static and random
(small-world structure). This experiment asks: can we learn better connectivity
through an evolutionary search?

Unlike RigL (gradient-based, during-training topology updates, step229 KILLED),
this approach is 'offline' — connectivity evolves between training rounds, not
during gradient descent. No interference with weight learning.

ALGORITHM
=========
Generation 0: N_CAND=10 random topologies
For each generation:
  1. Train each candidate from scratch (random weights) for TRAIN_EP=10 epochs
  2. Score edges: for each edge (u→v), score += val_acc_i × I(u→v in candidate_i)
  3. Aggregate topology: for each neuron u, select top K_hh edges by score
  4. Next gen: 1 aggregate + (N_CAND-1) random (with degree-biased exploration)
     - P(edge u→v) ∝ (1-EXPLORE) × node_score[v] + EXPLORE × uniform
After N_GEN generations:
  - Train aggregate topology for EVAL_EP epochs (Tier-0 scout quality)

COMPARE VS:
  Ref: static small-world conn_hh (standard SGNNET init), EVAL_EP epochs

CONFIGS (N=512, D=16, K_hh=2, K_iter=5, AH α=1.0)
  Ref        : static small-world topology (step199-style, K_local=1, K_random=1)
  ConnGA_rand: fully random sibling generation (EXPLORE=1.0 — pure random)
  ConnGA_deg : degree-biased sibling generation (EXPLORE=0.3 — exploits node popularity)

EXPECTED:
  If ConnGA_rand ≈ Ref: static random is already near-optimal topology
  If ConnGA_rand >> Ref: topology learning works → explore N=2048 and degree-biased
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

parser = argparse.ArgumentParser(description="Step 709: Evolutionary Connectivity GA")
parser.add_argument("--device",   default="auto")
parser.add_argument("--n_gen",    type=int, default=5,  help="Number of GA generations")
parser.add_argument("--n_cand",   type=int, default=10, help="Candidates per generation")
parser.add_argument("--train_ep", type=int, default=10, help="Epochs per candidate")
parser.add_argument("--eval_ep",  type=int, default=20, help="Final evaluation epochs")
parser.add_argument("--configs",  default="",
                    help="Comma-separated config keys. Empty = all.")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

N_GEN    = args.n_gen
N_CAND   = args.n_cand
TRAIN_EP = args.train_ep
EVAL_EP  = args.eval_ep

BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 512; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
EXPLORE_RAND  = 1.0   # pure random siblings
EXPLORE_DEG   = 0.3   # degree-biased: 70% node_score, 30% uniform

FLOPS    = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step709_conn_ga.json"


# ── Topology utilities ────────────────────────────────────────────────────────────

def random_conn_hh(N: int, K: int, seed: int) -> torch.Tensor:
    """[N, K] random connectivity, no self-loops, no duplicates."""
    rng = np.random.default_rng(seed)
    conn = np.zeros((N, K), dtype=np.int64)
    for h in range(N):
        pool = np.delete(np.arange(N), h)
        conn[h] = rng.choice(pool, size=K, replace=False)
    return torch.tensor(conn, dtype=torch.long)


def score_aggregate_conn(
    topologies: list[torch.Tensor],
    accuracies:  list[float],
    N: int, K: int
) -> torch.Tensor:
    """Build aggregate topology from accuracy-weighted edge scores.

    edge_score[u][v] = sum_i (acc_i * I(u→v in candidate_i))
    For each neuron u: select top K neighbors by edge_score.
    """
    # Accumulate scores in a dense N×N matrix
    edge_scores = torch.zeros(N, N, dtype=torch.float32)
    for conn_hh, acc in zip(topologies, accuracies):
        for u in range(N):
            for k in range(K):
                v = int(conn_hh[u, k].item())
                edge_scores[u, v] += acc
    # No self-loop (diagonal should be 0, but enforce it)
    edge_scores.fill_diagonal_(-1e9)
    # Top K per neuron
    top_idx = torch.topk(edge_scores, K, dim=1).indices  # [N, K]
    return top_idx.long()


def degree_biased_conn(
    aggregate_conn: torch.Tensor,
    edge_scores: torch.Tensor,
    N: int, K: int, explore: float, seed: int
) -> torch.Tensor:
    """Generate random topology biased toward high-degree nodes.

    P(edge u→v) ∝ (1-explore) × node_score[v] + explore × uniform
    node_score[v] = sum of edge_scores into v (in-degree popularity)
    """
    rng = np.random.default_rng(seed)
    # Node popularity: in-degree score (sum of incoming edge scores)
    node_score = edge_scores.sum(0).numpy()  # [N]
    node_score = np.clip(node_score, 0, None)
    total = node_score.sum()
    if total > 0:
        node_prob = (1 - explore) * node_score / total + explore / N
    else:
        node_prob = np.ones(N) / N

    conn = np.zeros((N, K), dtype=np.int64)
    for h in range(N):
        p = node_prob.copy()
        p[h] = 0  # no self-loop
        p /= p.sum()
        chosen = rng.choice(N, size=K, replace=False, p=p)
        conn[h] = chosen
    return torch.tensor(conn, dtype=torch.long)


# ── Model builder ─────────────────────────────────────────────────────────────────

def build_model(conn_hh_override=None, seed=SEED) -> nn.Module:
    """Build AH model; optionally inject custom conn_hh."""
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    sw = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    res = SGNNET_Resonant(
        sw, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, mode="dynamic_z_geo",
    )
    model = SGNNET_AntiHebbian(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    if conn_hh_override is not None:
        # Inject custom topology (overrides small-world init)
        model.m.base.conn_hh = conn_hh_override.clone()
    return model


def train_and_eval(model: nn.Module, tr, va, n_epochs: int) -> tuple[float, list]:
    """Train model for n_epochs. Returns (best_acc, history)."""
    kw = trainer_kwargs(N, n_epochs=n_epochs)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **kw)
    history = trainer.train(n_epochs=n_epochs, log_fn=None)
    top1h = [round(h.get("val_top1", 0.), 4) for h in history]
    return max(top1h) if top1h else 0., top1h


# ── Evolutionary search ───────────────────────────────────────────────────────────

def run_ga(tr, va, explore: float, ga_seed: int, label: str) -> dict:
    """Run connectivity GA for N_GEN generations, return result dict."""
    print(f"\n{'='*60}")
    print(f"ConnGA [{label}]  explore={explore:.1f}  {N_GEN}gen × {N_CAND}cand × {TRAIN_EP}ep")
    print(f"{'='*60}")

    # Initial random population
    topologies = [random_conn_hh(N, K_HH, seed=ga_seed + i) for i in range(N_CAND)]
    edge_scores = torch.zeros(N, N, dtype=torch.float32)
    gen_history = []

    for gen in range(N_GEN):
        print(f"\n  Gen {gen+1}/{N_GEN}:")
        accs = []
        for ci, conn_hh in enumerate(topologies):
            model = build_model(conn_hh_override=conn_hh,
                                seed=ga_seed + gen * 1000 + ci).to(DEVICE)
            best_acc, _ = train_and_eval(model, tr, va, TRAIN_EP)
            accs.append(best_acc)
            print(f"    cand {ci+1:2d}: acc={best_acc:.4f}", flush=True)

        mean_acc = float(np.mean(accs))
        max_acc  = float(np.max(accs))
        print(f"  Gen {gen+1} summary: mean={mean_acc:.4f} max={max_acc:.4f}")
        gen_history.append({"gen": gen + 1, "accs": accs, "mean": mean_acc, "max": max_acc})

        # Accumulate edge scores (weighted by accuracy)
        for conn_hh, acc in zip(topologies, accs):
            for u in range(N):
                for k in range(K_HH):
                    v = int(conn_hh[u, k].item())
                    edge_scores[u, v] += acc

        # Build aggregate topology
        es_nodiag = edge_scores.clone()
        es_nodiag.fill_diagonal_(-1e9)
        agg_conn  = torch.topk(es_nodiag, K_HH, dim=1).indices.long()

        # Build next generation: 1 aggregate + (N_CAND-1) siblings
        topologies = [agg_conn]
        for si in range(N_CAND - 1):
            s = ga_seed + (gen + 1) * 1000 + si + 7777
            if explore >= 0.99:  # pure random
                topologies.append(random_conn_hh(N, K_HH, seed=s))
            else:
                topologies.append(degree_biased_conn(
                    agg_conn, edge_scores, N, K_HH, explore=explore, seed=s))

    # Final evaluation: train aggregate topology for EVAL_EP epochs
    print(f"\n  Final eval: aggregate topology, {EVAL_EP}ep")
    final_model = build_model(conn_hh_override=agg_conn,
                               seed=ga_seed + 99999).to(DEVICE)
    final_acc, final_hist = train_and_eval(final_model, tr, va, EVAL_EP)
    print(f"  Final acc={final_acc:.4f}")

    return {
        "label": label,
        "explore": explore,
        "n_gen": N_GEN, "n_cand": N_CAND, "train_ep": TRAIN_EP, "eval_ep": EVAL_EP,
        "gen_history": gen_history,
        "final_acc": final_acc,
        "final_history": final_hist,
        "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER, "flops": FLOPS,
    }


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    all_keys = ["Ref", "ConnGA_rand", "ConnGA_deg"]
    run_keys = ([k.strip() for k in args.configs.split(",")]
                if args.configs else all_keys)

    print(f"\n{'='*70}")
    print(f"Step 709 — Evolutionary Connectivity GA")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {N_GEN}gen × {N_CAND}cand × {TRAIN_EP}ep + {EVAL_EP}ep eval")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.3f}M)  Device={DEVICE}")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total,
                         generator=torch.Generator().manual_seed(SEED))[:n_total // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True,
                                     num_workers=0,
                                     generator=torch.Generator().manual_seed(SEED))
    print(f"Train={len(subset)}  Val={len(va.dataset)}")

    results = {}

    if "Ref" in run_keys:
        print(f"\n{'─'*60}\nConfig Ref: static small-world topology, {EVAL_EP}ep\n{'─'*60}")
        ref_model = build_model(conn_hh_override=None, seed=SEED).to(DEVICE)
        n_p = sum(p.numel() for p in ref_model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")
        t0 = time.time()
        ref_acc, ref_hist = train_and_eval(ref_model, tr, va, EVAL_EP)
        elapsed = time.time() - t0
        results["Ref"] = {
            "label": "Ref: static small-world topology",
            "final_acc": ref_acc, "final_history": ref_hist,
            "elapsed_s": round(elapsed, 1),
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER, "flops": FLOPS,
        }
        print(f"  → Ref acc={ref_acc:.4f}  ({elapsed:.0f}s)")

    if "ConnGA_rand" in run_keys:
        t0 = time.time()
        results["ConnGA_rand"] = run_ga(tr, va, explore=EXPLORE_RAND,
                                         ga_seed=1000, label="ConnGA_rand")
        results["ConnGA_rand"]["elapsed_s"] = round(time.time() - t0, 1)

    if "ConnGA_deg" in run_keys:
        t0 = time.time()
        results["ConnGA_deg"] = run_ga(tr, va, explore=EXPLORE_DEG,
                                        ga_seed=2000, label="ConnGA_deg")
        results["ConnGA_deg"]["elapsed_s"] = round(time.time() - t0, 1)

    # Summary
    ref_acc = results.get("Ref", {}).get("final_acc", 0.)
    print(f"\n{'='*70}")
    print("STEP 709 SUMMARY — Evolutionary Connectivity GA")
    print(f"{'='*70}")
    print(f"  Ref (static): {ref_acc:.4f}")
    for key in ["ConnGA_rand", "ConnGA_deg"]:
        if key in results:
            fa = results[key]["final_acc"]
            delta = fa - ref_acc
            verdict = ("TOPOLOGY LEARNING WORKS" if delta > 0.010
                       else "NEUTRAL" if delta > -0.005
                       else "TOPOLOGY GA HURTS")
            print(f"  {key}: {fa:.4f}  Δ={delta:+.4f}  → {verdict}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
