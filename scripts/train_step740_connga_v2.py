"""Step 740: ConnGA v2 — Stabilised Evolutionary Topology Search.

MOTIVATION
==========
Step 709 ran ConnGA with TRAIN_EP=10 per child: too short for N=512 networks
to settle. Learning curves at N=512/D=16 typically need ~20-30ep before
validation accuracy stabilises. Result: evolutionary scoring was driven by
noise rather than topology quality, making parent selection unreliable.

This version uses 30ep per child so each network genuinely reflects its
topology before scoring. Population is trimmed to 8 (deeper per child instead
of wider) and generations reduced to 4 to hold total budget comparable.

Total budget:
  GA: 4 gen × 8 children × 30ep = 960 child-epochs
  Ref: 4 × 30 = 120ep (same total wall-time budget)

ALGORITHM
=========
Population: 8 conn_hh topologies (N×K_hh)

Each generation:
  1. Train each child from scratch for 30ep; record best val_top1.
  2. Score children by chosen scheme (--scoring flag):
       softmax  : weights = softmax(top1 / tau)
       rank     : linear 1/P, 2/P, ..., 1  (ascending rank order)
       top_k_avg: equal weight to top-k children only
  3. Select 2 parents by weighted sampling without replacement.
  4. Crossover: for each edge (u, k), inherit from parent A with p=0.5,
     else parent B.
  5. Mutate: per-edge swap with prob 0.05 (replace one neighbor with random).
  6. Next population: best child (elite) + 7 crossover-mutated offspring.

COMPARE VS:
  Ref: static small-world, same N=512 D=16 K_hh=2, trained 120ep straight.

CONFIGS
=======
  N=512, D=16, K_hh=2, K_iter=5, AH α=1.0, 50% data (Tier-1)
  --gens 4 --pop 8 --epochs 30 --scoring softmax --tau 0.5

EXPECTED
========
  If ConnGA_v2 >> Ref: settling time was the bottleneck in step709.
  If ConnGA_v2 ≈ Ref:  topology may not matter at this scale.
  Prior: step709 KILLED at 10ep; hypothesis = noise drove scoring.
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

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Step 740: ConnGA v2 — Stabilised Evolutionary Topology Search"
)
parser.add_argument("--device",  default="auto",
                    help="Device: auto / cpu / mps / cuda")
parser.add_argument("--epochs",  type=int, default=30,
                    help="Epochs per child per generation (default: 30)")
parser.add_argument("--gens",    type=int, default=4,
                    help="Number of GA generations (default: 4)")
parser.add_argument("--pop",     type=int, default=8,
                    help="Population size per generation (default: 8)")
parser.add_argument("--scoring", default="softmax",
                    choices=["softmax", "rank", "top_k_avg"],
                    help="Child scoring scheme (default: softmax)")
parser.add_argument("--tau",     type=float, default=0.5,
                    help="Temperature for softmax scoring (default: 0.5)")
parser.add_argument("--top_k",   type=int, default=3,
                    help="k for top_k_avg scoring (default: 3)")
parser.add_argument("--mut_prob", type=float, default=0.05,
                    help="Per-edge mutation probability (default: 0.05)")
parser.add_argument("--seed",    type=int, default=42)
args = parser.parse_args()

# ── Device ────────────────────────────────────────────────────────────────────

if args.device == "auto":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(args.device)

# ── Constants ─────────────────────────────────────────────────────────────────

EPOCHS   = args.epochs
N_GENS   = args.gens
POP_SIZE = args.pop
SCORING  = args.scoring
TAU      = args.tau
TOP_K    = args.top_k
MUT_PROB = args.mut_prob
SEED     = args.seed

BATCH  = 128
DATA   = "data/store.h5"

# Architecture (N=512, efficiency config scale-down)
N      = 512;  N_IN = 25088;  N_OUT = 10
D      = 16;   K_HH = 2;      K_IN  = 25;  K_ITER = 5
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
ALPHA_AHEBB   = 1.0

FLOPS     = 3 * N * K_HH * D * K_ITER   # 245,760 ≈ 0.25M
STEP_NAME = Path(__file__).stem
OUT_PATH  = ROOT / "results" / f"{STEP_NAME}.json"

# ── Topology utilities ────────────────────────────────────────────────────────

def random_conn_hh(n: int, k: int, seed: int) -> torch.Tensor:
    """[n, k] random connectivity — no self-loops, no duplicate neighbours."""
    rng = np.random.default_rng(seed)
    conn = np.zeros((n, k), dtype=np.int64)
    for h in range(n):
        pool = np.delete(np.arange(n), h)
        conn[h] = rng.choice(pool, size=k, replace=False)
    return torch.tensor(conn, dtype=torch.long)


def crossover(
    parent_a: torch.Tensor,
    parent_b: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Per-edge crossover: inherit from A with p=0.5, else B.

    Args:
        parent_a: [N, K] long tensor
        parent_b: [N, K] long tensor
        seed: RNG seed for reproducibility

    Returns:
        child conn_hh [N, K] with edges drawn from A or B
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(parent_a.shape) < 0.5   # True → take from A
    mask_t = torch.tensor(mask, dtype=torch.bool)
    child = torch.where(mask_t, parent_a, parent_b)
    return child


def mutate(
    conn: torch.Tensor,
    n: int,
    k: int,
    mut_prob: float,
    seed: int,
) -> torch.Tensor:
    """Random edge swap: replace one neighbour with a random new one.

    For each edge (u, j), with probability mut_prob, replace conn[u,j]
    with a random node ≠ u and not already in conn[u].
    """
    rng = np.random.default_rng(seed)
    conn_np = conn.numpy().copy()
    for u in range(n):
        for j in range(k):
            if rng.random() < mut_prob:
                existing = set(conn_np[u].tolist())
                existing.add(u)          # no self-loop
                candidates = [v for v in range(n) if v not in existing]
                if candidates:
                    new_v = rng.choice(candidates)
                    conn_np[u, j] = new_v
    return torch.tensor(conn_np, dtype=torch.long)


def score_children(accs: list[float], scoring: str, tau: float, top_k: int) -> np.ndarray:
    """Convert raw top-1 scores to selection weights.

    Args:
        accs:    list of validation top-1 accuracies (one per child)
        scoring: 'softmax' | 'rank' | 'top_k_avg'
        tau:     temperature (softmax only)
        top_k:   k (top_k_avg only)

    Returns:
        weights: np.ndarray of shape [P], sums to 1.0
    """
    accs_arr = np.array(accs, dtype=np.float64)
    P = len(accs_arr)

    if scoring == "softmax":
        shifted = accs_arr / tau
        shifted -= shifted.max()          # numerical stability
        exp_vals = np.exp(shifted)
        weights = exp_vals / exp_vals.sum()

    elif scoring == "rank":
        # Rank 1 = worst, Rank P = best; weights = rank / sum(ranks)
        order = np.argsort(accs_arr)      # ascending: worst first
        ranks = np.empty(P, dtype=np.float64)
        for pos, idx in enumerate(order):
            ranks[idx] = pos + 1          # 1..P
        weights = ranks / ranks.sum()

    elif scoring == "top_k_avg":
        k = min(top_k, P)
        top_indices = np.argsort(accs_arr)[-k:]
        weights = np.zeros(P, dtype=np.float64)
        weights[top_indices] = 1.0 / k

    else:
        raise ValueError(f"Unknown scoring: {scoring}")

    return weights

# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(conn_hh_override: torch.Tensor | None = None, seed: int = SEED) -> nn.Module:
    """Instantiate SGNNET_AH; optionally inject conn_hh topology."""
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4)
    K_l = K_HH - K_r
    ng  = max(8, N // 8)

    sw = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER,
        K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    res = SGNNET_Resonant(
        sw,
        K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
        beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    model = SGNNET_AntiHebbian(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos")

    if conn_hh_override is not None:
        # Overwrite the buffer in-place so downstream code sees the new topology
        model.m.base.conn_hh.copy_(conn_hh_override)

    return model


def train_child(
    conn_hh: torch.Tensor,
    tr,
    va,
    n_epochs: int,
    seed: int,
) -> tuple[float, list[float]]:
    """Train a single child from scratch. Returns (best_top1, history)."""
    model = build_model(conn_hh_override=conn_hh, seed=seed).to(DEVICE)
    kw = trainer_kwargs(N, n_epochs=n_epochs)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **kw)
    history = trainer.train(n_epochs=n_epochs, log_fn=None)
    top1h = [round(h.get("val_top1", 0.0), 4) for h in history]
    best  = max(top1h) if top1h else 0.0
    return best, top1h

# ── GA core ───────────────────────────────────────────────────────────────────

def run_connga(tr, va) -> dict:
    """Run ConnGA v2 for N_GENS generations.

    Returns result dict with per-generation stats and final best topology eval.
    """
    print(f"\n{'='*70}")
    print(f"ConnGA v2  |  N={N} D={D} K_hh={K_HH} K_iter={K_ITER}")
    print(f"  {N_GENS} gens × {POP_SIZE} children × {EPOCHS} ep  |  scoring={SCORING} τ={TAU}")
    print(f"  mutation p={MUT_PROB}  |  total child-epochs={N_GENS*POP_SIZE*EPOCHS}")
    print(f"{'='*70}")

    # Generation 0: random initial population
    population = [
        random_conn_hh(N, K_HH, seed=SEED + 100 + i)
        for i in range(POP_SIZE)
    ]

    gen_records = []
    best_global_acc  = -1.0
    best_global_conn = population[0].clone()

    for gen in range(N_GENS):
        print(f"\n  [Gen {gen+1}/{N_GENS}]")
        accs = []
        histories = []

        for ci, conn_hh in enumerate(population):
            child_seed = SEED + gen * 10_000 + ci
            best_acc, hist = train_child(conn_hh, tr, va, EPOCHS, seed=child_seed)
            accs.append(best_acc)
            histories.append(hist)
            print(f"    child {ci+1:2d}/{POP_SIZE}: best_top1={best_acc:.4f}", flush=True)

            # Track global best topology
            if best_acc > best_global_acc:
                best_global_acc  = best_acc
                best_global_conn = conn_hh.clone()

        best_g  = float(np.max(accs))
        mean_g  = float(np.mean(accs))
        worst_g = float(np.min(accs))
        best_ci = int(np.argmax(accs))
        print(f"  Gen {gen+1} summary: best={best_g:.4f}  mean={mean_g:.4f}  worst={worst_g:.4f}")

        gen_records.append({
            "gen":  gen + 1,
            "best": best_g,
            "mean": mean_g,
            "worst": worst_g,
            "accs": [round(a, 4) for a in accs],
        })

        # ── Build next population ─────────────────────────────────────────────
        if gen < N_GENS - 1:
            weights = score_children(accs, scoring=SCORING, tau=TAU, top_k=TOP_K)

            # Elite: carry forward the best child unchanged
            elite_conn = population[best_ci].clone()
            next_pop   = [elite_conn]

            rng_sel = np.random.default_rng(SEED + gen * 77777)
            for oi in range(POP_SIZE - 1):
                # Sample 2 parents weighted by scoring (without replacement)
                parents_idx = rng_sel.choice(
                    POP_SIZE, size=2, replace=False, p=weights
                )
                pa = population[parents_idx[0]]
                pb = population[parents_idx[1]]

                cross_seed = SEED + gen * 50_000 + oi
                child = crossover(pa, pb, seed=cross_seed)
                child = mutate(child, N, K_HH, MUT_PROB,
                               seed=cross_seed + 1_000_000)
                next_pop.append(child)

            population = next_pop

    # ── Final evaluation on best discovered topology ──────────────────────────
    print(f"\n  Final eval: best topology from all gens, {EPOCHS}ep")
    final_seed = SEED + 999_999
    final_acc, final_hist = train_child(
        best_global_conn, tr, va, EPOCHS, seed=final_seed
    )
    print(f"  Final best_top1={final_acc:.4f}  (vs GA best-seen={best_global_acc:.4f})")

    return {
        "algorithm": "ConnGA_v2",
        "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER, "flops": FLOPS,
        "n_gens": N_GENS, "pop_size": POP_SIZE, "epochs_per_child": EPOCHS,
        "scoring": SCORING, "tau": TAU, "top_k": TOP_K, "mut_prob": MUT_PROB,
        "gen_records": gen_records,
        "best_seen_acc": round(best_global_acc, 4),
        "final_acc": round(final_acc, 4),
        "final_history": final_hist,
        "best_conn_hh": best_global_conn.tolist(),
    }


def run_ref(tr, va) -> dict:
    """Train static small-world Ref for 4×30=120ep (same total budget)."""
    ref_epochs = N_GENS * EPOCHS
    print(f"\n{'='*70}")
    print(f"Ref — static small-world  N={N} D={D} K_hh={K_HH} K_iter={K_ITER}")
    print(f"  {ref_epochs}ep  (= {N_GENS}×{EPOCHS} = same total compute as GA)")
    print(f"{'='*70}")

    model = build_model(conn_hh_override=None, seed=SEED).to(DEVICE)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_p:,}")

    kw      = trainer_kwargs(N, n_epochs=ref_epochs)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **kw)

    def _log(m):
        ep = m["epoch"] + 1
        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)

    history = trainer.train(n_epochs=ref_epochs, log_fn=_log)
    top1h   = [round(h.get("val_top1", 0.0), 4) for h in history]
    best    = max(top1h) if top1h else 0.0
    bep     = int(np.argmax(top1h)) + 1
    print(f"  Ref best={best:.4f} @ ep{bep}")

    return {
        "algorithm": "Ref_static_smallworld",
        "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER, "flops": FLOPS,
        "n_epochs": ref_epochs,
        "best_acc": round(best, 4),
        "best_epoch": bep,
        "history": top1h,
        "n_params": n_p,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print(f"{STEP_NAME}")
    print(f"ConnGA v2 — stabilised evolutionary topology search")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER}  Device={DEVICE}")
    print(f"FLOPs per forward={FLOPS:,} (~{FLOPS/1e6:.3f}M)")
    print(f"Budget: {N_GENS}gen × {POP_SIZE}children × {EPOCHS}ep = "
          f"{N_GENS*POP_SIZE*EPOCHS} child-epochs")
    print(f"Scoring={SCORING} τ={TAU}  mutation_p={MUT_PROB}")
    print(f"{'='*70}")

    # 50% data subset — Tier-1 convention
    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(
        n_total, generator=torch.Generator().manual_seed(SEED)
    )[:n_total // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(
        subset, batch_size=BATCH, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"  Train subset={len(subset)}  Val={len(va.dataset)}")

    t0 = time.time()

    ref_result  = run_ref(tr, va)
    ref_elapsed = time.time() - t0
    ref_result["elapsed_s"] = round(ref_elapsed, 1)

    t1 = time.time()
    ga_result  = run_connga(tr, va)
    ga_elapsed = time.time() - t1
    ga_result["elapsed_s"] = round(ga_elapsed, 1)

    results = {
        "Ref":      ref_result,
        "ConnGA_v2": ga_result,
        "meta": {
            "step": STEP_NAME,
            "device": str(DEVICE),
            "seed": SEED,
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "epochs_per_child": EPOCHS,
            "n_gens": N_GENS,
            "pop_size": POP_SIZE,
            "scoring": SCORING,
            "tau": TAU,
            "top_k": TOP_K,
            "mut_prob": MUT_PROB,
            "total_elapsed_s": round(time.time() - t0, 1),
        },
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    # ── Summary ───────────────────────────────────────────────────────────────
    ref_best = ref_result["best_acc"]
    ga_best  = ga_result["final_acc"]
    delta    = ga_best - ref_best
    if delta > 0.010:
        verdict = "TOPOLOGY LEARNING WORKS — step709 was settling-time-limited"
    elif delta > -0.005:
        verdict = "NEUTRAL — topology at N=512 may not matter"
    else:
        verdict = "TOPOLOGY GA HURTS — investigate"

    print(f"\n{'='*70}")
    print(f"{STEP_NAME} SUMMARY")
    print(f"{'='*70}")
    print(f"  Ref (static):    {ref_best:.4f}")
    print(f"  ConnGA v2 final: {ga_best:.4f}  Δ={delta:+.4f}")
    print(f"  GA best-seen:    {ga_result['best_seen_acc']:.4f}")
    print(f"\n  Verdict: {verdict}")
    print(f"\n→ {OUT_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
