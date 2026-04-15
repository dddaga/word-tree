"""Step 741: ConnGA v2 with rank scoring.

Reuses step740's algorithm entirely. Only change: --scoring rank (linear rank
weights 1/P..P/P) instead of step740's default softmax.

Motivation: softmax with tau=0.5 can be dominated by a single outlier child.
Rank scoring spreads selection pressure more evenly, potentially stabilising
GA convergence especially in early generations where noise is high.

Config (identical to step740 except scoring):
  N=512, D=16, K_hh=2, K_iter=5, AH α=1.0
  4 gens × 8 children × 30ep = 960 child-epochs, 50% data (Tier-1)
  scoring=rank  (hardcoded default)

Compare to:
  step740: scoring=softmax (baseline)
  step742: scoring=top_k_avg k=2
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Step 741: ConnGA v2 with rank scoring (hardcoded default)"
)
parser.add_argument("--device",   default="auto")
parser.add_argument("--epochs",   type=int,   default=30)
parser.add_argument("--gens",     type=int,   default=4)
parser.add_argument("--pop",      type=int,   default=8)
parser.add_argument("--scoring",  default="rank",
                    choices=["softmax", "rank", "top_k_avg"],
                    help="Scoring scheme — default hardcoded to 'rank'")
parser.add_argument("--tau",      type=float, default=0.5)
parser.add_argument("--top_k",    type=int,   default=3)
parser.add_argument("--mut_prob", type=float, default=0.05)
parser.add_argument("--seed",     type=int,   default=42)
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
SCORING  = args.scoring   # "rank" by default
TAU      = args.tau
TOP_K    = args.top_k
MUT_PROB = args.mut_prob
SEED     = args.seed

BATCH = 128; DATA = "data/store.h5"
N = 512; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
FLOPS     = 3 * N * K_HH * D * K_ITER
STEP_NAME = Path(__file__).stem
OUT_PATH  = ROOT / "results" / f"{STEP_NAME}.json"

# ── Topology utilities ────────────────────────────────────────────────────────

def random_conn_hh(n, k, seed):
    rng = np.random.default_rng(seed)
    conn = np.zeros((n, k), dtype=np.int64)
    for h in range(n):
        pool = np.delete(np.arange(n), h)
        conn[h] = rng.choice(pool, size=k, replace=False)
    return torch.tensor(conn, dtype=torch.long)


def crossover(parent_a, parent_b, seed):
    rng = np.random.default_rng(seed)
    mask = rng.random(parent_a.shape) < 0.5
    return torch.where(torch.tensor(mask, dtype=torch.bool), parent_a, parent_b)


def mutate(conn, n, k, mut_prob, seed):
    rng = np.random.default_rng(seed)
    conn_np = conn.numpy().copy()
    for u in range(n):
        for j in range(k):
            if rng.random() < mut_prob:
                existing = set(conn_np[u].tolist()); existing.add(u)
                candidates = [v for v in range(n) if v not in existing]
                if candidates:
                    conn_np[u, j] = rng.choice(candidates)
    return torch.tensor(conn_np, dtype=torch.long)


def score_children(accs, scoring, tau, top_k):
    accs_arr = np.array(accs, dtype=np.float64); P = len(accs_arr)
    if scoring == "softmax":
        shifted = accs_arr / tau - (accs_arr / tau).max()
        exp_vals = np.exp(shifted)
        return exp_vals / exp_vals.sum()
    elif scoring == "rank":
        order = np.argsort(accs_arr)
        ranks = np.empty(P, dtype=np.float64)
        for pos, idx in enumerate(order):
            ranks[idx] = pos + 1
        return ranks / ranks.sum()
    elif scoring == "top_k_avg":
        k = min(top_k, P)
        w = np.zeros(P, dtype=np.float64)
        w[np.argsort(accs_arr)[-k:]] = 1.0 / k
        return w
    else:
        raise ValueError(f"Unknown scoring: {scoring}")

# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(conn_hh_override=None, seed=SEED):
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    sw = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                           K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                           n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(sw, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                          alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                          mode="dynamic_z_geo", resonance_threshold=0.0)
    model = SGNNET_AntiHebbian(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    if conn_hh_override is not None:
        model.m.base.conn_hh.copy_(conn_hh_override)
    return model


def train_child(conn_hh, tr, va, n_epochs, seed):
    model = build_model(conn_hh_override=conn_hh, seed=seed).to(DEVICE)
    history = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **trainer_kwargs(N, n_epochs=n_epochs)).train(
        n_epochs=n_epochs, log_fn=None)
    top1h = [round(h.get("val_top1", 0.0), 4) for h in history]
    return (max(top1h) if top1h else 0.0), top1h

# ── GA core ───────────────────────────────────────────────────────────────────

def run_connga(tr, va):
    print(f"\n{'='*70}")
    print(f"ConnGA v2 (rank scoring) | N={N} D={D} K_hh={K_HH} K_iter={K_ITER}")
    print(f"  {N_GENS} gens × {POP_SIZE} children × {EPOCHS} ep  |  scoring={SCORING}")
    print(f"{'='*70}")
    population = [random_conn_hh(N, K_HH, seed=SEED + 100 + i) for i in range(POP_SIZE)]
    gen_records = []; best_global_acc = -1.0; best_global_conn = population[0].clone()

    for gen in range(N_GENS):
        print(f"\n  [Gen {gen+1}/{N_GENS}]")
        accs = []
        for ci, conn_hh in enumerate(population):
            child_seed = SEED + gen * 10_000 + ci
            best_acc, _ = train_child(conn_hh, tr, va, EPOCHS, seed=child_seed)
            accs.append(best_acc)
            print(f"    child {ci+1:2d}/{POP_SIZE}: best_top1={best_acc:.4f}", flush=True)
            if best_acc > best_global_acc:
                best_global_acc = best_acc; best_global_conn = conn_hh.clone()

        best_g = float(np.max(accs)); mean_g = float(np.mean(accs)); worst_g = float(np.min(accs))
        best_ci = int(np.argmax(accs))
        print(f"  Gen {gen+1} summary: best={best_g:.4f}  mean={mean_g:.4f}  worst={worst_g:.4f}")
        gen_records.append({"gen": gen+1, "best": best_g, "mean": mean_g, "worst": worst_g,
                            "accs": [round(a, 4) for a in accs]})

        if gen < N_GENS - 1:
            weights = score_children(accs, scoring=SCORING, tau=TAU, top_k=TOP_K)
            elite_conn = population[best_ci].clone(); next_pop = [elite_conn]
            rng_sel = np.random.default_rng(SEED + gen * 77777)
            for oi in range(POP_SIZE - 1):
                parents_idx = rng_sel.choice(POP_SIZE, size=2, replace=False, p=weights)
                child = crossover(population[parents_idx[0]], population[parents_idx[1]],
                                  seed=SEED + gen * 50_000 + oi)
                child = mutate(child, N, K_HH, MUT_PROB, seed=SEED + gen * 50_000 + oi + 1_000_000)
                next_pop.append(child)
            population = next_pop

    print(f"\n  Final eval: best topology, {EPOCHS}ep")
    final_acc, final_hist = train_child(best_global_conn, tr, va, EPOCHS, seed=SEED + 999_999)
    print(f"  Final best_top1={final_acc:.4f}  (GA best-seen={best_global_acc:.4f})")
    return {"algorithm": "ConnGA_v2_rank", "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "flops": FLOPS, "n_gens": N_GENS, "pop_size": POP_SIZE, "epochs_per_child": EPOCHS,
            "scoring": SCORING, "mut_prob": MUT_PROB, "gen_records": gen_records,
            "best_seen_acc": round(best_global_acc, 4), "final_acc": round(final_acc, 4),
            "final_history": final_hist, "best_conn_hh": best_global_conn.tolist()}


def run_ref(tr, va):
    ref_epochs = N_GENS * EPOCHS
    print(f"\n{'='*70}")
    print(f"Ref — static small-world  N={N} D={D}  {ref_epochs}ep")
    print(f"{'='*70}")
    model = build_model(conn_hh_override=None, seed=SEED).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_p:,}")
    history = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **trainer_kwargs(N, n_epochs=ref_epochs)).train(
        n_epochs=ref_epochs,
        log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
                         if (m['epoch'] + 1) % 10 == 0 else None)
    top1h = [round(h.get("val_top1", 0.0), 4) for h in history]
    best = max(top1h) if top1h else 0.0; bep = int(np.argmax(top1h)) + 1
    print(f"  Ref best={best:.4f} @ ep{bep}")
    return {"algorithm": "Ref_static_smallworld", "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "flops": FLOPS, "n_epochs": ref_epochs, "best_acc": round(best, 4),
            "best_epoch": bep, "history": top1h, "n_params": n_p}


def main():
    print(f"\n{'='*70}")
    print(f"{STEP_NAME}")
    print(f"ConnGA v2 — rank scoring (step741)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER}  Device={DEVICE}")
    print(f"Scoring: {SCORING}  (step740 used softmax; step742 uses top_k_avg)")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))[:n_total // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(SEED))
    print(f"  Train subset={n_total//2}  Val={len(va.dataset)}")

    t0 = time.time()
    ref_result = run_ref(tr, va); ref_result["elapsed_s"] = round(time.time() - t0, 1)
    t1 = time.time()
    ga_result = run_connga(tr, va); ga_result["elapsed_s"] = round(time.time() - t1, 1)

    results = {"Ref": ref_result, "ConnGA_v2_rank": ga_result,
               "meta": {"step": STEP_NAME, "device": str(DEVICE), "seed": SEED,
                        "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
                        "epochs_per_child": EPOCHS, "n_gens": N_GENS, "pop_size": POP_SIZE,
                        "scoring": SCORING, "tau": TAU, "top_k": TOP_K, "mut_prob": MUT_PROB,
                        "total_elapsed_s": round(time.time() - t0, 1)}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = ref_result["best_acc"]; ga_best = ga_result["final_acc"]; delta = ga_best - ref_best
    verdict = ("RANK SCORING BETTER" if delta > 0.010 else
               "NEUTRAL vs softmax" if delta > -0.005 else "RANK SCORING HURTS")
    print(f"\n{'='*70}")
    print(f"{STEP_NAME} SUMMARY")
    print(f"  Ref (static):          {ref_best:.4f}")
    print(f"  ConnGA v2 rank final:  {ga_best:.4f}  Δ={delta:+.4f}")
    print(f"  GA best-seen:          {ga_result['best_seen_acc']:.4f}")
    print(f"\n  Verdict: {verdict}")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
