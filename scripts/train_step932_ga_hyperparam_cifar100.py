"""Step 932: Genetic Algorithm hyperparameter search — CIFAR-100.

MOTIVATION
==========
After step930 (K_hh×K_iter scout on CIFAR-100), GA jointly searches
(D, K_in, K_hh, K_iter, N) to find the best CIFAR-100 config.

step905 T1: N=4096=40.57% vs Linear=64.78% (−24.21pp).
The architectural gap may be tuning-addressable: CIFAR-100 is still vision
(VGG spatial features), unlike ESC-50 audio. The question is whether the
24pp gap reflects capacity (→ N), routing depth (→ K_iter), or connectivity (→ K_hh).

GENE SPACE
==========
  D:      [8, 16, 32]
  K_in:   [10, 15, 25]         # K_in crossover at N=4096: K_in=15 confirmed
  K_hh:   [1, 2, 4, 6]
  K_iter: [3, 5, 7, 10]
  N:      [2048, 4096, 8192]   # N-scaling confirmed positive for CIFAR-100

GA PROTOCOL
===========
  Population size:   8
  Generations:       4
  Selection:         top-4 survive (elitism)
  Crossover:         uniform (each gene from parent1 or parent2 uniformly)
  Mutation rate:     0.20 per gene
  Fitness:           T0 val accuracy (20ep, 50% data, seed=42)
  Seeding:           Gen0 includes canonical (D=16, K_in=15, K_hh=2, K_iter=5, N=4096)
                     and step930 best configs (seeded after step930 completes)

RESUMABLE: saves population state after each individual — safe to interrupt.

ADVANCE RULE
============
  Best GA config >44% → advance to T1 (>+3.4pp vs step905 N4096 T1)
  Best GA config >42% → T1 candidate (>+1.4pp)
  All GA configs <40% → CIFAR-100 gap is capacity-limited not tuning-limited
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path
from copy import deepcopy

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",   default="auto")
parser.add_argument("--epochs",   type=int, default=20)
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data",     default="data/store_cifar100.h5")
parser.add_argument("--pop_size", type=int, default=8)
parser.add_argument("--n_gen",    type=int, default=4)
parser.add_argument("--mut_rate", type=float, default=0.20)
parser.add_argument("--ga_seed",  type=int, default=0)
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS   = args.epochs
BATCH    = 128
SEED     = args.seed
N_IN     = 25088
N_OUT    = 100
ALPHA_REFLECT = 0.5

STEP905_LINEAR = 0.6478
STEP905_N4096  = 0.4057

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step932_ga_hyperparam_cifar100_seed{SEED}__{SLOT}.json"

GENE_SPACE = {
    "D":      [8, 16, 32],
    "K_in":   [10, 15, 25],
    "K_hh":   [1, 2, 4, 6],
    "K_iter": [3, 5, 7, 10],
    "N":      [2048, 4096, 8192],
}
GENE_KEYS = list(GENE_SPACE.keys())

SEED_INDIVIDUALS = [
    {"D": 16, "K_in": 15, "K_hh": 2, "K_iter": 5, "N": 4096},  # canonical
    {"D": 16, "K_in": 15, "K_hh": 2, "K_iter": 7, "N": 8192},  # deep + large N
    {"D": 32, "K_in": 15, "K_hh": 2, "K_iter": 5, "N": 4096},  # larger manifold
    {"D": 16, "K_in": 10, "K_hh": 4, "K_iter": 5, "N": 4096},  # denser K_hh
]


def _dw_proj(W_pos, conn_hh, n):
    W_h = W_pos[:n]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def make_model(ind: dict) -> nn.Module:
    D, K_in, K_hh, K_iter, N = ind["D"], ind["K_in"], ind["K_hh"], ind["K_iter"], ind["N"]
    torch.manual_seed(SEED)
    K_r = max(1, K_hh // 4); K_l = K_hh - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_in, K_iter=K_iter, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )

    class DeltaW(nn.Module):
        def __init__(self):
            super().__init__()
            self.m      = resonant
            self.k_iter = K_iter
            self.n      = N

        @property
        def W_pos(self): return self.m.W_pos
        def tick_epoch(self):
            if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

        def forward(self, x):
            Z         = self.m.base._seed(x)
            theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
            conn_hh   = self.m.base.conn_hh
            dw        = _dw_proj(self.m.W_pos, conn_hh, self.n)
            Z_ref     = torch.zeros_like(Z)
            for _ in range(self.k_iter):
                Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
                Z_nb  = Z_fwd[:, conn_hh, :]
                Z_agg = _dw_agg(Z_nb, dw)
                Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
                Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
            return self.m.base._readout(Z)

    return DeltaW()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        x, y = batch[0].to(DEVICE), batch[2].to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def eval_individual(ind, tr, va) -> tuple[float, int, float]:
    model = make_model(ind).to(DEVICE)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    t0    = time.time()
    hist  = []
    for ep in range(EPOCHS):
        model.train()
        for batch in tr:
            x, y = batch[0].to(DEVICE), batch[2].to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        val = evaluate(model, va)
        hist.append(val)
        if hasattr(model, "tick_epoch"): model.tick_epoch()
        print(f"    e{ep+1:3d}  top1={val:.4f}", flush=True)
    return max(hist), n_p, time.time() - t0


def random_individual(rng: random.Random) -> dict:
    return {k: rng.choice(GENE_SPACE[k]) for k in GENE_KEYS}


def crossover(p1: dict, p2: dict, rng: random.Random) -> dict:
    return {k: (p1[k] if rng.random() < 0.5 else p2[k]) for k in GENE_KEYS}


def mutate(ind: dict, rate: float, rng: random.Random) -> dict:
    child = deepcopy(ind)
    for k in GENE_KEYS:
        if rng.random() < rate:
            choices = [v for v in GENE_SPACE[k] if v != child[k]] or GENE_SPACE[k]
            child[k] = rng.choice(choices)
    return child


def breed_next_gen(survivors, pop_size, mut_rate, rng):
    next_gen = list(survivors)
    while len(next_gen) < pop_size:
        p1, p2 = rng.sample(survivors, 2)
        child  = mutate(crossover(p1, p2, rng), mut_rate, rng)
        next_gen.append(child)
    return next_gen


def ind_key(ind: dict) -> str:
    return f"D{ind['D']}_Kin{ind['K_in']}_Khh{ind['K_hh']}_Ki{ind['K_iter']}_N{ind['N']}"


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED, pin_memory=False)
    n_full  = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[: n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=10, pin_memory=False,
    )

    rng = random.Random(args.ga_seed)

    print(f"\n{'='*70}")
    print(f"step932 — GA Hyperparam Search (CIFAR-100)  pop={args.pop_size}  gen={args.n_gen}")
    print(f"  device={DEVICE}  T0_epochs={EPOCHS}  seed={SEED}  ga_seed={args.ga_seed}")
    print(f"  gene_space={GENE_SPACE}")
    print(f"  refs: Linear={STEP905_LINEAR:.4f}  N4096_T1={STEP905_N4096:.4f}")
    print(f"{'='*70}\n")

    state = {}
    if OUT_PATH.exists():
        state = json.loads(OUT_PATH.read_text())

    all_evals      = state.get("all_evals", {})
    linear_fitness = state.get("linear_fitness", None)

    if linear_fitness is None:
        print("Evaluating Linear baseline...")
        torch.manual_seed(SEED)
        model = nn.Linear(N_IN, N_OUT).to(DEVICE)
        opt   = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
        hist  = []
        for ep in range(EPOCHS):
            model.train()
            for batch in tr:
                x, y = batch[0].to(DEVICE), batch[2].to(DEVICE)
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            hist.append(evaluate(model, va))
        linear_fitness = max(hist)
        print(f"  Linear baseline: {linear_fitness:.4f}")
        state["linear_fitness"] = linear_fitness
        state["all_evals"]      = all_evals
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(state, indent=2))

    gen_populations = state.get("gen_populations", {})

    def build_gen0():
        pop = list(SEED_INDIVIDUALS[:args.pop_size])
        while len(pop) < args.pop_size:
            pop.append(random_individual(rng))
        return pop

    if "0" not in gen_populations:
        gen_populations["0"] = [{"genes": ind} for ind in build_gen0()]
        state["gen_populations"] = gen_populations
        OUT_PATH.write_text(json.dumps(state, indent=2))

    for gen_idx in range(args.n_gen):
        gen_key = str(gen_idx)
        if gen_key not in gen_populations:
            break
        pop_entries = gen_populations[gen_key]

        print(f"\n{'━'*70}")
        print(f"Generation {gen_idx}  ({len(pop_entries)} individuals)")
        print(f"{'━'*70}")

        for idx, entry in enumerate(pop_entries):
            ind = entry["genes"]
            key = ind_key(ind)
            if key in all_evals:
                print(f"  [{idx}] {key}  fitness={all_evals[key]['fitness']:.4f}  (cached)")
                continue

            print(f"\n  [{idx}] {key}")
            print(f"    D={ind['D']}  K_in={ind['K_in']}  K_hh={ind['K_hh']}  K_iter={ind['K_iter']}  N={ind['N']}")
            fitness, n_p, elapsed = eval_individual(ind, tr, va)
            delta = fitness - linear_fitness
            print(f"    → fitness={fitness:.4f}  Δ={delta*100:+.2f}pp  params={n_p:,}  {elapsed:.0f}s")

            all_evals[key] = {
                "fitness": round(fitness, 4), "delta_vs_linear": round(delta, 4),
                "n_params": n_p, "elapsed_s": round(elapsed), "genes": ind, "gen": gen_idx,
            }
            state["all_evals"] = all_evals
            OUT_PATH.write_text(json.dumps(state, indent=2))

        gen_fitnesses = []
        for entry in pop_entries:
            key = ind_key(entry["genes"])
            if key in all_evals:
                gen_fitnesses.append((all_evals[key]["fitness"], entry["genes"]))
        gen_fitnesses.sort(key=lambda x: x[0], reverse=True)
        n_survive = max(2, args.pop_size // 2)
        survivors = [g for _, g in gen_fitnesses[:n_survive]]

        print(f"\n  Gen {gen_idx} ranking:")
        for rank, (fit, g) in enumerate(gen_fitnesses):
            marker = "★" if rank == 0 else " "
            print(f"  {marker} [{rank+1}] {ind_key(g)}  fitness={fit:.4f}  Δ={((fit-linear_fitness)*100):+.2f}pp")

        next_gen_key = str(gen_idx + 1)
        if gen_idx + 1 < args.n_gen and next_gen_key not in gen_populations:
            next_gen = breed_next_gen(survivors, args.pop_size, args.mut_rate, rng)
            gen_populations[next_gen_key] = [{"genes": ind} for ind in next_gen]
            state["gen_populations"] = gen_populations
            OUT_PATH.write_text(json.dumps(state, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 932 SUMMARY — GA Hyperparam Search (CIFAR-100)")
    print(f"{'='*70}")
    print(f"  Linear baseline: {linear_fitness:.4f}")
    ranked = sorted(all_evals.items(), key=lambda x: x[1]["fitness"], reverse=True)
    print(f"\n  Top-5 configs:")
    for i, (key, r) in enumerate(ranked[:5]):
        d = r["delta_vs_linear"]
        verdict = ("T1 candidate" if r["fitness"] > 0.42 else
                   "below canonical" if r["fitness"] < STEP905_N4096 else "near canonical")
        print(f"  [{i+1}] {key}  fitness={r['fitness']:.4f}  Δ={d*100:+.2f}pp  gen={r['gen']}  {verdict}")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
