"""Step 931: Genetic Algorithm hyperparameter search — ESC-50.

MOTIVATION
==========
After step929 (K_hh×K_iter scout) reveals which parameters matter for audio,
the GA jointly searches (D, K_in, K_hh, K_iter, N) for the best ESC-50 config.

This is an exploratory search to answer:
  - Is there ANY parameter combination that closes the audio gap to ≤5pp?
  - If yes: is it a tunable gap (architecture wins) or structural (modality wins)?
  - Which parameters interact beneficially for audio features?

GENE SPACE
==========
  D:      [4, 8, 16]
  K_in:   [15, 25, 50]
  K_hh:   [1, 2, 4, 6]
  K_iter: [3, 5, 7, 10]
  N:      [512, 1024, 2048, 4096]

GA PROTOCOL
===========
  Population size:   8
  Generations:       4
  Selection:         top-4 survive (elitism)
  Crossover:         uniform (each gene from parent1 or parent2 uniformly)
  Mutation rate:     0.20 per gene
  Fitness:           T0 val accuracy (20ep, 50% data, seed=42)
  Seeding:           Gen0 includes canonical (D=8, K_in=25, K_hh=2, K_iter=5, N=2048)
                     and step928/929 best configs; rest random

RESUMABLE: saves population state after each individual — safe to interrupt.

ADVANCE RULE
============
  Best GA config within ±5pp of Linear → advance to T1 (step932_esc50_t1)
  Best GA config within ±8pp → marginal, architecture hypothesis survives
  All GA configs >10pp gap → STRUCTURAL audio gap CONFIRMED
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path
from copy import deepcopy

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant   import SGNNET_Resonant

parser = argparse.ArgumentParser()
parser.add_argument("--device",       default="auto")
parser.add_argument("--epochs",       type=int, default=20)
parser.add_argument("--seed",         type=int, default=42)
parser.add_argument("--data",         default="data/esc50/store_esc50_whisper.h5")
parser.add_argument("--pop_size",     type=int, default=8)
parser.add_argument("--n_gen",        type=int, default=4)
parser.add_argument("--mut_rate",     type=float, default=0.20)
parser.add_argument("--ga_seed",      type=int, default=0)
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS   = args.epochs
BATCH    = 64
SEED     = args.seed
N_IN     = 384
N_OUT    = 50
ALPHA_REFLECT = 0.5

# References
STEP928_LINEAR = 0.4775
STEP928_BEST   = 0.3625  # D=8, K_in=25 (step928 A_D8)

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step931_ga_hyperparam_esc50_seed{SEED}__{SLOT}.json"

# Gene domains
GENE_SPACE = {
    "D":      [4, 8, 16],
    "K_in":   [15, 25, 50],
    "K_hh":   [1, 2, 4, 6],
    "K_iter": [3, 5, 7, 10],
    "N":      [512, 1024, 2048, 4096],
}
GENE_KEYS = list(GENE_SPACE.keys())

# Seed population: canonical + step928/929 bests + random
SEED_INDIVIDUALS = [
    {"D": 8,  "K_in": 25, "K_hh": 2, "K_iter": 5, "N": 2048},  # step928 best
    {"D": 16, "K_in": 25, "K_hh": 2, "K_iter": 5, "N": 2048},  # canonical
    {"D": 4,  "K_in": 25, "K_hh": 1, "K_iter": 5, "N": 1024},  # minimal
    {"D": 8,  "K_in": 15, "K_hh": 2, "K_iter": 7, "N": 4096},  # large N, deep
]


class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(self, feats, labels):
        self.feats  = torch.tensor(feats,  dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.feats[idx], self.labels[idx]


def load_data(h5_path: Path, seed: int):
    with h5py.File(h5_path, "r") as f:
        tr_x = f["train_features"][:]
        tr_y = f["train_labels"][:]
        va_x = f["val_features"][:]
        va_y = f["val_labels"][:]
    tr_full = ESC50Dataset(tr_x, tr_y)
    va_ds   = ESC50Dataset(va_x, va_y)
    n_full  = len(tr_full)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(seed))[: n_full // 2]
    tr_sub  = torch.utils.data.Subset(tr_full, sub_idx.tolist())
    tr = torch.utils.data.DataLoader(tr_sub, batch_size=BATCH, shuffle=True,  num_workers=10, pin_memory=False)
    va = torch.utils.data.DataLoader(va_ds,  batch_size=BATCH, shuffle=False, num_workers=10, pin_memory=False)
    return tr, va


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
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def eval_individual(ind: dict, tr, va) -> tuple[float, int, float]:
    model = make_model(ind).to(DEVICE)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    t0    = time.time()
    hist  = []
    for ep in range(EPOCHS):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
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


# ── GA operators ─────────────────────────────────────────────────────────────

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


def breed_next_gen(survivors: list[dict], pop_size: int, mut_rate: float, rng: random.Random) -> list[dict]:
    next_gen = list(survivors)
    while len(next_gen) < pop_size:
        p1, p2 = rng.sample(survivors, 2)
        child  = crossover(p1, p2, rng)
        child  = mutate(child, mut_rate, rng)
        next_gen.append(child)
    return next_gen


def ind_key(ind: dict) -> str:
    return f"D{ind['D']}_Kin{ind['K_in']}_Khh{ind['K_hh']}_Ki{ind['K_iter']}_N{ind['N']}"


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr, va = load_data(data_path, SEED)

    rng = random.Random(args.ga_seed)
    torch.manual_seed(SEED)

    print(f"\n{'='*70}")
    print(f"step931 — GA Hyperparam Search (ESC-50)  pop={args.pop_size}  gen={args.n_gen}")
    print(f"  device={DEVICE}  T0_epochs={EPOCHS}  seed={SEED}  ga_seed={args.ga_seed}")
    print(f"  mut_rate={args.mut_rate}  gene_space={GENE_SPACE}")
    print(f"  refs: Linear={STEP928_LINEAR:.4f}  best_so_far={STEP928_BEST:.4f}")
    print(f"{'='*70}\n")

    # Load state if resuming
    state = {}
    if OUT_PATH.exists():
        state = json.loads(OUT_PATH.read_text())

    all_evals = state.get("all_evals", {})  # key → {fitness, n_params, elapsed_s, genes, gen}
    linear_fitness = state.get("linear_fitness", None)

    # Evaluate linear baseline first
    if linear_fitness is None:
        print("Evaluating Linear baseline...")
        model = nn.Linear(N_IN, N_OUT).to(DEVICE)
        torch.manual_seed(SEED)
        model = nn.Linear(N_IN, N_OUT).to(DEVICE)
        opt   = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
        hist  = []
        for ep in range(EPOCHS):
            model.train()
            for x, y in tr:
                x, y = x.to(DEVICE), y.to(DEVICE)
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

    # Build generation 0: seed + random fill
    def build_gen0():
        pop = list(SEED_INDIVIDUALS[:args.pop_size])
        while len(pop) < args.pop_size:
            cand = random_individual(rng)
            pop.append(cand)
        return pop

    gen_populations = state.get("gen_populations", {})

    if "0" not in gen_populations:
        gen_populations["0"] = [{"genes": ind} for ind in build_gen0()]
        state["gen_populations"] = gen_populations
        OUT_PATH.write_text(json.dumps(state, indent=2))

    best_overall = (STEP928_BEST, None)

    for gen_idx in range(args.n_gen):
        gen_key = str(gen_idx)
        if gen_key not in gen_populations:
            break
        pop_entries = gen_populations[gen_key]

        print(f"\n{'━'*70}")
        print(f"Generation {gen_idx}  ({len(pop_entries)} individuals)")
        print(f"{'━'*70}")

        for idx, entry in enumerate(pop_entries):
            ind    = entry["genes"]
            key    = ind_key(ind)
            if key in all_evals:
                fitness = all_evals[key]["fitness"]
                print(f"  [{idx}] {key}  fitness={fitness:.4f}  (cached)")
                continue

            print(f"\n  [{idx}] {key}")
            print(f"    genes: D={ind['D']}  K_in={ind['K_in']}  K_hh={ind['K_hh']}  K_iter={ind['K_iter']}  N={ind['N']}")
            t0 = time.time()
            fitness, n_p, elapsed = eval_individual(ind, tr, va)
            delta = fitness - linear_fitness
            print(f"    → fitness={fitness:.4f}  Δ={delta*100:+.2f}pp  params={n_p:,}  {elapsed:.0f}s")

            all_evals[key] = {
                "fitness": round(fitness, 4),
                "delta_vs_linear": round(delta, 4),
                "n_params": n_p,
                "elapsed_s": round(elapsed),
                "genes": ind,
                "gen": gen_idx,
            }
            state["all_evals"] = all_evals
            OUT_PATH.write_text(json.dumps(state, indent=2))

            if fitness > best_overall[0]:
                best_overall = (fitness, ind)

        # Score and select survivors for next generation
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

        # Breed next generation if needed
        next_gen_key = str(gen_idx + 1)
        if gen_idx + 1 < args.n_gen and next_gen_key not in gen_populations:
            next_gen = breed_next_gen(survivors, args.pop_size, args.mut_rate, rng)
            gen_populations[next_gen_key] = [{"genes": ind} for ind in next_gen]
            state["gen_populations"] = gen_populations
            OUT_PATH.write_text(json.dumps(state, indent=2))
            print(f"\n  Bred generation {gen_idx+1}: {len(next_gen)} individuals")

    # Final summary
    print(f"\n{'='*70}")
    print(f"STEP 931 SUMMARY — GA Hyperparam Search (ESC-50)")
    print(f"{'='*70}")
    print(f"  Linear baseline: {linear_fitness:.4f}")
    print(f"  Total individuals evaluated: {len(all_evals)}")

    ranked = sorted(all_evals.items(), key=lambda x: x[1]["fitness"], reverse=True)
    print(f"\n  Top-5 configs:")
    for i, (key, r) in enumerate(ranked[:5]):
        g = r["genes"]
        d = r["delta_vs_linear"]
        if d >= -0.05:
            verdict = "COMPETITIVE → T1"
        elif d >= -0.08:
            verdict = "marginal"
        else:
            verdict = "audio gap persists"
        print(f"  [{i+1}] {key}  fitness={r['fitness']:.4f}  Δ={d*100:+.2f}pp  gen={r['gen']}  {verdict}")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
