"""GA AutoRun v2 — Genetic algorithm search over SGNNET configuration space.

Population: 50 per generation
  - 20 elite   : best from previous generation (carried forward unchanged)
  - 20 crossbreed: uniform crossover of elite pairs
  - 10 mutations : random gene perturbation of elite members

Each individual trains for 20 epochs on 50% data (Tier-0 scout).
Fitness = accuracy * efficiency_multiplier (FLOPs-based bonus/penalty).

Usage:
    python scripts/ga_autorun_v2.py --device mps --generations 10 --epochs-per-eval 20
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset import make_loaders

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="GA AutoRun v2 -- SGNNET config search")
parser.add_argument("--device", default="auto", help="mps | cpu | cuda | auto")
parser.add_argument("--generations", type=int, default=10)
parser.add_argument("--epochs-per-eval", type=int, default=20,
                    help="Training epochs per individual (Tier-0 scout)")
parser.add_argument("--data", default="data/store.h5")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume-gen", type=int, default=None,
                    help="Resume from this generation number (loads JSON from results/)")
args = parser.parse_args()

if args.device == "auto":
    DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
else:
    DEVICE = torch.device(args.device)

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = ROOT / args.data
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

JOURNAL_PATH = RESULTS_DIR / "ga_autorun_v2_journal.tsv"

N_IN = 25088
N_OUT = 10
K_IN = 25   # input fan-in per hidden neuron (constant)

# ---------------------------------------------------------------------------
# Gene space
# ---------------------------------------------------------------------------

GENE_SPACE: dict[str, list] = {
    "N":              [1024, 2048, 4096],
    "D":              [8, 12, 16, 20, 24, 32],
    "K_hh":          [2, 3, 4],
    "K_iter":        [3, 4, 5, 6, 8],
    "alpha_ahebb":   [0.0, 0.5, 1.0],
    "alpha_reflect": [0.0, 0.3, 0.5],
    "routing_mode":  ["standard", "delta_proj", "wpos_proj"],
    "polarizer_alpha": [0.0, 0.5, 1.0, 1.5, 2.0],
}

GENE_KEYS = list(GENE_SPACE.keys())

# ---------------------------------------------------------------------------
# Seed population -- top-20 known configs as initial elite
# ---------------------------------------------------------------------------

SEED_ELITE: list[dict[str, Any]] = [
    # 1  step234-A: N=2048 D=16 K_hh=2 K_iter=5 AH=0.0 delta_proj a=1.5 -> 95.44%
    dict(N=2048, D=16, K_hh=2, K_iter=5, alpha_ahebb=0.0,  alpha_reflect=0.5, routing_mode="delta_proj",  polarizer_alpha=1.5),
    # 2  step217b-B: N=2048 D=16 K_hh=2 K_iter=5 AH=1.0 wpos_proj a=1.5 -> 95.92%
    dict(N=2048, D=16, K_hh=2, K_iter=5, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="wpos_proj",   polarizer_alpha=1.5),
    # 3  step199: N=2048 D=16 K_hh=2 K_iter=5 AH=1.0 standard -> 95.52%
    dict(N=2048, D=16, K_hh=2, K_iter=5, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="standard",    polarizer_alpha=0.0),
    # 4  step195: N=2048 D=16 K_hh=2 K_iter=6 AH=1.0 standard -> 96.08%
    dict(N=2048, D=16, K_hh=2, K_iter=6, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="standard",    polarizer_alpha=0.0),
    # 5  step205: N=4096 D=16 K_hh=2 K_iter=5 AH=1.0 standard -> 97.17%
    dict(N=4096, D=16, K_hh=2, K_iter=5, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="standard",    polarizer_alpha=0.0),
    # 6  step192: N=2048 D=16 K_hh=3 K_iter=5 AH=1.0 standard -> 95.90%
    dict(N=2048, D=16, K_hh=3, K_iter=5, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="standard",    polarizer_alpha=0.0),
    # 7  step185: N=2048 D=16 K_hh=4 K_iter=5 AH=1.0 standard -> 95.87%
    dict(N=2048, D=16, K_hh=4, K_iter=5, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="standard",    polarizer_alpha=0.0),
    # 8  step176: N=2048 D=32 K_hh=4 K_iter=5 AH=1.0 standard -> 96.18%
    dict(N=2048, D=32, K_hh=4, K_iter=5, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="standard",    polarizer_alpha=0.0),
    # 9  step181: N=2048 D=20 K_hh=4 K_iter=5 AH=1.0 standard -> 96.03%
    dict(N=2048, D=20, K_hh=4, K_iter=5, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="standard",    polarizer_alpha=0.0),
    # 10 step204: N=4096 D=16 K_hh=2 K_iter=6 AH=1.0 standard -> 97.15%
    dict(N=4096, D=16, K_hh=2, K_iter=6, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="standard",    polarizer_alpha=0.0),
    # 11-20: reasonable variations to fill elite slots
    dict(N=2048, D=24, K_hh=2, K_iter=5, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="standard",    polarizer_alpha=0.0),
    dict(N=2048, D=16, K_hh=2, K_iter=8, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="standard",    polarizer_alpha=0.0),
    dict(N=4096, D=16, K_hh=3, K_iter=5, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="standard",    polarizer_alpha=0.0),
    dict(N=2048, D=16, K_hh=2, K_iter=5, alpha_ahebb=1.0,  alpha_reflect=0.3, routing_mode="standard",    polarizer_alpha=0.0),
    dict(N=2048, D=16, K_hh=2, K_iter=4, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="standard",    polarizer_alpha=0.0),
    dict(N=2048, D=12, K_hh=2, K_iter=5, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="standard",    polarizer_alpha=0.0),
    dict(N=4096, D=16, K_hh=2, K_iter=8, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="standard",    polarizer_alpha=0.0),
    dict(N=2048, D=16, K_hh=2, K_iter=5, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="delta_proj",  polarizer_alpha=1.0),
    dict(N=2048, D=16, K_hh=2, K_iter=6, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="wpos_proj",   polarizer_alpha=1.5),
    dict(N=1024, D=16, K_hh=2, K_iter=5, alpha_ahebb=1.0,  alpha_reflect=0.5, routing_mode="standard",    polarizer_alpha=0.0),
]

assert len(SEED_ELITE) == 20, f"Expected 20 seed individuals, got {len(SEED_ELITE)}"

# ---------------------------------------------------------------------------
# Delta-vector polarizer model (self-contained copy from train_step234)
# ---------------------------------------------------------------------------

class SGNNET_DeltaPolarizer(nn.Module):
    """Polarizer using delta-W_pos or W_pos[receiver] as filter axis.

    Modes used by GA:
      "delta_proj"   -- project Z_nb onto delta-W direction
      "wpos_proj_ah" -- project onto W_pos[receiver] + AH (step217b style)

    When use_ah=False (alpha_ahebb=0.0), AH suppression is skipped.
    alpha_turing=0.0 always; W_phase is None.
    """

    def __init__(self, N_hidden, N_out, D, N_in, K_in, K_iter,
                 K_local, K_random, n_groups, alpha_reflect,
                 alpha_ahebb=1.0, mode="delta_proj",
                 polarizer_alpha=1.5, use_ah=True, seed=42):
        super().__init__()
        torch.manual_seed(seed)

        self.base = SGNNET_SmallWorld(
            N_hidden=N_hidden, N_out=N_out, D=D, N_in=N_in,
            K_in=K_in, K_iter=K_iter, K_local=K_local, K_random=K_random,
            n_groups=n_groups, norm_mode="l2", encoding_mode="fourier")

        self.resonant = SGNNET_Resonant(
            self.base, K_phase=8, alpha_reflect=alpha_reflect,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0)

        self.alpha_ahebb = alpha_ahebb if use_ah else 0.0
        self.alpha_reflect = alpha_reflect
        self.mode = mode
        self.polarizer_alpha = polarizer_alpha
        self.use_ah = use_ah and alpha_ahebb > 0.0

        # W_phase is None (alpha_turing=0.0 always)
        self.W_phase = None

    @property
    def W_pos(self):
        return self.base.W_pos

    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh       # [N, K_hh]
        N_h = self.base.N_hidden
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)

        W_h = self.base.W_pos[:N_h]       # [N, D]
        W_n = F.normalize(W_h, dim=-1)

        # Pre-compute AH suppression weights (static, wpos variant)
        supp_w = None
        if self.use_ah:
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)   # [N, K_hh]
            supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                      ).unsqueeze(0).unsqueeze(-1)                  # [1, N, K_hh, 1]

        # Pre-compute delta vectors for delta_proj mode
        delta_w_norm = None
        if self.mode == "delta_proj":
            delta_w = W_h.unsqueeze(1) - W_h[conn_hh]             # [N, K_hh, D]
            delta_w_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)  # [1, N, K_hh, D]

        # Pre-compute W_pos receiver axis for wpos_proj mode
        w_recv = None
        if self.mode == "wpos_proj_ah":
            w_recv = W_n.unsqueeze(0).unsqueeze(2)                 # [1, N, 1, D]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]   # [B, N, K_hh, D]

            # AH suppression first (if enabled)
            if supp_w is not None:
                Z_nb = Z_nb * supp_w

            # Polarizer
            if self.mode == "delta_proj":
                proj_coeff = (Z_nb * delta_w_norm).sum(dim=-1, keepdim=True)
                Z_projected = proj_coeff * delta_w_norm
                alpha = self.polarizer_alpha
                Z_nb = alpha * Z_projected + (1 - alpha) * Z_nb

            elif self.mode == "wpos_proj_ah":
                proj_coeff = (Z_nb * w_recv).sum(dim=-1, keepdim=True)
                Z_projected = proj_coeff * w_recv
                alpha = self.polarizer_alpha
                Z_nb = alpha * Z_projected + (1 - alpha) * Z_nb

            Z_struct = Z_nb.sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.base._readout(Z)


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------

def compute_flops(N: int, K_hh: int, D: int, K_iter: int) -> float:
    """Approximate FLOPs: 3 * N * K_hh * D * K_iter."""
    return 3.0 * N * K_hh * D * K_iter


def compute_params(N: int, D: int) -> int:
    """Approximate param count: W_pos (N+N_out)*D + theta N."""
    return (N + N_OUT) * D + N


def efficiency_multiplier(flops: float) -> float:
    """FLOPs-based efficiency bonus/penalty."""
    if flops <= 1e6:
        return 1.10
    elif flops <= 2e6:
        return 1.05
    elif flops <= 5e6:
        return 1.00
    else:
        return max(0.90, 1.0 - (flops - 5e6) / 50e6)


def fitness_score(accuracy: float, genes: dict) -> float:
    flops = compute_flops(genes["N"], genes["K_hh"], genes["D"], genes["K_iter"])
    bonus = efficiency_multiplier(flops)
    return accuracy * bonus


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(genes: dict, seed: int = 42) -> nn.Module:
    """Construct correct model from gene dict. alpha_turing=0.0 always."""
    N = genes["N"]
    D = genes["D"]
    K_hh = genes["K_hh"]
    K_iter = genes["K_iter"]
    alpha_ahebb = genes["alpha_ahebb"]
    alpha_reflect = genes["alpha_reflect"]
    routing_mode = genes["routing_mode"]
    polarizer_alpha = genes["polarizer_alpha"]

    K_r = max(1, K_hh // 4)
    K_l = K_hh - K_r
    n_groups = max(8, N // 8)

    torch.manual_seed(seed)

    if routing_mode == "standard":
        base = SGNNET_SmallWorld(
            N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
            K_in=K_IN, K_iter=K_iter, K_local=K_l, K_random=K_r,
            n_groups=n_groups, norm_mode="l2", encoding_mode="fourier")
        resonant = SGNNET_Resonant(
            base, K_phase=8, alpha_reflect=alpha_reflect,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0)
        if alpha_ahebb > 0.0:
            return SGNNET_AntiHebbian(resonant, alpha_ahebb=alpha_ahebb, variant="wpos")
        else:
            # No AH -- resonant with reflection only
            return resonant

    elif routing_mode in ("delta_proj", "wpos_proj"):
        internal_mode = "delta_proj" if routing_mode == "delta_proj" else "wpos_proj_ah"
        use_ah = alpha_ahebb > 0.0
        return SGNNET_DeltaPolarizer(
            N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
            K_in=K_IN, K_iter=K_iter, K_local=K_l, K_random=K_r,
            n_groups=n_groups, alpha_reflect=alpha_reflect,
            alpha_ahebb=alpha_ahebb, mode=internal_mode,
            polarizer_alpha=polarizer_alpha, use_ah=use_ah, seed=seed)

    else:
        raise ValueError(f"Unknown routing_mode: {routing_mode!r}")


# ---------------------------------------------------------------------------
# Individual evaluation
# ---------------------------------------------------------------------------

def evaluate_individual(
    genes: dict,
    train_loader,
    val_loader,
    device: torch.device,
    n_epochs: int,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Train one individual for n_epochs. Returns result dict.

    On any failure (OOM, NaN, exception) returns fitness=0, error=<tag>.
    """
    N = genes["N"]
    D = genes["D"]
    K_hh = genes["K_hh"]
    K_iter = genes["K_iter"]

    flops = compute_flops(N, K_hh, D, K_iter)
    n_params_est = compute_params(N, D)

    model = None
    try:
        model = build_model(genes, seed=seed).to(device)
        actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        kw = trainer_kwargs(N, n_epochs=n_epochs)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            **kw,
        )

        # Inject any extra learnable params not caught by Trainer
        if hasattr(model, "rotation_temp"):
            trainer.optimizer.add_param_group({
                "params": [model.rotation_temp],
                "lr": kw.get("lr_wpos", 2.36e-3),
                "weight_decay": 0.0,
            })

        t0 = time.time()
        history = trainer.train(n_epochs=n_epochs)
        elapsed = time.time() - t0

        top1_list = [h.get("val_top1", 0.0) for h in history]
        best_acc = max(top1_list) if top1_list else 0.0
        best_epoch = int(np.argmax(top1_list)) + 1 if top1_list else 0

        nan_detected = any(h.get("nan_detected", False) for h in history)
        if nan_detected:
            if verbose:
                print(f"    FAILED (nan_detected)")
            return dict(
                genes=genes, accuracy=0.0, fitness=0.0,
                flops=flops, params=actual_params,
                best_epoch=0, elapsed_s=round(elapsed, 1),
                error="nan_detected",
            )

        fit = fitness_score(best_acc, genes)

        if verbose:
            mult = efficiency_multiplier(flops)
            print(f"    acc={best_acc:.4f}  fit={fit:.4f}  "
                  f"flops={flops/1e6:.2f}M  eff_mult={mult:.3f}  "
                  f"ep={best_epoch}  t={elapsed:.0f}s")

        return dict(
            genes=genes,
            accuracy=best_acc,
            fitness=fit,
            flops=flops,
            params=actual_params,
            best_epoch=best_epoch,
            elapsed_s=round(elapsed, 1),
            error=None,
        )

    except RuntimeError as exc:
        err_str = str(exc)
        if "out of memory" in err_str.lower() or "oom" in err_str.lower():
            error_tag = "oom"
        else:
            error_tag = f"runtime_error: {err_str[:120]}"
        if verbose:
            print(f"    FAILED ({error_tag})")
        # Attempt to free memory
        if model is not None:
            del model
        try:
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass
        return dict(
            genes=genes, accuracy=0.0, fitness=0.0,
            flops=flops, params=n_params_est,
            best_epoch=0, elapsed_s=0.0,
            error=error_tag,
        )

    except Exception as exc:
        error_tag = f"exception: {type(exc).__name__}: {str(exc)[:120]}"
        if verbose:
            print(f"    FAILED ({error_tag})")
            traceback.print_exc()
        if model is not None:
            del model
        return dict(
            genes=genes, accuracy=0.0, fitness=0.0,
            flops=flops, params=n_params_est,
            best_epoch=0, elapsed_s=0.0,
            error=error_tag,
        )


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------

def random_individual(rng: random.Random) -> dict:
    """Sample a fully random individual from the gene space."""
    return {k: rng.choice(v) for k, v in GENE_SPACE.items()}


def crossover(parent_a: dict, parent_b: dict, rng: random.Random) -> dict:
    """Uniform crossover: each gene independently picked from one parent."""
    return {k: rng.choice([parent_a[k], parent_b[k]]) for k in GENE_KEYS}


def mutate(individual: dict, rng: random.Random, n_mutations: int | None = None) -> dict:
    """Mutate 1-3 random genes by sampling new values from the gene space."""
    child = copy.deepcopy(individual)
    if n_mutations is None:
        n_mutations = rng.randint(1, 3)
    keys_to_mutate = rng.sample(GENE_KEYS, k=min(n_mutations, len(GENE_KEYS)))
    for k in keys_to_mutate:
        current = child[k]
        candidates = [v for v in GENE_SPACE[k] if v != current]
        if candidates:
            child[k] = rng.choice(candidates)
        else:
            child[k] = rng.choice(GENE_SPACE[k])
    return child


def generate_population(
    elite: list[dict],
    n_crossbreed: int,
    n_mutations: int,
    rng: random.Random,
) -> list[dict]:
    """Build pop from elite: 20 elite + n_crossbreed crossbreeds + n_mutations mutations."""
    population = list(elite)

    for _ in range(n_crossbreed):
        parents = rng.sample(elite, k=2)
        population.append(crossover(parents[0], parents[1], rng))

    for _ in range(n_mutations):
        parent = rng.choice(elite)
        population.append(mutate(parent, rng))

    return population


# ---------------------------------------------------------------------------
# Journal logging
# ---------------------------------------------------------------------------

JOURNAL_HEADER = "\t".join([
    "gen", "idx", "N", "D", "K_hh", "K_iter",
    "alpha_ahebb", "alpha_reflect", "routing_mode", "polarizer_alpha",
    "accuracy", "fitness", "flops_M", "params", "best_epoch", "elapsed_s", "error",
])


def write_journal_header():
    if not JOURNAL_PATH.exists():
        with open(JOURNAL_PATH, "w") as f:
            f.write(JOURNAL_HEADER + "\n")


def append_journal(gen: int, idx: int, result: dict):
    genes = result["genes"]
    row = "\t".join([
        str(gen), str(idx),
        str(genes["N"]), str(genes["D"]), str(genes["K_hh"]), str(genes["K_iter"]),
        str(genes["alpha_ahebb"]), str(genes["alpha_reflect"]),
        genes["routing_mode"], str(genes["polarizer_alpha"]),
        f"{result['accuracy']:.6f}", f"{result['fitness']:.6f}",
        f"{result['flops']/1e6:.3f}", str(result["params"]),
        str(result["best_epoch"]), f"{result['elapsed_s']:.1f}",
        result["error"] or "",
    ])
    with open(JOURNAL_PATH, "a") as f:
        f.write(row + "\n")


# ---------------------------------------------------------------------------
# Generation summary
# ---------------------------------------------------------------------------

def print_generation_summary(gen: int, results: list[dict], elapsed: float):
    valid = [r for r in results if r["error"] is None]
    failed = len(results) - len(valid)
    sorted_valid = sorted(valid, key=lambda r: r["fitness"], reverse=True)

    print(f"\n{'='*76}")
    print(f"  Gen {gen:3d}  |  {len(results)} individuals  |  {failed} failed  |  {elapsed:.0f}s")
    print(f"{'='*76}")
    print(f"  {'#':>3}  {'N':>5}  {'D':>3}  {'Khh':>4}  {'Ki':>3}  "
          f"{'AH':>4}  {'mode':<12}  {'pa':>4}  {'acc':>7}  {'fit':>7}  {'MFLOPs':>7}")
    print(f"  {'-'*3}  {'-'*5}  {'-'*3}  {'-'*4}  {'-'*3}  "
          f"{'-'*4}  {'-'*12}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*7}")

    for rank, r in enumerate(sorted_valid[:20], 1):
        g = r["genes"]
        print(f"  {rank:>3}  {g['N']:>5}  {g['D']:>3}  {g['K_hh']:>4}  "
              f"{g['K_iter']:>3}  {g['alpha_ahebb']:>4.1f}  "
              f"{g['routing_mode']:<12}  {g['polarizer_alpha']:>4.1f}  "
              f"{r['accuracy']:>7.4f}  {r['fitness']:>7.4f}  "
              f"{r['flops']/1e6:>7.2f}")

    if sorted_valid:
        best = sorted_valid[0]
        g = best["genes"]
        print(f"\n  BEST: acc={best['accuracy']:.4f}  fit={best['fitness']:.4f}  "
              f"N={g['N']} D={g['D']} K_hh={g['K_hh']} K_iter={g['K_iter']} "
              f"AH={g['alpha_ahebb']} mode={g['routing_mode']}  "
              f"flops={best['flops']/1e6:.2f}M  params={best['params']:,}")
    print(f"{'='*76}\n")


# ---------------------------------------------------------------------------
# Data loading -- 50% subset for Tier-0
# ---------------------------------------------------------------------------

def make_tier0_loaders(data_path, batch_size: int, seed: int):
    """Full data load + 50% train subsample for Tier-0 scout protocol."""
    tr_full, va = make_loaders(str(data_path), batch_size=batch_size, seed=seed)
    n = len(tr_full.dataset)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    print(f"Tier-0: using {len(subset)}/{n} train samples (50%)")
    return tr, va


# ---------------------------------------------------------------------------
# Main GA loop
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*76}")
    print(f"  GA AutoRun v2 -- SGNNET Configuration Search")
    print(f"  device={DEVICE}  generations={args.generations}  "
          f"epochs={args.epochs_per_eval}  pop=50 (20 elite + 20 cross + 10 mut)")
    print(f"{'='*76}\n")

    tr, va = make_tier0_loaders(DATA_PATH, batch_size=args.batch_size, seed=SEED)
    write_journal_header()
    rng = random.Random(SEED)

    # Initialise or resume elite
    start_gen = 0
    elite_genes = list(SEED_ELITE)

    if args.resume_gen is not None:
        resume_path = RESULTS_DIR / f"ga_autorun_v2_gen{args.resume_gen}.json"
        if resume_path.exists():
            with open(resume_path) as f:
                prev = json.load(f)
            prev_results = [r for r in prev["results"] if r["error"] is None]
            prev_sorted = sorted(prev_results, key=lambda r: r["fitness"], reverse=True)[:20]
            elite_genes = [r["genes"] for r in prev_sorted]
            while len(elite_genes) < 20:
                elite_genes.append(random_individual(rng))
            start_gen = args.resume_gen + 1
            print(f"Resumed from gen {args.resume_gen}: {len(elite_genes)} elite members\n")
        else:
            print(f"WARNING: resume file not found ({resume_path}) -- starting fresh\n")

    # Generation loop
    for gen in range(start_gen, start_gen + args.generations):
        gen_t0 = time.time()
        population = generate_population(elite_genes, 20, 10, rng)

        print(f"--- Generation {gen:3d} ---  population={len(population)}")

        gen_results: list[dict] = []
        for i, genes in enumerate(population):
            flops = compute_flops(genes["N"], genes["K_hh"], genes["D"], genes["K_iter"])
            print(f"  [{i+1:2d}/{len(population)}] "
                  f"N={genes['N']} D={genes['D']} K_hh={genes['K_hh']} "
                  f"K_iter={genes['K_iter']} AH={genes['alpha_ahebb']} "
                  f"mode={genes['routing_mode']} pa={genes['polarizer_alpha']} "
                  f"flops={flops/1e6:.2f}M", flush=True)

            ind_seed = SEED + gen * 1000 + i
            result = evaluate_individual(
                genes=genes,
                train_loader=tr,
                val_loader=va,
                device=DEVICE,
                n_epochs=args.epochs_per_eval,
                seed=ind_seed,
                verbose=True,
            )
            gen_results.append(result)
            append_journal(gen, i, result)

        # Select new elite: top-20 by fitness
        valid = [r for r in gen_results if r["error"] is None and r["fitness"] > 0]
        if len(valid) >= 20:
            top20 = sorted(valid, key=lambda r: r["fitness"], reverse=True)[:20]
        else:
            top20 = sorted(gen_results, key=lambda r: r["fitness"], reverse=True)[:20]

        elite_genes = [r["genes"] for r in top20]
        while len(elite_genes) < 20:
            elite_genes.append(random_individual(rng))

        gen_elapsed = time.time() - gen_t0

        gen_output = {
            "generation": gen,
            "elapsed_s": round(gen_elapsed, 1),
            "n_individuals": len(gen_results),
            "n_failed": sum(1 for r in gen_results if r["error"] is not None),
            "best_fitness": top20[0]["fitness"] if top20 else 0.0,
            "best_accuracy": top20[0]["accuracy"] if top20 else 0.0,
            "elite_genes": elite_genes,
            "results": gen_results,
        }

        out_path = RESULTS_DIR / f"ga_autorun_v2_gen{gen}.json"
        out_path.write_text(json.dumps(gen_output, indent=2))
        print(f"  -> Saved: {out_path}")

        print_generation_summary(gen, gen_results, gen_elapsed)

    # Final cross-generation summary
    print(f"\n{'='*76}")
    print(f"  GA AutoRun v2 -- Complete")
    print(f"  Journal: {JOURNAL_PATH}")
    print(f"  Results: {RESULTS_DIR}/ga_autorun_v2_gen*.json")

    all_results: list[dict] = []
    for g in range(start_gen, start_gen + args.generations):
        p = RESULTS_DIR / f"ga_autorun_v2_gen{g}.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            for r in data["results"]:
                if r["error"] is None:
                    r["_gen"] = g
                    all_results.append(r)

    if all_results:
        all_sorted = sorted(all_results, key=lambda r: r["fitness"], reverse=True)
        print(f"\n  ALL-TIME TOP-5 (fitness):")
        print(f"  {'gen':>4}  {'N':>5}  {'D':>3}  {'Khh':>4}  {'Ki':>3}  "
              f"{'AH':>4}  {'mode':<12}  {'acc':>7}  {'fit':>7}  {'MFLOPs':>7}")
        for r in all_sorted[:5]:
            g = r["genes"]
            print(f"  {r['_gen']:>4}  {g['N']:>5}  {g['D']:>3}  {g['K_hh']:>4}  "
                  f"{g['K_iter']:>3}  {g['alpha_ahebb']:>4.1f}  "
                  f"{g['routing_mode']:<12}  {r['accuracy']:>7.4f}  "
                  f"{r['fitness']:>7.4f}  {r['flops']/1e6:>7.2f}")
    print(f"{'='*76}\n")


if __name__ == "__main__":
    main()
