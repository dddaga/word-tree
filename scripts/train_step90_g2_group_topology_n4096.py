"""Step 90 (G2): Group topology at N=4096 with K_hh=4 base.

MOTIVATION
==========
step82 showed random-group topology gives +3.01pp at N=1024:
  Ref (spatial, n_groups=128): 82.62%
  A   (random-group, n_groups=8): 85.63% (+3.01pp)
  B   (random-group, n_groups=16): 84.84% (+2.22pp)
  C   (random-group, n_groups=32): 84.15% (+1.53pp)
  D   (input-aligned, n_groups=8): KILLED 69.40% (−13.22pp)

This was NEVER tested at N=4096. If the +3pp gain scales, it could push
past the 97.38% project best. We also apply the K_hh=4 default (step86).

This is a CALIBRATION run (50%/75ep) to check if the gain survives scale.
If winner is confirmed, promote to 100%/150ep in the next step.

CONFIGS (N=4096, D=64, K_hh=4, K_iter=8, 50%/75ep)
======================================================
  Ref : spatial topology (K_local=2, K_random=2, n_groups=512)
  A   : random-group topology, n_groups=8
  B   : random-group topology, n_groups=16
  C   : random-group topology, n_groups=32
  D   : random-group topology, n_groups=64

All: turing=0.0, reflect=0.5, AH=1.0, K_in=50, D=64, K_hh=4 (K_local=2, K_random=2).

Note: D (input-align) was killed at N=1024. Skipped here.
Note: K_local=2 throughout — AH nullifies weak local edges anyway.

Comparison baseline: step86 A = 96.59% (K_hh=4, spatial, 50%/75ep).

To reproduce:
    python -u scripts/train_step90_g2_group_topology_n4096.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.sgnnet.model_smallworld       import SGNNET_SmallWorld
from src.sgnnet.model_resonant         import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory  import SGNNET_AntiHebbian
from src.training.trainer              import Trainer
from src.training.experiment_config    import trainer_kwargs, run_metadata
from src.training.dataset              import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS    = 75
BATCH     = 128
SEED      = 42
DATA      = "data/store.h5"
N         = 4096
N_IN      = 25088
N_OUT     = 10
D         = 64
K_IN      = 50
K_LOCAL   = 2       # K_hh=4: K_local=2 + K_random=2
K_RANDOM  = 2
K_ITER    = 8
K_PHASE   = 8
BEAM_SIZE = 16
GEO_GAMMA = 0.5
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
ALPHA_AHEBB   = 1.0
N_GROUPS_SPATIAL = max(8, N // 8)  # = 512 (spatial default)

# Baselines
STEP86A_REF = 0.9659   # step86 A: K_hh=4, spatial topology, N=4096, 50%/75ep


def _random_group_assignment(N: int, n_groups: int, seed: int) -> np.ndarray:
    rng      = np.random.default_rng(seed)
    perm     = rng.permutation(N)
    group_id = np.zeros(N, dtype=np.int64)
    grp_size = max(1, N // n_groups)
    for g in range(n_groups):
        lo = g * grp_size
        hi = N if g == n_groups - 1 else (g + 1) * grp_size
        group_id[perm[lo:hi]] = g
    return group_id


def _build_randomgroup_conn_hh(
    N: int, K_local: int, K_random: int, n_groups: int, seed: int = 0
) -> torch.Tensor:
    """K_local intra-group wires + K_random inter-group wires per neuron."""
    rng      = np.random.default_rng(seed)
    group_id = _random_group_assignment(N, n_groups, seed)
    members  = [np.where(group_id == g)[0].tolist() for g in range(n_groups)]

    K    = K_local + K_random
    conn = np.zeros((N, K), dtype=np.int64)

    for h in range(N):
        g    = group_id[h]
        same = [x for x in members[g] if x != h]
        diff = [x for x in range(N) if group_id[x] != g]

        loc = rng.choice(same, size=K_local, replace=(len(same) < K_local))
        rnd = rng.choice(diff, size=K_random, replace=False)
        conn[h] = np.concatenate([loc, rnd])

    return torch.tensor(conn, dtype=torch.long)


_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n   = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(
            subset, batch_size=BATCH, shuffle=True, num_workers=10
        )
        _loaders = (tr, va)
    return _loaders


def make_model(n_groups_hidden: int | None, seed_offset: int = 0) -> SGNNET_AntiHebbian:
    """n_groups_hidden=None → spatial topology (Ref). Otherwise random-group."""
    torch.manual_seed(SEED + seed_offset)
    n_groups = N_GROUPS_SPATIAL if n_groups_hidden is None else N_GROUPS_SPATIAL
    base = SGNNET_SmallWorld(
        N_in=N_IN, N_hidden=N, N_out=N_OUT,
        K_local=K_LOCAL, K_random=K_RANDOM,
        K_in=K_IN, K_iter=K_ITER,
        n_groups=n_groups,
        norm_mode="l2", D=D, encoding_mode="fourier",
    )

    if n_groups_hidden is not None:
        # Replace conn_hh with random-group topology
        new_conn_hh = _build_randomgroup_conn_hh(
            N, K_LOCAL, K_RANDOM,
            n_groups=n_groups_hidden, seed=SEED + seed_offset,
        )
        base.register_buffer("conn_hh", new_conn_hh)

    resonant = SGNNET_Resonant(
        base, K_phase=K_PHASE, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=BEAM_SIZE,
        geo_gamma=GEO_GAMMA, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(key: str, label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best      = max(top1_hist)
    best_ep   = int(np.argmax(top1_hist)) + 1
    frac      = best_ep / len(history)

    result = {
        "label":            label,
        "top1_best":        best,
        "top1_last":        history[-1].get("val_top1", 0.0),
        "final_task_loss":  float(np.mean([h.get("task_loss", 0.0) for h in history[-5:]])),
        "best_epoch":       best_ep,
        "epochs_run":       len(history),
        "elapsed_s":        round(elapsed, 1),
        "best_epoch_frac":  round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history":     top1_hist,
        "step86a_ref":      STEP86A_REF,
        "delta_vs_step86a": round(best - STEP86A_REF, 4),
        "params":           count_params(model),
        "_meta":            run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    result.update(meta)
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  vs_step86a={best-STEP86A_REF:+.4f}  t={elapsed:.0f}s"
    )
    return result


CONFIGS = [
    ("Ref", "Ref  spatial topology, n_groups=512 (step86-A control)", None),
    ("A",   "A    random-group, n_groups=8",   8),
    ("B",   "B    random-group, n_groups=16", 16),
    ("C",   "C    random-group, n_groups=32", 32),
    ("D",   "D    random-group, n_groups=64", 64),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  N={N}  Data: 50%")
    print(f"Step 90 (G2): Group topology scale test at N=4096, K_hh=4")
    print(f"K_local={K_LOCAL}  K_random={K_RANDOM}  K_iter={K_ITER}  D={D}")
    print(f"turing={ALPHA_TURING}  reflect={ALPHA_REFLECT}  AH={ALPHA_AHEBB}")
    print(f"step86 A baseline (K_hh=4, spatial, 50%/75ep): {STEP86A_REF:.4f}")
    print(f"step82 context: +3.01pp at N=1024 — does this scale?")
    print()
    for k, label, n_g in CONFIGS:
        print(f"  {k:4s}  n_groups={n_g if n_g else N_GROUPS_SPATIAL}  {label}")
    print()

    # NOTE: conn_hh build for N=4096 with large n_groups will be slow (O(N²) loop).
    # Expected: ~2-5 min per config for conn_hh construction.
    # For n_groups=8: each group has 512 neurons — large same/diff pools, fast sampling.
    # For n_groups=64: each group has 64 neurons — still manageable.
    print("Note: random-group conn_hh construction at N=4096 may take 2-5 min.")
    print()

    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step90_g2_group_topology_n4096.json"

    for i, (key, label, n_g) in enumerate(CONFIGS):
        model = make_model(n_g, seed_offset=i).to(DEVICE)
        meta  = {
            "key": key, "N": N, "D": D, "K_iter": K_ITER,
            "K_hh": K_LOCAL + K_RANDOM, "K_local": K_LOCAL,
            "K_random": K_RANDOM, "K_in": K_IN,
            "n_groups_hidden": n_g if n_g else N_GROUPS_SPATIAL,
            "alpha_turing": ALPHA_TURING, "alpha_ahebb": ALPHA_AHEBB,
            "data_frac": 0.5,
        }
        results[key] = run(key, label, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    print(f"\n{'='*70}")
    print(f"STEP 90 G2 COMPLETE — Group Topology at N=4096")
    print(f"step86 A baseline (spatial, K_hh=4): {STEP86A_REF:.4f}")
    print()
    print(f"  {'Key':4s}  {'n_groups':>9s}  {'top1':>8s}  {'vs_86a':>8s}")
    for key, label, n_g in CONFIGS:
        if key not in results:
            continue
        r = results[key]
        print(f"  {key:4s}  {n_g if n_g else N_GROUPS_SPATIAL:>9}  "
              f"{r['top1_best']:.4f}    {r['delta_vs_step86a']:+.4f}")
    winner = max(results, key=lambda k: results[k]["top1_best"])
    print(f"\n  Winner: {winner}")
    w_delta = results[winner]['delta_vs_step86a']
    if w_delta > 0.005:
        print(f"  → Group topology confirmed at N=4096 (+{w_delta:.4f}). Promote to 100%/150ep.")
    elif w_delta > 0:
        print(f"  → Marginal gain ({w_delta:+.4f}). Check at 100%/150ep before adopting.")
    else:
        print(f"  → Group topology does not scale to N=4096. Spatial topology remains optimal.")
