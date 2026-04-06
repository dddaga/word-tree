"""Step 82: Group-structured hidden topology.

MOTIVATION
==========
Current SGNNET_SmallWorld uses SPATIAL topology for conn_hh:
    local neighbours = neurons in the same position-based group (h // group_size)
    random wires    = long-range shortcuts (Watts-Strogatz)

step55 showed that GROUPED INPUT projections give +5pp over unified projection.
The hypothesis: explicit group structure creates an inductive bias for specialisation
without increasing parameters — neurons in a group co-learn a subspace representation,
K_random wires and K_iter routing steps handle cross-group integration.

This experiment replaces the SPATIAL conn_hh with a RANDOM-GROUP conn_hh:
    group assignment: neuron → group id (random permutation, not by position)
    local wires: K_local connections within the same group
    random wires: K_random connections guaranteed from a DIFFERENT group

The small-world property is preserved: random assignment means any neuron is
equally likely to share a group with any other, so log(N) mixing still holds.

Config D tests INPUT-GROUP ALIGNMENT: hidden neuron in group g samples inputs
preferentially from input region g (same block structure as current _build_fanin_conn
but using the random group assignment instead of position-based grouping).

CONFIGS (N=1024, D=64, K_iter=8, Gen4+ params, 50%/75ep)
==========================================================
  Ref : current spatial topology  (K_local=4, K_random=2, n_groups=128)
  A   : random-group topology, n_groups=8
  B   : random-group topology, n_groups=16
  C   : random-group topology, n_groups=32
  D   : random-group topology, n_groups=8 + input-group alignment

KEY QUESTION
============
Does group specialisation in the hidden layer give the same gain as step55's
grouped input projections? Or is the spatial structure essential?

To reproduce:
    python -u scripts/train_step82_group_topology.py --device mps
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
from src.training.experiment_config    import (
    trainer_kwargs, topology_kwargs, run_metadata,
)
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
N         = 1024
D         = 64

K_PHASE       = 8
BEAM_SIZE     = 16
GEO_GAMMA     = 0.5
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
ALPHA_AHEBB   = 1.0

STEP69_REF = 0.8336

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n   = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(
            subset, batch_size=BATCH, shuffle=True, num_workers=0
        )
        _loaders = (tr, va)
    return _loaders


# ── Random-group topology builders ────────────────────────────────────────────

def _random_group_assignment(N: int, n_groups: int, seed: int) -> np.ndarray:
    """Assign each of N neurons to one of n_groups groups randomly."""
    rng       = np.random.default_rng(seed)
    perm      = rng.permutation(N)
    group_id  = np.zeros(N, dtype=np.int64)
    grp_size  = max(1, N // n_groups)
    for g in range(n_groups):
        lo = g * grp_size
        hi = N if g == n_groups - 1 else (g + 1) * grp_size
        group_id[perm[lo:hi]] = g
    return group_id


def _build_randomgroup_conn_hh(
    N: int, K_local: int, K_random: int, n_groups: int, seed: int = 0
) -> torch.Tensor:
    """conn_hh [N, K_local+K_random] with random-group membership.

    K_local wires per neuron drawn from the SAME group (intra-group).
    K_random wires per neuron drawn from a DIFFERENT group (inter-group).
    """
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


def _build_aligned_fanin_conn(
    N_hidden: int, N_in: int, K_in: int, group_id_h: np.ndarray,
    n_groups: int, seed: int = 0,
) -> torch.Tensor:
    """conn_in [N_hidden, K_in] aligned to random group assignment.

    Neurons in hidden group g preferentially sample from input region g.
    Uses the same round-robin coverage guarantee as _build_fanin_conn.
    """
    rng          = np.random.default_rng(seed)
    group_size_in = max(1, N_in // n_groups)
    conn         = np.zeros((N_hidden, K_in), dtype=np.int64)

    # For each hidden group, collect its neurons and assign them to input region g
    for g in range(n_groups):
        neurons = np.where(group_id_h == g)[0].tolist()
        in_lo   = g * group_size_in
        in_hi   = N_in if g == n_groups - 1 else (g + 1) * group_size_in
        inputs  = np.arange(in_lo, in_hi)

        if len(neurons) == 0 or len(inputs) == 0:
            continue

        n_neurons = len(neurons)
        shuffled  = rng.permutation(inputs).tolist()

        for j, h in enumerate(neurons):
            coverage    = shuffled[j::n_neurons]
            n_remaining = K_in - len(coverage)
            if len(coverage) >= K_in:
                conn[h] = rng.choice(coverage, size=K_in, replace=False)
            else:
                covered_set = set(coverage)
                not_covered = [x for x in inputs.tolist() if x not in covered_set]
                if len(not_covered) >= n_remaining:
                    extra = rng.choice(not_covered, size=n_remaining, replace=False).tolist()
                else:
                    extra = not_covered + rng.integers(0, N_in, n_remaining - len(not_covered)).tolist()
                conn[h] = np.array(coverage + extra, dtype=np.int64)

    return torch.tensor(conn, dtype=torch.long)


# ── Model factory ──────────────────────────────────────────────────────────────

def make_model(
    n_groups_hidden: int | None,   # None = use default spatial topology
    align_input: bool = False,
    seed_offset: int = 0,
) -> SGNNET_AntiHebbian:
    """Build SGNNET_AntiHebbian, optionally replacing conn_hh with group topology."""
    torch.manual_seed(SEED + seed_offset)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=8, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )

    if n_groups_hidden is not None:
        # Replace conn_hh with random-group topology
        new_conn_hh = _build_randomgroup_conn_hh(
            N, tk["K_local"], tk["K_random"],
            n_groups=n_groups_hidden, seed=SEED + seed_offset,
        )
        base.register_buffer("conn_hh", new_conn_hh)

        if align_input:
            # Replace conn_in with input-aligned version
            group_id = _random_group_assignment(N, n_groups_hidden, SEED + seed_offset)
            new_conn_in = _build_aligned_fanin_conn(
                N, 25088, tk["K_in"], group_id, n_groups_hidden, seed=SEED + seed_offset,
            )
            base.register_buffer("conn_in", new_conn_in)

    resonant = SGNNET_Resonant(
        base, K_phase=K_PHASE, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=BEAM_SIZE,
        geo_gamma=GEO_GAMMA, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


# ── Training loop ──────────────────────────────────────────────────────────────

def run(label: str, model: nn.Module, meta: dict) -> dict:
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
        "step69_ref":       STEP69_REF,
        "delta_vs_ref":     round(best - STEP69_REF, 4),
        "_meta":            run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  vs_ref={best-STEP69_REF:+.4f}  t={elapsed:.0f}s"
    )
    return result


# ── Configs ────────────────────────────────────────────────────────────────────

CONFIGS = [
    # (key, label, n_groups_hidden, align_input, seed_offset)
    ("Ref", "Ref  spatial topology (n_groups=128, current default)",
     None,  False, 0),
    ("A",   "A    random-group topology, n_groups=8",
     8,     False, 1),
    ("B",   "B    random-group topology, n_groups=16",
     16,    False, 2),
    ("C",   "C    random-group topology, n_groups=32",
     32,    False, 3),
    ("D",   "D    random-group topology, n_groups=8 + input-group alignment",
     8,     True,  4),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 82: Group-structured hidden topology")
    print(f"Ref: step69 spatial topology = {STEP69_REF:.4f}")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step82_group_topology.json"

    for key, label, n_groups_h, align_inp, seed_off in CONFIGS:
        model = make_model(n_groups_h, align_input=align_inp,
                           seed_offset=seed_off).to(DEVICE)
        conn_desc = (f"random_group_n{n_groups_h}" + ("_aligned" if align_inp else "")
                     if n_groups_h is not None else "spatial")
        meta = {
            "N": N, "D": D, "K_iter": 8,
            "n_groups_hidden": n_groups_h,
            "align_input":     align_inp,
            "conn_hh_type":    conn_desc,
            "alpha_ahebb":     ALPHA_AHEBB,
            "data_frac":       0.5,
        }
        results[key] = run(label, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    print(f"\n{'='*70}")
    print(f"STEP 82 COMPLETE — Group-structured hidden topology")
    print(f"Ref (spatial) = {STEP69_REF:.4f}")
    print()
    print(f"  {'Key':4s}  {'top1':>8s}  {'vs_ref':>8s}  {'n_groups':>9s}  {'aligned':>8s}")
    for key, label, n_groups_h, align_inp, _ in CONFIGS:
        if key not in results:
            continue
        r = results[key]
        grp_str = str(n_groups_h) if n_groups_h is not None else "128(spat)"
        print(f"  {key:4s}  {r['top1_best']:.4f}    {r['delta_vs_ref']:+.4f}"
              f"  {grp_str:>9s}  {'yes' if align_inp else 'no':>8s}")
    print()
    print("  Winner n_groups → feeds into step83 (group state + inter-group routing)")
    print("  D > A: input-group alignment compounds the group-topology gain")
