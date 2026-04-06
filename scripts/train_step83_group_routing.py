"""Step 83: Group state + inter-group dynamic routing.

DEPENDENCY: Requires step82 results. Default assumes step82 Config B winner
(n_groups=16). Update N_GROUPS_HIDDEN below if step82 picks a different winner.

MOTIVATION
==========
Step 82 establishes group-structured hidden topology. Step 83 adds DYNAMIC
ROUTING between groups via group state vectors.

Gate-death bypass by design:
  - Wave-1 failure: N=4096 routing decisions per neuron per step → g^K collapse
  - Group routing: only G=n_groups routing decisions per step
  - softmax over G groups → Σ w_g = 1 → no signal attenuation, no gate-death

Architecture:
  1. At each K_iter step, compute group state: S_g = mean(Z[h] for h in group g)
     → [n_groups, D] — differentiable average, no new parameters
  2. Compute inter-group routing weights:
     w_{g→g'} = softmax(score(S_g, S_{g'}) / τ, dim=g') — [n_groups, n_groups]
  3. Each neuron h in group g receives:
     Z_inter[h] = Σ_{g'} w_{group(h)→g'} × S_{g'}   — [B, N, D]
  4. Combine with within-group AH structural routing:
     Z_new[h] = normalize(Z_struct_AH[h] + β × Z_inter[h])

β controls the mixing strength between within-group AH and inter-group routing.
When β=0 → reduces to step82 (within-group only). When β→∞ → pure group routing.

CONFIGS (N=1024, D=64, K_iter=8, Gen4+ params, 50%/75ep)
==========================================================
  Ref : step82 winner (group topology, no inter-group routing — β=0)
  A   : inter-group dot-product score, β=0.5, routing every K_iter step
  B   : inter-group dot-product score, β=0.5, routing FINAL step only
  C   : inter-group dot-product score, β=0.1 (weak mixing)
  D   : inter-group dot-product score, β=1.0 (strong mixing)
  E   : inter-group dot-product score, β=0.5, group state = max-pool (vs mean)

EXPECTED DIRECTION
==================
  A > Ref: dynamic inter-group routing adds to pure group topology
  B > A:   routing only at final step is sufficient (cheaper)
  D vs C:  find optimal β mixing coefficient
  E vs A:  max-pool group state captures dominant features better than mean

To reproduce (after step82 completes, update N_GROUPS_HIDDEN):
    python -u scripts/train_step83_group_routing.py --device mps
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
import torch.nn.functional as F

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

# ── UPDATE THIS after step82 results ──────────────────────────────────────────
N_GROUPS_HIDDEN = 16  # step82 assumed winner (Config B); update if different
# ─────────────────────────────────────────────────────────────────────────────

STEP69_REF   = 0.8336
STEP82_REF   = None   # fill in after step82 completes (group topology baseline)

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


# ── Group topology builder (reuse from step82) ────────────────────────────────

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


def _build_randomgroup_conn_hh(N, K_local, K_random, n_groups, seed=0):
    rng      = np.random.default_rng(seed)
    group_id = _random_group_assignment(N, n_groups, seed)
    members  = [np.where(group_id == g)[0].tolist() for g in range(n_groups)]
    K        = K_local + K_random
    conn     = np.zeros((N, K), dtype=np.int64)
    for h in range(N):
        g    = group_id[h]
        same = [x for x in members[g] if x != h]
        diff = [x for x in range(N) if group_id[x] != g]
        loc  = rng.choice(same, size=K_local, replace=(len(same) < K_local))
        rnd  = rng.choice(diff, size=K_random, replace=False)
        conn[h] = np.concatenate([loc, rnd])
    return torch.tensor(conn, dtype=torch.long), group_id


# ── Group routing wrapper ──────────────────────────────────────────────────────

class SGNNET_GroupRouting(nn.Module):
    """AH routing within groups + dynamic softmax routing between groups.

    Computes group state vectors S_g = pool(Z[h] for h in group g) and uses
    inter-group routing weights (softmax over G groups) to mix group signals.
    This is the first conservative dynamic routing mechanism at the group level:
    G² = n_groups² routing decisions instead of N×K_hh at the neuron level.

    Parameters
    ----------
    resonant      : SGNNET_Resonant backbone (with group conn_hh)
    group_id_arr  : int array [N], group assignment for each neuron
    n_groups      : number of groups
    beta          : mixing weight for inter-group signal (0 = AH only)
    tau           : softmax temperature for inter-group routing
    route_every_step: if False, only route at the FINAL K_iter step
    pool_mode     : 'mean' or 'max' for group state aggregation
    """

    def __init__(
        self,
        resonant: SGNNET_Resonant,
        group_id_arr: torch.Tensor,   # [N] int64
        n_groups: int,
        alpha_ahebb: float = 1.0,
        beta: float = 0.5,
        tau: float = 1.0,
        route_every_step: bool = True,
        pool_mode: str = "mean",
    ):
        super().__init__()
        self.m               = resonant
        self.alpha_ahebb     = alpha_ahebb
        self.beta            = beta
        self.tau             = tau
        self.n_groups        = n_groups
        self.route_every_step = route_every_step
        self.pool_mode       = pool_mode
        # Register group assignment as buffer (non-learnable, fixed topology)
        self.register_buffer("group_id", group_id_arr)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _compute_group_states(self, Z: torch.Tensor) -> torch.Tensor:
        """Compute per-group state: [B, G, D]."""
        B, N, D_ = Z.shape
        G        = self.n_groups
        # Scatter-mean: for each group g, average Z[h] where group_id[h] == g
        S = torch.zeros(B, G, D_, device=Z.device, dtype=Z.dtype)
        counts = torch.zeros(G, device=Z.device, dtype=Z.dtype)
        for g in range(G):
            mask = (self.group_id == g)   # [N]
            members_Z = Z[:, mask, :]      # [B, n_g, D]
            if self.pool_mode == "max":
                S[:, g, :] = members_Z.max(dim=1).values
            else:
                S[:, g, :] = members_Z.mean(dim=1)
        return S   # [B, G, D]

    def _inter_group_signal(self, Z: torch.Tensor) -> torch.Tensor:
        """Compute inter-group routed signal for each neuron: [B, N, D]."""
        S = self._compute_group_states(Z)   # [B, G, D]
        B, G, D_ = S.shape
        N        = Z.shape[1]

        # Routing scores: dot(S_g, S_{g'}) / tau → [B, G, G]
        S_norm  = F.normalize(S, dim=-1)
        scores  = torch.bmm(S_norm, S_norm.transpose(1, 2)) / self.tau   # [B, G, G]
        weights = F.softmax(scores, dim=2)   # [B, G, G], Σ_{g'} w_{g→g'} = 1

        # Routed group signals: Z_group[g] = Σ_{g'} w_{g→g'} × S_{g'} → [B, G, D]
        Z_group = torch.bmm(weights, S)   # [B, G, D]

        # Broadcast to neurons: each neuron gets its group's routed signal
        Z_inter = Z_group[:, self.group_id, :]   # [B, N, D]
        return Z_inter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        N_h       = self.m.base.N_hidden
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)

        # Pre-compute static AH suppression
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)
        K           = self.m.base.K_iter

        for k in range(K):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_nb     = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)   # AH within-group routing

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            # Inter-group routing: every step or only at final step
            if self.beta > 0 and (self.route_every_step or k == K - 1):
                Z_inter = self._inter_group_signal(Z)
                Z_new = Z_struct + Z_reflected + self.beta * Z_inter
            else:
                Z_new = Z_struct + Z_reflected

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ── Model factory ──────────────────────────────────────────────────────────────

def make_model(
    beta: float = 0.0,
    route_every_step: bool = True,
    pool_mode: str = "mean",
    seed_offset: int = 0,
) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=8, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    # Replace conn_hh with random-group topology (step82 winner)
    new_conn, group_id_arr = _build_randomgroup_conn_hh(
        N, tk["K_local"], tk["K_random"],
        n_groups=N_GROUPS_HIDDEN, seed=SEED + seed_offset,
    )
    base.register_buffer("conn_hh", new_conn)

    resonant = SGNNET_Resonant(
        base, K_phase=K_PHASE, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=BEAM_SIZE,
        geo_gamma=GEO_GAMMA, mode="dynamic_z_geo", resonance_threshold=0.0,
    )

    if beta == 0.0:
        # Ref: pure AH within-group routing (no inter-group)
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    else:
        return SGNNET_GroupRouting(
            resonant,
            group_id_arr=torch.tensor(group_id_arr, dtype=torch.long),
            n_groups=N_GROUPS_HIDDEN,
            alpha_ahebb=ALPHA_AHEBB,
            beta=beta,
            tau=1.0,
            route_every_step=route_every_step,
            pool_mode=pool_mode,
        )


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

    ref_delta = best - STEP69_REF
    step82_delta = (best - STEP82_REF) if STEP82_REF is not None else None

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
        "step82_ref":       STEP82_REF,
        "delta_vs_step69":  round(ref_delta, 4),
        "delta_vs_step82":  round(step82_delta, 4) if step82_delta is not None else None,
        "_meta":            run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  vs_step69={ref_delta:+.4f}  t={elapsed:.0f}s"
    )
    return result


# ── Configs ────────────────────────────────────────────────────────────────────

CONFIGS = [
    # (key, label, beta, route_every_step, pool_mode, seed_offset)
    ("Ref", "Ref  step82 winner — group topology, no inter-group routing (β=0)",
     0.0, True, "mean", 0),
    ("A",   "A    inter-group dot-product score, β=0.5, every K_iter step",
     0.5, True, "mean", 1),
    ("B",   "B    inter-group dot-product score, β=0.5, FINAL step only",
     0.5, False, "mean", 2),
    ("C",   "C    inter-group dot-product score, β=0.1 (weak mixing)",
     0.1, True, "mean", 3),
    ("D",   "D    inter-group dot-product score, β=1.0 (strong mixing)",
     1.0, True, "mean", 4),
    ("E",   "E    inter-group dot-product score, β=0.5, group state = max-pool",
     0.5, True, "max",  5),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 83: Group state + inter-group dynamic routing")
    print(f"n_groups assumed winner from step82: {N_GROUPS_HIDDEN}")
    print(f"⚠️  Update N_GROUPS_HIDDEN at top of file after step82 results")
    print(f"step69 Ref = {STEP69_REF:.4f}")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step83_group_routing.json"

    for key, label, beta, every_step, pool, seed_off in CONFIGS:
        model = make_model(beta, every_step, pool, seed_offset=seed_off).to(DEVICE)
        meta = {
            "N": N, "D": D, "K_iter": 8,
            "n_groups":        N_GROUPS_HIDDEN,
            "beta":            beta,
            "route_every_step": every_step,
            "pool_mode":       pool,
            "alpha_ahebb":     ALPHA_AHEBB,
            "data_frac":       0.5,
        }
        results[key] = run(label, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    print(f"\n{'='*70}")
    print(f"STEP 83 COMPLETE — Group state + inter-group routing")
    print(f"step69 Ref={STEP69_REF:.4f}  step82 Ref (group topo)={STEP82_REF}")
    print()
    print(f"  {'Key':4s}  {'top1':>8s}  {'vs_s69':>8s}  {'β':>5s}  {'every':>6s}  {'pool':>5s}")
    for key, label, beta, every_step, pool, _ in CONFIGS:
        if key not in results:
            continue
        r = results[key]
        print(f"  {key:4s}  {r['top1_best']:.4f}    {r['delta_vs_step69']:+.4f}"
              f"  {beta:>5.1f}  {'yes' if every_step else 'no':>6s}  {pool:>5s}")
    print()
    print("  Winner → feeds into step84 (phase-based inter-group routing)")
