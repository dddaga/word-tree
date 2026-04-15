"""Step 701: Parallel routing branches — can we replace K_iter=5 sequential passes
with K_iter_outer=1..3 passes over multiple parallel topologies?

HYPOTHESIS
==========
Sequential K_iter=5 gather-scatter is an inherent serialisation bottleneck.
Each pass depends on Z from the previous pass, preventing CUDA parallelism.

Alternative: at each outer iteration, compute several gather operations IN PARALLEL
over different conn_hh topologies (short-range, mixed, long-range) and combine
with learnable weights. Fewer sequential steps → same or better accuracy.

If a single outer iteration with 3+ parallel branches matches K_iter=5,
all branches can be launched as simultaneous GPU kernel calls — pure parallelism.

CONFIGS (Tier-0, 20ep, 50% data, N=2048, D=16, K_hh_total=2)
=============================================================
  Ref          : K_iter=5, standard mixed topology (step199 baseline)
  A_par3_k2    : 3 branches (short/mix/long), K_iter_outer=2, learnable α
  B_par3_k1    : 3 branches, K_iter_outer=1 — max speedup, tests if sequential needed
  C_par2_k3    : 2 branches (short + long, no middle), K_iter_outer=3
  D_par3_k3    : 3 branches, K_iter_outer=3 — moderate compression
  E_par5_k1    : 5 branches, K_iter_outer=1 — wide parallel, single outer pass

Each branch uses the same K_hh_total=2 but with different K_local/K_random splits.
Branch weights are learnable scalars passed through softmax so they stay positive
and sum to 1.

AH + reflection kept identical to step199 efficiency config:
  alpha_ahebb=1.0, alpha_reflect=0.5, alpha_turing=0.0
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

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Step 701 — parallel routing branches vs sequential K_iter")
parser.add_argument("--device",  default="auto",
                    help="mps | cpu | cuda | auto (default: auto)")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys to run (default: all)")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

EPOCHS        = args.epochs
BATCH         = 128
SEED          = 42
DATA          = "data/store.h5"

N      = 2048
N_IN   = 25088
N_OUT  = 10
D      = 16
K_HH   = 2          # per-branch fan-in (K_local + K_random = 2)
K_IN   = 25
N_GROUPS = max(8, N // 8)

ALPHA_AHEBB   = 1.0
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0

OUT_PATH = ROOT / "results" / "train_step701_parallel_branches.json"

# ---------------------------------------------------------------------------
# Branch topology definitions
# ---------------------------------------------------------------------------

# Each entry: (K_local, K_random)  — must sum to K_HH=2
BRANCH_DEFS = {
    "short": (2, 0),   # pure local
    "mix":   (1, 1),   # standard mixed (default step199)
    "long":  (0, 2),   # pure random shortcuts
    "lmix1": (2, 1),   # local-heavy with shortcut (K_HH=3 but we use K_HH=2 cap)
    "rmix1": (1, 2),   # random-heavy (K_HH=3 but we use K_HH=2 cap)
}

# For configs using only K_HH=2 branches (short/mix/long/lmix1/rmix1):
# Note: lmix1 and rmix1 have K_HH=3 (K_local+K_random=3), used only for E_par5_k1
# where we allow a slightly wider fan per branch.

# Config specs: (branch_names, K_iter_outer)
CONFIG_SPECS = {
    "Ref":        None,                           # built separately
    "A_par3_k2":  (["short", "mix", "long"], 2),
    "B_par3_k1":  (["short", "mix", "long"], 1),
    "C_par2_k3":  (["short", "long"],        3),
    "D_par3_k3":  (["short", "mix", "long"], 3),
    "E_par5_k1":  (["short", "mix", "long", "lmix1", "rmix1"], 1),
}


# ---------------------------------------------------------------------------
# SGNNET_ParallelBranch
# ---------------------------------------------------------------------------

class SGNNET_ParallelBranch(nn.Module):
    """SGNNET with multiple parallel routing branches per outer iteration.

    At each outer iteration k in range(K_iter_outer):
      1. Compute Z_fwd = relu(Z - theta_pos)  — same for all branches
      2. For each branch b: gather Z_fwd over conn_hh_b, apply AH suppression
      3. Combine: Z_struct = softmax(branch_weights) · [Z_struct_b for each b]
      4. Add reflection, clamp, L2-normalize

    AH suppression weights are per-branch (each branch has its own conn_hh
    so the neighbour cosine similarities differ).

    Parameters
    ----------
    base           : SGNNET_Resonant (carries theta, W_pos, alpha_reflect)
    branch_conns   : list of [N_hidden, K_hh] LongTensors, one per branch
    K_iter_outer   : number of sequential outer iterations (default 1)
    alpha_ahebb    : AH suppression strength
    """

    def __init__(
        self,
        base: SGNNET_Resonant,
        branch_conns: list,
        K_iter_outer: int = 1,
        alpha_ahebb: float = 1.0,
    ):
        super().__init__()
        self.m             = base
        self.K_iter_outer  = K_iter_outer
        self.alpha_ahebb   = alpha_ahebb
        n_branches         = len(branch_conns)

        # Learnable branch mixing weights — softmaxed before use
        self.branch_weights = nn.Parameter(
            torch.full((n_branches,), 1.0 / n_branches))

        # Register each conn_hh as a buffer (not learned, but device-portable)
        for i, conn in enumerate(branch_conns):
            self.register_buffer(f"conn_hh_{i}", conn)

        self._n_branches = n_branches

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def _get_conn(self, i: int) -> torch.Tensor:
        return getattr(self, f"conn_hh_{i}")

    def _precompute_supp(self, N_h: int) -> list:
        """Pre-compute static AH suppression weights for each branch."""
        W_n = F.normalize(self.m.W_pos[:N_h], dim=-1)  # [N_h, D]
        supp_list = []
        for i in range(self._n_branches):
            conn = self._get_conn(i)                         # [N_h, K_hh_i]
            pos_sim = (W_n.unsqueeze(1) * W_n[conn]).sum(-1) # [N_h, K_hh_i]
            supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                      ).unsqueeze(0).unsqueeze(-1)            # [1, N_h, K_hh_i, 1]
            supp_list.append(supp_w)
        return supp_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)  # [1, N, 1, 1]
        N_h       = self.m.base.N_hidden

        # Pre-compute AH suppression weights for all branches (static — outside loop)
        supp_list = self._precompute_supp(N_h)

        # Reflection accumulator — leaky memory of what relu suppressed each step
        Z_reflected = torch.zeros_like(Z)

        # Softmax branch mixing coefficients
        w = F.softmax(self.branch_weights, dim=0)   # [n_branches]

        for _ in range(self.K_iter_outer):
            Z_fwd = F.relu(Z - theta_pos)           # [B, N, D]

            # --- Parallel branch gathers (each branch uses same Z_fwd) ---
            Z_struct = torch.zeros_like(Z_fwd)
            for i in range(self._n_branches):
                conn   = self._get_conn(i)                   # [N_h, K_hh_i]
                Z_nb   = Z_fwd[:, conn, :]                   # [B, N, K_hh_i, D]
                Z_b    = (Z_nb * supp_list[i]).sum(dim=2)    # [B, N, D]
                Z_struct = Z_struct + w[i] * Z_b

            # Reflection
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _build_resonant(seed: int = SEED) -> SGNNET_Resonant:
    """Build the shared base (SmallWorld + Resonant wrapper)."""
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=5,      # K_iter used only for Ref; overridden in ParallelBranch
        K_local=K_l, K_random=K_r,
        n_groups=N_GROUPS, norm_mode="l2", encoding_mode="fourier")
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0)


def build_ref() -> nn.Module:
    """Standard K_iter=5 AntiHebbian reference."""
    resonant = _build_resonant()
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def build_parallel(branch_names: list, K_iter_outer: int, seed: int = SEED) -> nn.Module:
    """Build SGNNET_ParallelBranch with the requested branch topology names."""
    resonant = _build_resonant(seed)

    branch_conns = []
    for name in branch_names:
        K_l, K_r = BRANCH_DEFS[name]
        conn = _build_smallworld_conn(N, K_l, K_r, N_GROUPS, seed=seed)
        branch_conns.append(conn)

    return SGNNET_ParallelBranch(
        resonant, branch_conns, K_iter_outer=K_iter_outer, alpha_ahebb=ALPHA_AHEBB)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_config(key: str, model: nn.Module, tr, va) -> dict:
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_p:,}")

    model = model.to(DEVICE)
    kw      = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **kw)
    t0 = time.time()

    def _log(m):
        ep = m["epoch"] + 1
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)

    history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
    elapsed = time.time() - t0

    top1h = [round(h.get("val_top1", 0.), 4) for h in history]
    best  = max(top1h)
    bep   = int(np.argmax(top1h)) + 1

    # Extract final branch weights if available
    branch_w_final = None
    if hasattr(model, "branch_weights"):
        with torch.no_grad():
            branch_w_final = F.softmax(model.branch_weights, dim=0).cpu().tolist()
    elif hasattr(model, "m") and hasattr(model.m, "branch_weights"):
        with torch.no_grad():
            branch_w_final = F.softmax(model.m.branch_weights, dim=0).cpu().tolist()

    result = {
        "top1_best":      best,
        "top1_last":      top1h[-1],
        "best_epoch":     bep,
        "top1_history":   top1h,
        "elapsed_s":      round(elapsed, 1),
        "n_params":       n_p,
    }
    if branch_w_final is not None:
        result["branch_weights_final"] = [round(v, 4) for v in branch_w_final]

    print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")
    if branch_w_final is not None:
        branch_str = ", ".join(f"{v:.3f}" for v in branch_w_final)
        print(f"     branch_weights=[{branch_str}]")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    run_keys = list(CONFIG_SPECS.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"Step 701 — Parallel routing branches (Tier-0, {EPOCHS}ep, 50% data)")
    print(f"N={N}, D={D}, K_hh_total={K_HH}, AH={ALPHA_AHEBB}, reflect={ALPHA_REFLECT}")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    # Data loaders — 50% training split
    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n   = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr  = torch.utils.data.DataLoader(
        subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        print(f"\n{'─'*60}\nConfig {key}\n{'─'*60}")
        spec = CONFIG_SPECS[key]
        if spec is None:
            # Reference: standard sequential AH
            model = build_ref()
            print(f"  mode=sequential  K_iter=5  topology=mix")
        else:
            branch_names, K_iter_outer = spec
            model = build_parallel(branch_names, K_iter_outer)
            n_br  = len(branch_names)
            print(f"  mode=parallel  branches={branch_names}  K_iter_outer={K_iter_outer}")
            print(f"  topology_defs={[BRANCH_DEFS[b] for b in branch_names]}")

        results[key] = run_config(key, model, tr, va)

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    # Summary table
    ref_best = results.get("Ref", {}).get("top1_best", 0.0)
    print(f"\n{'='*70}")
    print(f"STEP 701 SUMMARY — Parallel branches vs K_iter=5 sequential")
    print(f"{'='*70}")
    print(f"  {'Config':<14}  {'Acc':>7}  {'Δ vs Ref':>10}  {'bep':>4}  {'Branch weights'}")
    print(f"  {'─'*14}  {'─'*7}  {'─'*10}  {'─'*4}  {'─'*28}")
    for key in run_keys:
        r     = results[key]
        delta = f"{r['top1_best'] - ref_best:+.4f}" if key != "Ref" else "  ---  "
        bw    = ""
        if "branch_weights_final" in r:
            bw = "[" + ", ".join(f"{v:.3f}" for v in r["branch_weights_final"]) + "]"
        print(f"  {key:<14}  {r['top1_best']:>7.4f}  {delta:>10}  "
              f"{r['best_epoch']:>4}  {bw}")

    print(f"\nInterpretation guide:")
    print(f"  B_par3_k1 ≥ Ref → sequential iters NOT needed; parallel branches sufficient")
    print(f"  A_par3_k2 ≥ Ref → 2 outer iters + parallel topology captures K_iter=5 info")
    print(f"  E_par5_k1 ≥ Ref → wide parallel sufficient; unlock full CUDA parallelism")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
