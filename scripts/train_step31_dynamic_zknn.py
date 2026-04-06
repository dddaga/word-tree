"""Step 31: Dynamic K-NN on Z — input-dependent topology at O(N×K×D) cost.

PROBLEM
=======
Static W_phase K-NN (step24) only recovered ~37% of the N² signed-coupling gain
because W_phase is input-independent: the graph topology is fixed (or at most
epoch-updated) and does not respond to the current activation state Z.

The fully dynamic N²D baseline (step18 signed coupling) achieved +10.93pp over
the D=64 N=1024 K_iter=8 reference ceiling (56.28%). Goal: close that gap
without N².

HYPOTHESIS
==========
If we build the K-NN from the current Z at every routing step — which changes
per sample AND per routing iteration — we get fully input-dependent dynamic
topology at O(N×K×D) cost, not O(N²×D).

If this recovers >80% of the N² gain, the core SGNNET hypothesis is validated:
  O(N×K) connectivity is sufficient when topology is input-dependent.

CONFIGS
=======
Ref : D=64  N=1024  K_iter=8  no Z-KNN             [step22E base = 56.28%, sanity check]
A   : + Z-KNN  K=8   per step   alpha=1.0           [replace conn_hh with dynamic KNN]
B   : + Z-KNN  K=16  per step   alpha=1.0
C   : + Z-KNN  K=32  per step   alpha=1.0
D   : + Z-KNN  K=8   first step only  alpha=1.0     [dynamic per input, static across K_iter]
E   : + Z-KNN  K=8   per step   alpha=0.3           [additive — Z_struct + 0.3*Z_dyn]

KEY METRIC
==========
N2_gain = 10.93  # pp from signed coupling (O(N²)) over the D=64 ceiling reference
vs_N2_recovery = (result["top1_best"] - ref) / N2_gain * 100   # % of N² gain recovered

All configs: 150 epochs, SEED=42, BATCH=128.
"""

import argparse, json, time, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.training.experiment_config import trainer_kwargs, topology_kwargs, run_metadata
from src.training.trainer            import Trainer
from src.training.dataset            import make_loaders
from src.sgnnet.model_resonant       import SGNNET_Resonant
from src.sgnnet.model_smallworld     import SGNNET_SmallWorld


# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="mps")
DEVICE = parser.parse_args().device

EPOCHS = 150
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"

_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


# ── helpers ───────────────────────────────────────────────────────────────────
def make_resonant(N: int = 1024, D: int = 64, K_iter: int = 8) -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_iter,
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )


# ── Dynamic Z-KNN wrapper ─────────────────────────────────────────────────────

class SGNNET_DynamicZKNN(nn.Module):
    """Replace (or augment) structural routing with K-NN built from current Z.

    At each routing step the K nearest neighbours of every neuron are found by
    cosine similarity on the live activation state Z — topology is therefore
    fully input-dependent and changes every routing iteration and every sample.

    Parameters
    ----------
    base           : SGNNET_Resonant backbone
    K_dyn          : number of dynamic nearest neighbours
    first_step_only: if True, build the K-NN graph once before the routing loop
                     and reuse the SAME indices for all K_iter steps.
                     Ablates "dynamic per step" vs "dynamic per input but static
                     across routing steps".
    alpha_dyn      : when < 1.0, the dynamic signal is ADDED to Z_struct
                     (additive / signed-coupling style). When 1.0 Z_struct is
                     fully replaced by Z_dyn (pure dynamic topology).
    """

    def __init__(
        self,
        base: SGNNET_Resonant,
        K_dyn: int = 8,
        first_step_only: bool = False,
        alpha_dyn: float = 1.0,
    ):
        super().__init__()
        self.m              = base
        self.K_dyn          = K_dyn
        self.first_step_only = first_step_only
        self.alpha_dyn      = alpha_dyn

    # Trainer requires W_pos and W_phase to be accessible on the outermost module
    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    # ------------------------------------------------------------------
    def _build_dyn_knn(self, Z: torch.Tensor):
        """Compute dynamic K-NN indices and similarity weights from current Z.

        Returns
        -------
        top_k_idx  : LongTensor  [B, N, K_dyn]  — indices of K nearest neighbours
        top_k_sim  : FloatTensor [B, N, K_dyn]  — cosine similarity weights (≥0)
        """
        Z_n = F.normalize(Z, dim=-1)                            # [B, N, D]
        with torch.no_grad():
            # All-pairs cosine similarity: [B, N, N]
            sim = torch.bmm(Z_n, Z_n.transpose(1, 2))
            # Exclude self-similarity so a neuron does not attend to itself
            sim.diagonal(dim1=1, dim2=2).fill_(-1.0)
            # Top-K neighbours per neuron
            top_k_sim, top_k_idx = sim.topk(self.K_dyn, dim=-1) # [B, N, K_dyn]
            # Only positive similarity contributes (negative cosine ≡ opposing direction)
            top_k_sim = top_k_sim.clamp(min=0.0)
        return top_k_idx, top_k_sim

    def _apply_dyn_knn(
        self,
        Z: torch.Tensor,
        top_k_idx: torch.Tensor,
        top_k_sim: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate Z values from dynamic K-NN neighbours.

        Parameters
        ----------
        Z          : [B, N, D]  — current (raw, pre-relu) activation state
        top_k_idx  : [B, N, K_dyn]
        top_k_sim  : [B, N, K_dyn]  (non-negative)

        Returns
        -------
        Z_dyn : [B, N, D]  — weighted sum of dynamic neighbour activations
        """
        B, N, D = Z.shape

        # Gather dynamic neighbours: Z[b, top_k_idx[b,n,k], :] → [B, N, K_dyn, D]
        flat_idx   = top_k_idx.view(B, -1)                       # [B, N*K_dyn]
        flat_idx_d = flat_idx.unsqueeze(-1).expand(-1, -1, D)    # [B, N*K_dyn, D]
        Z_dyn_nb   = Z.gather(1, flat_idx_d).view(B, N, self.K_dyn, D)  # [B, N, K_dyn, D]

        # Weighted sum over dynamic neighbours
        Z_dyn = (Z_dyn_nb * top_k_sim.unsqueeze(-1)).sum(2)      # [B, N, D]
        return Z_dyn

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                          # [B, N, D]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1) # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)           # [N, D]
        conn_hh   = self.m.base.conn_hh                           # [N, K_hh]

        Z_reflected = torch.zeros_like(Z)

        # For first_step_only: build the K-NN graph once here, before the loop
        static_idx, static_sim = None, None
        if self.first_step_only:
            static_idx, static_sim = self._build_dyn_knn(Z)

        for _ in range(self.m.base.K_iter):

            # ── 1. Excitatory gate ─────────────────────────────────────
            Z_fwd = F.relu(Z - theta_pos)                         # [B, N, D]

            # ── 2. Dynamic K-NN structural excitation ─────────────────
            if self.first_step_only:
                # Graph built once (input-dependent but routing-step-static)
                top_k_idx, top_k_sim = static_idx, static_sim
            else:
                # Truly dynamic: rebuilt at every routing step
                top_k_idx, top_k_sim = self._build_dyn_knn(Z)

            Z_dyn = self._apply_dyn_knn(Z_fwd, top_k_idx, top_k_sim)

            if self.alpha_dyn < 1.0:
                # Additive mode: keep static structural signal and augment with dynamic
                Z_struct = Z_fwd[:, conn_hh, :].sum(2)            # [B, N, D]
                Z_excite = Z_struct + self.alpha_dyn * Z_dyn
            else:
                # Replacement mode: dynamic K-NN replaces static conn_hh entirely
                Z_excite = Z_dyn

            # ── 3. Self-inhibition reflection ─────────────────────────
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            # ── 4. Long-range phase inhibition (unchanged) ─────────────
            Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            # ── 5. Combine & normalise ─────────────────────────────────
            Z_new = Z_excite + Z_reflected + self.m.alpha_turing * Z_inh
            Z     = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ── Run helper ────────────────────────────────────────────────────────────────
def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0
    best    = max(h["val_top1"] for h in history)
    best_ep = max(range(len(history)), key=lambda i: history[i]["val_top1"]) + 1
    frac    = best_ep / EPOCHS
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{EPOCHS} ({frac:.0%})"
          f"  t={elapsed:.0f}s")
    return {
        "label": label, "top1_best": best, "best_ep": best_ep,
        "ep_frac": frac, "t": elapsed,
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
        **meta,
    }


# ── Configs ───────────────────────────────────────────────────────────────────
# (key, label, N, D, K_iter, factory, extra_meta)
CONFIGS = [
    (
        "Ref",
        "Ref.  D=64  N=1024  K_iter=8  no Z-KNN                   [step22E base = 56.28%]",
        1024, 64, 8,
        lambda r: r,
        {"K_dyn": 0, "first_step_only": False, "alpha_dyn": 1.0},
    ),
    (
        "A",
        "A.    + Z-KNN  K=8   per step   alpha=1.0   [replace conn_hh with dynamic KNN]",
        1024, 64, 8,
        lambda r: SGNNET_DynamicZKNN(r, K_dyn=8,  first_step_only=False, alpha_dyn=1.0),
        {"K_dyn": 8,  "first_step_only": False, "alpha_dyn": 1.0},
    ),
    (
        "B",
        "B.    + Z-KNN  K=16  per step   alpha=1.0",
        1024, 64, 8,
        lambda r: SGNNET_DynamicZKNN(r, K_dyn=16, first_step_only=False, alpha_dyn=1.0),
        {"K_dyn": 16, "first_step_only": False, "alpha_dyn": 1.0},
    ),
    (
        "C",
        "C.    + Z-KNN  K=32  per step   alpha=1.0",
        1024, 64, 8,
        lambda r: SGNNET_DynamicZKNN(r, K_dyn=32, first_step_only=False, alpha_dyn=1.0),
        {"K_dyn": 32, "first_step_only": False, "alpha_dyn": 1.0},
    ),
    (
        "D",
        "D.    + Z-KNN  K=8   first step only  alpha=1.0   [dynamic per input, static across K_iter]",
        1024, 64, 8,
        lambda r: SGNNET_DynamicZKNN(r, K_dyn=8,  first_step_only=True,  alpha_dyn=1.0),
        {"K_dyn": 8,  "first_step_only": True,  "alpha_dyn": 1.0},
    ),
    (
        "E",
        "E.    + Z-KNN  K=8   per step   alpha=0.3   [additive — Z_struct + 0.3*Z_dyn]",
        1024, 64, 8,
        lambda r: SGNNET_DynamicZKNN(r, K_dyn=8,  first_step_only=False, alpha_dyn=0.3),
        {"K_dyn": 8,  "first_step_only": False, "alpha_dyn": 0.3},
    ),
]


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}")
    print("Goal: dynamic K-NN on current Z recovers N² gain at O(N×K×D) cost?")
    print(f"Base D=64 N=1024 K_iter=8 ~56.28%.  N² signed-coupling gain = +10.93pp.")
    print()

    results  = {}
    ref_top1 = None   # D=64 K_iter=8 reference

    for key, label, N, D, K_iter, factory, extra_meta in CONFIGS:
        resonant = make_resonant(N=N, D=D, K_iter=K_iter).to(DEVICE)
        model    = factory(resonant).to(DEVICE)
        meta     = {
            "N": N, "D": D, "K_iter": K_iter,
            "mechanism": "dynamic_z_knn",
            **extra_meta,
        }
        results[key] = run(label, model, meta)
        if key == "Ref":
            ref_top1 = results[key]["top1_best"]

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = ROOT / "results" / "train_step31_dynamic_zknn.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    N2_gain = 10.93   # pp gain from signed coupling (O(N²))

    ref = results["Ref"]["top1_best"]

    print(f"\n-- Dynamic Z-KNN  (ref_D64={ref:.4f},  N²_gain={N2_gain:.2f}pp) "
          + "-" * 30)
    hdr = f"  {'Config':<80}  {'top1':>6}  {'vs_ref':>8}  {'N²_rec%':>8}  "     \
          f"{'ep_frac':>8}  {'t(s)':>6}"
    print(hdr)
    print("  " + "-" * 120)

    for key, label, N, D, K_iter, _, _ in CONFIGS:
        r       = results[key]
        t1      = r["top1_best"]
        vs_ref  = t1 - ref
        vs_N2_recovery = vs_ref / N2_gain * 100
        ep_frac = r["ep_frac"]
        t_s     = r["t"]
        print(
            f"  {label:<80}  {t1:.4f}  {vs_ref:>+8.4f}  {vs_N2_recovery:>7.1f}%  "
            f"{ep_frac:>7.1%}  {t_s:>6.0f}"
        )

    print(f"\n  Signed-coupling N² ceiling (step18): ~0.4015"
          f"  (N²_gain={N2_gain:.2f}pp over D=16 ref; base here is D=64 ~0.5628)")

    print("\n  Interpretation guide:")
    print("  If A/B/C > Ref by >8.7pp  (>80% N² gain recovered): O(N×K) IS sufficient")
    print("  If A > D:  rebuilding K-NN per routing step beats reusing (truly dynamic)")
    print("  If D > Ref: even input-dependent-but-step-static topology helps")
    print("  If E > A:  additive augmentation of conn_hh beats full replacement")
    print("  B vs A / C vs B: diminishing returns as K grows (K=32 approaches N²?)")
