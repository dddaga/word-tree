"""Step 24: Sparse signed coupling via K-NN phase graph — O(N×K_phase) not O(N²).

MOTIVATION
==========
Step 18 showed that all-pairs signed coupling achieves 40.15% (+11.59pp), the biggest
win so far. BUT it costs O(N²) per routing step, which defeats the purpose of sparse
connectivity. The project's hard constraint is O(N × K) — sparsity must be preserved
for eventual hardware acceleration.

KEY INSIGHT
===========
The N² cost is NOT fundamental to signed coupling. The benefit comes from:
  cos(Z_h, Z_j) > 0 → neurons in phase EXCITE each other
  cos(Z_h, Z_j) < 0 → out-of-phase neurons INHIBIT each other

We can approximate this with K nearest neighbors in W_phase space:
  Z_coupled[h] = Σ_{j ∈ conn_phase[h]} cos(Z_h, Z_j) · Z_j

where conn_phase is the K-NN graph from W_phase (already computed, rebuilt each epoch).
Cost: O(N × K_phase × D) per routing step — same as the structural graph.

WHY THIS MIGHT WORK NEARLY AS WELL
===================================
W_phase is a LEARNED similarity metric. Over training it adapts to group neurons that
naturally become phase-correlated for class-relevant inputs. So K_phase=8 neighbors in
W_phase space ARE the most important coupling partners for each neuron — not random.
This is qualitatively different from random K-NN: it's a learned sparse approximation.

STEP 18 SPARSE FAILURE POST-MORTEM
====================================
Step 18's sparse variants (B/C/E) CRASHED at ep=1. The bug was in the scatter pattern:
  sparse.scatter_(-1, exc_i, exc_v.clamp(min=0))  ← in-place scatter breaks autograd
The fix here uses standard gather+sum (fully differentiable):
  Z_nb = Z[:, conn_phase, :]           # gather: [B, N, K_phase, D]
  cos_sim = (Z_n.unsqueeze(2) * normalize(Z_nb)).sum(-1)  # [B, N, K_phase]
  Z_coupled = (cos_sim.unsqueeze(-1) * Z_nb).sum(2)       # [B, N, D]

CONFIGS
=======
  Ref  : baseline dynamic_z_geo (no signed coupling)
  A    : sparse signed  K_phase=8   α=0.3  (use existing conn_phase)
  B    : sparse signed  K_phase=16  α=0.3  (wider phase K-NN)
  C    : sparse signed  K_phase=32  α=0.3  (32-NN approximation)
  D    : sparse signed  K_phase=64  α=0.3  (64-NN — close to full at N=512)
  E    : sparse signed  K_phase=8   α=0.1  (weaker coupling)
  F    : sparse signed  K_phase=8   α=0.3  + K_iter=8 (sparse + deep)

Configs A-E sweep K_phase to find the minimum K that captures the signed-coupling gain.
Config F tests sparse signed coupling at K_iter=8 — the most compute-efficient combo.

All: D=16 N=512 Fourier dynamic_z_geo 150ep plateau store.h5.
Reference: step18 full signed α=0.3 = 40.15% (O(N²) ceiling)

To reproduce:
    python -u scripts/train_step24_sparse_signed.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 150
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
D      = 16
N      = 512

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_resonant(K_phase: int = 8, K_iter: int = 3) -> SGNNET_Resonant:
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
        base=base, K_phase=K_phase, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )


class SGNNET_SparseSignedCoupling(nn.Module):
    """Sparse signed coupling using the W_phase K-NN graph.

    Cost: O(N × K_phase × D) per routing step.
    Method: gather K_phase neighbors from conn_phase, weight by cosine similarity,
            aggregate with sign (positive=excite, negative=inhibit).

    All ops use gather+sum — fully differentiable, no in-place scatter.
    """

    def __init__(self, base: SGNNET_Resonant, alpha_signed: float = 0.3):
        super().__init__()
        self.m            = base
        self.alpha_signed = alpha_signed

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()  # rebuilds conn_phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, self.m.base.conn_hh, :].sum(2)         # structural [B,N,D]
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)  # dynamic z inhibition

            # Sparse signed coupling via W_phase K-NN (O(N × K_phase))
            Z_n      = F.normalize(Z, dim=-1)                           # [B, N, D]
            Z_nb     = Z[:, self.m.conn_phase, :]                       # [B, N, K_phase, D]
            Z_nb_n   = F.normalize(Z_nb, dim=-1)                        # [B, N, K_phase, D]
            cos_sim  = (Z_n.unsqueeze(2) * Z_nb_n).sum(-1)             # [B, N, K_phase]
            # signed: positive cos → excite, negative → inhibit
            Z_coupled = (cos_sim.unsqueeze(-1) * Z_nb).sum(dim=2)      # [B, N, D]

            Z = F.normalize(
                (Z_struct
                 + self.m.alpha_turing * Z_inh
                 + self.alpha_signed * Z_coupled).clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0
    best    = max(h.get("val_top1", 0.0) for h in history)
    last5   = history[-5:]
    best_ep = int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1
    frac    = best_ep / len(history)
    result  = {
        "label": label, "top1_best": best,
        "top1_last": history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "best_epoch_frac": round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history": [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})  "
          f"diag={result['convergence_diag']}  t={elapsed:.0f}s")
    return result


# (label, K_phase, K_iter, alpha or None)
CONFIGS = [
    ("Ref. baseline dynamic_z_geo  [no signed coupling]",
     8,  3, None),
    ("A.  sparse-signed K_phase=8   α=0.3  K_iter=3  [lean O(N×8)]",
     8,  3, 0.3),
    ("B.  sparse-signed K_phase=16  α=0.3  K_iter=3  [O(N×16)]",
     16, 3, 0.3),
    ("C.  sparse-signed K_phase=32  α=0.3  K_iter=3  [O(N×32)]",
     32, 3, 0.3),
    ("D.  sparse-signed K_phase=64  α=0.3  K_iter=3  [O(N×64) near-dense]",
     64, 3, 0.3),
    ("E.  sparse-signed K_phase=8   α=0.1  K_iter=3  [weak coupling]",
     8,  3, 0.1),
    ("F.  sparse-signed K_phase=8   α=0.3  K_iter=8  [lean+deep]",
     8,  8, 0.3),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  D={D}  N={N}")
    print("Goal: O(N×K) sparse signed coupling — how much of step18's 40.15% can we recover?")
    print("Reference ceiling: step18 full all-pairs = 40.15%  (O(N²))")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    keys    = ["Ref", "A", "B", "C", "D", "E", "F"]
    for key, (label, K_phase, K_iter, alpha) in zip(keys, CONFIGS):
        resonant = make_resonant(K_phase=K_phase, K_iter=K_iter).to(DEVICE)
        model    = resonant if alpha is None else SGNNET_SparseSignedCoupling(resonant, alpha_signed=alpha)
        meta     = {"K_phase": K_phase, "K_iter": K_iter, "D": D, "N": N,
                    "alpha_signed": alpha if alpha is not None else 0.0,
                    "complexity": f"O(N×{K_phase})" if alpha is not None else "O(N×K_hh)"}
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = Path("results/train_step24_sparse_signed.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_full = 0.4015   # step18 full all-pairs ceiling
    ref_base = results.get("Ref", {}).get("top1_best", 0.2856)
    print(f"\n-- Sparse signed coupling sweep  (full_N²={ref_full:.4f}  base={ref_base:.4f}) ---")
    print("  %-52s  %9s  %9s  %9s  %8s  %6s" % (
        "Config", "top1", "vs_base", "vs_N²", "ep_frac", "t(s)"))
    print("  " + "-"*100)
    for k, r in results.items():
        d_base = r["top1_best"] - ref_base
        d_full = r["top1_best"] - ref_full
        print("  %-52s  %9.4f  %+9.4f  %+9.4f  %7.1f%%  %6.0f" % (
            r["label"][:52], r["top1_best"], d_base, d_full,
            r.get("best_epoch_frac", 0) * 100, r["elapsed_s"]))
    pct = lambda v: f"{100*(v-ref_base)/(ref_full-ref_base):.0f}%" if ref_full != ref_base else "n/a"
    print(f"\n  Recovery of N² gain: A={pct(results.get('A',{}).get('top1_best',0))}"
          f"  B={pct(results.get('B',{}).get('top1_best',0))}"
          f"  C={pct(results.get('C',{}).get('top1_best',0))}"
          f"  D={pct(results.get('D',{}).get('top1_best',0))}")
