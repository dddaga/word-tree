"""Step 14: Top-K gated conduction + excitatory radiation at D=16 Fourier.

ARCHITECTURE REDESIGN RATIONALE
================================

Current model has an asymmetry:
  Conduction (structural, conn_hh): ALL K_hh=6 neighbours contribute — no selection
  Radiation (dynamic Z, phase):     Top-beam_size by Z magnitude — inhibitory ONLY

Two problems:
  1. Conduction has no competitive gating — weak/irrelevant structural neighbours
     contribute as much as strong ones, diluting the signal
  2. Radiation only carries inhibitory signal — no dynamic excitatory pathway exists
     (W_phase tried to fill this role but failed because it learned a static graph)

MECHANISM 1: Top-K gated conduction
=====================================
Physics analogy: electrical conduction through a conductor only along paths where
the "impedance" (activation dissimilarity) is below threshold.

For each neuron h, rank its K_hh structural neighbours by current cosine similarity.
Only top-k_cond pass; the rest are gated off for this routing step.

k_cond=6 (all): current behaviour — backward compatible
k_cond=4: only the 4 most-resonant structural neighbours fire
k_cond=2: winner-takes-more within structural neighbourhood
k_cond=1: winner-takes-all

MECHANISM 2: Excitatory radiation (W_phase replacement)
=========================================================
Physics analogy: electromagnetic radiation — neurons "emit" in the direction of their
current activation, and receivers tuned to the same frequency absorb it (excitation).

For each neuron h, find the top-K_exc most DIRECTIONALLY SIMILAR neurons by current
normalised Z (not magnitude). These form transient excitatory connections:
  Z_h += alpha_exc * weighted_average(Z[similar neighbours])

This is dynamic_z for excitation (current dynamic_z is inhibitory). Directionally
similar neurons reinforce each other → soft clustering by activation direction.

Key difference from W_phase:
  W_phase: learns a static weight matrix → fixed graph baked in
  Excitatory radiation: rebuilt every forward pass from current Z → truly dynamic

EXPERIMENT CONFIGS
==================
All: D=16 Fourier N=512 dynamic_z_geo, 120ep, plateau, store.h5, MPS.
Reference: step9A full routing + geo 150ep = 29.22%

  A. Top-K cond=4  excrad=off  (conduction competitive, no new radiation)
  B. Top-K cond=2  excrad=off  (stronger competition in structural)
  C. Top-K cond=1  excrad=off  (winner-takes-all in structural)
  D. Top-K cond=4  excrad=on   beam_exc=8  alpha_exc=0.3
  E. Top-K cond=4  excrad=on   beam_exc=16 alpha_exc=0.3
  F. Top-K cond=6  excrad=on   beam_exc=16 alpha_exc=0.3  (excrad only, no cond change)
  G. Top-K cond=4  excrad=on   beam_exc=16 alpha_exc=0.1  (weaker excrad)

NOTE: This script adds two new parameters to SGNNET_Resonant forward():
  k_cond  (int): top-k within conn_hh for conduction. Default=None (use all = current)
  beam_exc (int): beam size for excitatory radiation. Default=0 (disabled)
  alpha_exc (float): weight for excitatory radiation contribution

These are implemented as local modifications inside the forward loop here to avoid
modifying model_resonant.py before the experiment validates the concept.

To reproduce:
    python -u scripts/train_step14_topk_cond_excrad.py --device mps
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant   import SGNNET_Resonant
from src.training.trainer        import Trainer
from src.training.experiment_config import trainer_kwargs, topology_kwargs
from src.training.dataset        import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args = parser.parse_args()

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 120
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
D      = 16

print(f"Device: {DEVICE}  Epochs: {EPOCHS}  D={D}  encoding=fourier")
print("Goal: top-K gated conduction + excitatory radiation (W_phase replacement)")

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


class SGNNET_TopKCond(nn.Module):
    """Wrapper around SGNNET_Resonant that overrides the routing forward pass
    to add top-K gated conduction and/or excitatory radiation.

    Parameters
    ----------
    base_model : trained/initialised SGNNET_Resonant
    k_cond     : top-k within structural conn_hh (None = all, current behaviour)
    beam_exc   : number of most similar neurons for excitatory radiation (0 = off)
    alpha_exc  : weight for excitatory radiation contribution
    """

    def __init__(
        self,
        base_model: SGNNET_Resonant,
        k_cond: int | None = None,
        beam_exc: int = 0,
        alpha_exc: float = 0.3,
    ):
        super().__init__()
        self.m          = base_model
        self.k_cond     = k_cond
        self.beam_exc   = beam_exc
        self.alpha_exc  = alpha_exc

    # Delegate parameter/buffer access to inner model
    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)   # [B, N, D]

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)   # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)

        Z_reflected = torch.zeros_like(Z)
        conn_hh = self.m.base.conn_hh   # [N, K_hh]
        K_hh    = conn_hh.shape[1]
        k_cond  = self.k_cond if self.k_cond is not None else K_hh

        for _ in range(self.m.base.K_iter):

            # ── 1. Excitatory gate ──────────────────────────────────────
            Z_fwd = F.relu(Z - theta_pos)               # [B, N, D]

            # ── 2. Top-K gated conduction ───────────────────────────────
            # Gather all K_hh structural neighbours
            Z_nb   = Z_fwd[:, conn_hh, :]               # [B, N, K_hh, D]

            if k_cond < K_hh:
                # Score each structural neighbour by cosine similarity to current Z
                Z_h_norm = F.normalize(Z, dim=-1)        # [B, N, D]
                Z_nb_norm = F.normalize(Z_nb, dim=-1)    # [B, N, K_hh, D]
                # Cosine sim: [B, N, K_hh]
                cond_sim = (Z_h_norm.unsqueeze(2) * Z_nb_norm).sum(-1)
                # Top-k_cond mask
                topk_idx = cond_sim.topk(k_cond, dim=-1).indices   # [B, N, k_cond]
                mask = torch.zeros_like(cond_sim)
                mask.scatter_(-1, topk_idx, 1.0)
                Z_struct = (Z_nb * mask.unsqueeze(-1)).sum(dim=2)   # [B, N, D]
            else:
                # No top-K filtering (all K_hh pass — original behaviour)
                Z_struct = Z_nb.sum(dim=2)               # [B, N, D]

            # ── 3. Excitatory radiation (dynamic Z-similarity based) ────
            if self.beam_exc > 0:
                Z_norm = F.normalize(Z, dim=-1)          # [B, N, D]
                # Pairwise cosine similarity: [B, N, N]
                sim_all = torch.bmm(Z_norm, Z_norm.transpose(1, 2))
                # Remove self-similarity
                eye = torch.eye(Z.shape[1], device=Z.device).unsqueeze(0)
                sim_all = sim_all - 1e9 * eye
                # Top beam_exc most directionally similar neurons
                exc_vals, exc_idx = sim_all.topk(self.beam_exc, dim=-1)  # [B, N, beam_exc]
                exc_vals = exc_vals.clamp(min=0)   # only positive resonance excites
                # Gather their activated Z vectors
                idx_exp = exc_idx.unsqueeze(-1).expand(-1, -1, -1, D)
                Z_exc_nb = Z_fwd.unsqueeze(1).expand(-1, Z.shape[1], -1, -1)
                Z_exc_nb = torch.gather(Z_exc_nb, 2, idx_exp)   # [B, N, beam_exc, D]
                exc_weight = exc_vals / (exc_vals.sum(-1, keepdim=True).clamp(min=1e-6))
                Z_radiate = (exc_weight.unsqueeze(-1) * Z_exc_nb).sum(dim=2)  # [B, N, D]
            else:
                Z_radiate = torch.zeros_like(Z)

            # ── 4. Self-inhibition reflection ───────────────────────────
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            # ── 5. Long-range phase inhibition (dynamic_z_geo) ──────────
            Z_inhibitory = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            # ── 6. Combine & normalise ───────────────────────────────────
            Z_new = (Z_struct +
                     self.alpha_exc * Z_radiate +
                     Z_reflected +
                     self.m.alpha_turing * Z_inhibitory)
            Z = F.normalize(Z_new, dim=-1)

        return self.m.base._readout(Z)


def make_base_resonant() -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk = topology_kwargs(512)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=512, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=tk["K_iter"],
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )


def run(label: str, model: nn.Module) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr_loader, va_loader = get_loaders()
    tk = trainer_kwargs(512, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr_loader, val_loader=va_loader,
                      device=DEVICE, **tk)
    t0 = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    best  = max(h.get("val_top1", 0.0) for h in history)
    last5 = history[-5:]
    result = {
        "label":           label,
        "top1_best":       best,
        "top1_last":       history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch":      int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1,
        "epochs_run":      len(history),
        "elapsed_s":       round(elapsed, 1),
        "top1_history":    [round(h.get("val_top1", 0.0), 4) for h in history],
    }
    print(f"  top1_best={best:.4f}  best_ep={result['best_epoch']}  "
          f"task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")
    return result


# (label, k_cond, beam_exc, alpha_exc)
CONFIGS = [
    # --- Top-K conduction only (no excitatory radiation) ---
    ("A. k_cond=4  excrad=off   [competitive conduction]",   4,  0, 0.0),
    ("B. k_cond=2  excrad=off   [strong cond competition]",  2,  0, 0.0),
    ("C. k_cond=1  excrad=off   [winner-take-all cond]",     1,  0, 0.0),
    # --- Excitatory radiation only (k_cond=6 = all) ---
    ("F. k_cond=6  excrad=16  alpha=0.3  [exc rad only]",    6, 16, 0.3),
    ("G. k_cond=6  excrad=16  alpha=0.1  [weak exc rad]",    6, 16, 0.1),
    # --- Combined top-K conduction + excitatory radiation ---
    ("D. k_cond=4  excrad=8   alpha=0.3  [cond+rad weak]",   4,  8, 0.3),
    ("E. k_cond=4  excrad=16  alpha=0.3  [cond+rad]",        4, 16, 0.3),
]

if __name__ == "__main__":
    get_loaders()
    n_train = len(_loaders[0].dataset)
    n_val   = len(_loaders[1].dataset)
    print(f"Dataset: train={n_train}  val={n_val}  (in-memory, {DATA})")

    results = {}
    for key, (label, k_cond, beam_exc, alpha_exc) in zip("ABCFGDE", CONFIGS):
        resonant = make_base_resonant().to(DEVICE)
        model = SGNNET_TopKCond(
            resonant, k_cond=k_cond, beam_exc=beam_exc, alpha_exc=alpha_exc
        )
        results[key] = run(label, model)
        results[key]["k_cond"]    = k_cond
        results[key]["beam_exc"]  = beam_exc
        results[key]["alpha_exc"] = alpha_exc

    out = Path("results/train_step14_topk_cond_excrad.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref = 0.2922  # step9A 150ep
    print(f"\n-- Top-K conduction + excitatory radiation (ref={ref:.4f}) ----------------")
    print("  %-55s  %6s  %7s  %9s  %+8s  %6s" %
          ("Config", "k_cond", "excrad", "top1_best", "vs_ref", "t(s)"))
    print("  " + "-"*95)
    for k, r in results.items():
        delta = r["top1_best"] - ref
        excrad = f"b={r['beam_exc']},a={r['alpha_exc']}" if r["beam_exc"] > 0 else "off"
        print("  %-55s  %6d  %7s  %9.4f  %+8.4f  %6.0f" % (
            k, r["k_cond"], excrad, r["top1_best"], delta, r["elapsed_s"]))

    print("\n  Top-K conduction effect (excrad=off):")
    for k in "ABC":
        if k in results:
            r = results[k]
            print(f"    k_cond={r['k_cond']}: {r['top1_best']:.4f}  (delta={r['top1_best']-ref:+.4f})")
    print("\n  Excitatory radiation effect (k_cond=6):")
    for k in "FG":
        if k in results:
            r = results[k]
            print(f"    beam_exc={r['beam_exc']} alpha={r['alpha_exc']}: {r['top1_best']:.4f}  "
                  f"(delta={r['top1_best']-ref:+.4f})")
