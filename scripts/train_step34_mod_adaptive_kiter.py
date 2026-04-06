"""Step 34: Mixture-of-Depths (MoD) adaptive routing depth for SGNNET.

QUESTION
========
Fixed K_iter wastes compute: some neurons likely "converge" early (their
pre-normalisation signal magnitude is already large) while others need more
routing steps to form useful representations.

Adaptive routing: at each step, measure ||Z_new_pre|| per neuron.
Neurons that exceed exit_threshold are "confident" — they stop updating and
hold their current state.  Active neurons continue routing up to K_iter_max.

This is analogous to Mixture-of-Depths (Raposo et al. 2024) but applied
per-neuron within a single forward pass rather than per-token across layers.

CONFIGS
=======
Ref     D=64 N=1024 K_iter=8  base              [step22E ≈56.28%]
RefK3   D=64 N=1024 K_iter=3  base              [diagnostic: K_iter=3 cost at D=64, expect ~45%?]
A       MoD  K_iter_max=8   exit_threshold=1.5  [adaptive depth up to 8]
B       MoD  K_iter_max=8   exit_threshold=1.0  [stricter exit]
C       MoD  K_iter_max=8   exit_threshold=2.0  [more lenient exit]
D       MoD  K_iter_max=16  exit_threshold=1.5  [deeper max with early exit]

KEY QUESTIONS
=============
1. Does adaptive depth match fixed K_iter=8 (Ref)?
   If MoD_A ≈ Ref: early exit is "free" — we get K_iter=8 quality at lower average steps.
   If MoD_A < Ref: some neurons need all K_iter=8 steps; early exit is lossy.
2. Does threshold tuning matter? (B vs A vs C)
3. Does deeper max (D) with early exit outperform fixed K_iter=8?

For MoD configs: K_iter=3 in the SmallWorld base (unused by wrapper).
K_iter_max controls the actual loop.
For Ref: K_iter=8 in SmallWorld.
For RefK3: K_iter=3 in SmallWorld (no MoD wrapper).

All configs 150ep.

To reproduce:
    python -u scripts/train_step34_mod_adaptive_kiter.py --device mps
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


# ── Model factory ──────────────────────────────────────────────────────────────
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


# ── MoD wrapper ────────────────────────────────────────────────────────────────

class SGNNET_MoD(nn.Module):
    """Adaptive routing depth: neurons exit early when pre-norm magnitude exceeds threshold.

    At each routing step, compute Z_new (pre-normalization). Neurons with
    ||Z_new|| > exit_threshold have "converged" and keep their current Z unchanged.
    Active neurons continue updating.

    This gives neurons 1..K_iter_max routing steps but each neuron may use fewer.

    Parameters
    ----------
    base            : SGNNET_Resonant backbone
    K_iter_max      : maximum number of routing steps (hard cap)
    exit_threshold  : pre-norm magnitude above which a neuron exits the loop
    """
    def __init__(self, base: SGNNET_Resonant, K_iter_max: int = 8,
                 exit_threshold: float = 1.5):
        super().__init__()
        self.m              = base
        self.K_iter_max     = K_iter_max
        self.exit_threshold = exit_threshold

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                               # [B, N, D]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)      # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)                # [N, D]
        conn_hh   = self.m.base.conn_hh

        # Track which neurons are still "active" (not yet exited)
        # 1.0 = active, 0.0 = exited (frozen); shape [B, N, 1]
        active = torch.ones(Z.shape[0], Z.shape[1], 1, device=Z.device)

        for _ in range(self.K_iter_max):
            Z_fwd    = F.relu(Z - theta_pos)                           # [B, N, D]
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)                    # [B, N, D]
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
            Z_new_pre = Z_struct + self.m.alpha_turing * Z_inh        # pre-norm [B, N, D]

            # Compute pre-norm magnitude to determine confidence
            mag = Z_new_pre.norm(dim=-1, keepdim=True)                 # [B, N, 1]
            # Neurons that exceed threshold "exit" — freeze at current Z
            still_active = (mag < self.exit_threshold).float()
            active = active * still_active  # once exited, stays exited

            Z_new = F.normalize(Z_new_pre.clamp(-10, 10), dim=-1)     # [B, N, D]
            # Update only active neurons; exited neurons keep their Z
            Z = active * Z_new + (1.0 - active) * Z

            if active.sum() == 0:
                break  # all neurons exited

        return self.m.base._readout(Z)


# ── Run helper ─────────────────────────────────────────────────────────────────
def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va   = get_loaders()
    tk       = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="plateau")
    trainer  = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0       = time.time()
    history  = trainer.train(n_epochs=EPOCHS)
    elapsed  = time.time() - t0
    best     = max(h["val_top1"] for h in history)
    best_ep  = max(range(len(history)), key=lambda i: history[i]["val_top1"]) + 1
    frac     = best_ep / EPOCHS
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{EPOCHS} ({frac:.0%})"
          f"  t={elapsed:.0f}s")
    return {
        "label": label, "top1_best": best, "best_ep": best_ep,
        "ep_frac": frac, "t": elapsed,
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
        **meta,
    }


# ── Configs ────────────────────────────────────────────────────────────────────
# (key, label, N, D, K_iter_base, model_factory, extra_meta)
CONFIGS = [
    ("Ref",
     "Ref    D=64 N=1024 K_iter=8  no MoD             [step22E ≈56.28%]",
     1024, 64, 8,
     lambda r: r,
     {"K_iter_max": 8, "exit_threshold": None}),

    ("RefK3",
     "RefK3  D=64 N=1024 K_iter=3  no MoD             [diagnostic: K_iter=3 cost at D=64]",
     1024, 64, 3,
     lambda r: r,
     {"K_iter_max": 3, "exit_threshold": None}),

    ("A",
     "A      MoD K_iter_max=8  exit_threshold=1.5      [adaptive depth up to 8]",
     1024, 64, 3,
     lambda r: SGNNET_MoD(r, K_iter_max=8, exit_threshold=1.5),
     {"K_iter_max": 8, "exit_threshold": 1.5}),

    ("B",
     "B      MoD K_iter_max=8  exit_threshold=1.0      [stricter exit]",
     1024, 64, 3,
     lambda r: SGNNET_MoD(r, K_iter_max=8, exit_threshold=1.0),
     {"K_iter_max": 8, "exit_threshold": 1.0}),

    ("C",
     "C      MoD K_iter_max=8  exit_threshold=2.0      [more lenient exit]",
     1024, 64, 3,
     lambda r: SGNNET_MoD(r, K_iter_max=8, exit_threshold=2.0),
     {"K_iter_max": 8, "exit_threshold": 2.0}),

    ("D",
     "D      MoD K_iter_max=16 exit_threshold=1.5      [deeper max with early exit]",
     1024, 64, 3,
     lambda r: SGNNET_MoD(r, K_iter_max=16, exit_threshold=1.5),
     {"K_iter_max": 16, "exit_threshold": 1.5}),
]


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}")
    print("Goal: does adaptive per-neuron routing depth match fixed K_iter=8?")
    print("Ref ≈56.28% (D=64 N=1024 K_iter=8).  RefK3 = diagnostic (K_iter=3 cost at D=64).")
    print("MoD wraps a K_iter=3 base but overrides the loop with adaptive exit.")

    results   = {}
    ref_top1  = None
    refk3_top1 = None

    for key, label, N, D, K_iter_base, factory, extra in CONFIGS:
        resonant = make_resonant(N=N, D=D, K_iter=K_iter_base).to(DEVICE)
        model    = factory(resonant).to(DEVICE)
        meta     = {
            "N": N, "D": D, "K_iter_base": K_iter_base,
            "mechanism": "mod_adaptive_kiter",
            **extra,
        }
        results[key] = run(label, model, meta)
        if key == "Ref":
            ref_top1 = results[key]["top1_best"]
        if key == "RefK3":
            refk3_top1 = results[key]["top1_best"]

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = ROOT / "results" / "train_step34_mod_adaptive.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    ref   = results["Ref"]["top1_best"]
    refk3 = results["RefK3"]["top1_best"]

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n-- MoD Adaptive Routing Depth  (ref={ref:.4f}  refK3={refk3:.4f}) ------")
    print(f"  {'Config':<58}  {'top1':>6}  {'vs_Ref':>7}  {'vs_RefK3':>9}  "
          f"{'ep%':>5}  {'t(s)':>6}")
    print("  " + "-"*95)
    for key, label, N, D, K_iter_base, _, _ in CONFIGS:
        r        = results[key]
        vs_ref   = f"{r['top1_best'] - ref:+.4f}"  if key != "Ref"   else "  base"
        vs_refk3 = f"{r['top1_best'] - refk3:+.4f}" if key != "RefK3" else "  base"
        print(f"  {label:<58}  {r['top1_best']:.4f}  {vs_ref:>7}  {vs_refk3:>9}  "
              f"{r['ep_frac']:>4.0%}  {r['t']:>6.0f}")

    print(f"\n  Diagnostics:")
    print(f"  K_iter=8 base (Ref):           {ref:.4f}")
    print(f"  K_iter=3 base (RefK3):         {refk3:.4f}")
    print(f"  Depth gap (Ref - RefK3):       {ref - refk3:+.4f}pp")

    mod_a = results.get("A", {}).get("top1_best")
    if mod_a is not None:
        print(f"  MoD A (adaptive exit=1.5):     {mod_a:.4f}")
        print(f"  MoD vs Ref:                    {mod_a - ref:+.4f}pp")
        print(f"\n  If A ≈ Ref:   adaptive exit recovers fixed-depth quality — efficiency win")
        print(f"  If A < Ref:   neurons need all K_iter=8 steps; exit is lossy")
        print(f"  If A > Ref:   dynamic depth adds expressivity beyond fixed K_iter=8")
