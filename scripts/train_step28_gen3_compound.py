"""Step 28: Gen3 compound — stacking all Gen1 winners on the signed-coupling base.

GENETIC ALGORITHM PRINCIPLE
============================
Each generation compounds the winners of the previous generation:

  Gen1 individual winners (from steps 14-20):
    signed coupling full α=0.3   → +10.93pp (step18)
    phase excitatory K=8 α=0.3  →  +2.98pp (step19)
    interneurons 50% readout=all →  +2.83pp (step20)
    fast W_phase attention τ=0.25 → +1.84pp (step17)

  Gen2 compound (step23 running): signed + K_iter=8 + N=1024 → ~40-41%
    This becomes the Gen3 baseline.

  Gen3 (this script): Gen2 base + add remaining Gen1 winners one at a time,
    then in pairs, then all together.

Key question: do mechanisms ADD independently or INTERFERE when compounded?
If additive: Ref(40%) + phase_exc(+3%) + intern(+3%) + fast_W(+2%) → ~48%
If sublinear: each addition helps but gains compress → ~43-45%
If interference: some pairs hurt → prune and retry

CONFIGS
=======
  Ref   : signed α=0.3 + N=1024 + K_iter=8   [Gen2 best = locked-in base]
  A     : Ref + phase excitatory K=8 α=0.3   [+step19 winner]
  B     : Ref + interneurons 50% readout=all  [+step20 winner]
  C     : Ref + fast W_phase attention τ=0.25 [+step17 winner]
  D     : Ref + phase_exc + interneurons      [2-way compound]
  E     : Ref + phase_exc + fast_phase        [2-way compound]
  F     : Ref + interneurons + fast_phase     [2-way compound]
  G     : Ref + ALL THREE                     [FULL Gen3 compound]

All: D=64 N=1024 K_iter=3 Fourier dynamic_z_geo 150ep plateau store.h5.
Gen2 base expected: ~40-41% (from step23 best). Ceiling hunt: can we push past 43%?

To reproduce:
    python -u scripts/train_step28_gen3_compound.py --device mps
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

EPOCHS  = 150
BATCH   = 128
SEED    = 42
DATA    = "data/store.h5"
D       = 64
N       = 1024
K_ITER  = 3

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_resonant() -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_ITER,
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )


class SGNNET_Gen3Compound(nn.Module):
    """Gen3: signed coupling (always on) + optional phase_exc / interneurons / fast_W_phase.

    Signed coupling (binding-by-synchrony): all-pairs cos(Z_h,Z_j)*Z_j — the Gen1 star.
    Phase excitatory: K_phase K-NN positive-cosine excitation (dual of inhibition).
    Interneurons: n_seeded neurons receive input; rest start Z=0 and accumulate via routing.
    Fast W_phase: attention-weighted Hopfield adaptation of W_phase within each forward pass.
    """

    def __init__(
        self,
        base:           SGNNET_Resonant,
        alpha_signed:   float = 0.3,
        alpha_exc:      float = 0.3,
        alpha_fast:     float = 0.1,
        tau_fast:       float = 0.25,
        beam_fast:      int   = 32,
        n_seeded:       int | None = None,   # None = all seeded
        use_phase_exc:  bool = False,
        use_fast_phase: bool = False,
    ):
        super().__init__()
        self.m             = base
        self.alpha_signed  = alpha_signed
        self.alpha_exc     = alpha_exc
        self.alpha_fast    = alpha_fast
        self.tau_fast      = tau_fast
        self.beam_fast     = beam_fast
        self.n_seeded      = n_seeded
        self.use_phase_exc  = use_phase_exc
        self.use_fast_phase = use_fast_phase

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)               # [B, N, D]
        B, N_, D_ = Z.shape

        # Interneurons: neurons [n_seeded:] receive no direct input, start at zero
        if self.n_seeded is not None:
            mask = torch.ones(N_, 1, device=Z.device, dtype=Z.dtype)
            mask[self.n_seeded:] = 0.0
            Z = Z * mask.unsqueeze(0)                  # [B, N, D]

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)   # [1, N, 1]

        # Fast W_phase: per-forward-pass copy (not backpropagated through adaptation)
        W_phase_curr = self.m.W_phase.clone() if self.use_fast_phase else self.m.W_phase

        for _ in range(self.m.base.K_iter):
            W_ph_norm = F.normalize(W_phase_curr, dim=-1)            # [N, D]

            # 1. Structural excitation (local + random K-NN graph)
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, self.m.base.conn_hh, :].sum(dim=2)  # [B, N, D]

            # 2. Phase inhibition — dynamic_z_geo (uses Z similarity + W_pos geometry)
            Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            # 3. Signed coupling: binding-by-synchrony, all-pairs O(N²)
            #    cos>0 → excite; cos<0 → inhibit; unified in one term
            Z_n  = F.normalize(Z, dim=-1)                            # [B, N, D]
            sim  = torch.bmm(Z_n, Z_n.transpose(1, 2))              # [B, N, N]
            Z_sc = torch.bmm(sim, Z)                                 # [B, N, D]

            # 4. Phase excitatory: K_phase K-NN positive-cosine portal (step19 winner)
            Z_exc = torch.zeros_like(Z)
            if self.use_phase_exc:
                Z_nb  = Z[:, self.m.conn_phase, :]                   # [B, N, K, D]
                cos_p = (Z_n.unsqueeze(2) * F.normalize(Z_nb, dim=-1)).sum(-1).clamp(min=0)
                Z_exc = (cos_p.unsqueeze(-1) * Z_nb).sum(dim=2)     # [B, N, D]

            Z = F.normalize(
                (Z_struct
                 + self.m.alpha_turing * Z_inh
                 + self.alpha_signed   * Z_sc
                 + self.alpha_exc      * Z_exc
                ).clamp(-10, 10),
                dim=-1,
            )

            # 5. Fast W_phase: attention-weighted Hopfield update (step17 winner)
            #    W_phase_curr attracts toward currently active neuron directions.
            #    This adapts the phase graph per-sample without touching the parameter.
            if self.use_fast_phase:
                with torch.no_grad():
                    act   = Z.norm(dim=-1)                           # [B, N]
                    _, bi = act.topk(self.beam_fast, dim=-1)         # [B, beam]
                    Z_b   = Z.gather(1, bi.unsqueeze(-1).expand(-1, -1, D_))
                    Z_b_n = F.normalize(Z_b, dim=-1)                 # [B, beam, D]
                    W_d   = F.normalize(W_phase_curr, dim=-1)        # [N, D]
                    # Attention: alignment of each W_phase dir with each beam Z
                    logits = torch.einsum('nd,bmd->bnm', W_d, Z_b_n) / self.tau_fast
                    att    = F.softmax(logits, dim=-1)               # [B, N, beam]
                    delta  = torch.einsum('bnm,bmd->nd', att, Z_b_n) / B
                    W_phase_curr = F.normalize(
                        W_phase_curr + self.alpha_fast * delta, dim=-1)

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


# (label, use_phase_exc, n_seeded, use_fast_phase)
CONFIGS = [
    ("Ref  signed+K_iter=8+N=1024           [Gen2 base]",    False, None,   False),
    ("A    Ref + phase_exc K=8 α=0.3        [+step19 win]",  True,  None,   False),
    ("B    Ref + interneurons 50%            [+step20 win]",  False, N//2,   False),
    ("C    Ref + fast W_phase att τ=0.25    [+step17 win]",   False, None,   True),
    ("D    Ref + phase_exc + interneurons   [2-way]",         True,  N//2,   False),
    ("E    Ref + phase_exc + fast_phase     [2-way]",         True,  None,   True),
    ("F    Ref + interneurons + fast_phase  [2-way]",         False, N//2,   True),
    ("G    Ref + phase_exc + intern + fast  [FULL GEN3]",     True,  N//2,   True),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  D={D}  N={N}  K_iter={K_ITER}")
    print("Gen3 compound: stacking ALL Gen1 winners on signed-coupling base")
    print("Gen2 base (step23 est ~40-41%) | Gen1 add-ons: +2.98, +2.83, +1.84pp")
    print("Question: additive gains → ~48%? or sublinear → ~43-45%? or interference?")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    keys    = ["Ref", "A", "B", "C", "D", "E", "F", "G"]
    for key, (label, use_exc, n_seeded, use_fast) in zip(keys, CONFIGS):
        resonant = make_resonant().to(DEVICE)
        model    = SGNNET_Gen3Compound(
            resonant, alpha_signed=0.3, alpha_exc=0.3,
            alpha_fast=0.1, tau_fast=0.25, beam_fast=32,
            n_seeded=n_seeded,
            use_phase_exc=use_exc, use_fast_phase=use_fast,
        )
        meta = {
            "N": N, "K_iter": K_ITER, "D": D,
            "use_phase_exc": use_exc, "n_seeded": n_seeded, "use_fast_phase": use_fast,
            "alpha_signed": 0.3, "alpha_exc": 0.3, "alpha_fast": 0.1,
        }
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = Path("results/train_step28_gen3_compound.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref = results.get("Ref", {}).get("top1_best", 0.40)
    ref9 = 0.2922
    print(f"\n-- Gen3 compound sweep (gen2_base={ref:.4f}  step9_ref={ref9:.4f}) ---")
    print("  %-52s  %9s  %9s  %8s  %6s" % (
        "Config", "top1", "vs_base", "ep_frac", "t(s)"))
    print("  " + "-"*90)
    for k, r in results.items():
        d = r["top1_best"] - ref
        print("  %-52s  %9.4f  %+9.4f  %7.1f%%  %6.0f" % (
            r["label"][:52], r["top1_best"], d,
            r.get("best_epoch_frac", 0) * 100, r["elapsed_s"]))

    print("\n  Interpretation guide:")
    print("  d > +1pp per mechanism → additive (ideal)")
    print("  d ~ +0.5pp             → sublinear but still beneficial")
    print("  d ~ 0 or negative      → interference — remove that mechanism from Gen4 combo")
