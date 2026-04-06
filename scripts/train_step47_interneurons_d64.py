"""Step 47: Comprehensive interneuron sweep at D=64 K_iter=8.

CONTEXT
=======
step20 tested interneurons at D=16 N=512 K_iter=3:
  50% interneurons readout=all        = 32.05% (+2.83pp)
  50% interneurons readout=interneurons = 27.98% (-1.24pp)
  25% interneurons readout=all        = 30.55% (+1.33pp)
  75% interneurons readout=all        = 29.12% (-0.10pp, neutral)

Winner: 50% frac, readout=all. But this was only at D=16 K_iter=3.
At D=64 K_iter=8, interneurons have 8 routing steps to integrate signal
(vs 3 at D=16), so higher interneuron fractions and interneuron-only readout
might work much better. The richer Fourier encoding at D=64 also gives
interneurons more signal to integrate through routing.

HYPOTHESIS
==========
At D=64 K_iter=8:
  1. 50% interneurons readout=all should compound with D=64 baseline
  2. readout=interneurons might NOW work (8 routing steps for integration)
  3. Higher fractions (75%) could work with K_iter=8 depth
  4. Interneurons + AntiHebb compound (AntiHebb α=0.5 = +11.6pp at D=64)

CONFIGS (D=64 N=1024 K_iter=8):
  Ref    no interneurons  [expect ~56.28%]
  A      25% interneurons  readout=all
  B      50% interneurons  readout=all      [step20 winner at D=16]
  C      75% interneurons  readout=all
  D      50% interneurons  readout=interneurons  [failed at D=16, retry with K_iter=8]
  E      75% interneurons  readout=interneurons  [pure routing integration]
  F      50% readout=all + AntiHebb α=0.5   [compound with strongest mechanism]
  G      50% readout=interneurons + AntiHebb α=0.5  [compound + compression]

To reproduce:
    python -u scripts/train_step47_interneurons_d64.py --device mps
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

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_resonant(N=1024, D=64, K_iter=8) -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk = topology_kwargs(N)
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


class SGNNET_Interneuron(nn.Module):
    """Interneuron fraction wrapper.

    n_input neurons receive input; remaining N-n_input are interneurons (zero seed).
    readout_from: 'all' (all neurons vote) or 'interneurons' (only interneurons vote).
    """

    def __init__(self, base_model: SGNNET_Resonant, n_input: int,
                 readout_from: str = "all"):
        super().__init__()
        self.m            = base_model
        self.n_input      = n_input
        self.readout_from = readout_from

        if readout_from == "interneurons":
            N_h = base_model.base.N_hidden
            mask = torch.zeros(N_h, dtype=torch.float32)
            mask[n_input:] = 1.0
            self.register_buffer("readout_mask", mask)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)                                      # [B, N, D]
        Z[:, self.n_input:, :] = 0.0                                  # interneurons blank

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, self.m.base.conn_hh, :].sum(2)
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
            Z        = F.normalize(
                (Z_struct + self.m.alpha_turing * Z_inh).clamp(-10, 10),
                dim=-1,
            )

        if self.readout_from == "interneurons":
            C_ho     = self.m.base.C_ho_mask.float()
            C_masked = C_ho * self.readout_mask.unsqueeze(1)
            A_out    = torch.einsum("bhd,ho->bod", Z, C_masked)
            W_out    = self.m.base.W_pos[self.m.base.N_hidden:]
            return (A_out * F.normalize(W_out, dim=-1).unsqueeze(0)).sum(dim=-1)
        else:
            return self.m.base._readout(Z)


class SGNNET_InterneuronAntiHebb(nn.Module):
    """Interneurons + Anti-Hebbian in one routing loop."""

    def __init__(self, base_model: SGNNET_Resonant, n_input: int,
                 readout_from: str = "all", alpha_ahebb: float = 0.5):
        super().__init__()
        self.m            = base_model
        self.n_input      = n_input
        self.readout_from = readout_from
        self.alpha_ahebb  = alpha_ahebb

        if readout_from == "interneurons":
            N_h = base_model.base.N_hidden
            mask = torch.zeros(N_h, dtype=torch.float32)
            mask[n_input:] = 1.0
            self.register_buffer("readout_mask", mask)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)
        Z[:, self.n_input:, :] = 0.0

        B, N, D   = Z.shape
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh
        W_pos_h   = self.m.W_pos[:N]

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            # Anti-Hebbian
            W_pos_nb = W_pos_h[conn_hh]
            pos_sim  = F.cosine_similarity(W_pos_h.unsqueeze(1), W_pos_nb, dim=-1)
            ahebb    = (pos_sim.unsqueeze(0).unsqueeze(-1) * Z_fwd[:, conn_hh, :]).sum(2)

            Z = F.normalize(
                (Z_struct + self.m.alpha_turing * Z_inh
                 - self.alpha_ahebb * ahebb).clamp(-10, 10),
                dim=-1,
            )

        if self.readout_from == "interneurons":
            C_ho     = self.m.base.C_ho_mask.float()
            C_masked = C_ho * self.readout_mask.unsqueeze(1)
            A_out    = torch.einsum("bhd,ho->bod", Z, C_masked)
            W_out    = self.m.base.W_pos[self.m.base.N_hidden:]
            return (A_out * F.normalize(W_out, dim=-1).unsqueeze(0)).sum(dim=-1)
        else:
            return self.m.base._readout(Z)


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
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
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
          f"  diag={result['convergence_diag']}  t={elapsed:.0f}s")
    return result


N = 1024

# (key, label, frac_input, readout, antihebb)
CONFIGS = [
    ("Ref", "Ref   D=64 N=1024 K_iter=8 no interneurons",       1.0,   "all",           False),
    ("A",   "A     25% interneurons  readout=all",                0.75,  "all",           False),
    ("B",   "B     50% interneurons  readout=all  [D=16 winner]", 0.50,  "all",           False),
    ("C",   "C     75% interneurons  readout=all",                0.25,  "all",           False),
    ("D",   "D     50% interneurons  readout=interneurons",       0.50,  "interneurons",  False),
    ("E",   "E     75% interneurons  readout=interneurons",       0.25,  "interneurons",  False),
    ("F",   "F     50% readout=all + AntiHebb α=0.5",            0.50,  "all",           True),
    ("G",   "G     50% readout=int + AntiHebb α=0.5",            0.50,  "interneurons",  True),
]


if __name__ == "__main__":
    REF_BASELINE = 0.5628
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print("Step 47: Comprehensive interneuron sweep at D=64 K_iter=8")
    print("step20 D=16 winner: 50% frac readout=all (+2.83pp)")
    print("Questions: does it compound at D=64? Does readout=interneurons work with K_iter=8?")
    print("           Does it stack with AntiHebb?")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    for key, label, frac_input, readout, use_ahebb in CONFIGS:
        resonant = make_resonant(N=N, D=64, K_iter=8).to(DEVICE)
        n_input  = int(N * frac_input)
        if frac_input >= 1.0:
            model = resonant
        elif use_ahebb:
            model = SGNNET_InterneuronAntiHebb(
                resonant, n_input=n_input, readout_from=readout, alpha_ahebb=0.5,
            ).to(DEVICE)
        else:
            model = SGNNET_Interneuron(
                resonant, n_input=n_input, readout_from=readout,
            ).to(DEVICE)

        meta = {"N": N, "D": 64, "K_iter": 8,
                "frac_input": frac_input, "n_input": n_input,
                "readout": readout, "antihebb": use_ahebb}
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step47_interneurons_d64.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    print(f"\n-- Interneuron sweep at D=64 K_iter=8 (ref={ref_val:.4f}) ---")
    print(f"  {'Config':<55}  {'top1':>6}  {'vs_Ref':>8}  {'best_ep':>8}  {'t(s)':>6}")
    print("  " + "-"*90)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        print(f"  {r['label'][:55]:<55}  {r['top1_best']:>6.4f}  {d:>+8.4f}  "
              f"{r.get('best_epoch', 0):>7d}    {r['elapsed_s']:>6.0f}")

    print("\n  Key questions answered:")
    print("  B > Ref?    → interneurons compound at D=64 (include in Gen4)")
    print("  D > step20? → readout=interneurons works with K_iter=8 depth")
    print("  C vs B      → 75% too many interneurons, or is more compression better?")
    print("  F > B?      → interneurons + AntiHebb stack")
    print("  G > D?      → compound stacking with interneuron-only readout")
    print("  F vs Ref    → combined gain of interneurons + AntiHebb at D=64")
