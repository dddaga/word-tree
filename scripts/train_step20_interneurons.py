"""Step 20: Interneuron fraction sweep at D=16.

MECHANISM
=========
Currently all N=512 neurons receive direct input (K_in=50 features each) AND
vote on the output. Two bottlenecks:
  1. Each neuron sees only 50/25088 = 0.2% of input — highly local receptive field.
  2. No compression bottleneck — input projections leak directly into readout.

Interneurons (frac_seeded < 1.0): first n_input neurons are seeded from input;
remaining interneurons start each forward pass at Z=0 and receive signal ONLY
via the routing graph (conn_hh + phase beam). They must integrate signals from
input neurons through routing to build class-relevant representations.

Two readout modes:
  readout=all:          all neurons (input + interneurons) vote on output (skip connection)
  readout=interneurons: only interneurons vote — input neurons excluded from output.
                        Forces compression through the routing bottleneck.

Hypothesis: 50% interneurons with readout=interneurons forces richer routing
integration, acting like a cortical column where sensory inputs drive interneurons
but the final classification comes from the integrating layer.

Reference: step9A 29.22% (D=16 Fourier N=512 dynamic_z_geo 150ep)
All configs: D=16 Fourier N=512 dynamic_z_geo 120ep plateau store.h5 MPS.

To reproduce:
    python -u scripts/train_step20_interneurons.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld     import SGNNET_SmallWorld
from src.sgnnet.model_resonant       import SGNNET_Resonant
from src.training.trainer            import Trainer
from src.training.experiment_config  import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset            import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 120
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


class SGNNET_Interneuron(nn.Module):
    """Interneuron fraction wrapper for SGNNET_Resonant.

    Parameters
    ----------
    base_model    : SGNNET_Resonant
    n_input       : number of neurons that receive input (first n_input)
                    remaining N - n_input are interneurons (zero seed)
    readout_from  : 'all' or 'interneurons'
    """

    def __init__(self, base_model: SGNNET_Resonant, n_input: int,
                 readout_from: str = "all"):
        super().__init__()
        self.m            = base_model
        self.n_input      = n_input
        self.readout_from = readout_from

        if readout_from == "interneurons":
            # Mask selects interneuron rows only in C_ho_mask [N, N_out]
            N_h = base_model.base.N_hidden
            mask = torch.zeros(N_h, dtype=torch.float32)
            mask[n_input:] = 1.0
            self.register_buffer("readout_mask", mask)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Seed all neurons, then zero interneurons
        Z         = self.m.base._seed(x)                              # [B, N, D]
        Z[:, self.n_input:, :] = 0.0                                  # interneurons blank

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, self.m.base.conn_hh, :].sum(2)
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
            Z_new    = Z_struct + self.m.alpha_turing * Z_inh
            Z        = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        if self.readout_from == "interneurons":
            # Mask out input neuron rows from C_ho, only interneurons vote
            C_ho     = self.m.base.C_ho_mask.float()
            C_masked = C_ho * self.readout_mask.unsqueeze(1)
            A_out    = torch.einsum("bhd,ho->bod", Z, C_masked)
            W_out    = self.m.base.W_pos[self.m.base.N_hidden:]
            return (A_out * F.normalize(W_out, dim=-1).unsqueeze(0)).sum(dim=-1)
        else:
            return self.m.base._readout(Z)


def make_resonant() -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
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
    result  = {
        "label": label, "top1_best": best,
        "top1_last": history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch": int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1,
        "epochs_run": len(history), "elapsed_s": round(elapsed, 1),
        "top1_history": [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta": run_metadata(__file__, {"D": D, "N": N, "epochs": EPOCHS, **meta}),
    }
    print(f"  top1_best={best:.4f}  best_ep={result['best_epoch']}  "
          f"task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")
    return result


# (label, n_input, readout_from, meta)
CONFIGS = [
    ("Ref. all=512 seeded  readout=all  [baseline]",
     N,          "all",           {"frac_seeded": 1.0, "readout": "all"}),
    ("A.  384/512 seeded  readout=all  [25% interneurons]",
     N * 3 // 4, "all",           {"frac_seeded": 0.75, "readout": "all"}),
    ("B.  256/512 seeded  readout=all  [50% interneurons]",
     N // 2,     "all",           {"frac_seeded": 0.50, "readout": "all"}),
    ("C.  256/512 seeded  readout=interneurons  [50% int, output=int only]",
     N // 2,     "interneurons",  {"frac_seeded": 0.50, "readout": "interneurons"}),
    ("D.  128/512 seeded  readout=interneurons  [75% int, output=int only]",
     N // 4,     "interneurons",  {"frac_seeded": 0.25, "readout": "interneurons"}),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  D={D}  N={N}  encoding=fourier")
    print("Goal: interneuron fraction sweep — does input→routing bottleneck help?")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    keys = ["Ref", "A", "B", "C", "D"]
    for key, (label, n_input, readout, meta) in zip(keys, CONFIGS):
        resonant = make_resonant().to(DEVICE)
        model    = resonant if n_input == N and readout == "all" \
                   else SGNNET_Interneuron(resonant, n_input=n_input, readout_from=readout)
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = Path("results/train_step20_interneurons.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref = results.get("Ref", {}).get("top1_best", 0.2922)
    print(f"\n-- Interneuron fraction sweep  (ref={ref:.4f}) -----------------------")
    print("  %-62s  %9s  %+8s  %6s" % ("Config", "top1_best", "vs_ref", "t(s)"))
    print("  " + "-"*90)
    for k, r in results.items():
        d = r["top1_best"] - ref
        print("  %-62s  %9.4f  %+8.4f  %6.0f" % (k, r["top1_best"], d, r["elapsed_s"]))
