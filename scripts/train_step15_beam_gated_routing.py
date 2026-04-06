"""Step 15: Beam-gated sparse routing — states of non-beam neurons.

MECHANISM
=========
The beam selects the top-K most active neurons each routing step.
ONLY beam neurons conduct (through structural conn_hh) AND radiate (dynamic_z).
Neurons outside the beam neither send structural signals nor participate in radiation.

The question: what happens to NON-BEAM neurons between routing steps?

Three options for non-beam neuron state:

OPTION A — Reset (zero state):
  Non-beam neurons are zeroed at the end of each step.
  They're blank slates — can only receive signals from beam neurons next step.
  Creates maximum sparsity: ~6.25% of neurons (32/512) carry information per step.
  "Earn your way into the beam or your state is forgotten."
  Analogous to spiking neurons: fire only if top-K active; non-firing neurons reset.

OPTION B — Retain (accumulate without decay):
  Non-beam neurons keep their Z state unchanged.
  They receive new signals (from beam neurons conducting into them) but don't send.
  They can "charge up" over multiple steps until they enter the beam.
  Analogous to integrate-and-fire: accumulate evidence until threshold.
  Creates two-speed dynamics: fast (beam, updated each step) + slow (accumulates).

OPTION C — Retain with exponential decay:
  Non-beam: Z_new = alpha_retain * Z_old + (1-alpha_retain) * received_signal
  Beam: Z_new = F.normalize(received_signal)  (full update, same as current)
  Smooth memory with forgetting — old state fades, new signal blends in.

REFERENCE CONFIGS:
  Current model: ALL neurons conduct and radiate regardless of beam rank.
  Beam is only used for radiation. Conduction uses ALL neurons.

COMPARISON CONFIGS (all: D=16 Fourier N=512 dynamic_z_geo beam=32 120ep):
  Ref  — current model (all neurons conduct/radiate, beam only for radiation)
  A1   — beam-only conduct+radiate, non-beam RESET (hard sparsity)
  B1   — beam-only conduct+radiate, non-beam RETAIN (integrate-and-fire)
  C1   — beam-only conduct+radiate, non-beam DECAY alpha=0.9 (slow memory)
  C2   — beam-only conduct+radiate, non-beam DECAY alpha=0.5 (fast forgetting)
  A2   — beam-only conduct+radiate, non-beam RESET, beam_size=64 (more neurons active)
  B2   — beam-only conduct+radiate, non-beam RETAIN, beam_size=16 (fewer neurons active)

To reproduce:
    python -u scripts/train_step15_beam_gated_routing.py --device mps
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
from src.training.experiment_config import trainer_kwargs, topology_kwargs, run_metadata
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
N      = 512

print(f"Device: {DEVICE}  Epochs: {EPOCHS}  D={D}  N={N}  encoding=fourier")
print("Goal: beam-gated sparse routing — non-beam state options A/B/C")

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


class SGNNET_BeamGated(nn.Module):
    """Beam-gated routing where ONLY beam neurons conduct and radiate.

    Parameters
    ----------
    base_model    : SGNNET_Resonant backbone
    nonbeam_mode  : 'reset' | 'retain' | 'decay'
    alpha_retain  : decay factor for 'decay' mode (0=full reset, 1=full retain)
    beam_size_override : override base_model.beam_size if set
    """

    def __init__(
        self,
        base_model: SGNNET_Resonant,
        nonbeam_mode: str = "reset",
        alpha_retain: float = 0.9,
        beam_size_override: int | None = None,
    ):
        super().__init__()
        self.m               = base_model
        self.nonbeam_mode    = nonbeam_mode
        self.alpha_retain    = alpha_retain
        self.beam_size_eff   = beam_size_override if beam_size_override else base_model.beam_size

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

        conn_hh = self.m.base.conn_hh   # [N, K_hh]
        M = min(self.beam_size_eff, Z.shape[1])
        B, N_h, D = Z.shape

        for _ in range(self.m.base.K_iter):

            # ── 0. Select beam neurons ──────────────────────────────────
            activity  = Z.norm(dim=-1)                               # [B, N]
            top_idx   = activity.topk(M, dim=-1).indices             # [B, M]

            # Build beam mask: 1 for beam neurons, 0 for non-beam
            beam_mask = torch.zeros(B, N_h, device=Z.device)
            beam_mask.scatter_(1, top_idx, 1.0)                      # [B, N]

            # ── 1. Excitatory gate (beam neurons only) ──────────────────
            # Only beam neurons' signals propagate
            Z_beam_signal = Z * beam_mask.unsqueeze(-1)              # [B, N, D]
            Z_fwd = F.relu(Z_beam_signal - theta_pos)                # [B, N, D]

            # ── 2. Structural conduction — only beam neurons send ───────
            # Each neuron receives from its structural neighbors,
            # but only IF those neighbors are in the beam (signal = 0 otherwise)
            Z_struct = Z_fwd[:, conn_hh, :].sum(dim=2)              # [B, N, D]

            # ── 3. Dynamic radiation — beam-gated ───────────────────────
            # Reuse the existing dynamic_z_geo logic but with beam-masked Z
            Z_inhibitory = self.m._phase_inhibit(
                Z_beam_signal, W_ph_norm, theta_pos
            )

            # ── 4. Combine ───────────────────────────────────────────────
            Z_new_all = Z_struct + self.m.alpha_turing * Z_inhibitory
            Z_new_all = F.normalize(Z_new_all.clamp(min=-10, max=10), dim=-1)

            # ── 5. Apply non-beam neuron state policy ───────────────────
            if self.nonbeam_mode == "reset":
                # Non-beam neurons are zeroed — they only have what they received
                # from beam neurons (via structural connections above)
                # Beam neurons get the full normalized update
                Z_beam_updated   = Z_new_all * beam_mask.unsqueeze(-1)
                Z_nonbeam_update = Z_new_all * (1 - beam_mask).unsqueeze(-1)
                # Non-beam: only their RECEIVED signal (from beam conduction)
                # Beam: fresh Z_new
                Z = Z_beam_updated + Z_nonbeam_update * 0.0  # zero non-beam

            elif self.nonbeam_mode == "retain":
                # Non-beam neurons keep their OLD state unchanged
                # They don't send, but accumulate received signals
                Z_beam_updated    = Z_new_all * beam_mask.unsqueeze(-1)
                Z_nonbeam_retain  = Z * (1 - beam_mask).unsqueeze(-1)
                Z = Z_beam_updated + Z_nonbeam_retain

            elif self.nonbeam_mode == "decay":
                # Non-beam neurons: blend old state with received signal
                Z_beam_updated = Z_new_all * beam_mask.unsqueeze(-1)
                Z_nonbeam_old  = Z * (1 - beam_mask).unsqueeze(-1)
                Z_nonbeam_new  = Z_new_all * (1 - beam_mask).unsqueeze(-1)
                Z_nonbeam      = self.alpha_retain * Z_nonbeam_old + (1 - self.alpha_retain) * Z_nonbeam_new
                Z = Z_beam_updated + Z_nonbeam

            else:
                raise ValueError(f"Unknown nonbeam_mode: {self.nonbeam_mode!r}")

        return self.m.base._readout(Z)


def make_base_resonant(beam_size: int = 32) -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=tk["K_iter"],
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=beam_size,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )


def run(label: str, model: nn.Module, extra_meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr_loader, va_loader = get_loaders()
    tk = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
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
        "_meta":           run_metadata(
            __file__,
            {"D": D, "N": N, "epochs": EPOCHS, "data": DATA, **extra_meta}
        ),
    }
    print(f"  top1_best={best:.4f}  best_ep={result['best_epoch']}  "
          f"task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")
    return result


# (label, nonbeam_mode, alpha_retain, beam_override, extra_meta)
CONFIGS = [
    ("Ref. current model  — all neurons conduct+radiate  [baseline]",
     None, 0.0, 32, {"mode": "reference_current"}),
    ("A1. beam-gated RESET   beam=32  [hard sparsity]",
     "reset", 0.0, 32, {"nonbeam": "reset", "beam": 32}),
    ("A2. beam-gated RESET   beam=64  [more beam neurons]",
     "reset", 0.0, 64, {"nonbeam": "reset", "beam": 64}),
    ("B1. beam-gated RETAIN  beam=32  [integrate-and-fire]",
     "retain", 0.0, 32, {"nonbeam": "retain", "beam": 32}),
    ("B2. beam-gated RETAIN  beam=16  [fewer beam neurons]",
     "retain", 0.0, 16, {"nonbeam": "retain", "beam": 16}),
    ("C1. beam-gated DECAY   beam=32  alpha=0.9  [slow memory]",
     "decay", 0.9, 32, {"nonbeam": "decay", "alpha": 0.9, "beam": 32}),
    ("C2. beam-gated DECAY   beam=32  alpha=0.5  [fast forgetting]",
     "decay", 0.5, 32, {"nonbeam": "decay", "alpha": 0.5, "beam": 32}),
]

if __name__ == "__main__":
    get_loaders()
    n_train = len(_loaders[0].dataset)
    n_val   = len(_loaders[1].dataset)
    print(f"Dataset: train={n_train}  val={n_val}  (in-memory, {DATA})")

    results = {}
    keys = ["Ref", "A1", "A2", "B1", "B2", "C1", "C2"]
    for key, (label, mode, alpha, beam, extra) in zip(keys, CONFIGS):
        resonant = make_base_resonant(beam_size=beam).to(DEVICE)
        if mode is None:
            model = resonant
        else:
            model = SGNNET_BeamGated(resonant, nonbeam_mode=mode, alpha_retain=alpha)
        results[key] = run(label, model, extra)
        results[key]["nonbeam_mode"] = mode or "all_active"
        results[key]["beam_size"]    = beam

    out = Path("results/train_step15_beam_gated_routing.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref = results.get("Ref", {}).get("top1_best", 0.2922)
    print(f"\n-- Beam-gated sparse routing (ref={ref:.4f}) -------------------------")
    print("  %-60s  %7s  %9s  %+8s  %6s" %
          ("Config", "nonbeam", "top1_best", "vs_ref", "t(s)"))
    print("  " + "-"*95)
    for k, r in results.items():
        delta = r["top1_best"] - ref
        print("  %-60s  %7s  %9.4f  %+8.4f  %6.0f" % (
            k, r["nonbeam_mode"], r["top1_best"], delta, r["elapsed_s"]))
