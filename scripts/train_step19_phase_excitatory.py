"""Step 19: W_phase as excitatory teleportation portals.

PROBLEM
=======
Current dynamic_z_geo uses radiation (dynamic Z beam) ONLY for inhibition.
W_phase builds a K-NN graph (conn_phase) rebuilt each epoch — but it drives
INHIBITORY routing in 'resonant' mode and is UNUSED in 'dynamic_z_geo' mode.

This experiment makes W_phase drive EXCITATORY long-range routing:
  Z_phase_exc = Z_fwd[:, conn_phase, :].sum(2)  # phase-similar neurons EXCITE each other

WHY THIS MATTERS
================
conn_hh (structural) = local K_hh=6 neighbours → 3 hops → diameter≈9 needed → bottleneck
conn_phase (W_phase)  = K_phase learned long-range partners → bypasses graph distance

W_phase teleportation: neuron h can directly excite/receive from its K_phase most
"phase-similar" neurons regardless of graph distance. With K_phase=8 at K_iter=3,
information can reach any neuron in 1-2 steps rather than 9.

Key: W_phase is LEARNED via gradient descent, so the teleportation graph adapts
to the task. This is distinct from fast-weight (step17) which is input-specific.

Reference: step9A 29.22% (D=16 Fourier N=512 dynamic_z_geo 150ep)

All configs: D=16 Fourier N=512 120ep plateau store.h5 MPS.

To reproduce:
    python -u scripts/train_step19_phase_excitatory.py --device mps
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


class SGNNET_PhaseExcitatory(nn.Module):
    """W_phase K-NN graph used for EXCITATORY routing.

    conn_phase (rebuilt by tick_epoch from W_phase similarity) provides
    learned long-range excitatory connections that bypass structural distance.
    The 'teleportation portal': any neuron can directly excite its K_phase
    most phase-similar partners in a single routing step.

    Parameters
    ----------
    base_model      : SGNNET_Resonant (any mode; we override routing)
    alpha_exc       : weight for phase excitatory contribution
    with_inhibition : if True, keep dynamic_z_geo inhibitory beam too
    """

    def __init__(self, base_model: SGNNET_Resonant, alpha_exc: float = 0.3,
                 with_inhibition: bool = True):
        super().__init__()
        self.m               = base_model
        self.alpha_exc       = alpha_exc
        self.with_inhibition = with_inhibition

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        self.m.tick_epoch()  # rebuilds conn_phase from W_phase K-NN each epoch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, self.m.base.conn_hh, :].sum(2)

            # W_phase teleportation: excite via learned long-range graph
            Z_exc = Z_fwd[:, self.m.conn_phase, :].sum(2)   # [B, N, D]

            if self.with_inhibition:
                Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
            else:
                Z_inh = torch.zeros_like(Z)

            Z_new = (Z_struct
                     + self.alpha_exc * Z_exc
                     + self.m.alpha_turing * Z_inh)
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


def make_resonant(K_phase: int = 8) -> SGNNET_Resonant:
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
        base=base, K_phase=K_phase, beam_size=32,
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


# (label, factory, meta)
CONFIGS = [
    ("Ref. dynamic_z_geo inhibitory only  [baseline]",
     lambda: make_resonant(8),
     {"mode": "ref_inhibitory_only"}),
    ("A. Phase exc K=8  α=0.3  inh=yes  [portal+inh]",
     lambda: SGNNET_PhaseExcitatory(make_resonant(8),  alpha_exc=0.3, with_inhibition=True),
     {"mode": "phase_exc", "K_phase": 8, "alpha": 0.3, "inh": True}),
    ("B. Phase exc K=8  α=0.3  inh=no   [portal only]",
     lambda: SGNNET_PhaseExcitatory(make_resonant(8),  alpha_exc=0.3, with_inhibition=False),
     {"mode": "phase_exc", "K_phase": 8, "alpha": 0.3, "inh": False}),
    ("C. Phase exc K=16 α=0.3  inh=yes  [wider portals]",
     lambda: SGNNET_PhaseExcitatory(make_resonant(16), alpha_exc=0.3, with_inhibition=True),
     {"mode": "phase_exc", "K_phase": 16, "alpha": 0.3, "inh": True}),
    ("D. Phase exc K=8  α=0.1  inh=yes  [weak portals]",
     lambda: SGNNET_PhaseExcitatory(make_resonant(8),  alpha_exc=0.1, with_inhibition=True),
     {"mode": "phase_exc", "K_phase": 8, "alpha": 0.1, "inh": True}),
    ("E. Phase exc K=4  α=0.3  inh=yes  [precise portals]",
     lambda: SGNNET_PhaseExcitatory(make_resonant(4),  alpha_exc=0.3, with_inhibition=True),
     {"mode": "phase_exc", "K_phase": 4, "alpha": 0.3, "inh": True}),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  D={D}  N={N}  encoding=fourier")
    print("Goal: W_phase as EXCITATORY teleportation portals (learned long-range routing)")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    keys = ["Ref", "A", "B", "C", "D", "E"]
    for key, (label, factory, meta) in zip(keys, CONFIGS):
        model = factory().to(DEVICE)
        results[key] = run(label, model, meta)
        if meta: results[key].update(meta)

    out = Path("results/train_step19_phase_excitatory.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref = results.get("Ref", {}).get("top1_best", 0.2922)
    print(f"\n-- Phase excitatory portals  (ref={ref:.4f}) -------------------------")
    print("  %-58s  %9s  %+8s  %6s" % ("Config", "top1_best", "vs_ref", "t(s)"))
    print("  " + "-"*85)
    for k, r in results.items():
        d = r["top1_best"] - ref
        print("  %-58s  %9.4f  %+8.4f  %6.0f" % (k, r["top1_best"], d, r["elapsed_s"]))
