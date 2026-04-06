"""Step 18: Signed coupling (binding-by-synchrony) + STDP causal excitation.

MECHANISM 1: Signed Coupling
=============================
Unified excitatory/inhibitory based on binding-by-synchrony (von der Malsburg 1981):
  cos(Z_h, Z_j) > 0 → same phase group → EXCITE
  cos(Z_h, Z_j) < 0 → different phase group → INHIBIT

Z_h += alpha * Σ_j cos(Z_h, Z_j) * Z_j     (positive = excite, negative = inhibit)

One mechanism replaces two separate pathways. No hyperparameter for exc/inh balance.
Sparse variant: top-32 positive (excite) + top-32 negative (inhibit) pairs only.

MECHANISM 2: STDP S2
=====================
Spike-Timing Dependent Plasticity analog: neurons in the beam at step k causally
excite neurons that become active and similar at step k+1.
Creates a 'routing agenda': early-step activations guide later-step activations.
No new learnable parameters.

Reference: step9A 29.22% (D=16 Fourier N=512 dynamic_z_geo 150ep)

All configs: D=16 Fourier N=512 dynamic_z_geo 120ep plateau store.h5 MPS.

To reproduce:
    python -u scripts/train_step18_signed_coupling.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_excitatory import SGNNET_SignedCoupling, SGNNET_STDP_S2
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset             import make_loaders

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


def make_base_resonant() -> SGNNET_Resonant:
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


class SGNNET_SignedSTDP(nn.Module):
    """Combined signed coupling + STDP S2 in a single routing loop."""

    def __init__(self, base: SGNNET_Resonant,
                 alpha_signed: float = 0.3, sparse_k: int = 32,
                 alpha_stdp: float = 0.2):
        super().__init__()
        self.m            = base
        self.alpha_signed = alpha_signed
        self.sparse_k     = sparse_k
        self.alpha_stdp   = alpha_stdp

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        B, N, D_  = Z.shape
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh
        M         = min(self.m.beam_size, N)
        Z_prev    = None

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            # Signed coupling
            Z_n   = F.normalize(Z, dim=-1)
            sim   = torch.bmm(Z_n, Z_n.transpose(1, 2))
            eye   = torch.eye(N, device=Z.device).unsqueeze(0)
            sim   = sim - 1e9 * eye
            K_    = min(self.sparse_k, N - 1)
            exc_v, exc_i = sim.topk(K_, dim=-1)
            inh_v, inh_i = (-sim).topk(K_, dim=-1)
            sparse = torch.zeros_like(sim)
            sparse.scatter_(-1, exc_i, exc_v.clamp(min=0))
            sparse.scatter_(-1, inh_i, (-inh_v).clamp(max=0))
            Z_coupled = torch.bmm(sparse, Z)

            Z_new = F.normalize(
                (Z_struct + self.m.alpha_turing * Z_inh + self.alpha_signed * Z_coupled)
                .clamp(-10, 10), dim=-1)

            # STDP
            if Z_prev is not None:
                idx_exp  = Z_prev.norm(dim=-1).topk(M, dim=-1).indices.unsqueeze(-1).expand(-1, -1, D_)
                Z_pb     = torch.gather(Z_prev, 1, idx_exp)
                cos_sim  = torch.bmm(F.normalize(Z_new, dim=-1),
                                     F.normalize(Z_pb, dim=-1).transpose(1, 2))
                Z_stdp   = torch.bmm(cos_sim.clamp(min=0), Z_pb)
                Z_new    = F.normalize((Z_new + self.alpha_stdp * Z_stdp).clamp(-10, 10), dim=-1)

            Z_prev = Z.clone()
            Z      = Z_new

        return self.m.base._readout(Z)


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0      = time.time()
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
        "_meta":           run_metadata(__file__, {"D": D, "N": N, "epochs": EPOCHS, **meta}),
    }
    print(f"  top1_best={best:.4f}  best_ep={result['best_epoch']}  "
          f"task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")
    return result


CONFIGS = [
    ("Ref. dynamic_z_geo  [baseline]",
     None, {}),
    ("A. Signed full  α=0.3  [all-pairs O(N²)]",
     lambda r: SGNNET_SignedCoupling(r, alpha_signed=0.3, sparse_k=0),
     {"mech": "signed_full", "alpha": 0.3}),
    ("B. Signed sparse  α=0.3  k=32  [top-32 exc + top-32 inh]",
     lambda r: SGNNET_SignedCoupling(r, alpha_signed=0.3, sparse_k=32),
     {"mech": "signed_sparse", "alpha": 0.3, "k": 32}),
    ("C. Signed sparse  α=0.1  k=32  [weaker coupling]",
     lambda r: SGNNET_SignedCoupling(r, alpha_signed=0.1, sparse_k=32),
     {"mech": "signed_sparse", "alpha": 0.1, "k": 32}),
    ("D. STDP S2  α=0.2  [causal cross-step excitation]",
     lambda r: SGNNET_STDP_S2(r, alpha_stdp=0.2),
     {"mech": "stdp_s2", "alpha": 0.2}),
    ("E. Signed sparse + STDP  α=0.3  α_stdp=0.2  [combined]",
     lambda r: SGNNET_SignedSTDP(r, alpha_signed=0.3, sparse_k=32, alpha_stdp=0.2),
     {"mech": "signed+stdp", "alpha_signed": 0.3, "alpha_stdp": 0.2}),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  D={D}  N={N}  encoding=fourier")
    print("Goal: signed coupling (exc+inh via cosine sign) + STDP causal excitation")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    keys = ["Ref", "A", "B", "C", "D", "E"]
    for key, (label, factory, meta) in zip(keys, CONFIGS):
        resonant = make_base_resonant().to(DEVICE)
        model    = resonant if factory is None else factory(resonant)
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = Path("results/train_step18_signed_coupling.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref = results.get("Ref", {}).get("top1_best", 0.2922)
    print(f"\n-- Signed coupling + STDP  (ref={ref:.4f}) ---------------------------")
    print("  %-60s  %9s  %+8s  %6s" % ("Config", "top1_best", "vs_ref", "t(s)"))
    print("  " + "-"*88)
    for k, r in results.items():
        d = r["top1_best"] - ref
        print("  %-60s  %9.4f  %+8.4f  %6.0f" % (k, r["top1_best"], d, r["elapsed_s"]))
