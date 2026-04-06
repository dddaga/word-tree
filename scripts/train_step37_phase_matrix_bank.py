"""Step 37: Phase-queried D×D matrix bank for SGNNET.

HYPOTHESIS
==========
The structural message Z_struct = Σ_{j∈K_hh} Z_j collapses all incoming
activations into a single D-dimensional sum, losing cross-dimensional structure.
A D×D transformation applied per-neuron before summing could rotate/project
incoming messages into the receiver's preferred subspace.

The phase vector W_phase[j] ∈ S^{D-1} encodes the neuron's oscillatory identity.
A bank of M learned D×D matrices, selected by W_phase[j] via soft attention, lets
each neuron apply a *different* linear mixing depending on its current phase state:

    alpha_j  = softmax(W_phase[j] @ phase_keys.T / sqrt(D))    # [M]  attention over bank
    W_eff_j  = einsum('m,mdc->dc', alpha_j, W_bank)             # [D,D] weighted mix
    Z_mix_j  = W_eff_j @ Z_struct_j                             # [D]   cross-dim msg

Cost: O(N × M × D²) to compute W_eff per neuron + O(B × N × D²) for matmul.
  At N=1024 M=16 D=64 B=128: ~67M + ~537M FLOPs per routing step — ~3×base cost.

Config D ablates phase-conditioning by using a single fixed W [D×D] instead of the
bank, isolating whether phase-querying adds value over plain cross-dim mixing.

CONFIGS
=======
  Ref   D=64  N=1024  K_iter=8   baseline (plain routing, no matrix bank)
  A     + phase-queried W_bank  M=4    [small bank]
  B     + phase-queried W_bank  M=16   [main hypothesis]
  C     + phase-queried W_bank  M=64   [large bank — overfit risk]
  D     + fixed single W [D×D]         [ablate phase-conditioning; pure cross-dim mixing]

Parameters added:
  A: 4×64²  + 4×64   =  16,640
  B: 16×64² + 16×64  =  66,560   (≈67k)
  C: 64×64² + 64×64  = 266,240
  D: 64×64            =   4,096

Key questions:
  1. Does any cross-dim mixing (D) improve over base?
  2. Does phase-querying (B vs D) add value on top of mixing?
  3. Does bank size matter (A vs B vs C)?

To reproduce:
    python -u scripts/train_step37_phase_matrix_bank.py --device mps
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


class SGNNET_PhaseMatrixBank(nn.Module):
    """Phase-queried D×D matrix bank.

    A bank of M D×D matrices is soft-selected via the source neuron's W_phase
    direction. The selected matrix mixes dimensions of the structural message
    Z_struct before it enters the routing update.

        alpha[n]  = softmax(W_phase[n] @ phase_keys.T / sqrt(D))   # [M]
        W_eff[n]  = einsum('m,mdc->dc', alpha[n], W_bank)           # [D,D]
        Z_mix     = einsum('ndc,bnd->bnc', W_eff, Z_struct)         # [B,N,D]

    Parameters
    ----------
    base_model  : SGNNET_Resonant
    n_matrices  : M — number of matrices in bank (4, 16, or 64)
    fixed_W     : if True, use a single fixed W [D×D] (ablate phase-conditioning)
    """

    def __init__(
        self,
        base_model: SGNNET_Resonant,
        n_matrices: int = 16,
        fixed_W: bool = False,
    ):
        super().__init__()
        self.m          = base_model
        self.n_matrices = n_matrices
        self.fixed_W    = fixed_W

        D = base_model.base.D
        if fixed_W:
            # Single shared D×D matrix, no phase-querying
            self.W_single = nn.Parameter(torch.eye(D) + torch.randn(D, D) * 0.01)
        else:
            M = n_matrices
            # Bank of M D×D matrices, init near identity to preserve signal
            eye = torch.eye(D).unsqueeze(0).expand(M, -1, -1)
            self.W_bank     = nn.Parameter(eye + torch.randn(M, D, D) * 0.01)
            # Learned keys for phase attention: [M, D]
            self.phase_keys = nn.Parameter(torch.randn(M, D) * (D ** -0.5))

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                              # [B, N, D]
        B, N, D   = Z.shape
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)     # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)               # [N, D]
        conn_hh   = self.m.base.conn_hh                               # [N, K_hh]

        if not self.fixed_W:
            # Pre-compute per-neuron matrix blend (static across routing steps;
            # W_phase is a slow learned weight, not updated within forward pass)
            keys_norm = F.normalize(self.phase_keys, dim=-1)          # [M, D]
            # alpha[n] = softmax(W_phase[n] @ keys.T / sqrt(D))
            attn_logits = (W_ph_norm @ keys_norm.T) / (D ** 0.5)     # [N, M]
            alpha       = F.softmax(attn_logits, dim=-1)              # [N, M]
            # W_eff[n] = sum_m alpha[n,m] * W_bank[m]
            W_eff = torch.einsum('nm,mdc->ndc', alpha, self.W_bank)   # [N, D, D]

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)                          # [B, N, D]
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)                   # [B, N, D]
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            # Apply per-neuron matrix to structural message
            if self.fixed_W:
                # Z_mix[b,n] = W_single @ Z_struct[b,n]
                Z_mix = torch.einsum('dc,bnd->bnc', self.W_single, Z_struct)
            else:
                # Z_mix[b,n] = W_eff[n] @ Z_struct[b,n]
                Z_mix = torch.einsum('ndc,bnd->bnc', W_eff, Z_struct)

            Z = F.normalize(
                (Z_mix + self.m.alpha_turing * Z_inh).clamp(-10, 10),
                dim=-1,
            )

        return self.m.base._readout(Z)


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="plateau")
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


# (key, label, n_matrices, fixed_W)
CONFIGS = [
    ("Ref", "Ref  D=64 N=1024 K_iter=8  [baseline step22E ≈56.28%]",     None, False),
    ("A",   "A    + phase-queried W_bank M=4    [small bank]",              4,   False),
    ("B",   "B    + phase-queried W_bank M=16   [main hypothesis]",        16,   False),
    ("C",   "C    + phase-queried W_bank M=64   [large bank]",             64,   False),
    ("D",   "D    + fixed single W [D×D]        [ablate phase-conditioning]", 1, True),
]


if __name__ == "__main__":
    REF_BASELINE = 0.5628
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Seed: {SEED}")
    print("Step 37: Phase-queried D×D matrix bank")
    print(f"Ref baseline (step22E / D=64 / N=1024 / K_iter=8): {REF_BASELINE:.4f}")
    print("Questions:")
    print("  1. Does any cross-dim mixing (D: fixed W) beat plain routing?")
    print("  2. Does phase-querying (B vs D) add value?")
    print("  3. Optimal bank size: M=4 vs M=16 vs M=64?")
    get_loaders()
    tr_ds = _loaders[0].dataset
    va_ds = _loaders[1].dataset
    print(f"Dataset: train={len(tr_ds)}  val={len(va_ds)}")

    results = {}
    for key, label, n_mats, fixed_W in CONFIGS:
        resonant = make_resonant(N=1024, D=64, K_iter=8).to(DEVICE)
        if n_mats is None:
            model = resonant
        else:
            model = SGNNET_PhaseMatrixBank(
                resonant, n_matrices=n_mats, fixed_W=fixed_W,
            ).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        meta = {
            "N": 1024, "D": 64, "K_iter": 8,
            "n_matrices": n_mats, "fixed_W": fixed_W,
            "n_params": n_params,
        }
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step37_phase_matrix_bank.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    print(f"\n-- Phase-queried matrix bank sweep (ref={ref_val:.4f}) ---")
    print("  %-58s  %9s  %9s  %7s  %8s  %6s" % (
        "Config", "top1", "vs_Ref", "params", "best_ep", "t(s)"))
    print("  " + "-"*105)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        print("  %-58s  %9.4f  %+9.4f  %7d  %7d    %6.0f" % (
            r["label"][:58], r["top1_best"], d,
            r.get("n_params", 0), r.get("best_epoch", 0), r["elapsed_s"]))

    print("\n  Interpretation guide:")
    print("  D wins       → cross-dim mixing helps regardless of phase; add W [D×D] to base")
    print("  B > D        → phase-conditioning adds value beyond plain mixing; matrix bank works")
    print("  A ≈ B ≈ C    → bank size doesn't matter much; M=4 is sufficient")
    print("  C >> B       → expressivity matters; use larger bank")
    print("  all ~ Ref    → D×D mixing doesn't help at this scale; routing captures enough")
