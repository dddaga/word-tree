"""Step 36: Input-gated adjacency for SGNNET.

HYPOTHESIS
==========
Static K-NN topology (conn_hh) is fixed after construction — every neuron j always
sends to the same K neighbors regardless of the current input.  An input-dependent
gate can selectively open/close each j→h edge based on whether the receiving neuron h
is currently "interested" in neuron j's signal, measured by the cosine similarity
between Z_h and a learned gate vector W_gate[j].

Gate condition (soft):  gate = sigmoid((Z_h_norm · W_gate_norm[j] - θ) / τ)
Gate condition (hard):  gate = 1 if Z_h_norm · W_gate_norm[j] > θ else 0

Cost: O(N × K_hh × D) — same as the base routing loop (gates are evaluated only for
the K existing neighbors, not all N² pairs).

CONFIGS
=======
  Ref   D=64  N=1024  K_iter=8   baseline [step22E ≈56.28%]
  A     + soft gate  τ=1.0  θ=0.0   [default soft gating]
  B     + soft gate  τ=0.5  θ=0.0   [sharper gating]
  C     + soft gate  τ=2.0  θ=0.0   [smoother gating]
  D     + hard gate  θ=0.0          [binary mask — ablates differentiability]
  E     + hard gate  θ=0.3          [stricter binary gate threshold]

Key question: does learned input-gating on existing edges recover gain over static
              topology at D=64/N=1024?

To reproduce:
    python -u scripts/train_step36_input_gated.py --device mps
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


class SGNNET_InputGated(nn.Module):
    """Input-dependent gating on conn_hh edges.

    For each connection j→h in conn_hh, gate = sigmoid(Z_h · W_gate[j] / tau).
    Makes topology input-dependent at O(N×K×D) — same cost as base routing.

    gate_mode:
      'hard'  — binary gate (Z_h_norm · W_gate_norm[j] > gate_theta → open)
      'soft'  — sigmoid gate (differentiable, smoother gradient)
    """

    def __init__(self, base: SGNNET_Resonant, gate_mode: str = 'soft',
                 gate_theta: float = 0.0, tau: float = 1.0):
        super().__init__()
        self.m          = base
        self.gate_mode  = gate_mode
        self.gate_theta = gate_theta
        self.tau        = tau
        N = base.base.N_hidden
        D = base.base.D
        self.W_gate = nn.Parameter(torch.randn(N, D) * 0.1)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)          # [B, N, D]

        theta_pos  = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        W_ph_norm  = F.normalize(self.m.W_phase, dim=-1)            # [N, D]
        conn_hh    = self.m.base.conn_hh                            # [N, K_hh]

        # Pre-compute gate vectors for each neuron's K_hh neighbors (static shape)
        W_gate_norm = F.normalize(self.W_gate, dim=-1)  # [N, D]
        W_gate_nb   = W_gate_norm[conn_hh]              # [N, K_hh, D]

        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)               # [B, N, D]
            Z_nb  = Z_fwd[:, conn_hh, :]                # [B, N, K_hh, D]

            # Gate scores: dot(Z_h_norm, W_gate_nb[h, k]) for each (b, h, k)
            Z_n          = F.normalize(Z, dim=-1)        # [B, N, D]
            # gate_scores[b, h, k] = Z_n[b,h] · W_gate_nb[h,k]
            gate_scores  = (Z_n.unsqueeze(2) * W_gate_nb.unsqueeze(0)).sum(-1)  # [B, N, K_hh]

            if self.gate_mode == 'hard':
                gate = (gate_scores > self.gate_theta).float()
            else:  # soft sigmoid
                gate = torch.sigmoid((gate_scores - self.gate_theta) / self.tau)

            Z_struct = (Z_nb * gate.unsqueeze(-1)).sum(2)            # [B, N, D]
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
            Z = F.normalize(
                (Z_struct + self.m.alpha_turing * Z_inh).clamp(-10, 10),
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


# (label, N, D, K_iter, gate_mode, gate_theta, tau)
CONFIGS = [
    ("Ref  D=64 N=1024 K_iter=8  [baseline step22E ≈56.28%]",
     1024, 64, 8, None,   0.0, 1.0),
    ("A    + soft gate τ=1.0 θ=0.0   [default soft gating]",
     1024, 64, 8, 'soft', 0.0, 1.0),
    ("B    + soft gate τ=0.5 θ=0.0   [sharper gating]",
     1024, 64, 8, 'soft', 0.0, 0.5),
    ("C    + soft gate τ=2.0 θ=0.0   [smoother gating]",
     1024, 64, 8, 'soft', 0.0, 2.0),
    ("D    + hard gate θ=0.0         [binary — ablate diff'ability]",
     1024, 64, 8, 'hard', 0.0, 1.0),
    ("E    + hard gate θ=0.3         [stricter binary gate]",
     1024, 64, 8, 'hard', 0.3, 1.0),
]
KEYS = ["Ref", "A", "B", "C", "D", "E"]


if __name__ == "__main__":
    REF_BASELINE = 0.5628
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Seed: {SEED}")
    print("Step 36: Input-gated adjacency — learned gate per sending neuron")
    print(f"Ref baseline (step22E/D=64/N=1024/K_iter=8): {REF_BASELINE:.4f}")
    print("Question: does input-dependent gating on K-NN edges improve over static topology?")
    get_loaders()
    tr_ds = _loaders[0].dataset
    va_ds = _loaders[1].dataset
    print(f"Dataset: train={len(tr_ds)}  val={len(va_ds)}")

    results = {}
    for key, (label, N, D, K_iter, gate_mode, gate_theta, tau) in zip(KEYS, CONFIGS):
        resonant = make_resonant(N=N, D=D, K_iter=K_iter).to(DEVICE)
        if gate_mode is None:
            # Baseline: plain resonant model
            model = resonant
        else:
            model = SGNNET_InputGated(
                resonant, gate_mode=gate_mode, gate_theta=gate_theta, tau=tau,
            )
        meta = {
            "N": N, "D": D, "K_iter": K_iter,
            "gate_mode": gate_mode, "gate_theta": gate_theta, "tau": tau,
        }
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step36_input_gated.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    print(f"\n-- Input-gated adjacency sweep (ref={ref_val:.4f}) ---")
    print("  %-58s  %9s  %9s  %8s  %6s" % (
        "Config", "top1", "vs_Ref", "best_ep", "t(s)"))
    print("  " + "-"*100)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        print("  %-58s  %9.4f  %+9.4f  %7d    %6.0f" % (
            r["label"][:58], r["top1_best"], d,
            r.get("best_epoch", 0), r["elapsed_s"]))

    print("\n  Interpretation guide:")
    print("  d > +1pp  → input-gating helps; keep and scale")
    print("  d ~ 0     → gating adds no signal; topology is already near-optimal")
    print("  d < 0     → gating hurts (over-pruning edges); lower θ or increase τ")
    print("  hard vs soft gap → if hard ≈ soft: gate values polarise naturally")
