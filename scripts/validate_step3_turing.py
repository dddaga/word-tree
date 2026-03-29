"""Step 3 validation: two-scale Turing mechanism.

Local excitatory (Z_fwd = relu(Z - theta)) via structural conn_hh.
Long-range inhibitory (Z_ref = -relu(-(Z + theta))) via phase conn_phase.

Hypothesis: the competition between local excitation and global inhibition
produces non-overlapping feature detectors (Turing pattern formation).

Compares at N=512:
  A. Baseline (pass-through, no threshold — current)
  B. Threshold only: Z_fwd = relu(Z - theta), no reflection
  C. Full two-scale: Z_fwd local + Z_ref phase (Turing)
  D. Learnable theta (per-neuron threshold)

Pass criterion: C > A or B by ≥ 1% top-1.
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.training.trainer import Trainer
from src.training.experiment_config import trainer_kwargs, topology_kwargs

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=60)
args = parser.parse_args()

if args.device == "auto":
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else:
    DEVICE = torch.device(args.device)

EPOCHS   = args.epochs
BATCH    = 128
N_HIDDEN = 512
print(f"Device: {DEVICE}  Epochs: {EPOCHS}")


def load_data():
    with h5py.File("data/store.h5", "r") as f:
        return (
            torch.from_numpy(f["train/features"][:]),
            torch.from_numpy(f["train/soft_labels"][:]),
            torch.from_numpy(f["train/labels"][:]).long(),
            torch.from_numpy(f["val/features"][:]),
            torch.from_numpy(f["val/soft_labels"][:]),
            torch.from_numpy(f["val/labels"][:]).long(),
        )

def make_loaders(data):
    tf, tsl, tl, vf, vsl, vl = data
    tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tf, tsl, tl),
        batch_size=BATCH, shuffle=True, num_workers=0,
    )
    va = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(vf, vsl, vl),
        batch_size=BATCH, num_workers=0,
    )
    return tr, va


class TuringSmallWorld(nn.Module):
    """SmallWorld with two-scale Turing mechanism.

    mode:
      'baseline'   — original pass-through (Z propagates as-is)
      'threshold'  — only Z_fwd = relu(Z - theta) propagates locally
      'turing'     — Z_fwd local (excitatory) + Z_ref global inhibitory via phase
      'learnable'  — same as turing but theta is a learnable per-neuron parameter
      'reflection' — Z_fwd propagates forward; relu remainder reflects back to
                     source neuron and is added to its next-iteration state.
                     Negative "memory" accumulates on the neuron, gating future
                     incoming signals. No long-range phase needed.
    """

    def __init__(self, base: SGNNET_SmallWorld, mode: str = "turing",
                 K_phase: int = 8, theta: float = 0.1, alpha: float = 0.3):
        super().__init__()
        self.base    = base
        self.mode    = mode
        self.K_phase = K_phase
        self.alpha   = alpha
        N = base.N_hidden
        D = base.W_pos.shape[1]

        self.W_phase = nn.Parameter(torch.rand(N, D))

        if mode == "learnable":
            # Per-neuron threshold, initialized to fixed theta
            self.theta = nn.Parameter(torch.full((N,), theta))
        else:
            self.theta = theta

        self._build_phase_graph()

    def _build_phase_graph(self):
        with torch.no_grad():
            Wp = F.normalize(self.W_phase.detach(), dim=-1)
            sim = Wp @ Wp.T
            sim.fill_diagonal_(-1e9)
            _, idx = sim.topk(self.K_phase, dim=-1)
        self.register_buffer("conn_phase", idx)

    @property
    def W_pos(self): return self.base.W_pos

    def tick_epoch(self):
        self._build_phase_graph()
        if hasattr(self.base, "tick_epoch"):
            self.base.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        N = self.base.N_hidden

        Z = self.base._seed(x)   # [B, N, D]

        theta = self.theta
        if self.mode == "learnable":
            theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)   # [1, N, 1]
        else:
            theta_pos = theta

        W_ph_norm = F.normalize(self.W_phase, dim=-1)

        Z_reflected = torch.zeros_like(Z)  # accumulated self-reflection across iters

        for _ in range(self.base.K_iter):

            if self.mode == "baseline":
                Z_fwd = Z
            else:
                Z_fwd = F.relu(Z - theta_pos)                 # excitatory: clearly present

            # Local structural aggregation (excitatory)
            Z_struct = Z_fwd[:, self.base.conn_hh, :].sum(dim=2)   # [B, N, D]

            if self.mode == "reflection":
                # What relu discarded: the negative remainder
                # Z_remainder[h] = -(what couldn't propagate forward)
                # Reflects back onto the source neuron — self-inhibition
                Z_remainder = Z_fwd - Z                       # = relu(Z-θ) - Z ≤ 0
                Z_reflected  = self.alpha * Z_reflected + Z_remainder  # leaky accumulation

                # Combine: forward neighbours + self-reflection
                # Reflection gates what the neuron accepts next iteration
                Z_new = Z_struct + Z_reflected

            elif self.mode in ("turing", "learnable"):
                # Reflective (inhibitory) signal: strongly negative activations
                Z_ref = -F.relu(-(Z + theta_pos))             # [B, N, D], ≤ 0

                # Beam: only top-M active neurons broadcast their reflection
                M = min(64, N)
                activity = Z.norm(dim=-1)                     # [B, N]
                top_idx  = activity.topk(M, dim=-1).indices   # [B, M]

                # For each batch: gather Z_ref of top-M active neurons
                Z_ref_beam = torch.gather(
                    Z_ref, 1,
                    top_idx.unsqueeze(-1).expand(-1, -1, Z.shape[-1])
                )  # [B, M, D]

                # Phase collapse: project received signal onto W_phase direction
                score = torch.einsum("bmd,nd->bmn", Z_ref_beam, W_ph_norm)   # [B, M, N]
                gate  = score.clamp(min=0)
                Z_inhibitory = torch.einsum("bmn,bmd->bnd", gate, Z_ref_beam)  # [B, N, D]
                gate_sum = gate.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1.0)
                Z_inhibitory = Z_inhibitory / gate_sum.squeeze(1)

                Z_new = Z_struct + self.alpha * Z_inhibitory
            else:
                Z_new = Z_struct

            Z = F.normalize(Z_new, dim=-1)

        return self.base._readout(Z)


def run(label: str, model, data) -> dict:
    print(f"\n{'='*55}\n{label}\n{'='*55}")
    tr_loader, va_loader = make_loaders(data)
    tk = trainer_kwargs(N_HIDDEN)

    trainer = Trainer(model=model, train_loader=tr_loader, val_loader=va_loader,
                      device=DEVICE, **tk)
    t0 = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    last = history[-5:]
    result = {
        "label": label,
        "top1": history[-1].get("val_top1", 0.0),
        "final_train_loss": float(np.mean([h["train_loss"] for h in last])),
        "final_task_loss":  float(np.mean([h.get("task_loss", 0) for h in last])),
        "final_safety_loss":float(np.mean([h.get("safety_loss", 0) for h in last])),
        "epochs": len(history),
        "elapsed_s": round(elapsed, 1),
    }
    print(f"  top1={result['top1']:.4f}  train={result['final_train_loss']:.3f}  "
          f"task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")
    return result


if __name__ == "__main__":
    data = load_data()
    print(f"Loaded: train={len(data[0])}  val={len(data[3])}")

    tk = topology_kwargs(N_HIDDEN)
    results = {}

    for mode in ["baseline", "threshold", "reflection", "turing", "learnable"]:
        base = SGNNET_SmallWorld(
            N_in=25088, N_hidden=N_HIDDEN, N_out=10,
            K_local=tk["K_local"], K_random=tk["K_random"],
            K_in=tk["K_in"], K_iter=tk["K_iter"], n_groups=tk["n_groups"], norm_mode="l2",
        )
        model = TuringSmallWorld(base, mode=mode, K_phase=8, theta=0.1, alpha=0.3)
        model = model.to(DEVICE)
        results[mode] = run(f"Turing mode={mode}", model, data)

    out = Path("results/validate_step3_turing.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    print("\n── Step 3 summary ──────────────────────────────────")
    for mode, r in results.items():
        print(f"  {mode:12s}  top1={r['top1']:.4f}")
    best = max(results.values(), key=lambda r: r["top1"])
    print(f"\n  Best: {best['label']}  top1={best['top1']:.4f}")
