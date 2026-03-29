"""Step 2 validation: W_phase receiver + conn_phase.

Adds a learned phase receiver (W_phase) alongside the existing W_pos backbone.
During routing each neuron receives from its K_phase nearest phase-space partners,
gated by dot(Z[b,k], W_phase[h]) — the activation-queries-weight mechanism.

Compare at N=512:
  - SmallWorld baseline (no W_phase): Step 1 result
  - SmallWorld + W_phase receiver (K_phase=8)
  - SmallWorld + W_phase receiver (K_phase=16)

Pass criterion: top-1 ≥ Step1 result + 1% (phase receiver adds signal)
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


# ── Data ──────────────────────────────────────────────────────────────────────

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


# ── W_phase receiver wrapper ──────────────────────────────────────────────────

class SmallWorldWithPhase(nn.Module):
    """SGNNET_SmallWorld + W_phase receiver channel.

    W_phase[h] is a D-dim learnable receiver filter.
    conn_phase[h] = K_phase nearest neighbors in W_phase space (built at init).
    During each routing iteration, neuron h also receives from its phase partners:
        score[b,h,k] = dot(Z[b,k], W_phase_norm[h])
        Z_phase[b,h] = sum_k softmax(score)[k] * Z[b,k]
    Combined: Z_new = normalize(Z_struct + alpha * Z_phase)
    """

    def __init__(self, base: SGNNET_SmallWorld, K_phase: int = 8, alpha: float = 0.5):
        super().__init__()
        self.base    = base
        self.K_phase = K_phase
        self.alpha   = alpha
        N  = base.N_hidden
        D  = base.W_pos.shape[1]

        # Learnable phase receiver vectors — init uniform like W_pos
        self.W_phase = nn.Parameter(torch.rand(N, D))

        # Build initial phase graph (k-NN in W_phase space)
        self._build_phase_graph()

    def _build_phase_graph(self):
        with torch.no_grad():
            Wp = F.normalize(self.W_phase.detach(), dim=-1)
            sim = Wp @ Wp.T                                      # [N, N]
            sim.fill_diagonal_(-1e9)
            _, idx = sim.topk(self.K_phase, dim=-1)             # [N, K_phase]
        self.register_buffer("conn_phase", idx)

    @property
    def W_pos(self): return self.base.W_pos

    def tick_epoch(self):
        """Rebuild phase graph + delegate to base."""
        self._build_phase_graph()
        if hasattr(self.base, "tick_epoch"):
            self.base.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Seed: delegate to base's _seed if accessible, else run base forward
        # For simplicity: run base forward but intercept at the routing stage
        # We patch via the base model's internal buffers
        return self.base.forward_with_phase(x, self.W_phase, self.conn_phase, self.alpha)


# Patch SGNNET_SmallWorld with forward_with_phase method
def _forward_with_phase(self, x, W_phase, conn_phase, alpha):
    """Forward pass using both structural and phase connections."""
    B = x.shape[0]
    N = self.N_hidden
    D = self.W_pos.shape[1]

    # Seed hidden neurons from input
    Z = self._seed(x)   # [B, N, D]

    W_ph_norm = F.normalize(W_phase, dim=-1)   # [N, D]

    for _ in range(self.K_iter):
        # Structural: gather from conn_hh
        Z_struct = Z[:, self.conn_hh, :].sum(dim=2)   # [B, N, D]

        # Phase receiver: score each phase partner
        # score[b,h,k] = Z[b,k] · W_phase_norm[h]
        Z_partners = Z[:, conn_phase, :]               # [B, N, K_phase, D]
        score = (Z_partners * W_ph_norm.unsqueeze(0).unsqueeze(2)).sum(-1)  # [B, N, K_phase]
        gate  = F.softmax(score, dim=-1)               # [B, N, K_phase]
        Z_phase = (gate.unsqueeze(-1) * Z_partners).sum(dim=2)  # [B, N, D]

        Z = F.normalize(Z_struct + alpha * Z_phase, dim=-1)

    return self._readout(Z)


SGNNET_SmallWorld.forward_with_phase = _forward_with_phase


# ── Run one config ─────────────────────────────────────────────────────────────

def run(label: str, model, data) -> dict:
    print(f"\n{'='*55}\n{label}\n{'='*55}")
    tr_loader, va_loader = make_loaders(data)
    tk = trainer_kwargs(N_HIDDEN)

    trainer = Trainer(
        model=model,
        train_loader=tr_loader,
        val_loader=va_loader,
        device=DEVICE,
        **tk,
    )
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
    ratio = result["final_safety_loss"] / max(result["final_task_loss"], 1e-8)
    print(f"  top1={result['top1']:.4f}  task={result['final_task_loss']:.3f}  "
          f"safety={result['final_safety_loss']:.3f}  safety/task={ratio:.2f}  t={elapsed:.0f}s")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_data()
    print(f"Loaded: train={len(data[0])}  val={len(data[3])}")

    tk   = topology_kwargs(N_HIDDEN)
    results = {}

    # Baseline: no W_phase
    base_model = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N_HIDDEN, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=tk["K_iter"], n_groups=tk["n_groups"], norm_mode="l2",
    ).to(DEVICE)
    results["baseline"] = run("Baseline (no W_phase)", base_model, data)

    # With W_phase K_phase=8
    base2 = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N_HIDDEN, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=tk["K_iter"], n_groups=tk["n_groups"], norm_mode="l2",
    )
    model_p8 = SmallWorldWithPhase(base2, K_phase=8, alpha=0.5).to(DEVICE)
    results["wphase_k8"] = run("W_phase receiver K_phase=8 alpha=0.5", model_p8, data)

    # With W_phase K_phase=16
    base3 = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N_HIDDEN, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=tk["K_iter"], n_groups=tk["n_groups"], norm_mode="l2",
    )
    model_p16 = SmallWorldWithPhase(base3, K_phase=16, alpha=0.5).to(DEVICE)
    results["wphase_k16"] = run("W_phase receiver K_phase=16 alpha=0.5", model_p16, data)

    out = Path("results/validate_step2_wphase.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    print("\n── Step 2 summary ──────────────────────────────────")
    for k, r in results.items():
        print(f"  {k:20s}  top1={r['top1']:.4f}  "
              f"safety/task={r['final_safety_loss']/max(r['final_task_loss'],1e-8):.2f}")
    baseline_top1 = results["baseline"]["top1"]
    print(f"\n  Pass: W_phase variants ≥ {baseline_top1 + 0.01:.4f} (baseline + 1%)")
