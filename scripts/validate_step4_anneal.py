"""Step 4 validation: simulated annealing schedule.

Couples three parameters to a temperature schedule tau(epoch):
  - gate_temp (softmax tau for phase routing)
  - lambda_safety_eff = lambda_base * (1 + beta * tau)  [stronger repulsion early]
  - beam_size = max(beam_min, int(beam_max * tau))       [wider search early]

Compares at N=512 against the best Step 3 configuration:
  A. Best Step 3 config, no annealing (fixed tau=1.0)
  B. Exponential decay: tau = exp(-decay * epoch)
  C. Cosine decay:      tau = 0.5 * (1 + cos(pi * epoch / N_epochs))
  D. Stepwise:          tau = 1 if epoch < warmup else 0.1

Pass criterion: annealed variants converge faster OR reach higher final top-1.
"""

from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.training.experiment_config import trainer_kwargs, topology_kwargs
from src.sgnnet.losses import safety_valve_loss, load_balance_loss

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


# ── Annealing schedules ───────────────────────────────────────────────────────

def tau_fixed(epoch, n_epochs):       return 1.0
def tau_exp(epoch, n_epochs):         return math.exp(-3.0 * epoch / n_epochs)
def tau_cosine(epoch, n_epochs):      return 0.5 * (1 + math.cos(math.pi * epoch / n_epochs))
def tau_stepwise(epoch, n_epochs):
    warmup = int(0.2 * n_epochs)
    return 1.0 if epoch < warmup else 0.1


# ── Annealed trainer (custom loop) ───────────────────────────────────────────

def run_annealed(label: str, model, data, tau_fn, base_lambda: float) -> dict:
    print(f"\n{'='*55}\n{label}\n{'='*55}")
    tr_loader, va_loader = make_loaders(data)

    device = DEVICE
    model = model.to(device)

    opt = torch.optim.AdamW([
        {"params": [model.W_pos], "lr": 2.364e-3, "weight_decay": 0.0},
    ])

    history = []
    t0 = time.time()

    for epoch in range(EPOCHS):
        tau = tau_fn(epoch, EPOCHS)
        lambda_eff  = base_lambda * (1 + 0.5 * tau)   # stronger early repulsion
        gate_temp   = max(0.1, tau)                    # routing temperature
        beam_now    = max(8, int(32 * tau))            # shrinking beam

        # Training epoch
        model.train()
        total_loss_sum = task_sum = safety_sum = 0.0
        n_batches = 0

        for feats, slabels, _ in tr_loader:
            feats   = feats.to(device)
            slabels = slabels.to(device)
            opt.zero_grad()

            scores = model(feats, gate_temp=gate_temp, beam_size=beam_now)
            task   = F.kl_div(F.log_softmax(scores, dim=-1), slabels, reduction="batchmean")
            safety = safety_valve_loss(model.W_pos, task_loss=task)
            loss   = task + lambda_eff * safety

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                model.W_pos.clamp_(0, 1.0)

            total_loss_sum += loss.item()
            task_sum       += task.item()
            safety_sum     += safety.item()
            n_batches      += 1

        # Eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for feats, _, labels in va_loader:
                feats = feats.to(device)
                preds = model(feats, gate_temp=0.1, beam_size=8).argmax(dim=-1).cpu()
                correct += (preds == labels).sum().item()
                total   += len(labels)

        n = max(n_batches, 1)
        h = {
            "epoch": epoch,
            "tau": tau,
            "train_loss": total_loss_sum / n,
            "task_loss": task_sum / n,
            "safety_loss": safety_sum / n,
            "val_top1": correct / max(total, 1),
        }
        history.append(h)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  e{epoch+1:3d}  tau={tau:.3f}  loss={h['train_loss']:.3f}  "
                  f"top1={h['val_top1']:.4f}  beam={beam_now}")

    elapsed = time.time() - t0
    last = history[-5:]
    return {
        "label": label,
        "top1": history[-1]["val_top1"],
        "final_train_loss": float(np.mean([h["train_loss"] for h in last])),
        "epochs": len(history),
        "elapsed_s": round(elapsed, 1),
        "history": history,
    }


# ── Annealing-aware model wrapper ─────────────────────────────────────────────

class AnnealedSmallWorld(nn.Module):
    """SmallWorld with gate_temp and beam_size as forward-time arguments."""

    def __init__(self, base: SGNNET_SmallWorld, K_phase: int = 8):
        super().__init__()
        self.base    = base
        self.K_phase = K_phase
        N = base.N_hidden
        D = base.W_pos.shape[1]
        self.W_phase = nn.Parameter(torch.rand(N, D))
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

    def forward(self, x, gate_temp: float = 1.0, beam_size: int = 16):
        B = x.shape[0]
        N = self.base.N_hidden

        Z = self.base._seed(x)
        W_ph_norm = F.normalize(self.W_phase, dim=-1)

        for _ in range(self.base.K_iter):
            # Structural
            Z_struct = Z[:, self.base.conn_hh, :].sum(dim=2)

            # Phase receiver with temperature
            Z_partners = Z[:, self.conn_phase, :]
            score = (Z_partners * W_ph_norm.unsqueeze(0).unsqueeze(2)).sum(-1)
            gate  = F.softmax(score / max(gate_temp, 0.01), dim=-1)
            Z_phase = (gate.unsqueeze(-1) * Z_partners).sum(dim=2)

            Z = F.normalize(Z_struct + 0.5 * Z_phase, dim=-1)

        return self.base._readout(Z)


if __name__ == "__main__":
    data = load_data()
    print(f"Loaded: train={len(data[0])}  val={len(data[3])}")

    tk = topology_kwargs(N_HIDDEN)
    base_lambda = trainer_kwargs(N_HIDDEN)["lambda_safety"]

    schedules = {
        "fixed":    tau_fixed,
        "exp":      tau_exp,
        "cosine":   tau_cosine,
        "stepwise": tau_stepwise,
    }

    results = {}
    for name, tau_fn in schedules.items():
        base = SGNNET_SmallWorld(
            N_in=25088, N_hidden=N_HIDDEN, N_out=10,
            K_local=tk["K_local"], K_random=tk["K_random"],
            K_in=tk["K_in"], K_iter=tk["K_iter"], n_groups=tk["n_groups"], norm_mode="l2",
        )
        model = AnnealedSmallWorld(base, K_phase=8)
        results[name] = run_annealed(
            f"Annealing schedule={name}", model, data, tau_fn, base_lambda
        )

    out = Path("results/validate_step4_anneal.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    print("\n── Step 4 summary ──────────────────────────────────")
    for name, r in results.items():
        print(f"  {name:12s}  top1={r['top1']:.4f}  loss={r['final_train_loss']:.3f}")
    best = max(results.values(), key=lambda r: r["top1"])
    print(f"\n  Best: {best['label']}  top1={best['top1']:.4f}")
