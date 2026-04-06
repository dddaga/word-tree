"""Step 39: Mechanism-aware auxiliary losses for SGNNET.

HYPOTHESIS (InfraNodus Gap: loss/task ↔ mechanisms)
=====================================================
All prior experiments modify the forward pass (routing mechanisms) but use the
same loss function. The loss and mechanism clusters are structurally disconnected.

Hypothesis: auxiliary losses that directly reward mechanism behavior can unlock
gains that forward-pass-only mechanisms miss. Specifically:

  L_phase    Phase coherence — maximize cosine alignment between Z[j] and
             its structural neighbors Z[conn_hh[j]]. Encourages neurons to
             converge to shared subspaces rather than diverge after routing.

  L_sparse   Inhibition sparsity — penalize large post-threshold activations
             (Z_fwd = relu(Z - theta)). Rewards routing steps that genuinely
             zero out uninformative neurons. Pure L1 on Z_fwd.

  L_div      Routing diversity — penalize collapse: if all neurons always route
             to the same K neighbors, entropy of the routing distribution is 0.
             Measured as negative entropy of per-neuron activation magnitudes
             after gathering: entropy(softmax(|Z_nb|.sum(-1), dim=-1)).

All auxiliary losses are added to the standard loss in AuxTrainer:
  total = task + λ_safety·safety + λ_lb·lb + λ_aux·aux

DESIGN
======
The wrapper model computes aux loss during forward pass and stores it as
self.last_aux_loss (a scalar tensor). AuxTrainer (subclass of Trainer) reads
this after model(features) and adds it to the total loss.

CONFIGS
=======
  Ref    baseline — no auxiliary loss (standard training)
  A      + phase coherence   λ=0.01
  B      + phase coherence   λ=0.1
  C      + inhibition sparsity  λ=0.01
  D      + inhibition sparsity  λ=0.1
  E      + routing diversity    λ=0.01
  F      A+C combined  (phase coherence + inhibition sparsity, λ=0.01 each)

Base: D=64 N=1024 K_iter=8 (uncalibrated base — same as step29b)

To reproduce:
    python -u scripts/train_step39_mechanism_loss.py --device mps
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
from src.sgnnet.losses              import load_balance_loss, safety_valve_loss

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


# ── Mechanism-aware loss wrapper ─────────────────────────────────────────────

class SGNNET_MechLoss(nn.Module):
    """Wraps SGNNET_Resonant and accumulates auxiliary mechanism loss during forward.

    Aux loss types:
      'phase'   — phase coherence: -mean(cos_sim(Z[j], Z[conn_hh[j]]))
      'sparse'  — inhibition sparsity: mean(relu(Z - theta)) — L1 on Z_fwd
      'div'     — routing diversity: -entropy(softmax(|Z_nb|.sum(-1)))
    """

    def __init__(self, base: SGNNET_Resonant, aux_type: str, lambda_aux: float):
        super().__init__()
        self.m          = base
        self.aux_type   = aux_type    # 'phase', 'sparse', 'div', or 'phase+sparse'
        self.lambda_aux = lambda_aux
        self.last_aux_loss = torch.tensor(0.0)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                                  # [B, N, D]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)         # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)                   # [N, D]
        conn_hh   = self.m.base.conn_hh                                   # [N, K_hh]

        aux_accum = []

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)                              # [B, N, D]
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)                       # [B, N, D]
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
            Z = F.normalize(
                (Z_struct + self.m.alpha_turing * Z_inh).clamp(-10, 10),
                dim=-1,
            )

            # Compute aux contribution this routing step
            if 'phase' in self.aux_type:
                # Phase coherence: align Z[j] with its structural neighbors
                Z_norm = Z   # already normalized
                Z_nb   = Z_norm[:, conn_hh, :]                            # [B, N, K_hh, D]
                cos_sim = (Z_norm.unsqueeze(2) * Z_nb).sum(-1)           # [B, N, K_hh]
                aux_accum.append(-cos_sim.mean())

            if 'sparse' in self.aux_type:
                # Inhibition sparsity: L1 on post-threshold activations
                # (large Z_fwd = neuron activated = not sparse)
                aux_accum.append(Z_fwd.abs().mean())

            if 'div' in self.aux_type:
                # Routing diversity: reward even distribution of activation across neighbors
                Z_nb_mag = Z_fwd[:, conn_hh, :].norm(dim=-1)            # [B, N, K_hh]
                # Entropy of softmax over neighbor magnitudes per neuron
                log_prob = F.log_softmax(Z_nb_mag + 1e-8, dim=-1)
                prob     = log_prob.exp()
                entropy  = -(prob * log_prob).sum(-1).mean()             # scalar
                aux_accum.append(-entropy)   # maximize entropy = minimize negative

        # Average aux loss across routing steps
        if aux_accum:
            self.last_aux_loss = torch.stack(aux_accum).mean()
        else:
            self.last_aux_loss = torch.tensor(0.0, device=x.device)

        return self.m.base._readout(Z)


# ── AuxTrainer: adds model.last_aux_loss to total loss ───────────────────────

class AuxTrainer(Trainer):
    """Trainer subclass that adds model.last_aux_loss to the total loss."""

    def __init__(self, lambda_aux: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_aux = lambda_aux

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss_sum = task_loss_sum = safety_loss_sum = lb_loss_sum = aux_loss_sum = 0.0
        n_batches = 0

        for features, soft_labels, _labels in self.train_loader:
            features    = features.to(self.device)
            soft_labels = soft_labels.to(self.device)
            self.optimizer.zero_grad()

            _amp_ctx = (
                torch.autocast(str(self.device).split(":")[0], dtype=torch.float16)
                if self.use_amp
                else torch.autocast("cpu", enabled=False)
            )
            with _amp_ctx:
                scores     = self.model(features)
                task_loss  = F.kl_div(F.log_softmax(scores, dim=-1),
                                      soft_labels, reduction="batchmean")
                safety     = safety_valve_loss(self.model.W_pos, self.box_size,
                                               task_loss=task_loss)
                lb_loss    = load_balance_loss(scores.abs().sum(dim=0))
                aux_loss   = (getattr(self.model, "last_aux_loss", torch.tensor(0.0))
                              .to(self.device))
                loss       = (task_loss
                              + self.lambda_safety * safety
                              + self.lambda_lb * lb_loss
                              + self.lambda_aux * aux_loss)

            _opt_params = [p for g in self.optimizer.param_groups for p in g["params"]]
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(_opt_params, self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_opt_params, self.grad_clip_norm)
                self.optimizer.step()

            if hasattr(self.model, "tick_step"):
                self.model.tick_step()
            with torch.no_grad():
                self.model.W_pos.clamp_(0, self.box_size)

            total_loss_sum  += loss.item()
            task_loss_sum   += task_loss.item()
            safety_loss_sum += safety.item()
            lb_loss_sum     += lb_loss.item()
            aux_loss_sum    += aux_loss.item()
            n_batches += 1

        n = max(n_batches, 1)
        return {
            "train_loss": total_loss_sum / n,
            "task_loss":  task_loss_sum / n,
            "safety_loss": safety_loss_sum / n,
            "lb_loss":    lb_loss_sum / n,
            "aux_loss":   aux_loss_sum / n,
        }


def run(label: str, model: nn.Module, meta: dict, lambda_aux: float = 0.0) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="plateau")
    trainer = AuxTrainer(
        lambda_aux=lambda_aux, model=model,
        train_loader=tr, val_loader=va, device=DEVICE, **tk,
    )
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
        "final_aux_loss":  float(np.mean([h.get("aux_loss",  0.0) for h in last5])),
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


# (key, label, aux_type, lambda_aux)
CONFIGS = [
    ("Ref", "Ref   D=64 N=1024 K_iter=8  [baseline ≈56.28%]",
     None, 0.0),
    ("A",   "A     + phase coherence loss  λ=0.01",
     "phase", 0.01),
    ("B",   "B     + phase coherence loss  λ=0.1",
     "phase", 0.1),
    ("C",   "C     + inhibition sparsity   λ=0.01",
     "sparse", 0.01),
    ("D",   "D     + inhibition sparsity   λ=0.1",
     "sparse", 0.1),
    ("E",   "E     + routing diversity     λ=0.01",
     "div", 0.01),
    ("F",   "F     + phase coherence + inhibition sparsity  λ=0.01 each",
     "phase+sparse", 0.01),
]
KEYS = ["Ref", "A", "B", "C", "D", "E", "F"]


if __name__ == "__main__":
    REF_BASELINE = 0.5628
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Seed: {SEED}")
    print("Step 39: Mechanism-aware auxiliary losses")
    print("InfraNodus gap: loss/task cluster disconnected from mechanisms cluster")
    print(f"Ref baseline (step22E / D=64 / N=1024 / K_iter=8): {REF_BASELINE:.4f}")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    for key, label, aux_type, lambda_aux in CONFIGS:
        resonant = make_resonant(N=1024, D=64, K_iter=8).to(DEVICE)
        if aux_type is None:
            model = resonant
        else:
            model = SGNNET_MechLoss(resonant, aux_type=aux_type,
                                    lambda_aux=lambda_aux).to(DEVICE)
        meta = {"N": 1024, "D": 64, "K_iter": 8,
                "aux_type": aux_type, "lambda_aux": lambda_aux}
        results[key] = run(label, model, meta, lambda_aux=lambda_aux)
        results[key].update(meta)

    out = ROOT / "results" / "train_step39_mechanism_loss.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    print(f"\n-- Mechanism-aware auxiliary loss sweep (ref={ref_val:.4f}) ---")
    print("  %-55s  %9s  %9s  %8s  %6s" % ("Config", "top1", "vs_Ref", "best_ep", "t(s)"))
    print("  " + "-"*95)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        print("  %-55s  %9.4f  %+9.4f  %7d    %6.0f" % (
            r["label"][:55], r["top1_best"], d,
            r.get("best_epoch", 0), r["elapsed_s"]))

    print("\n  Interpretation:")
    print("  d > +1pp  → aux loss bridges mechanism/task gap; add to Gen4 base loss")
    print("  d ~ 0     → mechanism already optimised by task loss implicitly")
    print("  d < 0     → aux loss conflicts with task; λ too high or wrong signal")
    print("  B > A     → λ=0.1 better than 0.01 → phase coherence wants strong signal")
    print("  F > A+C   → combining phase+sparse is additive; else they conflict")
