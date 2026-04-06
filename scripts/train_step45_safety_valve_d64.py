"""Step 45: Safety valve redesign for D=64.

GAP G7 — Safety Valve Ineffective at D=64
===========================================
Safety valve r* = 0.5 / N^(1/D). At D=64 N=1024: r* ≈ 0.497 — nearly the box
size. On S^63, random points are already far apart, so Coulomb repulsion never
activates. Confirmed: safety_loss = 0.0016-0.0028 at D=64 (effectively zero).

The safety valve was designed for D=4 where N=256 neurons in [0,1]^4 can easily
cluster. At D=64, the curse of dimensionality makes clustering unlikely but
W_pos could still degenerate in other ways (collapse to a hyperplane, concentrate
in corners, etc).

Hypothesis: replacing the Coulomb safety valve with a D-appropriate regularizer
could improve W_pos utilization at D=64:
  A) Variance regularizer: penalize low variance of W_pos across neurons per dim
  B) Entropy regularizer: penalize low entropy of W_pos coordinate distribution
  C) No safety valve: just remove it (it's dead weight that adds 0 gradient)
  D) Spectral regularizer: penalize low rank of W_pos matrix (encourage spread)

CONFIGS (D=64 N=1024 K_iter=8):
  Ref    standard safety valve (dead at D=64) [expect ~56%]
  A      no safety valve at all (λ_safety=0)
  B      + variance regularizer λ=0.01
  C      + variance regularizer λ=0.1
  D      + spectral regularizer (rank penalty) λ=0.01

To reproduce:
    python -u scripts/train_step45_safety_valve_d64.py --device mps
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


# ── Custom Trainer with pluggable safety regularizer ─────────────────────────

class SafetyTrainer(Trainer):
    """Trainer that replaces safety_valve_loss with a custom regularizer."""

    def __init__(self, safety_mode: str = "default", lambda_reg: float = 0.01,
                 **kwargs):
        super().__init__(**kwargs)
        self.safety_mode = safety_mode
        self.lambda_reg  = lambda_reg

    def _compute_reg(self, W_pos: torch.Tensor) -> torch.Tensor:
        """Custom safety regularizer based on safety_mode."""
        if self.safety_mode == "none":
            return torch.tensor(0.0, device=W_pos.device)

        elif self.safety_mode == "variance":
            # Penalize low per-dimension variance across neurons
            # Want W_pos to spread out, not cluster
            var_per_dim = W_pos.var(dim=0)       # [D]
            return -var_per_dim.mean()            # maximize variance

        elif self.safety_mode == "spectral":
            # Penalize low effective rank of W_pos
            # Low rank = neurons concentrated in a subspace
            # Approximate via nuclear norm / Frobenius norm
            W_centered = W_pos - W_pos.mean(dim=0, keepdim=True)
            fro = W_centered.norm(p='fro')
            # SVD is expensive at D=64; use squared Frobenius / trace(W^T W) proxy
            cov = W_centered.T @ W_centered / W_pos.shape[0]  # [D, D]
            eigenvalues = torch.linalg.eigvalsh(cov)           # [D], sorted ascending
            # Effective rank = exp(entropy of normalized eigenvalues)
            eig_norm = eigenvalues.clamp(min=1e-8) / eigenvalues.sum()
            entropy  = -(eig_norm * eig_norm.log()).sum()
            return -entropy   # maximize entropy of eigenvalues = spread across dims

        else:  # "default" — standard safety valve
            return safety_valve_loss(W_pos, self.box_size)

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss_sum = task_loss_sum = safety_loss_sum = lb_loss_sum = 0.0
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
                scores    = self.model(features)
                task_loss = F.kl_div(F.log_softmax(scores, dim=-1),
                                     soft_labels, reduction="batchmean")
                if self.safety_mode == "default":
                    safety = safety_valve_loss(self.model.W_pos, self.box_size,
                                              task_loss=task_loss)
                    reg_loss = self.lambda_safety * safety
                else:
                    safety = self._compute_reg(self.model.W_pos)
                    reg_loss = self.lambda_reg * safety
                lb_loss   = load_balance_loss(scores.abs().sum(dim=0))
                loss      = task_loss + reg_loss + self.lambda_lb * lb_loss

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
            n_batches += 1

        n = max(n_batches, 1)
        return {
            "train_loss":  total_loss_sum / n,
            "task_loss":   task_loss_sum / n,
            "safety_loss": safety_loss_sum / n,
            "lb_loss":     lb_loss_sum / n,
        }


def run(label: str, model: nn.Module, meta: dict,
        safety_mode: str = "default", lambda_reg: float = 0.01) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="plateau")
    trainer = SafetyTrainer(
        safety_mode=safety_mode, lambda_reg=lambda_reg,
        model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk,
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
        "final_safety_loss": float(np.mean([h.get("safety_loss", 0.0) for h in last5])),
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


CONFIGS = [
    ("Ref",  "Ref   standard safety valve (dead at D=64)",  "default", 0.0),
    ("A",    "A     no safety valve (λ_safety=0)",           "none",    0.0),
    ("B",    "B     + variance regularizer λ=0.01",          "variance", 0.01),
    ("C",    "C     + variance regularizer λ=0.1",           "variance", 0.1),
    ("D",    "D     + spectral regularizer λ=0.01",          "spectral", 0.01),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print("Step 45: Safety valve redesign for D=64")
    print("Gap G7: Coulomb r* ≈ 0.497 at D=64 — valve is dead (safety=0.002)")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    for key, label, safety_mode, lambda_reg in CONFIGS:
        model = make_resonant(N=1024, D=64, K_iter=8).to(DEVICE)
        meta  = {"N": 1024, "D": 64, "K_iter": 8,
                 "safety_mode": safety_mode, "lambda_reg": lambda_reg}
        results[key] = run(label, model, meta,
                           safety_mode=safety_mode, lambda_reg=lambda_reg)
        results[key].update(meta)

    out = ROOT / "results" / "train_step45_safety_valve_d64.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_val = results.get("Ref", {}).get("top1_best", 0.5628)
    print(f"\n-- Safety valve redesign (ref={ref_val:.4f}) ---")
    print(f"  {'Config':<50}  {'top1':>6}  {'vs_Ref':>8}  {'safety':>8}  {'t(s)':>6}")
    print("  " + "-"*85)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        print(f"  {r['label'][:50]:<50}  {r['top1_best']:>6.4f}  {d:>+8.4f}  "
              f"{r.get('final_safety_loss',0):>8.5f}  {r['elapsed_s']:>6.0f}")

    print("\n  Interpretation:")
    print("  A ≈ Ref  → safety valve is truly dead at D=64; removing it is free")
    print("  B/C > Ref → variance regularizer helps W_pos spread; include in Gen4")
    print("  D > Ref  → spectral reg prevents subspace collapse; include in Gen4")
