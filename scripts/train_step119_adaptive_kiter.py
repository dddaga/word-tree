"""Step 119: Adaptive K_iter Per Sample (ACT-style early exit).

MOTIVATION
==========
K_iter=12 is fixed for all samples. Easy images (e.g., T-shirt) may need fewer
iterations than hard ones (e.g., sandal). MoD (step34) was KILLED because it made
per-NEURON exit decisions, destroying graph structure. Per-SAMPLE exit preserves
coherence — all neurons in a sample run the same number of steps.

MECHANISM: ACT (Adaptive Computation Time)
==========================================
At each K_iter step t:
  1. Compute halt probability: p_t = sigmoid(W_halt @ Z.mean(dim=1))  [B, 1]
  2. Accumulate: R_t = R_{t-1} + p_t
  3. When R_t > 1-epsilon for a sample, that sample halts (remainder trick)
  4. Ponder cost: loss += tau * mean(N_steps_per_sample)

Training uses soft ACT formulation: readout = weighted sum of Z at each step,
weighted by the halt probability (remainder for the final step). This is fully
differentiable. At inference: same mechanism, saves FLOPs on easy samples.

CONFIGS (N=1024, D=32, K_hh=4, K_iter=12 max, AH=1.0, 50%/75ep)
=================================================================
  Ref : Fixed K_iter=12 (standard AH, D=32)
  A   : ACT with tau=0.01 (weak ponder cost, prefer accuracy)
  B   : ACT with tau=0.1  (moderate ponder cost)
  C   : ACT with tau=0.5  (strong ponder cost, prefer efficiency)
  D   : Fixed K_iter=8    (simple baseline — is K_iter=12 overkill for D=32?)

To reproduce:
    python -u scripts/train_step119_adaptive_kiter.py --device mps
    python -u scripts/train_step119_adaptive_kiter.py --device mps --epochs 20  # scout
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.sgnnet.losses                import load_balance_loss, safety_valve_loss
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs, topology_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75,
                    help="Training epochs (default 75; use 20 for Tier-0 scout)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys to run (e.g. A,D). Empty = run all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 32; K_ITER = 12; K_IN = 50
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0


# ---------------------------------------------------------------------------
# Model: SGNNET with ACT (Adaptive Computation Time) per-sample early exit
# ---------------------------------------------------------------------------

class SGNNET_ACT(nn.Module):
    """Anti-Hebbian SGNNET with per-sample adaptive K_iter via ACT.

    At each iteration, a halt probability p_t is computed from the mean
    activation Z.mean(dim=1). The cumulative halt R_t determines when each
    sample stops. Training uses the soft ACT formulation (weighted sum of
    readouts). The ponder cost tau penalizes extra computation.

    Parameters
    ----------
    resonant : SGNNET_Resonant
        Base resonant model.
    alpha_ahebb : float
        Anti-Hebbian suppression strength.
    tau : float
        Ponder cost weight added to loss.
    epsilon : float
        Halting threshold: sample halts when R_t > 1 - epsilon.
    max_steps : int
        Maximum K_iter steps (hard ceiling).
    """

    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float = 1.0,
                 tau: float = 0.01, epsilon: float = 0.01, max_steps: int = 12):
        super().__init__()
        self.m             = resonant
        self.alpha_ahebb   = alpha_ahebb
        self.tau           = tau
        self.epsilon       = epsilon
        self.max_steps     = max_steps

        # Halt decision head: maps mean Z [D] -> scalar halt probability
        self.W_halt = nn.Linear(resonant.base.W_pos.shape[1], 1, bias=True)
        # Initialize bias negative so early steps don't halt prematurely
        nn.init.constant_(self.W_halt.bias, -3.0)
        nn.init.normal_(self.W_halt.weight, std=0.01)

        # Store ponder cost for external loss access
        self._ponder_cost = 0.0
        self._mean_steps  = 0.0

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base      = self.m.base
        Z         = base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = base.conn_hh
        N_h       = base.N_hidden
        B         = x.shape[0]

        # Pre-compute static AH suppression weights (wpos variant)
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)       # [N_h, K_hh]
        supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)                      # [1, N, K_hh, 1]

        Z_reflected = torch.zeros_like(Z)

        # ACT state
        halted       = torch.zeros(B, 1, device=x.device)          # [B, 1] binary
        cumul_halt   = torch.zeros(B, 1, device=x.device)          # R_t
        remainders   = torch.zeros(B, 1, device=x.device)          # remainder weights
        n_updates    = torch.zeros(B, 1, device=x.device)          # ponder count

        # Collect weighted readouts for soft ACT
        readout_accum = torch.zeros(B, N_OUT, device=x.device)

        for t in range(self.max_steps):
            # --- Standard AH iteration ---
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]                           # [B, N, K_hh, D]
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected
            Z     = F.normalize(Z_new.clamp(-10, 10), dim=-1)

            # --- ACT halt decision ---
            # Compute halt probability from mean activation
            z_mean = Z.mean(dim=1)                                  # [B, D]
            p_t    = torch.sigmoid(self.W_halt(z_mean))             # [B, 1]

            # For samples not yet halted
            still_running = 1.0 - halted                            # [B, 1]

            # Check if this step would push cumulative past threshold
            new_cumul = cumul_halt + p_t * still_running

            # Samples that cross threshold this step
            crosses = ((new_cumul > 1.0 - self.epsilon) &
                       (halted < 0.5)).float()                      # [B, 1]

            # Remainder for samples that halt this step
            # r_t = 1 - R_{t-1} (the remaining probability mass)
            r_t = (1.0 - cumul_halt) * crosses

            # Weight for non-halting steps: p_t
            # Weight for the halting step: remainder r_t
            step_weight = p_t * still_running * (1.0 - crosses) + r_t  # [B, 1]

            # Accumulate weighted readout
            step_readout = base._readout(Z)                         # [B, N_OUT]
            readout_accum = readout_accum + step_weight * step_readout

            # Update ACT state
            cumul_halt = new_cumul
            n_updates  = n_updates + still_running
            halted     = halted + crosses

            # If all samples halted, break early (saves compute at inference)
            if halted.all():
                break

        # For samples that never halted (ran all max_steps), use remainder
        never_halted = (halted < 0.5).float()
        if never_halted.any():
            final_remainder = (1.0 - cumul_halt + p_t * still_running) * never_halted
            readout_accum = readout_accum + final_remainder * base._readout(Z)
            n_updates = n_updates  # already counted

        # Store ponder cost for loss computation
        self._ponder_cost = n_updates.mean()
        self._mean_steps  = n_updates.mean().item()

        return readout_accum


# ---------------------------------------------------------------------------
# Custom trainer with ponder cost
# ---------------------------------------------------------------------------

class ACTTrainer(Trainer):
    """Trainer subclass that adds ACT ponder cost to the total loss."""

    def __init__(self, tau: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss_sum = task_loss_sum = safety_loss_sum = 0.0
        lb_loss_sum = ponder_loss_sum = 0.0
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
                ponder     = (getattr(self.model, "_ponder_cost", torch.tensor(0.0))
                              .to(self.device))
                loss       = (task_loss
                              + self.lambda_safety * safety
                              + self.lambda_lb * lb_loss
                              + self.tau * ponder)

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
            ponder_loss_sum += ponder.item() if hasattr(ponder, "item") else float(ponder)
            n_batches += 1

        n = max(n_batches, 1)
        return {
            "train_loss":   total_loss_sum / n,
            "task_loss":    task_loss_sum / n,
            "safety_loss":  safety_loss_sum / n,
            "lb_loss":      lb_loss_sum / n,
            "ponder_loss":  ponder_loss_sum / n,
        }


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    use_act: bool
    tau: float = 0.0
    k_iter_override: Optional[int] = None   # None = use K_ITER (12)


CONFIGS = [
    Config("Ref", "Ref  Fixed K_iter=12 (standard AH, D=32)", use_act=False),
    Config("A",   "A    ACT tau=0.01 (weak ponder cost)",      use_act=True, tau=0.01),
    Config("B",   "B    ACT tau=0.1  (moderate ponder cost)",  use_act=True, tau=0.1),
    Config("C",   "C    ACT tau=0.5  (strong ponder cost)",    use_act=True, tau=0.5),
    Config("D",   "D    Fixed K_iter=8 (simple baseline)",     use_act=False, k_iter_override=8),
]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    k_iter = cfg.k_iter_override if cfg.k_iter_override is not None else K_ITER
    topo = topology_kwargs(N)
    topo.pop("K_in", None)
    topo.pop("K_iter", None)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=k_iter, encoding_mode="fourier",
        **topo,
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    if cfg.use_act:
        return SGNNET_ACT(
            resonant, alpha_ahebb=ALPHA_AHEBB,
            tau=cfg.tau, epsilon=0.01, max_steps=K_ITER,
        )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data loaders (cached, 50% data)
# ---------------------------------------------------------------------------

_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 119 — Adaptive K_iter Per Sample (ACT)")
    print(f"N={N}  D={D}  K_iter={K_ITER} max  K_hh=4  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    print("Configs:")
    for c in CONFIGS:
        k_str = "ACT" if c.use_act else str(c.k_iter_override or K_ITER)
        print(f"  {c.key:4s}  act={c.use_act!s:5s}  tau={c.tau:<5.2f}  "
              f"k_iter={k_str:>4s}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step119_adaptive_kiter.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  act={cfg.use_act}  tau={cfg.tau}")
        print(f"{'─'*60}")

        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=EPOCHS)

        # Use ACTTrainer for ACT configs, standard Trainer otherwise
        if cfg.use_act:
            trainer = ACTTrainer(
                tau=cfg.tau, model=model,
                train_loader=get_loaders()[0],
                val_loader=get_loaders()[1],
                device=DEVICE, **kw,
            )
        else:
            trainer = Trainer(
                model=model,
                train_loader=get_loaders()[0],
                val_loader=get_loaders()[1],
                device=DEVICE, **kw,
            )

        history   = trainer.train(n_epochs=EPOCHS)
        elapsed   = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1

        # Get mean steps for ACT configs
        mean_steps = model._mean_steps if hasattr(model, "_mean_steps") else (
            cfg.k_iter_override or K_ITER)

        results[cfg.key] = {
            "N": N, "D": D, "K_iter_max": K_ITER,
            "use_act": cfg.use_act, "tau": cfg.tau,
            "k_iter_effective": cfg.k_iter_override or K_ITER,
            "alpha_ahebb": ALPHA_AHEBB,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "mean_steps_final": round(mean_steps, 2) if isinstance(mean_steps, float) else mean_steps,
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params,
            "label": cfg.label,
        }

        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs_ref = top1_best - ref_best if ref_best > 0 else 0

        steps_str = f"  mean_steps={mean_steps:.1f}" if cfg.use_act else ""
        print(f"\n  top1_best={top1_best:.4f}  vs_Ref={vs_ref:+.4f}{steps_str}  "
              f"elapsed={elapsed/60:.1f}min")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary table
    print(f"\n{'='*70}")
    print(f"STEP 119 SUMMARY (D={D})")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Config':<6}  {'act':>3}  {'tau':>5}  {'steps':>5}  "
          f"{'params':>8}  {'top1':>7}  {'vs_Ref':>8}")
    print(f"{'─'*70}")
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        steps = r.get("mean_steps_final", r.get("k_iter_effective", K_ITER))
        print(f"{key:<6}  {str(r['use_act']):>3s}  {r['tau']:>5.2f}  {steps:>5}  "
              f"{r['n_params']:>8,}  {r['top1_best']:.4f}  {vs:>+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
