"""Step 113: Intermediate Supervision for K_iter routing.

MOTIVATION
==========
SGNNET trained only on CE(Z_final) provides no gradient signal about what
Z should hold at intermediate steps. The K_iter routing loop is an unrolled
RNN — each step is a time step. Without supervision at intermediate states,
the network has no incentive to:
  - Route signal to the right neurons early
  - Maintain classifiable representations across steps
  - Use the sparse beam efficiently at each step

Intermediate supervision (deep supervision analogy, like GoogLeNet/DenseNet):
  loss = CE(Z_final) + λ * CE(Z_mid)

Z_mid = readout of graph state at step K_iter//2 (step 6 of 12).
If λ_aux helps: temporal credit assignment was the bottleneck — the routing
needed a signal about what to hold at mid-pass, not just what to output finally.

This is distinct from GCNII residual (step91): residual stabilises gradient
flow geometrically. Intermediate supervision gives explicit semantic signal
at intermediate states. They can compound.

CONFIGS (N=1024, D=64, K_hh=4, K_iter=12, AH=1.0, turing=0.0, 50%/75ep)
=========================================================================
  Ref : no intermediate supervision (standard AH K_iter=12 baseline)
  A   : λ_aux=0.1, mid at step 6 (K_iter//2)
  B   : λ_aux=0.5, mid at step 6             (stronger supervision signal)
  C   : λ_aux=0.1, mid at steps 3 AND 9      (two checkpoints)
  D   : λ_aux=0.1, mid at step 6 + GCNII α=0.05  (compound with step91 winner)

To reproduce:
    python -u scripts/train_step113_intermediate_supervision.py --device mps
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs, topology_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75,
                    help="Training epochs (default 75; use 20 for Tier-0 scout)")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 64; K_ITER = 12; K_IN = 50
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
STEP69_REF = 0.8336


# ---------------------------------------------------------------------------
# Model: AH routing with intermediate readout side-channel
# ---------------------------------------------------------------------------

class SGNNET_AH_IntermSupervision(nn.Module):
    """AH routing that exposes intermediate Z readouts as a side-channel.

    During training, stores logits at each mid_step as self._aux_logits (list).
    During eval, _aux_logits is empty — forward returns only final logits.

    The training loop picks up _aux_logits to compute the auxiliary loss.
    No architecture change to the routing — only readout hooks added.
    """

    def __init__(
        self,
        resonant,
        alpha_ahebb: float,
        mid_steps: List[int],       # which K_iter steps to readout (0-indexed)
        residual_alpha: float = 0.0,
    ):
        super().__init__()
        self.m              = resonant
        self.alpha          = alpha_ahebb
        self.mid_steps      = set(mid_steps)
        self.residual_alpha = residual_alpha
        self._aux_logits: List[torch.Tensor] = []   # populated during train forward

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base      = self.m.base
        N_h       = base.N_hidden
        conn_hh   = base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)

        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)       # [N, K_hh]
        supp_w  = (1.0 - self.alpha * pos_sim.clamp(min=0)
                   ).unsqueeze(0).unsqueeze(-1)                    # [1, N, K_hh, 1]

        Z           = base._seed(x)
        h_0         = Z.clone() if self.residual_alpha > 0 else None
        Z_reflected = torch.zeros_like(Z)

        # Clear aux logits each forward pass
        self._aux_logits = []

        for t in range(K_ITER):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_nb     = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected

            if self.residual_alpha > 0 and h_0 is not None:
                Z_new = (1.0 - self.residual_alpha) * Z_new + self.residual_alpha * h_0

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

            # Intermediate readout: only during training, at designated steps
            if self.training and t in self.mid_steps:
                self._aux_logits.append(base._readout(Z))

        return base._readout(Z)


# ---------------------------------------------------------------------------
# Trainer subclass: adds auxiliary CE loss from intermediate readouts
# ---------------------------------------------------------------------------

class TrainerIntermSupervision(Trainer):
    """Trainer that adds λ_aux * KL(Z_mid, soft_labels) to the main loss.

    The model stores intermediate readouts in model._aux_logits during
    forward(). This trainer picks them up before the backward pass.
    """

    def __init__(self, *args, lambda_aux: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_aux = lambda_aux

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss_sum = 0.0
        task_loss_sum  = 0.0
        aux_loss_sum   = 0.0
        n_batches      = 0

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
                task_loss = F.kl_div(
                    F.log_softmax(scores, dim=-1),
                    soft_labels,
                    reduction="batchmean",
                )

                # Auxiliary loss from intermediate readouts
                aux_loss = torch.tensor(0.0, device=self.device)
                if self.lambda_aux > 0 and hasattr(self.model, "_aux_logits"):
                    for aux_logits in self.model._aux_logits:
                        aux_loss = aux_loss + F.kl_div(
                            F.log_softmax(aux_logits, dim=-1),
                            soft_labels,
                            reduction="batchmean",
                        )
                    if len(self.model._aux_logits) > 0:
                        aux_loss = aux_loss / len(self.model._aux_logits)

                from src.sgnnet.losses import safety_valve_loss, load_balance_loss
                safety  = safety_valve_loss(self.model.W_pos, self.box_size, task_loss=task_loss)
                lb_loss = load_balance_loss(scores.abs().sum(dim=0))

                loss = (
                    task_loss
                    + self.lambda_aux * aux_loss
                    + self.lambda_safety * safety
                    + self.lambda_lb * lb_loss
                )

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

            total_loss_sum += loss.item()
            task_loss_sum  += task_loss.item()
            aux_loss_sum   += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
            n_batches      += 1

        return {
            "train_loss":  total_loss_sum / max(n_batches, 1),
            "task_loss":   task_loss_sum  / max(n_batches, 1),
            "aux_loss":    aux_loss_sum   / max(n_batches, 1),
        }


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    mid_steps: List[int]
    lambda_aux: float
    residual_alpha: float = 0.0
    use_intermsup: bool = True


CONFIGS = [
    Config("Ref", "Ref  no intermediate supervision",                  [],         0.0,  use_intermsup=False),
    Config("A",   "A    λ=0.1 mid@step6",                              [5],        0.1),
    Config("B",   "B    λ=0.5 mid@step6  (strong signal)",             [5],        0.5),
    Config("C",   "C    λ=0.1 mid@steps3+9  (two checkpoints)",        [2, 8],     0.1),
    Config("D",   "D    λ=0.1 mid@step6 + GCNII α=0.05",               [5],        0.1,  residual_alpha=0.05),
]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    topo = topology_kwargs(N)
    topo.pop("K_in", None); topo.pop("K_iter", None)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, encoding_mode="fourier",
        **topo,
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    if not cfg.use_intermsup:
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_AH_IntermSupervision(
        resonant,
        alpha_ahebb=ALPHA_AHEBB,
        mid_steps=cfg.mid_steps,
        residual_alpha=cfg.residual_alpha,
    )


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data loaders (cached)
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
    print(f"Step 113 — Intermediate Supervision for K_iter routing")
    print(f"N={N}  D={D}  K_iter={K_ITER}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"Baseline (step69-A): {STEP69_REF:.4f}")
    print(f"{'='*70}\n")

    print("Configs:")
    for c in CONFIGS:
        print(f"  {c.key:4s}  λ_aux={c.lambda_aux:.2f}  mid_steps={c.mid_steps}  "
              f"res_α={c.residual_alpha:.2f}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step113_intermediate_supervision.json"

    for i, cfg in enumerate(CONFIGS):
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  mid_steps={cfg.mid_steps}  λ_aux={cfg.lambda_aux}")
        print(f"{'─'*60}")

        t0  = time.time()
        kw  = trainer_kwargs(N, n_epochs=EPOCHS)

        if cfg.use_intermsup:
            trainer = TrainerIntermSupervision(
                model=model,
                train_loader=get_loaders()[0],
                val_loader=get_loaders()[1],
                device=DEVICE,
                lambda_aux=cfg.lambda_aux,
                **kw,
            )
        else:
            trainer = Trainer(
                model=model,
                train_loader=get_loaders()[0],
                val_loader=get_loaders()[1],
                device=DEVICE,
                **kw,
            )

        history   = trainer.train(n_epochs=EPOCHS)
        elapsed   = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1
        vs_ref    = top1_best - STEP69_REF

        results[cfg.key] = {
            "N": N, "D": D, "K_iter": K_ITER,
            "mid_steps": cfg.mid_steps,
            "lambda_aux": cfg.lambda_aux,
            "residual_alpha": cfg.residual_alpha,
            "alpha_ahebb": ALPHA_AHEBB,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "vs_ref": round(vs_ref, 6),
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params,
            "label": cfg.label,
        }

        print(f"\n  top1_best={top1_best:.4f}  vs_ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary table
    print(f"\n{'='*70}")
    print(f"STEP 113 SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<6}  {'λ_aux':>5}  {'mid_steps':<14}  {'top1':>7}  {'vs_ref':>8}  label")
    print(f"{'─'*70}")
    for key, r in results.items():
        print(f"{key:<6}  {r['lambda_aux']:>5.2f}  {str(r['mid_steps']):<14}  "
              f"{r.get('top1_best',0):.4f}  {r['vs_ref']:>+.4f}  {r['label']}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
