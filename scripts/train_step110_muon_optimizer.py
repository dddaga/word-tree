"""Step 110: Muon optimizer for W_pos (autoresearch-inspired).

MOTIVATION
==========
Karpathy's autoresearch (github.com/karpathy/autoresearch) discovered that
orthogonalizing matrix gradients via Newton-Schulz iteration (Muon optimizer)
consistently beats AdamW for matrix parameters. SGNNET's primary trainable
parameter is W_pos — a 2D matrix [N_hidden, D] of neuron positions.

Muon hypothesis for SGNNET: W_pos lives in a high-dimensional space where
neurons compete for territory. Orthogonalized gradient updates naturally
promote diversity (orthogonal updates → spread out in D-dim space), which
directly amplifies AH's anti-correlation signal. AdamW with element-wise
scaling may allow correlated W_pos clusters to persist.

MECHANISM
=========
  zeropower(G) = G ÷ ‖G‖_F, orthogonalized via 5-step Newton-Schulz:
    X₀ = G / ‖G‖_F
    Xₙ₊₁ = aXₙ + bAXₙ + cA²Xₙ   where A = XₙXₙᵀ, (a,b,c) = (3.4445, −4.7750, 2.0315)
  Update: W_pos -= lr * (μ * buf + zeropower(grad))   (Nesterov-style)

Key property: every update step moves W_pos by a FIXED amount regardless of
gradient magnitude. This is especially useful if early gradients are large
(noisy), as Muon normalizes them away.

AMP NOTE: Muon bypasses GradScaler. AMP is disabled for Muon configs.
This is safe — MPS/CPU both run float32 natively; AMP is only a speedup.

CONFIGS (N=1024, D=64, K_iter=12, AH=1.0, turing=0.0, reflect=0.5, 50%/75ep)
===============================================================================
  Ref : AdamW on W_pos (standard baseline)
  A   : Muon on W_pos, lr=2.364e-3, momentum=0.95 (direct swap)
  B   : Muon, lr=0.02  (Muon's canonical lr from autoresearch)
  C   : Muon, lr=2.364e-3, momentum warmup 0.85→0.95 over 300 steps
  D   : Muon, lr=2.364e-3 + squared ReLU activation (relu(x)²)

To reproduce:
    python -u scripts/train_step110_muon_optimizer.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.experiment_config   import trainer_kwargs, topology_kwargs, run_metadata, GA_BEST, scaled_lambda_safety
from src.training.trainer             import safety_valve_loss, load_balance_loss
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75,
                    help="Training epochs (default 75; use 20 for Tier-0 scout)")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 64; K_ITER = 12
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
LR_WPOS = GA_BEST["lr_Wpos"]   # 2.364e-3
STEP69_REF = 0.8336


# ── Muon: Newton-Schulz orthogonalization ────────────────────────────────────

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize matrix G via 5-step Newton-Schulz iteration.

    Returns a matrix with the same shape as G but orthogonal rows/cols.
    Coefficients (a,b,c) chosen so the polynomial maps singular values to 1.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float() / (G.norm() + 1e-7)
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ A @ X)
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X.to(G.dtype)


class MuonParam:
    """State for a single Muon-tracked parameter."""
    def __init__(self, p: torch.Tensor):
        self.buf = torch.zeros_like(p)
        self.step = 0


def muon_step(p: torch.Tensor, state: MuonParam,
              lr: float, momentum: float,
              momentum_end: float, total_steps: int,
              ns_steps: int = 5) -> None:
    """Apply one Muon update to parameter p in-place.

    Momentum warms up linearly from `momentum` to `momentum_end` over
    `total_steps` steps. Set momentum == momentum_end for constant schedule.
    """
    g = p.grad
    if g is None:
        return
    # Momentum warmup
    frac = min(state.step / max(total_steps, 1), 1.0)
    mu   = momentum + (momentum_end - momentum) * frac
    state.step += 1

    state.buf.mul_(mu).add_(g)
    # Nesterov: use lookahead gradient
    g_nes = g.add(state.buf, alpha=mu)
    # Orthogonalize
    g_orth = zeropower_via_newtonschulz5(g_nes, steps=ns_steps)
    p.data.add_(g_orth, alpha=-lr)


# ── Squared ReLU activation wrapper ──────────────────────────────────────────

class SGNNET_AH_SquaredReLU(nn.Module):
    """AH routing with relu(x)² activation instead of relu(x)."""

    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float):
        super().__init__()
        self.m     = resonant
        self.alpha = alpha_ahebb

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden

        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            act    = F.relu(Z - theta_pos)
            Z_fwd  = act * act                                # squared ReLU
            Z_nb   = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = act - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_muon(model: nn.Module, tr, va, cfg, device) -> list[dict]:
    """Minimal training loop with Muon optimizer for W_pos."""
    model = model.to(device)
    lambda_safety = scaled_lambda_safety(N)
    lambda_lb     = 0.01
    box_size      = 1.0

    # Muon state for W_pos
    muon_state = MuonParam(model.W_pos)
    total_steps = EPOCHS * len(tr)   # for momentum warmup schedule

    # Scheduler: ReduceLROnPlateau on train_loss
    lr  = cfg.lr
    sched_patience = 10
    sched_factor   = 0.5
    min_lr         = 1e-7
    no_improve     = 0
    best_train_loss = float("inf")
    patience_count  = 0

    early_stop_patience = 50
    early_stop_delta    = 5e-4
    best_val            = 0.0
    es_count            = 0

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for feats, soft_labels, _ in tr:
            feats       = feats.to(device)
            soft_labels = soft_labels.to(device)

            # Zero W_pos grad manually
            if model.W_pos.grad is not None:
                model.W_pos.grad.zero_()

            scores    = model(feats)
            task_loss = F.kl_div(F.log_softmax(scores, dim=-1),
                                 soft_labels, reduction="batchmean")
            safety    = safety_valve_loss(model.W_pos, box_size, task_loss=task_loss)
            lb_loss   = load_balance_loss(scores.abs().sum(dim=0))
            loss      = task_loss + lambda_safety * safety + lambda_lb * lb_loss

            loss.backward()

            # Muon update on W_pos
            muon_step(model.W_pos, muon_state,
                      lr=lr, momentum=cfg.momentum,
                      momentum_end=cfg.momentum_end,
                      total_steps=total_steps)

            # Clamp to box
            with torch.no_grad():
                model.W_pos.clamp_(0.0, box_size)

            train_loss_sum += loss.item()
            n_batches += 1

        avg_train_loss = train_loss_sum / max(n_batches, 1)

        # ReduceLROnPlateau
        if avg_train_loss < best_train_loss - 1e-4:
            best_train_loss = avg_train_loss
            patience_count  = 0
        else:
            patience_count += 1
            if patience_count >= sched_patience and lr > min_lr:
                lr = max(lr * sched_factor, min_lr)
                patience_count = 0

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for feats, _, labels in va:
                feats  = feats.to(device)
                labels = labels.to(device)
                preds  = model(feats).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        val_top1 = correct / max(total, 1)

        history.append({"val_top1": val_top1, "train_loss": avg_train_loss, "lr": lr})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  e{epoch:3d}  loss={avg_train_loss:.4f}  top1={val_top1:.4f}  lr={lr:.2e}")

        # Early stopping
        if val_top1 > best_val + early_stop_delta:
            best_val  = val_top1
            es_count  = 0
        else:
            es_count += 1
            if es_count >= early_stop_patience:
                print(f"  [early stop at e{epoch}]")
                break

        if hasattr(model, "tick_epoch"):
            model.tick_epoch()

    return history


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    key: str; label: str
    use_muon: bool
    lr: float
    momentum: float = 0.95
    momentum_end: float = 0.95
    squared_relu: bool = False


CONFIGS = [
    Config("Ref", "Ref  AdamW baseline (K_iter=12)",
           use_muon=False, lr=LR_WPOS),
    Config("A",   "A    Muon lr=2.364e-3 momentum=0.95",
           use_muon=True,  lr=LR_WPOS),
    Config("B",   "B    Muon lr=0.02 (autoresearch canonical)",
           use_muon=True,  lr=0.02),
    Config("C",   "C    Muon lr=2.364e-3 momentum warmup 0.85→0.95",
           use_muon=True,  lr=LR_WPOS, momentum=0.85, momentum_end=0.95),
    Config("D",   "D    Muon lr=2.364e-3 + squared ReLU",
           use_muon=True,  lr=LR_WPOS, squared_relu=True),
]

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=N_IN, N_hidden=N, N_out=N_OUT,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_ITER, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    if cfg.squared_relu:
        return SGNNET_AH_SquaredReLU(resonant, ALPHA_AHEBB)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run(cfg: Config, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{cfg.label}\n{'='*70}")
    tr, va = get_loaders()

    t0 = time.time()
    if cfg.use_muon:
        history = train_muon(model, tr, va, cfg, DEVICE)
    else:
        # Standard Trainer path for Ref
        from src.training.trainer import Trainer
        tk = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
        history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best      = max(top1_hist)
    best_ep   = int(np.argmax(top1_hist)) + 1

    result = {
        "label": cfg.label,
        "use_muon": cfg.use_muon, "lr": cfg.lr,
        "momentum": cfg.momentum, "momentum_end": cfg.momentum_end,
        "squared_relu": cfg.squared_relu,
        "top1_best": best, "top1_last": top1_hist[-1],
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "best_epoch_frac": round(best_ep / len(history), 3),
        "step69_ref": STEP69_REF,
        "delta_vs_ref": round(best - STEP69_REF, 4),
        "params": count_params(model),
        "top1_history": top1_hist,
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1={best:.4f}  ep={best_ep}/{len(history)}  vs_ref={best-STEP69_REF:+.4f}  t={elapsed:.0f}s")
    return result


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  N={N}  K_iter={K_ITER}  Data: 50%")
    print(f"Step 110: Muon optimizer vs AdamW for W_pos")
    print(f"Baseline: step69 Ref = {STEP69_REF:.4f}  (lr_wpos={LR_WPOS:.3e})\n")
    for c in CONFIGS:
        tag = f"muon={'Y' if c.use_muon else 'N'}  lr={c.lr:.3e}  mu={c.momentum:.2f}→{c.momentum_end:.2f}  relu²={c.squared_relu}"
        print(f"  {c.key:4s}  {tag}")
    print()
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")
    results = {}
    out_path = ROOT / "results" / "train_step110_muon_optimizer.json"
    for i, cfg in enumerate(CONFIGS):
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        meta  = {"N": N, "D": D, "K_iter": K_ITER, "lr": cfg.lr,
                 "momentum": cfg.momentum, "use_muon": cfg.use_muon,
                 "squared_relu": cfg.squared_relu, "data_frac": 0.5}
        results[cfg.key] = run(cfg, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")
    print(f"\n{'='*70}\nSTEP 110 COMPLETE\n")
    print(f"  {'Key':4s}  {'muon':>4s}  {'lr':>9s}  {'top1':>8s}  {'vs_ref':>8s}  {'ep':>4s}")
    for c in CONFIGS:
        if c.key in results:
            r = results[c.key]
            print(f"  {c.key:4s}  {'Y' if c.use_muon else 'N':>4s}  {c.lr:>9.3e}"
                  f"  {r['top1_best']:.4f}  {r['delta_vs_ref']:+.4f}  {r['best_epoch']:>4d}")
    w = max(results, key=lambda k: results[k]["top1_best"])
    print(f"\n  Winner: {w}")
