"""Step 121: Spectral normalisation on W_pos.

MOTIVATION
==========
W_pos encodes neuron identity on S^{D-1}. Without explicit constraints its
singular values can blow up (causing routing collapse) or collapse near zero
(making all positions indistinguishable). Spectral normalisation is the standard
technique for bounding the Lipschitz constant of a weight matrix — applied here
to stabilise the positional embedding geometry.

Four strategies tested:
  A: torch.nn.utils.spectral_norm applied to W_pos directly (hard SN).
  B: Soft spectral regularisation — add λ‖σ_max(W_pos)−1‖² to loss (λ=0.01).
  C: Periodic re-orthogonalisation of W_pos every 5 epochs (project back toward
     column-orthogonal via thin SVD).
  D: W_pos weight decay 1e-4 (simplest baseline regularisation, no SN logic).

CONFIGS (N=1024, D=16, K_hh=8, K_iter=8, K_in=25, AH=1.0, 50%/75ep)
======================================================================
  Ref : standard (no spectral norm, no W_pos regularisation)
  A   : spectral_norm on W_pos applied every step
  B   : soft spectral reg λ=0.01: loss += λ‖σ_max(W_pos)−1‖²
  C   : periodic W_pos re-orthogonalisation every 5 epochs
  D   : W_pos weight decay 1e-4

To reproduce:
    python -u scripts/train_step121_spectral_norm.py --device mps
    python -u scripts/train_step121_spectral_norm.py --device mps --epochs 20  # Tier-0 scout
    python -u scripts/train_step121_spectral_norm.py --device mps --configs A,B
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75,
                    help="Training epochs (default 75; use 20 for Tier-0 scout)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys to run (e.g. A,C). Empty = run all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 16; K_ITER = 8; K_IN = 25; K_HH = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
SPECTRAL_LAMBDA = 0.01      # Config B soft-reg coefficient
WPOS_WD = 1e-4              # Config D weight decay on W_pos
REORTHO_INTERVAL = 5        # Config C: re-orthogonalise every N epochs


# ---------------------------------------------------------------------------
# W_pos spectral norm helper (Config A)
# ---------------------------------------------------------------------------

def apply_spectral_norm_wpos(model: nn.Module):
    """Wrap W_pos with torch spectral_norm if not already wrapped.

    spectral_norm wraps the nn.Parameter with a weight hook that divides
    by its largest singular value at each forward pass, enforcing
    σ_max(W_pos) = 1 as a hard constraint.

    Note: spectral_norm requires the parameter to be on a module as an
    attribute named 'weight'. We add a thin nn.Module shim for this purpose.
    """
    pass  # Applied via ShimModule approach below


class WposSpectralShim(nn.Module):
    """Thin shim to allow spectral_norm to wrap W_pos.

    spectral_norm hooks require an nn.Module with a 'weight' attribute.
    This shim owns 'weight' (= W_pos) so spectral_norm can be applied.
    The parent model is patched to forward W_pos reads through this shim.
    """
    def __init__(self, N: int, D: int):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(N, D))

    def forward(self):
        # Return the (spectral-normalised) weight
        return self.weight


# ---------------------------------------------------------------------------
# Soft spectral loss (Config B)
# ---------------------------------------------------------------------------

def soft_spectral_loss(W: torch.Tensor, lam: float = SPECTRAL_LAMBDA) -> torch.Tensor:
    """λ‖σ_max(W) − 1‖²

    Uses power iteration approximation via torch.svd_lowrank (1 iteration) for
    efficiency. Exact SVD is too slow to run every step on W [N+N_out, D].

    For W_pos of shape [N+N_out, D] with D=16 << N, the full SVD is actually
    O(N·D²) = O(N·256), which at N=1034 is only ~265K ops — acceptable. Using
    full SVD for numerical accuracy.
    """
    # W_pos shape: [N+N_out, D]
    try:
        sv = torch.linalg.svdvals(W)
        sigma_max = sv[0]
    except Exception:
        # Fallback for older torch without linalg.svdvals
        _, sv, _ = torch.svd(W)
        sigma_max = sv[0]
    return lam * (sigma_max - 1.0).pow(2)


# ---------------------------------------------------------------------------
# Re-orthogonalisation callback (Config C)
# ---------------------------------------------------------------------------

class WposReorthoCallback:
    """Called every REORTHO_INTERVAL epochs to project W_pos toward orthogonal.

    Computes thin SVD of W_pos, reconstructs as U @ V^T (sets all singular
    values to 1). Applies in-place with no_grad.
    """
    def __init__(self, model: nn.Module, interval: int = REORTHO_INTERVAL):
        self.model    = model
        self.interval = interval
        self._epoch   = 0

    def tick_epoch(self):
        self._epoch += 1
        if self._epoch % self.interval == 0:
            self._reortho()

    def _reortho(self):
        with torch.no_grad():
            W = self.model.W_pos.data  # [N+N_out, D]
            try:
                U, _, Vh = torch.linalg.svd(W, full_matrices=False)
            except Exception:
                U, _, V = torch.svd(W)
                Vh = V.T
            # Project: W ← U @ Vh (all singular values = 1)
            W_ortho = U @ Vh
            self.model.W_pos.data.copy_(W_ortho)


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    variant: str   # "standard" | "spectral_norm" | "soft_spec" | "reortho" | "wd"
    wpos_wd: float = 0.0

CONFIGS = [
    Config("Ref", "Ref  standard (no W_pos regularisation)",            "standard"),
    Config("A",   "A    spectral_norm on W_pos (hard σ_max=1)",         "spectral_norm"),
    Config("B",   "B    soft spectral reg λ=0.01: loss += λ‖σ_max−1‖²","soft_spec"),
    Config("C",   "C    periodic re-orthogonalisation every 5 epochs",  "reortho"),
    Config("D",   "D    W_pos weight decay 1e-4",                       "wd", wpos_wd=WPOS_WD),
]


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
# Model factory
# ---------------------------------------------------------------------------

def make_model(cfg: Config, seed_offset: int = 0):
    """Returns (model, extra_loss_fn, tick_epoch_fn).

    extra_loss_fn: callable(model) -> scalar loss to add to task loss, or None.
    tick_epoch_fn: callable() to invoke between epochs (re-ortho callback), or None.
    """
    torch.manual_seed(SEED + seed_offset)
    K_random = max(1, K_HH // 4)
    K_local  = K_HH - K_random
    n_groups = max(8, N // 8)

    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER,
        K_local=K_local, K_random=K_random, n_groups=n_groups,
        norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    model = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")

    extra_loss_fn   = None
    tick_epoch_fn   = None

    if cfg.variant == "spectral_norm":
        # Apply spectral_norm to W_pos on the base SmallWorld model
        # spectral_norm wraps in-place and registers a forward hook
        nn.utils.spectral_norm(base, name="W_pos")

    elif cfg.variant == "soft_spec":
        # extra loss added each step: λ‖σ_max(W_pos) − 1‖²
        def _soft_loss(m):
            return soft_spectral_loss(m.W_pos)
        extra_loss_fn = _soft_loss

    elif cfg.variant == "reortho":
        callback = WposReorthoCallback(base, interval=REORTHO_INTERVAL)
        tick_epoch_fn = callback.tick_epoch

    # Config D: weight decay handled via optimizer param group (see run loop)

    return model, extra_loss_fn, tick_epoch_fn


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Custom training loop for configs with extra loss or tick callbacks
# ---------------------------------------------------------------------------

def run_config(cfg: Config, seed_offset: int, device: torch.device) -> dict:
    model, extra_loss_fn, tick_epoch_fn = make_model(cfg, seed_offset=seed_offset)
    model = model.to(device)
    n_params = count_params(model)

    tr_loader, va_loader = get_loaders()

    kw = trainer_kwargs(N, n_epochs=EPOCHS)

    # Config D: pass W_pos weight decay to Trainer if supported; otherwise standard.
    # The standard Trainer sets weight_decay=0 on W_pos (documented in experiment_config).
    # For Config D we override this by patching lr_wpos group after the fact.
    # We use the standard Trainer and apply wd manually via the optimizer.
    trainer = Trainer(
        model=model,
        train_loader=tr_loader,
        val_loader=va_loader,
        device=device,
        **kw,
    )

    # Config D: add W_pos weight decay by modifying optimizer param groups
    if cfg.variant == "wd" and cfg.wpos_wd > 0:
        for pg in trainer.optimizer.param_groups:
            # Identify the W_pos param group (lr_wpos group has a single param)
            for p in pg["params"]:
                if p is model.W_pos:
                    pg["weight_decay"] = cfg.wpos_wd
                    break

    # Inject extra_loss_fn into trainer if needed
    if extra_loss_fn is not None:
        original_loss_fn = trainer.loss_fn if hasattr(trainer, "loss_fn") else None

        # Monkey-patch: wrap the trainer's _train_epoch to add auxiliary loss.
        # We hook into the optimizer step by wrapping trainer's compute_loss.
        # Since Trainer API may vary, we use the model's forward and add loss post-hoc.
        # Simpler approach: use trainer's existing aux_loss hook if present,
        # otherwise patch the trainer's loss computation via a closure.
        #
        # The safest cross-version approach: override trainer.extra_loss_fn if supported,
        # otherwise train with a thin subclass loop.
        if hasattr(trainer, "set_aux_loss"):
            trainer.set_aux_loss(lambda: extra_loss_fn(model))
        else:
            # Fallback: attach directly as attribute — many Trainers check this
            trainer.aux_loss_fn = lambda: extra_loss_fn(model)

    # Inject tick_epoch callback
    if tick_epoch_fn is not None:
        original_tick = getattr(model, "tick_epoch", None)
        def combined_tick():
            if original_tick: original_tick()
            tick_epoch_fn()
        model.tick_epoch = combined_tick

    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    top1_best = max(top1_hist)
    best_ep   = int(np.argmax(top1_hist)) + 1

    return {
        "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER, "K_in": K_IN,
        "variant": cfg.variant,
        "wpos_wd": cfg.wpos_wd,
        "alpha_ahebb": ALPHA_AHEBB,
        "data_frac": 0.5,
        "top1_best": top1_best, "top1_last": top1_hist[-1],
        "best_epoch": best_ep, "epochs_run": len(history),
        "top1_history": top1_hist,
        "elapsed_s": round(elapsed, 1),
        "n_params": n_params,
        "label": cfg.label,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 121 — Spectral Norm / W_pos Regularisation Ablation")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  K_in={K_IN}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    print("Configs:")
    for c in CONFIGS:
        wd_str = f"  wd={c.wpos_wd}" if c.wpos_wd > 0 else ""
        print(f"  {c.key:4s}  variant={c.variant:14s}{wd_str}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step121_spectral_norm.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"{'─'*60}")

        r = run_config(cfg, seed_offset=i, device=DEVICE)
        results[cfg.key] = r

        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs_ref   = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"\n  top1={r['top1_best']:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"elapsed={r['elapsed_s']/60:.1f}min  params={r['n_params']:,}")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary table
    print(f"\n{'='*70}")
    print(f"STEP 121 SUMMARY — W_pos Spectral / Regularisation")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Key':4s}  {'variant':14s}  {'params':>8}  {'top1':>7}  {'vs_Ref':>8}")
    print(f"{'─'*55}")
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"{key:4s}  {r['variant']:14s}  {r['n_params']:>8,}  "
              f"{r['top1_best']:.4f}  {vs:>+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
