"""Step 141: Split-D architecture + residual on hypersphere.

MOTIVATION
==========
Two structural changes addressing identified ceilings:

1. SPLIT-D INPUT: Currently at D=16, the input is [1_feature, 15_spatial]. The feature
   value is crushed into 1/16th of the representation — input collapse. Proposal:
   split D into D_act (activation channels) + D_pos (positional channels).
   - D_act: multiple learned projections of input features (richer content)
   - D_pos: Fourier spatial encoding (routing identity)
   This separates "what" (signal) from "where" (position).

2. RESIDUAL HYPERSPHERE: F.normalize(Z_new) every step is aggressive centering that
   provably causes oversmoothing (ICLR 2025). Replace with:
   Z = normalize(alpha * Z_old + (1-alpha) * Z_new)
   This preserves initial signal while re-projecting to S^{D-1}.

CONFIGS (N=1024, D=16, K_hh=8, K_iter=8, K_in=25, AH=1.0, 50%/75ep)
=====================================================================
  Ref     : Standard D=16 (1 feat + 15 spatial), normalize every step
  A       : Split D=16 (12 act + 4 pos), normalize every step
  B       : Split D=16 (8 act + 8 pos), normalize every step
  C       : Standard D=16, residual normalize (α=0.3)
  D       : Split D=16 (12 act + 4 pos) + residual normalize (α=0.3)
  E       : Split D=16 (12 act + 4 pos), normalize act/pos separately

To reproduce:
    python -u scripts/train_step141_splitD_residual.py --device mps
    python -u scripts/train_step141_splitD_residual.py --device mps --epochs 20
"""
from __future__ import annotations

import argparse
import json
import math
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
from src.sgnnet.encoding              import compute_fourier_encoding
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs, topology_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75,
                    help="Training epochs (default 75; use 20 for scout)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys. Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 16; K_ITER = 8; K_IN = 25; K_HH = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0


# ---------------------------------------------------------------------------
# Split-D Model: separate activation and positional channels
# ---------------------------------------------------------------------------

class SGNNET_SplitD(nn.Module):
    """SGNNET with split activation/positional dimensions.

    Input encoding:
      - D_pos dims: Fourier spatial encoding (identity of input pixel)
      - D_act dims: learned linear projections of feature value (content)

    Routing:
      - AH suppression uses ONLY positional dims (cosine similarity on D_pos)
      - Signal propagates through ALL dims (D_act + D_pos)

    Normalization modes:
      - "joint":    F.normalize over full D (standard)
      - "residual": Z = normalize(alpha * Z_old + (1-alpha) * Z_new)
      - "split":    normalize D_act and D_pos subspaces independently
    """

    def __init__(self, N_hidden: int, N_out: int, D: int, N_in: int,
                 K_in: int, K_iter: int, K_local: int, K_random: int,
                 n_groups: int, D_act: int, D_pos: int,
                 alpha_ahebb: float = 1.0, alpha_reflect: float = 0.5,
                 norm_mode: str = "joint", residual_alpha: float = 0.3):
        super().__init__()
        assert D_act + D_pos == D, f"D_act({D_act}) + D_pos({D_pos}) != D({D})"
        self.N_hidden     = N_hidden
        self.N_out        = N_out
        self.D            = D
        self.D_act        = D_act
        self.D_pos        = D_pos
        self.K_iter       = K_iter
        self.K_in         = K_in
        self.alpha_ahebb  = alpha_ahebb
        self.alpha_reflect = alpha_reflect
        self.norm_mode    = norm_mode
        self.residual_alpha = residual_alpha

        # Positional encoding: W_pos [N+N_out, D] — used for topology + AH
        self.W_pos = nn.Parameter(torch.rand(N_hidden + N_out, D))

        # Theta for routing threshold
        self.theta = nn.Parameter(torch.full((N_hidden,), 0.1))

        # Input feature projection: 1 scalar → D_act dims
        # Each input pixel gets D_act learned projections instead of 1
        self.input_proj = nn.Linear(1, D_act, bias=False)

        # Spatial encoding for positional dims: [N_in, D_pos]
        spatial_full = compute_fourier_encoding(N_in, D=D_pos + 1)  # D_pos+1 because it reserves 1
        # Take only D_pos dims (drop the feature-value slot)
        self.register_buffer("spatial_pos", spatial_full[:, :D_pos])

        # Input connectivity: [N_hidden, K_in]
        conn_in = torch.stack([
            torch.randperm(N_in)[:K_in] for _ in range(N_hidden)
        ])
        self.register_buffer("conn_in", conn_in)

        # Hidden connectivity: small-world topology
        from src.sgnnet.model_smallworld import _build_smallworld_conn
        conn_hh = _build_smallworld_conn(
            N_hidden, K_local=K_local, K_random=K_random,
            n_groups=n_groups,
        )
        self.register_buffer("conn_hh", conn_hh)

        # Readout
        self.fc_out = nn.Linear(D, N_out, bias=True)

    @property
    def W_phase(self):
        return self.W_pos  # compatibility

    def tick_epoch(self):
        pass

    def _seed(self, x: torch.Tensor) -> torch.Tensor:
        """Seed with split activation/positional encoding.

        x: [B, N_in] raw features.
        Returns: [B, N_hidden, D] where D = D_act + D_pos.
        """
        B = x.shape[0]

        # Activation channels: project each feature scalar to D_act dims
        # x: [B, N_in] → [B, N_in, 1] → [B, N_in, D_act]
        feat_proj = self.input_proj(x.unsqueeze(-1))  # [B, N_in, D_act]

        # Positional channels: spatial encoding [N_in, D_pos] → [B, N_in, D_pos]
        spatial = self.spatial_pos.unsqueeze(0).expand(B, -1, -1)

        # Combine: [B, N_in, D_act + D_pos]
        A_input = torch.cat([feat_proj, spatial], dim=-1)  # [B, N_in, D]

        # Gather and sum over K_in inputs per neuron
        Z = A_input[:, self.conn_in, :].sum(dim=2)  # [B, N_hidden, D]

        return F.normalize(Z, dim=-1)

    def _normalize(self, Z_new: torch.Tensor, Z_old: torch.Tensor) -> torch.Tensor:
        """Apply configured normalization."""
        Z_clamped = Z_new.clamp(-10, 10)
        if self.norm_mode == "joint":
            return F.normalize(Z_clamped, dim=-1)
        elif self.norm_mode == "residual":
            Z_mix = self.residual_alpha * Z_old + (1 - self.residual_alpha) * Z_clamped
            return F.normalize(Z_mix, dim=-1)
        elif self.norm_mode == "split":
            # Normalize activation and positional subspaces independently
            act = F.normalize(Z_clamped[..., :self.D_act], dim=-1)
            pos = F.normalize(Z_clamped[..., self.D_act:], dim=-1)
            return torch.cat([act, pos], dim=-1)
        else:
            return F.normalize(Z_clamped, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self._seed(x)

        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh = self.conn_hh
        N_h = self.N_hidden

        # AH suppression: use ONLY positional dims for similarity
        if self.D_pos > 0:
            W_n_pos = F.normalize(self.W_pos[:N_h, self.D_act:], dim=-1)
            pos_sim = (W_n_pos.unsqueeze(1) * W_n_pos[conn_hh]).sum(-1)
        else:
            # Fallback: use full W_pos
            W_n = F.normalize(self.W_pos[:N_h], dim=-1)
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)

        supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                 ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.K_iter):
            Z_fwd  = F.relu(Z - theta_pos)
            Z_nb   = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected

            Z = self._normalize(Z_new, Z)

        # Readout: mean pool + linear
        Z_mean = Z.mean(dim=1)  # [B, D]
        return self.fc_out(Z_mean)


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    D_act: int
    D_pos: int
    norm_mode: str
    residual_alpha: float = 0.3
    use_split: bool = True  # False → use standard SGNNET_AntiHebbian

CONFIGS = [
    Config("Ref", "Ref  standard D=16 (1feat+15spatial), joint norm",
           0, 0, "joint", use_split=False),
    Config("A",   "A    split 12act+4pos, joint norm",
           12, 4, "joint"),
    Config("B",   "B    split 8act+8pos, joint norm",
           8, 8, "joint"),
    Config("C",   "C    standard D=16, residual norm α=0.3",
           0, 0, "residual", residual_alpha=0.3, use_split=False),
    Config("D",   "D    split 12act+4pos + residual norm α=0.3",
           12, 4, "residual", residual_alpha=0.3),
    Config("E",   "E    split 12act+4pos, split norm (act/pos independent)",
           12, 4, "split"),
]


# ---------------------------------------------------------------------------
# Data (cached, 50%)
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

def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    K_random = max(1, K_HH // 4)
    K_local  = K_HH - K_random
    n_groups = max(8, N // 8)

    if not cfg.use_split:
        # Standard SGNNET_AntiHebbian for Ref and C
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
        # For Config C: we can't easily patch residual norm into existing model
        # So C uses the SplitD model with D_act=1, D_pos=15 (equivalent to standard)
        if cfg.norm_mode == "residual":
            model = SGNNET_SplitD(
                N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                K_in=K_IN, K_iter=K_ITER,
                K_local=K_local, K_random=K_random, n_groups=n_groups,
                D_act=1, D_pos=D-1,
                alpha_ahebb=ALPHA_AHEBB, alpha_reflect=ALPHA_REFLECT,
                norm_mode="residual", residual_alpha=cfg.residual_alpha,
            )
        return model

    return SGNNET_SplitD(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER,
        K_local=K_local, K_random=K_random, n_groups=n_groups,
        D_act=cfg.D_act, D_pos=cfg.D_pos,
        alpha_ahebb=ALPHA_AHEBB, alpha_reflect=ALPHA_REFLECT,
        norm_mode=cfg.norm_mode, residual_alpha=cfg.residual_alpha,
    )


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 141 — Split-D Architecture + Residual Hypersphere")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  K_in={K_IN}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    for c in CONFIGS:
        split = f"({c.D_act}act+{c.D_pos}pos)" if c.use_split or c.norm_mode == "residual" else "(standard)"
        print(f"  {c.key:4s}  {split:15s}  norm={c.norm_mode:8s}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step141_splitD_residual.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  norm={cfg.norm_mode}")
        print(f"{'─'*60}")

        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(
            model=model, train_loader=get_loaders()[0],
            val_loader=get_loaders()[1], device=DEVICE, **kw,
        )
        history = trainer.train(n_epochs=EPOCHS)
        elapsed = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1

        results[cfg.key] = {
            "N": N, "D": D, "D_act": cfg.D_act, "D_pos": cfg.D_pos,
            "K_hh": K_HH, "K_iter": K_ITER, "K_in": K_IN,
            "norm_mode": cfg.norm_mode,
            "residual_alpha": cfg.residual_alpha,
            "alpha_ahebb": ALPHA_AHEBB,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params,
            "label": cfg.label,
        }

        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs_ref = top1_best - ref_best if ref_best > 0 else 0
        print(f"\n  top1={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min  params={n_params:,}")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 141 SUMMARY — Split-D + Residual Hypersphere")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Key':4s}  {'D_act':>5}  {'D_pos':>5}  {'norm':>8}  "
          f"{'params':>8}  {'top1':>7}  {'vs_Ref':>8}")
    print(f"{'─'*60}")
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        d_a = r.get("D_act", 1)
        d_p = r.get("D_pos", D-1)
        print(f"{key:4s}  {d_a:>5}  {d_p:>5}  {r['norm_mode']:>8}  "
              f"{r['n_params']:>8}  {r['top1_best']:.4f}  {vs:>+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
