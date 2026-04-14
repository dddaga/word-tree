"""Step 149: Input de-squashification — richer input encoding.

MOTIVATION
==========
Current _seed(): each input pixel → 1 scalar + (D-1) spatial dims. The scalar
is 1/D of the representation — massive information loss (392× compression further
squashed to 1/16th bandwidth at D=16).

Three remedies tested here:
  1. Multi-feature projection: each pixel → K_feat learned features (not just 1 scalar)
  2. Attention-weighted scatter: learned importance weights over K_in inputs per neuron
  3. Combined: multi-feature + attention

CONFIGS (N=1024, D=16, K_hh=8, K_iter=8, K_in=25, AH=1.0, 50%/75ep)
=====================================================================
  Ref : Standard scatter-sum (1 scalar + 15 spatial)
  A   : Multi-feature: nn.Linear(1, 4) per pixel → 4 feat + 12 spatial
  B   : Multi-feature: nn.Linear(1, 8) per pixel → 8 feat + 8 spatial
  C   : Attention scatter: learned α[h,k] weights over K_in inputs
  D   : Multi-feature(4) + attention scatter (compound)

To reproduce:
    python -u scripts/train_step149_input_desquash.py --device mps
    python -u scripts/train_step149_input_desquash.py --device mps --epochs 20
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

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld, _build_smallworld_conn
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.sgnnet.encoding              import compute_fourier_encoding
from src.sgnnet.model_wave            import _make_binary_c
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


class SGNNET_InputDesquash(nn.Module):
    """SGNNET with enriched input encoding.

    Modes:
      "standard"    — 1 scalar + (D-1) spatial (current)
      "multi_feat"  — K_feat learned projections + (D-K_feat) spatial
      "attn_scatter" — learned attention weights over K_in inputs per neuron
      "both"        — multi_feat + attn_scatter
    """

    def __init__(self, N_hidden, N_out, D, N_in, K_in, K_iter,
                 K_local, K_random, n_groups,
                 mode="standard", K_feat=4,
                 alpha_ahebb=1.0, alpha_reflect=0.5):
        super().__init__()
        self.N_hidden = N_hidden
        self.D = D
        self.K_in = K_in
        self.K_iter = K_iter
        self.mode = mode
        self.K_feat = K_feat
        self.alpha_ahebb = alpha_ahebb
        self.alpha_reflect = alpha_reflect

        self.W_pos = nn.Parameter(torch.rand(N_hidden + N_out, D))
        self.theta = nn.Parameter(torch.full((N_hidden,), 0.1))

        # Input feature projection
        D_spatial = D - K_feat if mode in ("multi_feat", "both") else D - 1
        D_feat = K_feat if mode in ("multi_feat", "both") else 1

        if mode in ("multi_feat", "both"):
            self.feat_proj = nn.Linear(1, K_feat, bias=False)
        else:
            self.feat_proj = None

        # Spatial encoding
        spatial_full = compute_fourier_encoding(N_in, D=D_spatial + 1)
        self.register_buffer("spatial_enc", spatial_full[:, :D_spatial])
        self.D_feat = D_feat
        self.D_spatial = D_spatial

        # Attention weights for scatter
        if mode in ("attn_scatter", "both"):
            # Per-neuron attention over its K_in inputs: [N_hidden, K_in]
            self.attn_logits = nn.Parameter(torch.zeros(N_hidden, K_in))
        else:
            self.attn_logits = None

        # Connectivity
        conn_in = torch.stack([
            torch.randperm(N_in)[:K_in] for _ in range(N_hidden)
        ])
        self.register_buffer("conn_in", conn_in)

        conn_hh = _build_smallworld_conn(
            N_hidden, K_local=K_local, K_random=K_random,
            n_groups=n_groups,
        )
        self.register_buffer("conn_hh", conn_hh)

        # C_ho: dense hidden→output projection (N_out=10 stays small)
        self.register_buffer("C_ho_mask", _make_binary_c(N_hidden, N_out, sparsity=0.0))
        self.N_out = N_out

    @property
    def W_phase(self):
        return self.W_pos

    def tick_epoch(self):
        pass

    def _seed(self, x):
        B = x.shape[0]

        # Feature encoding
        if self.feat_proj is not None:
            feat = self.feat_proj(x.unsqueeze(-1))  # [B, N_in, K_feat]
        else:
            feat = x.unsqueeze(-1)  # [B, N_in, 1]

        # Spatial encoding
        spatial = self.spatial_enc.unsqueeze(0).expand(B, -1, -1)  # [B, N_in, D_spatial]

        # Combine
        A_input = torch.cat([feat, spatial], dim=-1)  # [B, N_in, D]

        # Gather: [B, N_hidden, K_in, D]
        gathered = A_input[:, self.conn_in, :]

        # Aggregate
        if self.attn_logits is not None:
            # Attention-weighted scatter
            attn_w = F.softmax(self.attn_logits, dim=-1)  # [N, K_in]
            attn_w = attn_w.unsqueeze(0).unsqueeze(-1)     # [1, N, K_in, 1]
            Z = (gathered * attn_w).sum(dim=2)              # [B, N, D]
        else:
            Z = gathered.sum(dim=2)

        return F.normalize(Z, dim=-1)

    def forward(self, x):
        Z = self._seed(x)

        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh = self.conn_hh
        N_h = self.N_hidden

        W_n = F.normalize(self.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                 ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        C_ho = self.C_ho_mask.float()
        A_out = torch.einsum("bhd,ho->bod", Z, C_ho)      # [B, N_out, D]
        W_out = self.W_pos[self.N_hidden:]
        W_out_norm = F.normalize(W_out, dim=-1)
        return (A_out * W_out_norm.unsqueeze(0)).sum(dim=-1)  # [B, N_out]


@dataclass
class Config:
    key: str
    label: str
    mode: str
    K_feat: int
    use_custom: bool = True

CONFIGS = [
    Config("Ref", "Ref  standard (1 feat + 15 spatial)",   "standard",     1, use_custom=False),
    Config("A",   "A    multi-feat K=4 (4 feat + 12 spa)", "multi_feat",   4),
    Config("B",   "B    multi-feat K=8 (8 feat + 8 spa)",  "multi_feat",   8),
    Config("C",   "C    attention scatter over K_in",       "attn_scatter", 1),
    Config("D",   "D    multi-feat K=4 + attention",        "both",         4),
]


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


def make_model(cfg, seed_offset=0):
    torch.manual_seed(SEED + seed_offset)
    K_random = max(1, K_HH // 4)
    K_local = K_HH - K_random
    n_groups = max(8, N // 8)

    if not cfg.use_custom:
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
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")

    return SGNNET_InputDesquash(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER,
        K_local=K_local, K_random=K_random, n_groups=n_groups,
        mode=cfg.mode, K_feat=cfg.K_feat,
        alpha_ahebb=ALPHA_AHEBB, alpha_reflect=ALPHA_REFLECT,
    )


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    print(f"\n{'='*70}")
    print(f"Step 149 — Input De-squashification")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  K_in={K_IN}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    for c in CONFIGS:
        print(f"  {c.key:4s}  {c.label}")
    print()

    get_loaders()
    results = {}
    out_path = ROOT / "results" / "train_step149_input_desquash.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active = [(i, c) for i, c in enumerate(CONFIGS)
              if not cfg_filter or c.key in cfg_filter]

    for i, cfg in active:
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}  params={n_params:,}")
        print(f"{'─'*60}")

        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=get_loaders()[0],
                         val_loader=get_loaders()[1], device=DEVICE, **kw)
        history = trainer.train(n_epochs=EPOCHS)
        elapsed = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep = int(np.argmax(top1_hist)) + 1

        results[cfg.key] = {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "mode": cfg.mode, "K_feat": cfg.K_feat,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params, "label": cfg.label,
        }

        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs = top1_best - ref_best if ref_best > 0 else 0
        print(f"\n  top1={top1_best:.4f}  vs_Ref={vs:+.4f}  params={n_params:,}")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}\nSTEP 149 SUMMARY\n{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for k, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"  {k:4s}  {r['mode']:>14s}  K_feat={r['K_feat']}  "
              f"params={r['n_params']:>8}  top1={r['top1_best']:.4f}  vs_Ref={vs:+.4f}")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
