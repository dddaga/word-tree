"""Step 150: Positional max-pool readout — max-pool + source neuron identity.

MOTIVATION
==========
Mean-pool readout discards all graph structure — which neuron contributed what.
Attention readout (step118) catastrophically failed (−60pp) because learned
attention over N neurons is too many parameters to learn.

Positional max-pool is simpler and parameter-free:
  1. For each of D dims, find which neuron has the MAXIMUM activation
  2. Record that neuron's positional encoding (W_pos index)
  3. The readout sees: [max_value per dim, position of winner per dim]
  4. This tells the classifier WHAT the strongest signal is AND WHERE it came from

The positional information is compact: we encode the winner neuron index as
a small learned embedding or use the neuron's W_pos directly.

CONFIGS (N=1024, D=16, K_hh=8, K_iter=8, K_in=25, AH=1.0, 50%/75ep)
=====================================================================
  Ref : Standard mean-pool + linear (baseline)
  A   : Max-pool only (no position info) — ablation
  B   : Max-pool + winner W_pos (D_pos dims per winning dim)
  C   : Max-pool + winner index embedding (learned, 4-dim per winner)
  D   : Max-pool + mean-pool concatenated (both signals)
  E   : Top-2 pool: for each dim, top-2 neurons + their positions

To reproduce:
    python -u scripts/train_step150_positional_maxpool.py --device mps
    python -u scripts/train_step150_positional_maxpool.py --device mps --epochs 20
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


class SGNNET_PosMaxPool(nn.Module):
    """SGNNET with positional max-pool readout.

    Readout modes:
      "mean"       — standard mean-pool (baseline)
      "max"        — max-pool per dimension (no position info)
      "max_wpos"   — max-pool + W_pos of winner neurons
      "max_embed"  — max-pool + learned embedding of winner index
      "max_mean"   — max-pool || mean-pool concatenated
      "top2_wpos"  — top-2 per dim + their W_pos
    """

    def __init__(self, base_model: nn.Module, W_pos: nn.Parameter,
                 readout_mode: str = "mean", embed_dim: int = 4):
        super().__init__()
        self.base = base_model
        self.readout_mode = readout_mode

        # Compute readout input dimension based on mode
        D_ = D
        if readout_mode == "mean":
            readout_dim = D_
        elif readout_mode == "max":
            readout_dim = D_
        elif readout_mode == "max_wpos":
            # D max values + D * D_pos winner positions (use 4 pos dims)
            self.pos_dim = min(4, D_)
            readout_dim = D_ + D_ * self.pos_dim
        elif readout_mode == "max_embed":
            self.embed_dim = embed_dim
            self.winner_embed = nn.Embedding(N, embed_dim)
            readout_dim = D_ + D_ * embed_dim
        elif readout_mode == "max_mean":
            readout_dim = D_ * 2  # max || mean
        elif readout_mode == "top2_wpos":
            self.pos_dim = min(4, D_)
            readout_dim = D_ * 2 + D_ * 2 * self.pos_dim  # 2 values + 2 positions per dim
        else:
            readout_dim = D_

        self.fc_out = nn.Linear(readout_dim, N_OUT, bias=True)

    @property
    def W_pos(self):
        return self.base.W_pos if hasattr(self.base, 'W_pos') else self.base.m.W_pos

    @property
    def W_phase(self):
        return self.W_pos

    def tick_epoch(self):
        if hasattr(self.base, "tick_epoch"):
            self.base.tick_epoch()

    def _readout(self, Z: torch.Tensor) -> torch.Tensor:
        """Custom readout with positional max-pool.

        Z: [B, N, D] — final activations after routing.
        """
        B = Z.shape[0]
        W_pos = self.W_pos[:N]  # [N, D]

        if self.readout_mode == "mean":
            return self.fc_out(Z.mean(dim=1))

        elif self.readout_mode == "max":
            max_vals, _ = Z.max(dim=1)  # [B, D]
            return self.fc_out(max_vals)

        elif self.readout_mode == "max_wpos":
            max_vals, max_idx = Z.max(dim=1)  # [B, D], [B, D]
            # For each dim d, get W_pos of the winning neuron (first pos_dim dims)
            # max_idx[b,d] is the neuron index that won for dim d in batch b
            W_pos_short = W_pos[:, :self.pos_dim]  # [N, pos_dim]
            # Gather: for each (b,d), look up W_pos_short[max_idx[b,d]]
            # max_idx: [B, D] → expand for pos_dim
            idx_expanded = max_idx.unsqueeze(-1).expand(-1, -1, self.pos_dim)  # [B, D, pos_dim]
            winner_pos = torch.gather(
                W_pos_short.unsqueeze(0).expand(B, -1, -1),  # [B, N, pos_dim]
                1,
                idx_expanded  # [B, D, pos_dim] — gather dim 1 (N)
            )  # [B, D, pos_dim]
            # Flatten: [B, D + D*pos_dim]
            features = torch.cat([max_vals, winner_pos.flatten(1)], dim=-1)
            return self.fc_out(features)

        elif self.readout_mode == "max_embed":
            max_vals, max_idx = Z.max(dim=1)  # [B, D], [B, D]
            # Embed winner neuron indices
            winner_emb = self.winner_embed(max_idx)  # [B, D, embed_dim]
            features = torch.cat([max_vals, winner_emb.flatten(1)], dim=-1)
            return self.fc_out(features)

        elif self.readout_mode == "max_mean":
            max_vals, _ = Z.max(dim=1)
            mean_vals = Z.mean(dim=1)
            features = torch.cat([max_vals, mean_vals], dim=-1)
            return self.fc_out(features)

        elif self.readout_mode == "top2_wpos":
            # Top-2 per dimension
            top2_vals, top2_idx = Z.topk(2, dim=1)  # [B, 2, D]
            top2_vals = top2_vals.permute(0, 2, 1).flatten(1)  # [B, D*2]
            # Positions of top-2 winners
            W_pos_short = W_pos[:, :self.pos_dim]
            top2_idx_flat = top2_idx.permute(0, 2, 1).flatten(1)  # [B, D*2]
            idx_exp = top2_idx_flat.unsqueeze(-1).expand(-1, -1, self.pos_dim)
            winner_pos = torch.gather(
                W_pos_short.unsqueeze(0).expand(B, -1, -1),
                1, idx_exp
            )  # [B, D*2, pos_dim]
            features = torch.cat([top2_vals, winner_pos.flatten(1)], dim=-1)
            return self.fc_out(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run the base model's routing but intercept before its readout
        # We need to get Z after routing, before readout
        base = self.base
        # Navigate to the actual SmallWorld base
        if hasattr(base, 'm'):  # SGNNET_AntiHebbian
            resonant = base.m
            sw = resonant.base
        else:
            sw = base

        Z = sw._seed(x)

        # Run AH routing loop (from SGNNET_AntiHebbian)
        if hasattr(base, 'm'):
            theta_pos = resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
            conn_hh = sw.conn_hh
            N_h = sw.N_hidden

            W_n = F.normalize(resonant.W_pos[:N_h], dim=-1)
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
            supp_w = (1.0 - base.alpha_ahebb * pos_sim.clamp(min=0)
                     ).unsqueeze(0).unsqueeze(-1)

            Z_reflected = torch.zeros_like(Z)
            for _ in range(sw.K_iter):
                Z_fwd = F.relu(Z - theta_pos)
                Z_nb = Z_fwd[:, conn_hh, :]
                Z_struct = (Z_nb * supp_w).sum(dim=2)
                Z_remainder = Z_fwd - Z
                Z_reflected = resonant.alpha_reflect * Z_reflected + Z_remainder
                Z_new = Z_struct + Z_reflected
                Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self._readout(Z)


@dataclass
class Config:
    key: str
    label: str
    readout_mode: str
    embed_dim: int = 4
    use_custom: bool = True

CONFIGS = [
    Config("Ref", "Ref  mean-pool (baseline)",           "mean",      use_custom=False),
    Config("A",   "A    max-pool only (no position)",    "max"),
    Config("B",   "B    max-pool + winner W_pos",        "max_wpos"),
    Config("C",   "C    max-pool + learned winner embed", "max_embed", embed_dim=4),
    Config("D",   "D    max-pool || mean-pool concat",   "max_mean"),
    Config("E",   "E    top-2 pool + winner W_pos",      "top2_wpos"),
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
    ah = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")

    if not cfg.use_custom:
        return ah

    return SGNNET_PosMaxPool(ah, resonant.W_pos,
                              readout_mode=cfg.readout_mode,
                              embed_dim=cfg.embed_dim)


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    print(f"\n{'='*70}")
    print(f"Step 150 — Positional Max-Pool Readout")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  K_in={K_IN}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    for c in CONFIGS:
        print(f"  {c.key:4s}  {c.label}")
    print()

    get_loaders()
    results = {}
    out_path = ROOT / "results" / "train_step150_positional_maxpool.json"

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
            "readout_mode": cfg.readout_mode,
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

    print(f"\n{'='*70}\nSTEP 150 SUMMARY\n{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for k, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"  {k:4s}  {r['readout_mode']:>12s}  params={r['n_params']:>8}  "
              f"top1={r['top1_best']:.4f}  vs_Ref={vs:+.4f}")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
