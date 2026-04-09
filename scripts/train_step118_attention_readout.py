"""Step 118: Attention-Pooling Readout for SGNNET.

MOTIVATION
==========
Current readout is W_out @ Z.mean(dim=1) — mean-pool all N neurons then linear.
This discards all graph structure. Transformers use attention pooling / CLS token
for richer readout. Can we get a better readout with negligible extra params?

The routing loop is IDENTICAL to SGNNET_AntiHebbian. Only the final readout changes.

CONFIGS (N=1024, D=64, K_hh=4, K_iter=12, AH=1.0, 50%/75ep)
==============================================================
  Ref : Standard mean-pool + linear (SGNNET_AntiHebbian directly)
  A   : Attention-pool: a = softmax(W_att @ Z^T) [B,1,N], Z_pooled = a @ Z [B,1,D], then W_out.
        W_att is [1,D] = 64 params.
  B   : Top-k pool: take k=N//4 highest-magnitude neurons, mean-pool those. Zero extra params.
  C   : Multi-head readout: 4 heads, each attends to Z with separate [D//4] query.
        Concat -> W_out. 4x16=64 params.
  D   : Learnable query: q=nn.Parameter(D), scores = Z @ q / sqrt(D),
        w = softmax(scores), Z_pooled = (w.unsqueeze(-1) * Z).sum(1). 64 params.

To reproduce:
    python -u scripts/train_step118_attention_readout.py --device mps
    python -u scripts/train_step118_attention_readout.py --device mps --epochs 20  # scout
    python -u scripts/train_step118_attention_readout.py --device mps --configs A,D
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
from src.sgnnet.mechanisms_inhibitory  import SGNNET_AntiHebbian
from src.training.trainer              import Trainer
from src.training.experiment_config    import trainer_kwargs, topology_kwargs
from src.training.dataset              import make_loaders

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
N = 1024; N_IN = 25088; N_OUT = 10; D = 64; K_ITER = 12; K_IN = 50
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
STEP69_REF = 0.8336


# ---------------------------------------------------------------------------
# Model: AH routing + configurable readout
# ---------------------------------------------------------------------------

class SGNNET_AH_AttentionReadout(nn.Module):
    """SGNNET with standard AntiHebbian routing but configurable readout pooling.

    Routing loop is identical to SGNNET_AntiHebbian (wpos variant).
    Only _readout() changes based on pool_mode.

    Pool modes:
      "mean"      — standard: Z.mean(dim=1) then W_out (baseline)
      "attention"  — W_att [1,D] attention over N neurons, then W_out
      "topk"      — top N//4 neurons by magnitude, mean-pool, then W_out
      "multihead" — 4 heads with D//4 queries each, concat, then W_out
      "query"     — learnable query vector q [D], softmax attention, then W_out
    """

    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float,
                 pool_mode: str = "mean"):
        super().__init__()
        self.m          = resonant
        self.alpha      = alpha_ahebb
        self.pool_mode  = pool_mode

        base = resonant.base
        self.N_hidden = base.N_hidden
        self.D        = base.W_pos.shape[1]

        # Pool-mode-specific parameters
        if pool_mode == "attention":
            # [1, D] query for single-head attention
            self.W_att = nn.Parameter(torch.randn(1, self.D) * 0.01)

        elif pool_mode == "multihead":
            # 4 heads, each with D//4 query
            self.n_heads = 4
            self.head_dim = self.D // self.n_heads
            # [n_heads, head_dim] — each head gets its own query
            self.W_heads = nn.Parameter(torch.randn(self.n_heads, self.head_dim) * 0.01)

        elif pool_mode == "query":
            # Learnable query vector [D]
            self.q = nn.Parameter(torch.randn(self.D) * 0.01)

        # topk and mean need no extra params

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _route(self, x: torch.Tensor) -> torch.Tensor:
        """Run standard AntiHebbian wpos routing. Returns Z [B, N, D]."""
        base      = self.m.base
        Z         = base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = base.conn_hh
        N_h       = base.N_hidden

        # Static wpos suppression (precomputed)
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)       # [N_h, K_hh]
        supp_w  = (1.0 - self.alpha * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)                      # [1, N, K_hh, 1]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]                           # [B, N, K_hh, D]
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            # Reflection accumulator
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            # Phase inhibition — skip when alpha_turing=0
            if self.m.alpha_turing != 0.0:
                Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
                Z_new = Z_struct + Z_reflected + self.m.alpha_turing * Z_inh
            else:
                Z_new = Z_struct + Z_reflected

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return Z

    def _pool(self, Z: torch.Tensor) -> torch.Tensor:
        """Pool Z [B, N, D] -> Z_pooled [B, D] based on pool_mode."""
        if self.pool_mode == "mean":
            return Z.mean(dim=1)                                    # [B, D]

        elif self.pool_mode == "attention":
            # a = softmax(W_att @ Z^T) -> [B, 1, N]
            scores = torch.matmul(self.W_att, Z.transpose(1, 2))   # [B, 1, N]
            a = F.softmax(scores, dim=-1)                           # [B, 1, N]
            Z_pooled = torch.matmul(a, Z).squeeze(1)               # [B, D]
            return Z_pooled

        elif self.pool_mode == "topk":
            # Top N//4 neurons by L2 magnitude
            k = max(1, self.N_hidden // 4)
            magnitudes = Z.norm(dim=-1)                             # [B, N]
            _, top_idx = magnitudes.topk(k, dim=1)                  # [B, k]
            # Gather top-k neurons and mean-pool
            top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, self.D)  # [B, k, D]
            Z_top = torch.gather(Z, 1, top_idx_exp)                # [B, k, D]
            return Z_top.mean(dim=1)                                # [B, D]

        elif self.pool_mode == "multihead":
            # Split Z into n_heads chunks along D, attend each with its query
            B = Z.shape[0]
            # Z: [B, N, D] -> [B, N, n_heads, head_dim]
            Z_split = Z.view(B, -1, self.n_heads, self.head_dim)   # [B, N, 4, 16]
            # W_heads: [n_heads, head_dim] -> scores per head
            # scores[b, n, h] = Z_split[b, n, h, :] @ W_heads[h, :]
            scores = (Z_split * self.W_heads).sum(dim=-1)           # [B, N, n_heads]
            a = F.softmax(scores, dim=1)                            # [B, N, n_heads]
            # Weighted sum per head: [B, n_heads, head_dim]
            Z_attended = (a.unsqueeze(-1) * Z_split).sum(dim=1)    # [B, n_heads, head_dim]
            # Concat heads -> [B, D]
            return Z_attended.reshape(B, -1)

        elif self.pool_mode == "query":
            # scores = Z @ q / sqrt(D), w = softmax(scores)
            scores = torch.matmul(Z, self.q) / math.sqrt(self.D)   # [B, N]
            w = F.softmax(scores, dim=-1)                           # [B, N]
            Z_pooled = (w.unsqueeze(-1) * Z).sum(dim=1)            # [B, D]
            return Z_pooled

        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

    def _readout(self, Z_pooled: torch.Tensor) -> torch.Tensor:
        """Z_pooled [B, D] -> logits [B, N_out] via W_pos output neurons."""
        base = self.m.base
        W_out = base.W_pos[base.N_hidden:]                         # [N_out, D]
        W_out_norm = F.normalize(W_out, dim=-1)                    # [N_out, D]
        # dot product: [B, D] @ [D, N_out] -> [B, N_out]
        return torch.matmul(Z_pooled, W_out_norm.t())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self._route(x)          # [B, N, D]
        Z_pooled = self._pool(Z)    # [B, D]
        return self._readout(Z_pooled)


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    pool_mode: str
    use_wrapper: bool = True   # True = use SGNNET_AH_AttentionReadout; False = use SGNNET_AntiHebbian


CONFIGS = [
    Config("Ref", "Ref  mean-pool + linear (baseline AH)",
           "mean", use_wrapper=False),
    Config("A",   "A    attention-pool: W_att [1,D] softmax over N (+64 params)",
           "attention"),
    Config("B",   "B    top-k pool: N//4 highest-magnitude neurons (0 params)",
           "topk"),
    Config("C",   "C    multi-head readout: 4 heads x D//4 query, concat (+64 params)",
           "multihead"),
    Config("D",   "D    learnable query: q [D] scaled-dot attention (+64 params)",
           "query"),
]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _make_base(seed_offset: int = 0):
    """Build SmallWorld + Resonant base (shared across all configs)."""
    torch.manual_seed(SEED + seed_offset)
    topo = topology_kwargs(N)
    topo.pop("K_in", None)
    topo.pop("K_iter", None)
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
    return resonant


def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    resonant = _make_base(seed_offset)
    if not cfg.use_wrapper:
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_AH_AttentionReadout(resonant, alpha_ahebb=ALPHA_AHEBB,
                                       pool_mode=cfg.pool_mode)


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
    print(f"Step 118 — Attention-Pooling Readout")
    print(f"N={N}  D={D}  K_iter={K_ITER}  K_hh=4  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    print("Configs:")
    for c in CONFIGS:
        print(f"  {c.key:4s}  pool={c.pool_mode:10s}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step118_attention_readout.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  pool_mode={cfg.pool_mode}")
        print(f"{'─'*60}")

        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=EPOCHS)

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

        results[cfg.key] = {
            "N": N, "D": D, "K_iter": K_ITER,
            "pool_mode": cfg.pool_mode,
            "alpha_ahebb": ALPHA_AHEBB,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params,
            "label": cfg.label,
        }

        # vs Ref
        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs_ref = top1_best - ref_best if ref_best > 0 else 0

        print(f"\n  top1_best={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary table
    print(f"\n{'='*70}")
    print(f"STEP 118 SUMMARY")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Config':<6}  {'pool':>10}  {'params':>8}  {'top1':>7}  {'vs_Ref':>8}")
    print(f"{'─'*50}")
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"{key:<6}  {r['pool_mode']:>10}  {r['n_params']:>8,}  "
              f"{r['top1_best']:.4f}  {vs:>+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
