"""Step 404: GCN / GAT / GIN baselines at matched params (reviewer-critical).

MOTIVATION (V3 Gap 2.10)
========================
SGNNET is a GNN. Paper reviewers WILL ask "why not compare to GCN/GAT?"
Missing this baseline = rejection risk. step401 has MLP baselines but no
GNN — fills a critical gap.

DESIGN: reuse SGNNET's small-world graph (same conn_hh, same N nodes, same
K_in input gather) but replace the SGNNET routing with STANDARD message-
passing operators. Everything else identical: seed gather, readout, training.

This is the fairest possible comparison: same graph, same data, same
optimizer — only the routing operator differs.

VARIANTS (N=2048, D=16, K_hh=2, K_iter=5, ~34-70K params each)
  GCN : Z' = σ(Â Z W)   — normalized-adjacency + linear W
  GAT : Z' = Σ_j α_{ij} Z_j W    α_{ij} = softmax(LeakyReLU(a·[Z_i, Z_j]W))
  GIN : Z' = MLP((1+ε) Z_i + Σ_j Z_j)   — most expressive standard GNN

All three use the SAME conn_hh neighborhood as SGNNET (K=2 sparse per node).

To run:
    python -u scripts/train_step404_gnn_baselines.py --configs GCN,GAT,GIN
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--data",   default="data/store.h5")
parser.add_argument("--configs", default="GCN,GAT,GIN")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step404_gnn_baselines_seed{SEED}__{SLOT}.json"


class GNNHead(nn.Module):
    """Shared scaffold: SGNNET_SmallWorld seed + graph, replaceable routing,
    SGNNET readout. Variants differ only in the _route(...) method."""

    def __init__(self, variant: str, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
        self.base = SGNNET_SmallWorld(
            N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
            K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
            n_groups=ng, norm_mode="l2", encoding_mode="fourier")
        self.theta = nn.Parameter(torch.full((N,), 0.1))
        self.variant = variant

        # Per-variant learnable components
        if variant == "GCN":
            # Z' = (Â Z) W   — W is shared across iterations [D, D]
            self.W_gcn = nn.Parameter(torch.eye(D) + 0.01 * torch.randn(D, D))
        elif variant == "GAT":
            # Single-head GAT (for fair params budget)
            self.W_gat = nn.Parameter(torch.eye(D) + 0.01 * torch.randn(D, D))
            self.a_gat = nn.Parameter(torch.randn(2 * D) * 0.01)
            self.leaky_slope = 0.2
        elif variant == "GIN":
            # GIN: MLP on sum-aggregate. Use small MLP for param parity.
            self.eps = nn.Parameter(torch.zeros(1))
            self.mlp = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        else:
            raise ValueError(f"unknown variant: {variant}")

    # These attrs are needed so Trainer doesn't crash looking for them
    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return torch.zeros(N, D, device=self.base.W_pos.device)

    def tick_epoch(self): pass

    def _route_gcn(self, Z, conn_hh):
        """Â Z W. Â here is the sparse K-NN adjacency (no self-loop for now)."""
        Z_nb = Z[:, conn_hh, :]                    # [B, N, K_hh, D]
        agg = Z_nb.mean(dim=2)                     # mean-aggregate neighbors
        return agg @ self.W_gcn

    def _route_gat(self, Z, conn_hh):
        """Single-head attention aggregate."""
        Zw = Z @ self.W_gat                        # [B, N, D]
        Zw_nb = Zw[:, conn_hh, :]                  # [B, N, K_hh, D]
        Zw_exp = Zw.unsqueeze(2).expand_as(Zw_nb)  # [B, N, K_hh, D]
        pair = torch.cat([Zw_exp, Zw_nb], dim=-1)  # [B, N, K_hh, 2D]
        e = (pair * self.a_gat).sum(-1)            # [B, N, K_hh]
        e = F.leaky_relu(e, negative_slope=self.leaky_slope)
        alpha = F.softmax(e, dim=-1)               # [B, N, K_hh]
        return (alpha.unsqueeze(-1) * Zw_nb).sum(dim=2)

    def _route_gin(self, Z, conn_hh):
        """(1+ε) Z_i + Σ_j Z_j passed through MLP."""
        Z_nb = Z[:, conn_hh, :]                    # [B, N, K_hh, D]
        agg = (1.0 + self.eps) * Z + Z_nb.sum(dim=2)
        return self.mlp(agg)

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh
        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)

        for _ in range(K_ITER):
            Z_fwd = F.relu(Z - theta_pos)
            if self.variant == "GCN":
                Z_new = self._route_gcn(Z_fwd, conn_hh)
            elif self.variant == "GAT":
                Z_new = self._route_gat(Z_fwd, conn_hh)
            else:  # GIN
                Z_new = self._route_gin(Z_fwd, conn_hh)
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.base._readout(Z)


def main():
    run_keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    print(f"Step 404 — GNN baselines (GCN/GAT/GIN) at matched params")
    print(f"  data={args.data}  seed={SEED}  epochs={EPOCHS}  device={DEVICE}")
    print(f"  Reusing SGNNET small-world graph: N={N} K_hh={K_HH} K_iter={K_ITER}")

    tr, va = make_loaders(ROOT / args.data, batch_size=BATCH, seed=SEED)
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")

    results = {}
    for variant in run_keys:
        print(f"\n{'─'*60}\nVariant: {variant}\n{'─'*60}")
        t0 = time.time()
        model = GNNHead(variant=variant, seed=SEED).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_params:,}")
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 10 == 0 else None
        ))
        top1h = [h.get("val_top1", 0.0) for h in history]
        top1_best = max(top1h) if top1h else 0.0
        best_ep = int(np.argmax(top1h)) + 1 if top1h else 0
        elapsed = time.time() - t0
        print(f"  → best={top1_best:.4f} @ep{best_ep}  elapsed={elapsed:.0f}s")
        results[variant] = {
            "variant": variant, "n_params": n_params,
            "top1_best": top1_best, "best_epoch": best_ep,
            "elapsed_s": elapsed,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUT_PATH, "w") as f: json.dump(results, f, indent=2)

    print(f"\n========== STEP 404 SUMMARY (vs SGNNET_Ref 95.52%) ==========")
    for v, r in results.items():
        d = (r["top1_best"] - 0.9552) * 100
        print(f"  {v:<6}  params={r['n_params']:>7,}  best={r['top1_best']:.4f}  "
              f"Δ_vs_SGNNET={d:+.2f}pp")


if __name__ == "__main__":
    main()
