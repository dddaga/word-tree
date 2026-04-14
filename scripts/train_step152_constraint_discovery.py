"""Step 152: Constraint Discovery Mechanisms — architectural tools for low-rank representation.

MOTIVATION
==========
Core hypothesis: physical data has compact constraint structure. SGNNET routes
efficiently in space (sparse graph) but doesn't compress in representation
(all D dims, full rank, no bottleneck). We need architectural tools that
encourage finding low-rank, objective-relevant constraint representations.

Six mechanisms tested:
  1. Nuclear norm regularization on Z — direct rank pressure on activations
  2. Mid-routing information bottleneck — forced compression at K_iter midpoint
  3. Dimensional gating — learned per-step importance over D dimensions
  4. L1 activation sparsity — sparse coding encourages dimension specialization
  5. Contrastive routing loss — objective-aware representation shaping via SupCon
  6. Compound: nuclear norm + bottleneck + dim gating

CONFIGS (N=1024, D=16, K_hh=8, K_iter=8, K_in=25, AH=1.0, 50%/75ep)
=====================================================================
  Ref : Standard (no constraint tools)
  A   : Nuclear norm λ=0.01 (rank pressure)
  B   : Mid-routing bottleneck D→8→D at step K//2
  C   : Dimensional gating (learned per-step gates)
  D   : L1 activation sparsity λ=0.1
  E   : Contrastive routing loss λ=0.1, τ=0.1
  F   : Nuclear norm + bottleneck + dim gating (compound)

To reproduce:
    python -u scripts/train_step152_constraint_discovery.py --device mps
    python -u scripts/train_step152_constraint_discovery.py --device mps --epochs 20
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

from src.sgnnet.model_smallworld import _build_smallworld_conn
from src.sgnnet.encoding import compute_fourier_encoding
from src.training.experiment_config import trainer_kwargs
from src.training.dataset import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75,
                    help="Training epochs (default 75; use 20 for scout)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys. Empty = all.")
args = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 16; K_ITER = 8; K_IN = 25; K_HH = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0


class SGNNET_ConstraintDiscovery(nn.Module):
    """SGNNET with constraint discovery mechanisms inserted into the routing loop.

    Mechanisms (independently toggleable):
      - bottleneck: D→bottleneck_d→D projection at routing midpoint
      - dim_gate: learned per-step sigmoid gates over D dimensions
      - nuclear_lambda: nuclear norm penalty on Z (rank pressure)
      - l1_lambda: L1 penalty on Z (activation sparsity)
      - contrastive_lambda: SupCon loss on mean-pooled features
    """

    def __init__(self, N_hidden, N_out, D, N_in, K_in, K_iter,
                 K_local, K_random, n_groups,
                 bottleneck_d=0, use_dim_gate=False,
                 nuclear_lambda=0.0, l1_lambda=0.0,
                 contrastive_lambda=0.0, contrastive_temp=0.1,
                 alpha_ahebb=1.0, alpha_reflect=0.5):
        super().__init__()
        self.N_hidden = N_hidden
        self.D = D
        self.K_in = K_in
        self.K_iter = K_iter
        self.alpha_ahebb = alpha_ahebb
        self.alpha_reflect = alpha_reflect
        self.nuclear_lambda = nuclear_lambda
        self.l1_lambda = l1_lambda
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temp = contrastive_temp

        self.W_pos = nn.Parameter(torch.rand(N_hidden + N_out, D))
        self.theta = nn.Parameter(torch.full((N_hidden,), 0.1))

        # Input encoding: 1 feature dim + (D-1) spatial dims
        spatial_full = compute_fourier_encoding(N_in, D=D)
        self.register_buffer("spatial_enc", spatial_full[:, :D - 1])

        # Input connectivity
        conn_in = torch.stack([
            torch.randperm(N_in)[:K_in] for _ in range(N_hidden)
        ])
        self.register_buffer("conn_in", conn_in)

        # Hidden-hidden connectivity
        conn_hh = _build_smallworld_conn(
            N_hidden, K_local=K_local, K_random=K_random,
            n_groups=n_groups,
        )
        self.register_buffer("conn_hh", conn_hh)

        # --- Constraint discovery modules ---

        # Mid-routing bottleneck: D → bottleneck_d → D
        self.has_bottleneck = bottleneck_d > 0
        if self.has_bottleneck:
            self.bottleneck_down = nn.Linear(D, bottleneck_d, bias=False)
            self.bottleneck_up = nn.Linear(bottleneck_d, D, bias=False)
            self.bottleneck_step = K_iter // 2

        # Per-step dimensional gating
        self.use_dim_gate = use_dim_gate
        if use_dim_gate:
            # Initialized at 0 → sigmoid(0)=0.5 → all dims half-open initially
            self.dim_gates = nn.Parameter(torch.zeros(K_iter, D))

        self.fc_out = nn.Linear(D, N_out, bias=True)
        self._last_Z = None  # stored for aux loss computation

    @property
    def W_phase(self):
        return self.W_pos

    def tick_epoch(self):
        pass

    def _seed(self, x):
        B = x.shape[0]
        feat = x.unsqueeze(-1)                                      # [B, N_in, 1]
        spatial = self.spatial_enc.unsqueeze(0).expand(B, -1, -1)    # [B, N_in, D-1]
        A_input = torch.cat([feat, spatial], dim=-1)                 # [B, N_in, D]
        gathered = A_input[:, self.conn_in, :]                       # [B, N, K_in, D]
        Z = gathered.sum(dim=2)                                      # [B, N, D]
        return F.normalize(Z, dim=-1)

    def forward(self, x):
        Z = self._seed(x)

        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)
        N_h = self.N_hidden

        # Precompute AH suppression weights (static, position-based)
        W_n = F.normalize(self.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[self.conn_hh]).sum(-1)
        supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)
        for k in range(self.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, self.conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

            # --- Constraint discovery insertions ---

            # Bottleneck at routing midpoint: force information through D' < D
            if self.has_bottleneck and k == self.bottleneck_step:
                Z = self.bottleneck_up(F.relu(self.bottleneck_down(Z)))
                Z = F.normalize(Z, dim=-1)

            # Dimensional gating: learned per-step importance
            if self.use_dim_gate:
                gate = torch.sigmoid(self.dim_gates[k])  # [D]
                Z = Z * gate
                Z = F.normalize(Z, dim=-1)

        self._last_Z = Z  # keep in grad graph for aux losses
        return self.fc_out(Z.mean(dim=1))

    def compute_aux_loss(self, labels=None):
        """Compute auxiliary constraint-discovery losses from stored activations."""
        Z = self._last_Z
        if Z is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        loss = torch.tensor(0.0, device=Z.device)

        # Nuclear norm: penalizes rank of activation matrix
        # Computed via eigenvalues of Z^T Z (guaranteed MPS-compatible)
        if self.nuclear_lambda > 0:
            Z_mean = Z.mean(dim=0)           # [N, D]
            gram = Z_mean.T @ Z_mean         # [D, D] — tiny 16×16 matrix
            eigvals = torch.linalg.eigvalsh(gram).clamp(min=0)
            nuclear_norm = eigvals.sqrt().sum()
            loss = loss + self.nuclear_lambda * nuclear_norm

        # L1 sparsity: encourages each neuron to specialize in fewer dims
        if self.l1_lambda > 0:
            loss = loss + self.l1_lambda * Z.abs().mean()

        # Supervised contrastive: shapes representation for objective
        if self.contrastive_lambda > 0 and labels is not None:
            feats = F.normalize(Z.mean(dim=1), dim=-1)  # [B, D]
            B = feats.shape[0]
            sim = feats @ feats.T / self.contrastive_temp  # [B, B]

            # Masks
            self_mask = torch.eye(B, device=sim.device, dtype=torch.bool)
            pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask

            # Numerically stable SupCon
            sim = sim.masked_fill(self_mask, -1e9)
            log_denom = torch.logsumexp(sim, dim=1)          # [B]
            log_prob = sim - log_denom.unsqueeze(1)           # [B, B]

            n_pos = pos_mask.float().sum(dim=1)
            valid = n_pos > 0
            if valid.any():
                mean_log_prob = (log_prob * pos_mask.float()).sum(1) / (n_pos + 1e-8)
                loss = loss + self.contrastive_lambda * (-mean_log_prob[valid].mean())

        return loss


@dataclass
class Config:
    key: str
    label: str
    bottleneck_d: int = 0
    use_dim_gate: bool = False
    nuclear_lambda: float = 0.0
    l1_lambda: float = 0.0
    contrastive_lambda: float = 0.0


CONFIGS = [
    Config("Ref", "Ref  standard (no constraint tools)"),
    Config("A",   "A    nuclear norm λ=0.01",           nuclear_lambda=0.01),
    Config("B",   "B    bottleneck D→8→D at K//2",      bottleneck_d=8),
    Config("C",   "C    dimensional gating per-step",   use_dim_gate=True),
    Config("D",   "D    L1 sparsity λ=0.1",            l1_lambda=0.1),
    Config("E",   "E    contrastive routing λ=0.1",     contrastive_lambda=0.1),
    Config("F",   "F    nuclear+bottleneck+dim_gate",   bottleneck_d=8, use_dim_gate=True,
           nuclear_lambda=0.01),
]


_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True,
                                          num_workers=0)
        _loaders = (tr, va)
    return _loaders


def make_model(cfg, seed_offset=0):
    torch.manual_seed(SEED + seed_offset)
    K_random = max(1, K_HH // 4)
    K_local = K_HH - K_random
    n_groups = max(8, N // 8)
    return SGNNET_ConstraintDiscovery(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER,
        K_local=K_local, K_random=K_random, n_groups=n_groups,
        bottleneck_d=cfg.bottleneck_d,
        use_dim_gate=cfg.use_dim_gate,
        nuclear_lambda=cfg.nuclear_lambda,
        l1_lambda=cfg.l1_lambda,
        contrastive_lambda=cfg.contrastive_lambda,
        alpha_ahebb=ALPHA_AHEBB, alpha_reflect=ALPHA_REFLECT,
    )


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_with_aux(model, device, n_epochs):
    """Custom training loop supporting auxiliary constraint losses."""
    tr, va = get_loaders()
    kw = trainer_kwargs(N, n_epochs=n_epochs)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=kw.get("lr", 1e-3),
        weight_decay=kw.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    history = []
    for ep in range(1, n_epochs + 1):
        model.train()
        if hasattr(model, "tick_epoch"):
            model.tick_epoch()

        total_task = 0.0
        total_aux = 0.0

        for batch in tr:
            x = batch[0].to(device)
            soft_labels = batch[1].to(device)
            hard_labels = batch[2].to(device)

            optimizer.zero_grad()
            logits = model(x)
            task_loss = criterion(logits, soft_labels)
            aux_loss = model.compute_aux_loss(labels=hard_labels)
            loss = task_loss + aux_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_task += task_loss.item()
            total_aux += aux_loss.item()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in va:
                x = batch[0].to(device)
                labels = batch[2].to(device)
                out = model(x)
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)

        val_top1 = correct / total
        avg_task = total_task / len(tr)
        avg_aux = total_aux / len(tr)
        scheduler.step(total_task)
        history.append({
            "val_top1": val_top1,
            "train_loss": avg_task,
            "aux_loss": avg_aux,
        })

        if ep % 5 == 0 or ep <= 3 or ep == n_epochs:
            print(f"  e{ep:3d}  task={avg_task:.4f}  aux={avg_aux:.4f}  "
                  f"top1={val_top1:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

    return history


def main():
    print(f"\n{'='*70}")
    print(f"Step 152 — Constraint Discovery Mechanisms")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  K_in={K_IN}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    for c in CONFIGS:
        print(f"  {c.key:4s}  {c.label}")
    print()

    get_loaders()
    results = {}
    out_path = ROOT / "results" / "train_step152_constraint_discovery.json"

    cfg_filter = ([k.strip() for k in args.configs.split(",") if k.strip()]
                  if args.configs else [])
    active = [(i, c) for i, c in enumerate(CONFIGS)
              if not cfg_filter or c.key in cfg_filter]

    for i, cfg in active:
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}  params={n_params:,}")
        print(f"{'─'*60}")

        t0 = time.time()
        history = train_with_aux(model, DEVICE, EPOCHS)
        elapsed = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep = int(np.argmax(top1_hist)) + 1

        results[cfg.key] = {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "bottleneck_d": cfg.bottleneck_d,
            "use_dim_gate": cfg.use_dim_gate,
            "nuclear_lambda": cfg.nuclear_lambda,
            "l1_lambda": cfg.l1_lambda,
            "contrastive_lambda": cfg.contrastive_lambda,
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

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 152 SUMMARY — Constraint Discovery Mechanisms")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for k, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        tags = []
        if r["nuclear_lambda"] > 0: tags.append("nuc")
        if r["bottleneck_d"] > 0: tags.append("bneck")
        if r["use_dim_gate"]: tags.append("gate")
        if r["l1_lambda"] > 0: tags.append("L1")
        if r["contrastive_lambda"] > 0: tags.append("con")
        tag = "+".join(tags) if tags else "base"
        print(f"  {k:4s}  {tag:>20s}  params={r['n_params']:>8,}  "
              f"top1={r['top1_best']:.4f}  vs_Ref={vs:+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
