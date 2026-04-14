"""Step 153: Progressive Capacity Reduction — start large, prune to compact.

MOTIVATION
==========
Core hypothesis: data has compact constraint structure, but discovering it may
require more capacity than the final representation needs. Start with excess
capacity (more dims, more neurons), let the network explore freely, then
progressively prune away unused capacity based on contribution.

Inspired by: Matformer (nested dim loss), GMP (cubic pruning schedule),
Nested Dropout (dimension ordering), Slimmable Networks (elastic width).

Key design:
  - Extra dimensions initialized to 1.0 ("vacant slots" for signal to evolve into)
  - Positional dims fixed; activation dims can be pruned
  - Pruning criterion: accumulated activation magnitude per neuron/dim
  - Pruning schedule: cubic (aggressive early, gentle late), every prune_interval epochs
  - Connection updates batched (not every batch — every prune_interval epochs)

CONFIGS (50%/75ep)
==================
  Ref : N=1024, D=16 fixed (standard baseline)
  A   : N=1024, D=32→16 progressive dim pruning (GMP cubic, prune every 5ep)
  B   : N=1024, D=32 + Nested Dropout (force dimension ordering, no explicit prune)
  C   : N=1024, D=32 + Matformer nested loss at D={32,24,16} simultaneously
  D   : N=2048→1024 progressive neuron pruning, D=16 fixed
  E   : N=2048→1024 + D=32→16 full progressive (neurons + dims)

To reproduce:
    python -u scripts/train_step153_progressive_capacity.py --device mps
    python -u scripts/train_step153_progressive_capacity.py --device mps --epochs 20
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
N_IN = 25088; N_OUT = 10; K_IN = 25; K_HH = 8; K_ITER = 8
ALPHA_REFLECT = 0.5; ALPHA_AHEBB = 1.0


def _gmp_cubic_target(t, t0, T, start, final):
    """GMP cubic schedule: returns target count at epoch t."""
    if t < t0:
        return start
    if t >= T:
        return final
    frac = (t - t0) / (T - t0)
    return int(final + (start - final) * (1 - frac) ** 3)


class SGNNET_Progressive(nn.Module):
    """SGNNET with progressive capacity reduction.

    Modes:
      "fixed"           — standard, no pruning
      "prune_dims"      — progressively prune dimensions (D_start → D_final)
      "nested_dropout"  — nested dropout on dims (forces ordering, no hard prune)
      "matformer"       — multi-granularity loss at multiple D values
      "prune_neurons"   — progressively prune neurons (N_start → N_final)
      "prune_both"      — prune neurons + dims simultaneously
    """

    def __init__(self, N_hidden, N_out, D, N_in, K_in, K_iter,
                 K_local, K_random, n_groups,
                 mode="fixed",
                 D_start=None, D_final=None,
                 N_start=None, N_final=None,
                 prune_warmup=5, prune_interval=5,
                 nested_drop_rate=0.5,
                 matformer_granularities=None,
                 alpha_ahebb=1.0, alpha_reflect=0.5):
        super().__init__()
        # Use the larger of start/final for allocation
        self.N_alloc = N_start if N_start else N_hidden
        self.D_alloc = D_start if D_start else D
        self.N_final = N_final if N_final else N_hidden
        self.D_final = D_final if D_final else D
        self.N_start = self.N_alloc
        self.D_start = self.D_alloc
        self.N_hidden = self.N_alloc
        self.D = self.D_alloc
        self.K_in = K_in
        self.K_iter = K_iter
        self.alpha_ahebb = alpha_ahebb
        self.alpha_reflect = alpha_reflect
        self.mode = mode
        self.prune_warmup = prune_warmup
        self.prune_interval = prune_interval
        self.nested_drop_rate = nested_drop_rate
        self.matformer_granularities = matformer_granularities or []

        self.W_pos = nn.Parameter(torch.rand(self.N_alloc + N_out, self.D_alloc))
        self.theta = nn.Parameter(torch.full((self.N_alloc,), 0.1))

        # Input encoding: 1 feat + (D_alloc-1) spatial
        spatial_full = compute_fourier_encoding(N_in, D=self.D_alloc)
        self.register_buffer("spatial_enc", spatial_full[:, :self.D_alloc - 1])

        # Initialize extra dimensions to 1.0 (vacant slots)
        # First D_final dims are "real", remaining are vacant
        if self.D_alloc > self.D_final and mode != "fixed":
            with torch.no_grad():
                self.W_pos.data[:, self.D_final:] = 1.0

        # Connectivity
        conn_in = torch.stack([
            torch.randperm(N_in)[:K_in] for _ in range(self.N_alloc)
        ])
        self.register_buffer("conn_in", conn_in)

        conn_hh = _build_smallworld_conn(
            self.N_alloc, K_local=K_local, K_random=K_random,
            n_groups=n_groups,
        )
        self.register_buffer("conn_hh", conn_hh)

        # Active masks (updated during pruning)
        self.register_buffer("neuron_mask", torch.ones(self.N_alloc, dtype=torch.bool))
        self.register_buffer("dim_mask", torch.ones(self.D_alloc, dtype=torch.bool))

        # Importance accumulators
        self.register_buffer("neuron_importance", torch.zeros(self.N_alloc))
        self.register_buffer("dim_importance", torch.zeros(self.D_alloc))
        self.register_buffer("importance_count", torch.tensor(0))

        self.fc_out = nn.Linear(self.D_alloc, N_out, bias=True)
        self._current_epoch = 0
        self._last_Z = None  # for matformer multi-granularity

    @property
    def W_phase(self):
        return self.W_pos

    def tick_epoch(self):
        self._current_epoch += 1
        # Check if pruning should happen
        ep = self._current_epoch
        if self.mode in ("prune_dims", "prune_neurons", "prune_both"):
            if ep >= self.prune_warmup and ep % self.prune_interval == 0:
                self._prune_step(ep)

    def _prune_step(self, ep):
        """Execute one pruning step based on cubic schedule."""
        T = self.prune_warmup + (
            (max(self.N_start - self.N_final, self.D_start - self.D_final))
            / max(1, self.prune_interval) * self.prune_interval
        )
        # More robust: prune until 80% through training
        T = int(0.8 * (self.prune_warmup + 75))  # assume ~75ep training

        if self.mode in ("prune_neurons", "prune_both"):
            target_n = _gmp_cubic_target(
                ep, self.prune_warmup, T, self.N_start, self.N_final)
            current_n = self.neuron_mask.sum().item()
            if target_n < current_n and self.importance_count > 0:
                scores = self.neuron_importance / self.importance_count.float()
                scores = scores * self.neuron_mask.float()  # only active neurons
                # Keep top target_n neurons
                _, keep_idx = scores.topk(min(target_n, current_n))
                self.neuron_mask.zero_()
                self.neuron_mask[keep_idx] = True
                print(f"    [prune] ep={ep}: neurons {current_n}→{self.neuron_mask.sum().item()}")

        if self.mode in ("prune_dims", "prune_both"):
            target_d = _gmp_cubic_target(
                ep, self.prune_warmup, T, self.D_start, self.D_final)
            current_d = self.dim_mask.sum().item()
            if target_d < current_d and self.importance_count > 0:
                scores = self.dim_importance / self.importance_count.float()
                scores = scores * self.dim_mask.float()
                _, keep_idx = scores.topk(min(target_d, current_d))
                self.dim_mask.zero_()
                self.dim_mask[keep_idx] = True
                print(f"    [prune] ep={ep}: dims {current_d}→{self.dim_mask.sum().item()}")

        # Reset accumulators
        self.neuron_importance.zero_()
        self.dim_importance.zero_()
        self.importance_count.zero_()

    def _seed(self, x):
        B = x.shape[0]
        feat = x.unsqueeze(-1)                                      # [B, N_in, 1]
        spatial = self.spatial_enc.unsqueeze(0).expand(B, -1, -1)    # [B, N_in, D-1]
        A_input = torch.cat([feat, spatial], dim=-1)                 # [B, N_in, D]
        gathered = A_input[:, self.conn_in, :]                       # [B, N, K_in, D]
        Z = gathered.sum(dim=2)                                      # [B, N, D]
        return F.normalize(Z, dim=-1)

    def _apply_masks(self, Z):
        """Apply neuron and dimension masks."""
        if not self.neuron_mask.all():
            Z = Z * self.neuron_mask.float().unsqueeze(0).unsqueeze(-1)
        if not self.dim_mask.all():
            Z = Z * self.dim_mask.float().unsqueeze(0).unsqueeze(0)
        return Z

    def _apply_nested_dropout(self, Z):
        """Nested dropout: drop dims from the end with increasing probability."""
        if not self.training:
            return Z
        D = Z.shape[-1]
        # Each dim d is kept with prob 1 - (d/D)*drop_rate
        # Dim 0 always kept, dim D-1 dropped with prob drop_rate
        keep_prob = 1.0 - torch.arange(D, device=Z.device).float() / D * self.nested_drop_rate
        mask = torch.bernoulli(keep_prob.expand(1, 1, D)).to(Z.device)
        # Enforce nesting: if dim d is dropped, all dims > d are also dropped
        # Find first dropped dim and zero everything after
        cummin, _ = mask.squeeze().cummin(dim=0)
        Z = Z * cummin.unsqueeze(0).unsqueeze(0)
        return Z

    def forward(self, x):
        Z = self._seed(x)
        Z = self._apply_masks(Z)

        if self.mode == "nested_dropout":
            Z = self._apply_nested_dropout(Z)

        Z = F.normalize(Z, dim=-1)

        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)
        N_h = self.N_alloc

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

            # Re-apply masks after each step
            Z = self._apply_masks(Z)
            if self.mode == "nested_dropout":
                Z = self._apply_nested_dropout(Z)
            Z = F.normalize(Z, dim=-1)

        self._last_Z = Z

        # Accumulate importance (for pruning modes)
        if self.training and self.mode in ("prune_dims", "prune_neurons", "prune_both"):
            with torch.no_grad():
                self.neuron_importance += Z.abs().mean(dim=(0, 2))
                self.dim_importance += Z.abs().mean(dim=(0, 1))
                self.importance_count += 1

        # Readout: mean over active neurons only
        if not self.neuron_mask.all():
            n_active = self.neuron_mask.sum().clamp(min=1)
            pooled = (Z * self.neuron_mask.float().unsqueeze(0).unsqueeze(-1)
                      ).sum(dim=1) / n_active
        else:
            pooled = Z.mean(dim=1)

        return self.fc_out(pooled)

    def compute_matformer_loss(self, x, labels, criterion):
        """Matformer-style: compute loss at multiple D granularities."""
        if not self.matformer_granularities:
            return torch.tensor(0.0, device=x.device)

        total_aux = torch.tensor(0.0, device=x.device)
        Z = self._last_Z
        if Z is None:
            return total_aux

        for d_sub in self.matformer_granularities:
            if d_sub >= self.D_alloc:
                continue
            # Truncate to first d_sub dims
            Z_sub = Z[:, :, :d_sub]
            Z_sub = F.normalize(Z_sub, dim=-1)
            pooled_sub = Z_sub.mean(dim=1)
            # Need a separate readout head for this granularity?
            # Simpler: use first d_sub dims of fc_out weight
            logits_sub = F.linear(pooled_sub,
                                   self.fc_out.weight[:, :d_sub],
                                   self.fc_out.bias)
            total_aux = total_aux + criterion(logits_sub, labels)

        return total_aux / max(1, len(self.matformer_granularities))

    def active_neurons(self):
        return self.neuron_mask.sum().item()

    def active_dims(self):
        return self.dim_mask.sum().item()


@dataclass
class Config:
    key: str
    label: str
    mode: str
    N_hidden: int = 1024
    D: int = 16
    N_start: int = 0     # 0 = same as N_hidden
    N_final: int = 0     # 0 = same as N_hidden
    D_start: int = 0     # 0 = same as D
    D_final: int = 0     # 0 = same as D
    nested_drop_rate: float = 0.5
    matformer_grans: tuple = ()


CONFIGS = [
    Config("Ref", "Ref  N=1024 D=16 fixed",
           mode="fixed", N_hidden=1024, D=16),
    Config("A",   "A    D=32→16 progressive dim prune",
           mode="prune_dims", N_hidden=1024, D=32,
           D_start=32, D_final=16),
    Config("B",   "B    D=32 + nested dropout (dim ordering)",
           mode="nested_dropout", N_hidden=1024, D=32,
           nested_drop_rate=0.5),
    Config("C",   "C    D=32 + matformer loss at D={32,24,16}",
           mode="matformer", N_hidden=1024, D=32,
           matformer_grans=(24, 16)),
    Config("D",   "D    N=2048→1024 neuron prune, D=16",
           mode="prune_neurons", N_hidden=2048, D=16,
           N_start=2048, N_final=1024),
    Config("E",   "E    N=2048→1024 + D=32→16 full progressive",
           mode="prune_both", N_hidden=2048, D=32,
           N_start=2048, N_final=1024, D_start=32, D_final=16),
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
    N = cfg.N_hidden
    D = cfg.D
    K_random = max(1, K_HH // 4)
    K_local = K_HH - K_random
    n_groups = max(8, N // 8)

    return SGNNET_Progressive(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER,
        K_local=K_local, K_random=K_random, n_groups=n_groups,
        mode=cfg.mode,
        D_start=cfg.D_start or D,
        D_final=cfg.D_final or D,
        N_start=cfg.N_start or N,
        N_final=cfg.N_final or N,
        prune_warmup=5, prune_interval=5,
        nested_drop_rate=cfg.nested_drop_rate,
        matformer_granularities=list(cfg.matformer_grans),
        alpha_ahebb=ALPHA_AHEBB, alpha_reflect=ALPHA_REFLECT,
    )


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_progressive(model, device, n_epochs):
    """Training loop with progressive pruning + optional matformer loss."""
    tr, va = get_loaders()
    kw = trainer_kwargs(1024, n_epochs=n_epochs)  # use N=1024 for lr/schedule
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
        model.tick_epoch()

        total_task = 0.0
        total_aux = 0.0

        for batch in tr:
            x = batch[0].to(device)
            soft_labels = batch[1].to(device)

            optimizer.zero_grad()
            logits = model(x)
            task_loss = criterion(logits, soft_labels)

            # Matformer multi-granularity loss
            aux_loss = torch.tensor(0.0, device=device)
            if model.mode == "matformer":
                aux_loss = model.compute_matformer_loss(x, soft_labels, criterion)

            loss = task_loss + 0.5 * aux_loss  # weight auxiliary losses at 0.5
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
        scheduler.step(total_task)
        history.append({
            "val_top1": val_top1,
            "train_loss": avg_task,
            "active_neurons": model.active_neurons(),
            "active_dims": model.active_dims(),
        })

        if ep % 5 == 0 or ep <= 3 or ep == n_epochs:
            print(f"  e{ep:3d}  task={avg_task:.4f}  top1={val_top1:.4f}  "
                  f"N_act={model.active_neurons()}  D_act={model.active_dims()}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

    return history


def main():
    print(f"\n{'='*70}")
    print(f"Step 153 — Progressive Capacity Reduction")
    print(f"K_hh={K_HH}  K_iter={K_ITER}  K_in={K_IN}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    for c in CONFIGS:
        print(f"  {c.key:4s}  {c.label}")
    print()

    get_loaders()
    results = {}
    out_path = ROOT / "results" / "train_step153_progressive_capacity.json"

    cfg_filter = ([k.strip() for k in args.configs.split(",") if k.strip()]
                  if args.configs else [])
    active = [(i, c) for i, c in enumerate(CONFIGS)
              if not cfg_filter or c.key in cfg_filter]

    for i, cfg in active:
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  N={cfg.N_hidden} D={cfg.D}  mode={cfg.mode}  params={n_params:,}")
        print(f"{'─'*60}")

        t0 = time.time()
        history = train_progressive(model, DEVICE, EPOCHS)
        elapsed = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep = int(np.argmax(top1_hist)) + 1

        results[cfg.key] = {
            "N_start": cfg.N_start or cfg.N_hidden,
            "N_final": cfg.N_final or cfg.N_hidden,
            "D_start": cfg.D_start or cfg.D,
            "D_final": cfg.D_final or cfg.D,
            "mode": cfg.mode, "K_hh": K_HH, "K_iter": K_ITER,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "final_active_neurons": history[-1]["active_neurons"],
            "final_active_dims": history[-1]["active_dims"],
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params, "label": cfg.label,
        }

        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs = top1_best - ref_best if ref_best > 0 else 0
        print(f"\n  top1={top1_best:.4f}  vs_Ref={vs:+.4f}  "
              f"final: N={history[-1]['active_neurons']} D={history[-1]['active_dims']}  "
              f"params={n_params:,}")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 153 SUMMARY — Progressive Capacity Reduction")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for k, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"  {k:4s}  {r['mode']:>16s}  "
              f"N={r['final_active_neurons']:>5d}  D={r['final_active_dims']:>3d}  "
              f"params={r['n_params']:>8,}  top1={r['top1_best']:.4f}  vs_Ref={vs:+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
