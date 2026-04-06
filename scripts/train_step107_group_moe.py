"""Step 107: Group-as-Expert MoE routing (Gemma4-inspired).

MOTIVATION
==========
Gemma 4 uses 128 experts with top-2 activation (1.6% sparse). SGNNET has
n_groups=8 (step82 winner: +3.01pp at N=1024). Each group = an expert.
A ReLU router selects top-k groups per input token. Only selected groups
participate in message passing → massive FLOPs reduction.

Why this differs from step83 (KILLED):
  1. ReLU not softmax — constant gradient for active routes
  2. L1 load balancing — prevents group domination
  3. Shared expert — group 0 always active (like Gemma4)
  4. Token-level routing — per-image group selection, not batch-level

Gate-death safe: ReLU(x) for x>0 has gradient=1. For x<=0, cleanly off.
No multiplicative compounding. Sparse activation (on/off), not scaling.

CONFIGS (N=1024, D=64, K_hh=4, K_iter=12, n_groups=8, 50%/75ep)
=================================================================
  Ref : all 8 groups active (no routing, step82 baseline)
  A   : top-3 + shared expert (group 0), L1=0.01
  B   : top-2 + shared expert, L1=0.01 (sparser)
  C   : top-3, NO shared expert, L1=0.01
  D   : top-3 + shared expert, L1=0.001 (weaker L1)
  E   : top-3 + shared expert, L1=0.01, routing delayed 30ep

To reproduce:
    python -u scripts/train_step107_group_moe.py --device mps
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
from src.training.experiment_config   import trainer_kwargs, topology_kwargs, run_metadata
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
N_GROUPS = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
STEP69_REF = 0.8336


class SGNNET_AH_GroupMoE(nn.Module):
    """AH routing with top-k group selection per token (MoE-style).

    Architecture:
      S_g = mean(Z[neurons in group g])           [B, n_groups, D]
      logits = (W_route @ S_g^T).diagonal          [B, n_groups]
      active = top-k of ReLU(logits) + shared expert (group 0)
      Only active groups run message passing. Inactive: Z frozen.
    """

    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float,
                 n_groups: int, top_k: int, l1_lambda: float,
                 shared_expert: bool, delay_epochs: int = 0):
        super().__init__()
        self.m     = resonant
        self.alpha = alpha_ahebb
        self.n_groups = n_groups
        self.top_k = top_k
        self.l1_lambda = l1_lambda
        self.shared_expert = shared_expert
        self.delay_epochs = delay_epochs
        self._current_epoch = 0

        # Router: one D-dim vector per group
        self.W_route = nn.Parameter(torch.randn(n_groups, D) * 0.01)

        N_h = resonant.base.N_hidden
        self.register_buffer("group_ids",
            torch.arange(N_h) % n_groups)                         # [N_h]
        # Group masks [n_groups, N_h] — boolean-like
        masks = torch.zeros(n_groups, N_h)
        for g in range(n_groups):
            masks[g, self.group_ids == g] = 1.0
        self.register_buffer("group_masks", masks)
        counts = masks.sum(dim=1, keepdim=True).clamp(min=1)
        self.register_buffer("group_counts", counts)               # [n_groups, 1]

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase
    @property
    def routing_l1(self):
        return self._routing_l1

    def tick_epoch(self):
        self._current_epoch += 1
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _get_active_mask(self, Z: torch.Tensor) -> torch.Tensor:
        """Returns [B, n_groups] boolean mask of active groups per token."""
        B = Z.shape[0]
        N_h = self.m.base.N_hidden

        # Group centroids: [B, n_groups, D]
        S_g = torch.einsum("bnd,gn->bgd", Z[:, :N_h, :], self.group_masks)
        S_g = S_g / self.group_counts.unsqueeze(0)

        # ReLU routing scores: [B, n_groups]
        logits = torch.einsum("bgd,gd->bg", S_g, self.W_route)
        scores = F.relu(logits)

        # L1 penalty on scores
        self._routing_l1 = self.l1_lambda * scores.mean()

        # Top-k selection
        if self.shared_expert:
            # Group 0 always active; select top_k from remaining
            remaining_scores = scores[:, 1:]                       # [B, n_groups-1]
            _, topk_idx = remaining_scores.topk(min(self.top_k, self.n_groups - 1), dim=1)
            active = torch.zeros(B, self.n_groups, device=Z.device, dtype=torch.bool)
            active[:, 0] = True                                    # shared expert
            # Scatter top-k into active mask (offset by 1 for group 0)
            active.scatter_(1, topk_idx + 1, True)
        else:
            _, topk_idx = scores.topk(min(self.top_k, self.n_groups), dim=1)
            active = torch.zeros(B, self.n_groups, device=Z.device, dtype=torch.bool)
            active.scatter_(1, topk_idx, True)

        return active                                              # [B, n_groups]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                          # [B, N, D]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden

        # AH wpos suppression (static)
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        # Determine active groups (or all groups if delayed)
        routing_active = self._current_epoch >= self.delay_epochs
        if routing_active:
            active_groups = self._get_active_mask(Z)               # [B, n_groups]
        else:
            active_groups = torch.ones(Z.shape[0], self.n_groups,
                                       device=x.device, dtype=torch.bool)
            self._routing_l1 = torch.tensor(0.0, device=x.device)

        # Neuron active mask: [B, N_h] — neuron active if its group is active
        # group_ids: [N_h], active_groups: [B, n_groups]
        neuron_active = active_groups[:, self.group_ids]           # [B, N_h]
        # Pad for input neurons (always active)
        N_total = Z.shape[1]
        if N_total > N_h:
            pad = torch.ones(Z.shape[0], N_total - N_h, device=x.device, dtype=torch.bool)
            neuron_active = torch.cat([neuron_active, pad], dim=1)

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z_fwd  = F.relu(Z - theta_pos)
            Z_nb   = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected

            # Mask: inactive neurons keep their Z, active neurons update
            neuron_mask = neuron_active[:, :N_h].unsqueeze(-1).float()  # [B, N_h, 1]
            if N_total > N_h:
                neuron_mask = torch.cat([neuron_mask,
                    torch.ones(Z.shape[0], N_total - N_h, 1, device=x.device)], dim=1)
            Z_normed = F.normalize(Z_new.clamp(-10, 10), dim=-1)
            Z = neuron_mask * Z_normed + (1.0 - neuron_mask) * Z

        return self.m.base._readout(Z)


@dataclass
class Config:
    key: str; label: str; top_k: int; shared_expert: bool
    l1: float; delay_epochs: int
    use_moe: bool = True


CONFIGS = [
    Config("Ref", "Ref  all groups active (no MoE)", 8, False, 0.0, 0, use_moe=False),
    Config("A",   "A    top-3 + shared, L1=0.01",   3, True,  0.01,  0),
    Config("B",   "B    top-2 + shared, L1=0.01",   2, True,  0.01,  0),
    Config("C",   "C    top-3, no shared, L1=0.01",  3, False, 0.01,  0),
    Config("D",   "D    top-3 + shared, L1=0.001",  3, True,  0.001, 0),
    Config("E",   "E    top-3 + shared, L1=0.01, delay=30", 3, True, 0.01, 30),
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
    if not cfg.use_moe:
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_AH_GroupMoE(
        resonant, ALPHA_AHEBB, N_GROUPS, cfg.top_k, cfg.l1,
        cfg.shared_expert, cfg.delay_epochs)


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run(cfg, model, meta):
    print(f"\n{'='*70}\n{cfg.label}\n{'='*70}")
    tr, va = get_loaders()
    tk = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0 = time.time(); history = trainer.train(n_epochs=EPOCHS); elapsed = time.time() - t0
    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best = max(top1_hist); best_ep = int(np.argmax(top1_hist)) + 1
    frac = best_ep / len(history)
    result = {
        "label": cfg.label, "top_k": cfg.top_k,
        "shared_expert": cfg.shared_expert, "l1_lambda": cfg.l1,
        "delay_epochs": cfg.delay_epochs,
        "top1_best": best, "top1_last": history[-1].get("val_top1", 0.0),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1), "best_epoch_frac": round(frac, 3),
        "step69_ref": STEP69_REF, "delta_vs_ref": round(best - STEP69_REF, 4),
        "params": count_params(model), "top1_history": top1_hist,
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1={best:.4f}  ep={best_ep}/{len(history)}  vs_ref={best-STEP69_REF:+.4f}  t={elapsed:.0f}s")
    return result


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  N={N}  K_iter={K_ITER}  n_groups={N_GROUPS}  Data: 50%")
    print(f"Step 107: Group-as-Expert MoE routing (Gemma4-inspired)")
    print(f"Baseline: step69 Ref = {STEP69_REF:.4f}\n")
    for c in CONFIGS:
        if c.use_moe:
            tag = f"top-{c.top_k}  shared={c.shared_expert}  L1={c.l1}  delay={c.delay_epochs}"
        else:
            tag = "disabled (all groups active)"
        print(f"  {c.key:4s}  {tag}")
    print()
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")
    results = {}
    out_path = ROOT / "results" / "train_step107_group_moe.json"
    for i, cfg in enumerate(CONFIGS):
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        meta = {"N": N, "D": D, "K_iter": K_ITER, "n_groups": N_GROUPS,
                "top_k": cfg.top_k, "shared_expert": cfg.shared_expert,
                "l1_lambda": cfg.l1, "delay_epochs": cfg.delay_epochs,
                "alpha_ahebb": ALPHA_AHEBB, "data_frac": 0.5}
        results[cfg.key] = run(cfg, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")
    print(f"\n{'='*70}\nSTEP 107 COMPLETE\n")
    print(f"  {'Key':4s}  {'top_k':>5s}  {'shared':>6s}  {'L1':>6s}  {'top1':>8s}  {'vs_ref':>8s}")
    for c in CONFIGS:
        if c.key in results:
            r = results[c.key]
            print(f"  {c.key:4s}  {c.top_k:>5d}  {'Y' if c.shared_expert else 'N':>6s}  "
                  f"{c.l1:>6.3f}  {r['top1_best']:.4f}  {r['delta_vs_ref']:+.4f}")
    w = max(results, key=lambda k: results[k]["top1_best"])
    print(f"\n  Winner: {w}")
