"""Step 165: Compound warm-start efficiency stack at N=1024 D=16.

MOTIVATION
==========
step163-E: warm-start (K=12→K=8) alone gives +7.82pp (87.26% total, N=1024 D=16).
step144-C: W_proj + RigL synergistic at N=1024 D=32 (+5.55pp compound).
step143-B: twopop_weight het neurons +6.42pp at N=1024 D=16.

QUESTION: Do W_proj, RigL, and het-neurons compound on top of warm-start?
At N=4096, compounding onto AH kills gain. At N=1024 D=32, W_proj+RigL was synergistic.
D=16 with warm-start init is a new operating point — outcome is unknown.

CONFIGS (N=1024, D=16, K_hh=8, K_iter=8, K_in=25, AH=1.0, 50%/75ep)
======================================================================
  Ref : K=8 scratch (no warm-start, no extra mechanisms)
  A   : K=8 warm-start from K=12 teacher (confirm step163-E ~87%)
  B   : K=8 warm-start + W_proj [D,D]
  C   : K=8 warm-start + W_proj + RigL (every 5ep, freeze last 30%)
  D   : K=8 warm-start + twopop_weight (relay×2 / specialist×0.5 edge scales)

Baseline: step163/164 Ref ≈ 79-81% (K=8 scratch, N=1024 D=16).
Teacher checkpoint cached at results/train_step165_teacher.pt.

To reproduce:
    python -u scripts/train_step165_compound_warmstart.py --device mps
    python -u scripts/train_step165_compound_warmstart.py --device cpu
    python -u scripts/train_step165_compound_warmstart.py --configs A,B --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--configs", default="",
                    help="Comma-sep config keys (e.g. A,B). Empty = all.")
parser.add_argument("--force-retrain-teacher", action="store_true")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 8; K_IN = 25; K_ITER = 8; K_ITER_TEACHER = 12
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
N_CANDIDATES  = 16   # RigL: non-neighbor candidates per neuron per refinement
RELAY_SCALE   = 2.0; SPEC_SCALE = 0.5   # twopop_weight init

TEACHER_PATH = ROOT / "results" / "train_step165_teacher.pt"
OUT_PATH     = ROOT / "results" / "train_step165_compound_warmstart.json"


# ---------------------------------------------------------------------------
# Compound model: warm-start + optional W_proj + optional twopop_weight
# ---------------------------------------------------------------------------

class SGNNET_CompoundWarmStack(nn.Module):
    """SGNNET_AntiHebbian core + optional W_proj and/or twopop_weight het neurons.

    Warm-start: call warm_load(teacher_state_dict) after construction to copy
    matching params (W_pos, theta, conn_hh, conn_in) from the teacher.
    Extra params (proj.weight, edge_scale) are left at their init values.

    use_proj=True:     apply nn.Linear(D,D,bias=False) after _seed().
                       Init near-zero to avoid dominating the warm-started Z.
    use_twopop=True:   two-population edge-weight scaling (relay×2 / spec×0.5).
    """

    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float = 1.0,
                 use_proj: bool = False, use_twopop: bool = False):
        super().__init__()
        self.m           = resonant
        self.alpha_ahebb = alpha_ahebb
        self.use_proj    = use_proj
        self.use_twopop  = use_twopop

        D_ = resonant.base.D
        N_h = resonant.base.N_hidden

        if use_proj:
            self.proj = nn.Linear(D_, D_, bias=False)
            # Init near-zero so warm-started Z is not disrupted on first pass
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)

        if use_twopop:
            half = N_h // 2
            pop_mask = torch.zeros(N_h, dtype=torch.long)
            pop_mask[half:] = 1   # pop 0 = relay, pop 1 = specialist
            self.register_buffer("pop_mask", pop_mask)
            # Learnable per-pop edge-weight scales
            self.edge_scale = nn.Parameter(torch.tensor([RELAY_SCALE, SPEC_SCALE]))

    @property
    def W_pos(self):   return self.m.W_pos

    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def warm_load(self, teacher_state: dict) -> None:
        """Load matching params from teacher state_dict (strict=False)."""
        missing, unexpected = self.load_state_dict(teacher_state, strict=False)
        loaded = [k for k in teacher_state if k not in unexpected]
        print(f"    warm_load: loaded {len(loaded)} keys, "
              f"missing {len(missing)} (new params), "
              f"unexpected {len(unexpected)} (teacher-only)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base      = self.m.base
        Z         = base._seed(x)

        if self.use_proj:
            Z = self.proj(Z)

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = base.conn_hh
        N_h       = base.N_hidden

        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        if self.use_twopop:
            # Per-neuron edge-weight scale [1, N_h, 1, 1]
            es = self.edge_scale.abs()[self.pop_mask]
            edge_mult = es.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)

        for _ in range(base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_nb     = Z_fwd[:, conn_hh, :]    # [B, N_h, K_hh, D]

            if self.use_twopop:
                Z_nb = Z_nb * edge_mult

            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z           = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return base._readout(Z)


# ---------------------------------------------------------------------------
# RigL topology refinement (adapted from step144)
# ---------------------------------------------------------------------------

@torch.no_grad()
def refine_topology_scored(model: SGNNET_CompoundWarmStack,
                           loader, device: torch.device,
                           max_swaps: int = 1) -> int:
    batch_x = next(iter(loader))[0].to(device)
    base     = model.m.base
    resonant = model.m
    conn_hh  = base.conn_hh
    N_h      = base.N_hidden

    Z         = base._seed(batch_x)
    theta_pos = resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
    for _ in range(base.K_iter // 2):
        Z_fwd = F.relu(Z - theta_pos)
        Z_nb  = Z_fwd[:, conn_hh, :]
        Z     = F.normalize(Z_nb.sum(dim=2).clamp(-10, 10), dim=-1)

    Z_neighbors     = Z[:, conn_hh, :]
    Z_expanded      = Z.unsqueeze(2).expand_as(Z_neighbors)
    existing_scores = (Z_expanded - Z_neighbors).abs().mean(dim=(0, -1))

    conn_np  = conn_hh.cpu().numpy()
    new_conn = conn_hh.clone()
    total_swaps = 0

    for h in range(N_h):
        nbr_set     = set(conn_np[h].tolist()); nbr_set.add(h)
        non_nbrs    = [i for i in range(N_h) if i not in nbr_set]
        if not non_nbrs:
            continue
        n_s         = min(N_CANDIDATES, len(non_nbrs))
        cand_idx    = np.random.choice(non_nbrs, size=n_s, replace=False)
        cand_idx_t  = torch.tensor(cand_idx, dtype=torch.long, device=device)
        Z_h         = Z[:, h, :]
        Z_cands     = Z[:, cand_idx_t, :]
        cand_scores = (Z_h.unsqueeze(1) - Z_cands).abs().mean(dim=(0, -1))
        e_sorted    = existing_scores[h].argsort()
        c_sorted    = cand_scores.argsort(descending=True)
        for s in range(min(max_swaps, K_HH)):
            wp  = e_sorted[s].item()
            cp  = c_sorted[s].item() if s < len(c_sorted) else None
            if cp is None:
                break
            worst_n  = conn_np[h][wp]
            best_c   = cand_idx[cp]
            existing_scores_h = existing_scores[h].clone()
            if cand_scores[cp] > existing_scores_h[wp]:
                new_conn[h, wp] = best_c
                total_swaps    += 1

    base.conn_hh.copy_(new_conn)
    return total_swaps


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key:       str
    label:     str
    warm_init: bool
    use_proj:  bool
    use_rigl:  bool
    use_twopop: bool


CONFIGS = [
    Config("Ref", "Ref  K=8 scratch — no warm-start, no mechanisms",
           False, False, False, False),
    Config("A",   "A    K=8 warm-start only (confirm step163-E ~87%)",
           True,  False, False, False),
    Config("B",   "B    K=8 warm-start + W_proj [D,D]",
           True,  True,  False, False),
    Config("C",   "C    K=8 warm-start + W_proj + RigL (every 5ep)",
           True,  True,  True,  False),
    Config("D",   "D    K=8 warm-start + twopop_weight (relay×2 / spec×0.5)",
           True,  False, False, True),
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_loaders_cache = None

def get_loaders():
    global _loaders_cache
    if _loaders_cache is None:
        tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
        n   = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        sub = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr  = torch.utils.data.DataLoader(sub, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders_cache = (tr, va)
    return _loaders_cache


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_base_model(k_iter: int = K_ITER, seed: int = SEED) -> SGNNET_AntiHebbian:
    """Build a plain SGNNET_AntiHebbian (used as teacher, and as Ref)."""
    torch.manual_seed(seed)
    K_random = max(1, K_HH // 4)
    K_local  = K_HH - K_random
    n_groups = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=k_iter,
        K_local=K_local, K_random=K_random, n_groups=n_groups,
        norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def make_student(cfg: Config, seed_offset: int = 0) -> SGNNET_CompoundWarmStack:
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
    return SGNNET_CompoundWarmStack(
        resonant, alpha_ahebb=ALPHA_AHEBB,
        use_proj=cfg.use_proj, use_twopop=cfg.use_twopop,
    )


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Teacher: train or load from cache
# ---------------------------------------------------------------------------

def get_or_train_teacher(force: bool = False) -> SGNNET_AntiHebbian:
    teacher = make_base_model(k_iter=K_ITER_TEACHER, seed=SEED).to(DEVICE)

    if not force and TEACHER_PATH.exists():
        print(f"  Loading cached teacher from {TEACHER_PATH}")
        state = torch.load(TEACHER_PATH, map_location=DEVICE, weights_only=True)
        teacher.load_state_dict(state)
        return teacher

    print(f"\n{'─'*60}")
    print(f"Training teacher  K_iter={K_ITER_TEACHER}  epochs={EPOCHS}")
    print(f"{'─'*60}")
    t0      = time.time()
    kw      = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(
        model=teacher, train_loader=get_loaders()[0],
        val_loader=get_loaders()[1], device=DEVICE, **kw,
    )
    def _log(m):
        if str(DEVICE) == "mps":
            torch.mps.empty_cache()
        if (m["epoch"] + 1) % 10 == 0:
            print(f"    ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
    teacher_history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
    elapsed  = time.time() - t0
    best_top1 = max(h["val_top1"] for h in teacher_history)
    print(f"  Teacher done: top1={best_top1:.4f}  elapsed={elapsed/60:.1f}min")
    TEACHER_PATH.parent.mkdir(exist_ok=True)
    torch.save(teacher.state_dict(), TEACHER_PATH)
    return teacher


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 165 — Compound warm-start efficiency stack (N=1024 D=16)")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}")

    print("\nConfigs:")
    for c in CONFIGS:
        mech = []
        if c.warm_init:  mech.append("warm-start")
        if c.use_proj:   mech.append("W_proj")
        if c.use_rigl:   mech.append("RigL")
        if c.use_twopop: mech.append("twopop_wt")
        print(f"  {c.key:4s}  {', '.join(mech) or 'scratch':30s}  {c.label}")
    print()

    cfg_filter    = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    # Load / train teacher once (needed for warm-start configs)
    need_teacher = any(cfg.warm_init for _, cfg in active_configs)
    teacher      = None
    if need_teacher:
        teacher = get_or_train_teacher(force=args.force_retrain_teacher)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    get_loaders()
    results  = {}

    for i, cfg in active_configs:
        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"{'─'*60}")

        # Build model
        if not cfg.warm_init and not cfg.use_proj and not cfg.use_twopop:
            # Ref: plain AntiHebbian, no wrapper
            model = make_base_model(k_iter=K_ITER, seed=SEED + i).to(DEVICE)
        else:
            model = make_student(cfg, seed_offset=i).to(DEVICE)
            if cfg.warm_init and teacher is not None:
                print(f"  Warm-loading from teacher...")
                model.warm_load(teacher.state_dict())

        n_params = count_params(model)
        print(f"  params={n_params:,}  warm={cfg.warm_init}  proj={cfg.use_proj}  "
              f"rigl={cfg.use_rigl}  twopop={cfg.use_twopop}")

        # RigL freeze epoch
        freeze_rigl_epoch = int(EPOCHS * 0.70) if cfg.use_rigl else None
        rigl_interval     = 5  # refine every N epochs

        t0  = time.time()
        kw  = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(
            model=model, train_loader=get_loaders()[0],
            val_loader=get_loaders()[1], device=DEVICE, **kw,
        )

        # Custom training loop to inject RigL at epoch boundaries
        if cfg.use_rigl:
            # Bypass Trainer.train() to hook epoch boundaries
            history    = []
            ep_start   = 0
            for epoch in range(ep_start, EPOCHS):
                train_m = trainer.train_epoch()
                val_m   = trainer.evaluate()
                train_loss = train_m["train_loss"]

                if trainer.sched_type not in ("cosine", "warm_restarts") and trainer.scheduler:
                    trainer.scheduler.step(train_loss)

                combined = {"epoch": epoch, **train_m, **val_m, "lr": trainer.current_lr()}
                history.append(combined)

                # Sync before RigL to avoid MPS pipeline stall
                if str(DEVICE) == "mps":
                    torch.mps.empty_cache()

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"  ep{epoch+1:3d}  val={val_m['val_top1']:.4f}  "
                          f"lr={trainer.current_lr():.2e}", flush=True)

                # RigL topology refinement
                if (freeze_rigl_epoch is None or epoch < freeze_rigl_epoch):
                    if (epoch + 1) % rigl_interval == 0:
                        n_swaps = refine_topology_scored(model, get_loaders()[0], DEVICE)
                        if n_swaps > 0:
                            print(f"    RigL ep{epoch+1}: {n_swaps} edge swaps", flush=True)

                if trainer.early_stopping.step(train_loss, model):
                    print(f"  Early stop ep{epoch+1}")
                    break
        else:
            def _log(m):
                if str(DEVICE) == "mps":
                    torch.mps.empty_cache()
                if (m["epoch"] + 1) % 10 == 0:
                    print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            history = trainer.train(n_epochs=EPOCHS, log_fn=_log)

        elapsed = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1

        results[cfg.key] = {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER, "K_in": K_IN,
            "warm_init": cfg.warm_init, "use_proj": cfg.use_proj,
            "use_rigl": cfg.use_rigl, "use_twopop": cfg.use_twopop,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1), "n_params": n_params,
            "label": cfg.label,
        }

        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs_ref   = top1_best - ref_best if ref_best > 0 else 0
        print(f"\n  top1={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min  params={n_params:,}")

        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 165 SUMMARY — Compound warm-start at N=1024 D=16")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Key':4s}  {'mechanisms':30s}  {'params':>8}  {'top1':>7}  {'vs_Ref':>7}")
    print(f"{'─'*60}")
    for key, r in results.items():
        mechs = []
        if r["warm_init"]:  mechs.append("warm")
        if r["use_proj"]:   mechs.append("W_proj")
        if r["use_rigl"]:   mechs.append("RigL")
        if r["use_twopop"]: mechs.append("twopop")
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"{key:4s}  {','.join(mechs) or 'scratch':30s}  "
              f"{r['n_params']:>8,}  {r['top1_best']:.4f}  {vs:>+.4f}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {OUT_PATH}")
    print(f"\nDecision: if B or C > A by ≥+1pp → W_proj/RigL compound on warm-start.")
    print(f"If all ≈ A → warm-start W_pos is the ceiling, mechanisms don't compound.")


if __name__ == "__main__":
    main()
