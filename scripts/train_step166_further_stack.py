"""Step 166: Further stacking on step165-B winner (warm+W_proj = 90.96%) at N=1024 D=16.

MOTIVATION
==========
step165-B (warm-start + W_proj): 90.96% at N=1024 D=16 (~3.1M FLOPs) — efficiency record.
Gap to phase-exit criterion (≥95%): +4.04pp.

Confirmed per-mechanism winners at N=1024 (standalone):
  twopop_weight  +6.42pp  (step143-B)
  curriculum     +2.85pp  (step142-C: ramp 2→4→8→12)
  soft_spec      +3.54pp  (step121-B: spectral reg on W_pos)
  weighted_neg   +3.97pp  (step131-A)

QUESTION: Which of these further compounds on top of warm+W_proj?
RigL excluded (killed W_proj in step165-C).

CONFIGS (N=1024, D=16, K_hh=8, K_iter=8, AH=1.0, 50%/75ep)
=============================================================
  Ref : warm+W_proj only (reproduce step165-B, ~90.96%)
  E   : warm+W_proj+twopop_weight
  F   : warm+W_proj+curriculum K_iter (ramp 2→4→8 over 75ep)
  G   : warm+W_proj+soft_spec (spectral reg λ=0.01 on W_pos)
  H   : warm+W_proj+twopop_weight+curriculum (E+F compound)

Teacher checkpoint reused from step165: results/train_step165_teacher.pt.

To reproduce:
    python -u scripts/train_step166_further_stack.py --device mps
    python -u scripts/train_step166_further_stack.py --device cpu
    python -u scripts/train_step166_further_stack.py --configs Ref,E --device mps
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

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--configs", default="",
                    help="Comma-sep config keys (e.g. E,F). Empty = all.")
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
RELAY_SCALE   = 2.0; SPEC_SCALE = 0.5
SOFT_SPEC_LAM = 0.01   # spectral regularization lambda

TEACHER_PATH = ROOT / "results" / "train_step165_teacher.pt"
OUT_PATH     = ROOT / "results" / "train_step166_further_stack.json"


# ---------------------------------------------------------------------------
# Model: warm+W_proj base + optional twopop / curriculum / soft_spec
# ---------------------------------------------------------------------------

class SGNNET_FurtherStack(nn.Module):
    """Extends step165 CompoundWarmStack with additional mechanisms:
      use_twopop    : two-population edge-weight scaling (relay×2 / spec×0.5)
      use_curriculum: ramp K_iter from k_start→K_ITER over training
      soft_spec_lam : spectral regularization on W_pos (||W_pos^T W_pos - I||_F * λ)
    """

    def __init__(self, resonant: "SGNNET_Resonant", alpha_ahebb: float = 1.0,
                 use_twopop: bool = False,
                 k_start: int = 0,           # 0 = no curriculum
                 soft_spec_lam: float = 0.0):
        super().__init__()
        self.m            = resonant
        self.alpha_ahebb  = alpha_ahebb
        self.use_twopop   = use_twopop
        self.soft_spec_lam = soft_spec_lam
        self._k_iter_curr = resonant.base.K_iter   # current K_iter (mutable for curriculum)
        self.k_start      = k_start
        self._epoch        = 0

        D_  = resonant.base.D
        N_h = resonant.base.N_hidden

        # W_proj: always present (this is the step165-B mechanism)
        self.proj = nn.Linear(D_, D_, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)

        if use_twopop:
            half = N_h // 2
            pop_mask = torch.zeros(N_h, dtype=torch.long)
            pop_mask[half:] = 1
            self.register_buffer("pop_mask", pop_mask)
            self.edge_scale = nn.Parameter(torch.tensor([RELAY_SCALE, SPEC_SCALE]))

    @property
    def W_pos(self):   return self.m.W_pos

    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        """Called at start of each epoch to update curriculum K_iter."""
        self._epoch += 1
        if self.k_start > 0:
            # Linear ramp: k_start → K_ITER over first EPOCHS epochs
            frac = min(1.0, self._epoch / EPOCHS)
            self._k_iter_curr = max(self.k_start,
                                    int(self.k_start + frac * (K_ITER - self.k_start)))

    def warm_load(self, teacher_state: dict) -> None:
        missing, unexpected = self.load_state_dict(teacher_state, strict=False)
        loaded = [k for k in teacher_state if k not in unexpected]
        print(f"    warm_load: {len(loaded)} keys loaded, "
              f"{len(missing)} new (proj/edge_scale), {len(unexpected)} teacher-only")

    def spectral_loss(self) -> torch.Tensor:
        """||W_pos[:N_h]^T @ W_pos[:N_h] / N_h - I_D||_F^2 * lambda"""
        if self.soft_spec_lam <= 0.0:
            return torch.tensor(0.0, device=self.proj.weight.device)
        N_h  = self.m.base.N_hidden
        W    = self.m.W_pos[:N_h]          # [N_h, D]
        W_n  = F.normalize(W, dim=-1)
        gram = W_n.T @ W_n / N_h           # [D, D]
        eye  = torch.eye(W.shape[-1], device=W.device)
        return self.soft_spec_lam * (gram - eye).pow(2).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base     = self.m.base
        Z        = base._seed(x)
        Z        = self.proj(Z)

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = base.conn_hh
        N_h       = base.N_hidden

        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        if self.use_twopop:
            es         = self.edge_scale.abs()[self.pop_mask]
            edge_mult  = es.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)
        k_use = self._k_iter_curr

        for _ in range(k_use):
            Z_fwd   = F.relu(Z - theta_pos)
            Z_nb    = Z_fwd[:, conn_hh, :]

            if self.use_twopop:
                Z_nb = Z_nb * edge_mult

            Z_struct    = (Z_nb * supp_w).sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z           = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return base._readout(Z)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key:         str
    label:       str
    use_twopop:  bool
    k_start:     int     # 0 = no curriculum; >0 = ramp from k_start
    spec_lam:    float

CONFIGS = [
    Config("Ref", "Ref  warm+W_proj only (step165-B repro)",
           False, 0,    0.0),
    Config("E",   "E    warm+W_proj+twopop_weight",
           True,  0,    0.0),
    Config("F",   "F    warm+W_proj+curriculum (ramp 2→8)",
           False, 2,    0.0),
    Config("G",   "G    warm+W_proj+soft_spec (λ=0.01)",
           False, 0,    SOFT_SPEC_LAM),
    Config("H",   "H    warm+W_proj+twopop+curriculum (E+F)",
           True,  2,    0.0),
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

def make_student(cfg: Config, seed_offset: int = 0) -> SGNNET_FurtherStack:
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
    return SGNNET_FurtherStack(
        resonant, alpha_ahebb=ALPHA_AHEBB,
        use_twopop=cfg.use_twopop,
        k_start=cfg.k_start,
        soft_spec_lam=cfg.spec_lam,
    )


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Load teacher
# ---------------------------------------------------------------------------

def load_teacher() -> dict:
    if not TEACHER_PATH.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {TEACHER_PATH}\n"
            f"Run train_step165_compound_warmstart.py first to generate it.")
    state = torch.load(TEACHER_PATH, map_location=DEVICE, weights_only=True)
    print(f"  Loaded teacher from {TEACHER_PATH}")
    return state


# ---------------------------------------------------------------------------
# Training loop (with curriculum tick + spectral loss injection)
# ---------------------------------------------------------------------------

def train_with_hooks(model: SGNNET_FurtherStack, trainer: "Trainer",
                     n_epochs: int, device: torch.device) -> list:
    """Custom loop: tick curriculum each epoch, add spectral loss if enabled."""
    history   = []
    use_curriculum = (model.k_start > 0)
    use_spec       = (model.soft_spec_lam > 0)

    for epoch in range(n_epochs):
        model.tick_epoch()   # update curriculum K_iter before training epoch

        if use_spec:
            # Inject spectral loss by patching the trainer's criterion temporarily
            # We do this by wrapping the model's forward to add the aux loss.
            # Simpler: manually do train_epoch + add spectral grad step.
            # Most compatible: override after each batch is complex; instead,
            # add spectral loss in a post-batch hook via separate backward pass.
            # Cleanest: add loss via model-level auxiliary hook in training step.
            # For now: run train_epoch normally, then do one spectral backward.
            train_m = trainer.train_epoch()
            # Spectral regularization step (extra backward, same optimizer)
            model.train()
            spec_loss = model.spectral_loss()
            if spec_loss.item() > 0:
                trainer.optimizer.zero_grad()
                spec_loss.backward()
                trainer.optimizer.step()
        else:
            train_m = trainer.train_epoch()

        val_m = trainer.evaluate()
        if str(device) == "mps":
            torch.mps.empty_cache()

        if trainer.scheduler and trainer.sched_type not in ("cosine", "warm_restarts"):
            trainer.scheduler.step(train_m.get("train_loss", 0))

        combined = {"epoch": epoch, **train_m, **val_m, "lr": trainer.current_lr()}
        history.append(combined)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            k_disp = model._k_iter_curr if use_curriculum else K_ITER
            print(f"  ep{epoch+1:3d}  val={val_m['val_top1']:.4f}  "
                  f"k_iter={k_disp}  lr={trainer.current_lr():.2e}", flush=True)

        if trainer.early_stopping.step(train_m.get("train_loss", 0), model):
            print(f"  Early stop ep{epoch+1}")
            break

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 166 — Further stacking on warm+W_proj (step165-B=90.96%)")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}")

    print("\nConfigs:")
    for c in CONFIGS:
        mech = ["warm+W_proj"]
        if c.use_twopop: mech.append("twopop_wt")
        if c.k_start > 0: mech.append(f"curric({c.k_start}→{K_ITER})")
        if c.spec_lam > 0: mech.append(f"spec_reg(λ={c.spec_lam})")
        print(f"  {c.key:4s}  {'+'.join(mech):45s}  {c.label}")
    print()

    teacher_state = load_teacher()

    cfg_filter     = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    get_loaders()
    results = {}

    for i, cfg in active_configs:
        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"{'─'*60}")

        model    = make_student(cfg, seed_offset=i).to(DEVICE)
        model.warm_load(teacher_state)
        n_params = count_params(model)

        print(f"  params={n_params:,}  twopop={cfg.use_twopop}  "
              f"k_start={cfg.k_start}  spec_lam={cfg.spec_lam}")

        t0      = time.time()
        kw      = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(
            model=model, train_loader=get_loaders()[0],
            val_loader=get_loaders()[1], device=DEVICE, **kw,
        )

        history = train_with_hooks(model, trainer, EPOCHS, DEVICE)
        elapsed = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1

        results[cfg.key] = {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER, "K_in": K_IN,
            "use_twopop": cfg.use_twopop,
            "k_start": cfg.k_start,
            "spec_lam": cfg.spec_lam,
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
    print(f"STEP 166 SUMMARY — Further stacking on warm+W_proj")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"step165-B reference: 90.96%")
    print(f"{'Key':4s}  {'params':>8}  {'top1':>7}  {'vs_165B':>8}")
    print(f"{'─'*40}")
    for key, r in results.items():
        vs165b = r["top1_best"] - 0.9096
        print(f"{key:4s}  {r['n_params']:>8,}  {r['top1_best']:.4f}  {vs165b:>+.4f}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {OUT_PATH}")
    print(f"\nDecision: winner(s) with ≥+1pp vs step165-B → advance to full data 150ep.")
    print(f"Target: ≥95% (phase exit criterion for efficiency track).")


if __name__ == "__main__":
    main()
