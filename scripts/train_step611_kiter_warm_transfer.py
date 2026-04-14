"""Step 611: K_iter warm transfer (teacher→student weight handoff).

MOTIVATION
==========
Step163 warm-start (K=12 teacher → K=8 student, same weights) won +7.82pp over
constant K=8 at N=1024 D=16 K_hh=8. This re-tests on the efficiency config
(N=2048, D=16, K_hh=2) and pushes the student K further down to K=3.

The mechanism: a higher-K teacher develops a rich representation that enables the
student to converge faster with fewer routing steps. Weight transfer, not
distillation — same model, just K_iter reduced at the transition point.

CONFIGS (N=2048, D=16, K_hh=2, AH=1.0, 75ep total, 50% data — Tier-1)
=======================================================================
  Ref      : K_iter=5 constant throughout (75ep single phase)
  A_12to5  : Teacher K=12 for 40ep, Student K=5 for 35ep
  B_12to3  : Teacher K=12 for 40ep, Student K=3 for 35ep
  C_8to5   : Teacher K=8  for 40ep, Student K=5 for 35ep (smaller compression)
  D_16to5  : Teacher K=16 for 30ep, Student K=5 for 45ep (bigger teacher)

CRITICAL: Optimizer state (LR, Adam momentum) is NOT reset at transition.
The switch is just model.m.base.K_iter = STUDENT_K between epochs.

Tracks: effective_final_flops (student K × N × K_hh × D factor).
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

from src.sgnnet.model_smallworld       import SGNNET_SmallWorld
from src.sgnnet.model_resonant         import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory  import SGNNET_AntiHebbian
from src.training.trainer              import Trainer
from src.training.experiment_config    import trainer_kwargs
from src.training.dataset              import make_loaders

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75,
                    help="Total epochs per config (default 75)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys. Empty = all.")
args = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25
ALPHA_AHEBB = 1.0; ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

OUT_PATH = ROOT / "results" / "train_step611_kiter_warm_transfer.json"


# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------
@dataclass
class Config:
    key: str
    label: str
    teacher_k: int      # K_iter during teacher phase
    teacher_ep: int     # epochs in teacher phase (student gets EPOCHS - teacher_ep)
    student_k: int      # K_iter during student phase

    @property
    def student_ep(self) -> int:
        return EPOCHS - self.teacher_ep


CONFIGS = [
    Config("Ref",     "Ref     K_iter=5 constant (75ep)",    teacher_k=5,  teacher_ep=EPOCHS, student_k=5),
    Config("A_12to5", "A_12to5 Teacher K=12 40ep → Student K=5 35ep", teacher_k=12, teacher_ep=40, student_k=5),
    Config("B_12to3", "B_12to3 Teacher K=12 40ep → Student K=3 35ep", teacher_k=12, teacher_ep=40, student_k=3),
    Config("C_8to5",  "C_8to5  Teacher K=8  40ep → Student K=5 35ep", teacher_k=8,  teacher_ep=40, student_k=5),
    Config("D_16to5", "D_16to5 Teacher K=16 30ep → Student K=5 45ep", teacher_k=16, teacher_ep=30, student_k=5),
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
def make_model(init_k: int, seed_offset: int = 0):
    torch.manual_seed(SEED + seed_offset)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=init_k,
                              K_local=K_l, K_random=K_r, n_groups=ng,
                              norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def _get_base(model) -> SGNNET_SmallWorld:
    return model.m.base


def compute_flops(k_iter: int) -> int:
    seed     = N * K_IN * D
    per_step = N * K_HH * D * 2 + N * D + N * D * 2
    routing  = k_iter * per_step
    readout  = N * N_OUT * D
    return seed + routing + readout


# ---------------------------------------------------------------------------
# Two-phase training
# ---------------------------------------------------------------------------
def train_two_phase(model, cfg: Config) -> dict:
    """Run teacher phase then student phase. Optimizer state preserved across."""
    tr, va = get_loaders()
    # Build trainer for the full budget (LR schedule spans all epochs)
    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **kw)
    base = _get_base(model)
    top1_hist = []
    k_hist = []

    # Teacher phase
    base.K_iter = cfg.teacher_k
    print(f"    [teacher phase] K_iter={cfg.teacher_k} for ep1-{cfg.teacher_ep}", flush=True)
    for ep in range(1, cfg.teacher_ep + 1):
        ep_hist = trainer.train(n_epochs=1)
        v = round(ep_hist[-1].get("val_top1", 0.0), 4)
        top1_hist.append(v)
        k_hist.append(cfg.teacher_k)
        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep:3d} [teacher K={cfg.teacher_k}]  val={v:.4f}", flush=True)

    if cfg.student_ep > 0:
        # Student phase — no optimizer reset
        base.K_iter = cfg.student_k
        print(f"    [student phase] K_iter → {cfg.student_k} at ep{cfg.teacher_ep+1}", flush=True)
        for ep in range(cfg.teacher_ep + 1, EPOCHS + 1):
            ep_hist = trainer.train(n_epochs=1)
            v = round(ep_hist[-1].get("val_top1", 0.0), 4)
            top1_hist.append(v)
            k_hist.append(cfg.student_k)
            if (ep - cfg.teacher_ep) % 10 == 0 or ep == cfg.teacher_ep + 1:
                print(f"  ep{ep:3d} [student K={cfg.student_k}]  val={v:.4f}", flush=True)

    best_idx = int(np.argmax(top1_hist))
    return {
        "top1_history": top1_hist,
        "k_iter_history": k_hist,
        "best_top1": max(top1_hist),
        "best_epoch": best_idx + 1,
        "k_at_best": k_hist[best_idx],
        "final_k_iter": k_hist[-1],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    # Need to update Ref's teacher_ep to match EPOCHS (handles --epochs override)
    for c in CONFIGS:
        if c.key == "Ref":
            c.teacher_ep = EPOCHS
    active = [(i, c) for i, c in enumerate(CONFIGS)
              if not cfg_filter or c.key in cfg_filter]

    print(f"\n{'='*70}")
    print(f"Step 611 — K_iter warm transfer (teacher→student)")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_in={K_IN}  AH={ALPHA_AHEBB}")
    print(f"Total epochs={EPOCHS}  Device={DEVICE}  Data=50%  Tier-1")
    print(f"Running: {[c.key for _, c in active]}")
    print(f"{'='*70}\n")

    get_loaders()
    results = {}

    for i, cfg in active:
        n_params = sum(p.numel() for p in make_model(cfg.teacher_k).parameters()
                       if p.requires_grad)
        student_flops = compute_flops(cfg.student_k)
        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  teacher_k={cfg.teacher_k}  teacher_ep={cfg.teacher_ep}")
        print(f"  student_k={cfg.student_k}  student_ep={cfg.student_ep}")
        print(f"  params={n_params:,}  student_flops={student_flops/1e6:.2f}M")
        print(f"{'─'*60}")

        model = make_model(cfg.teacher_k, seed_offset=i).to(DEVICE)
        t0 = time.time()
        r = train_two_phase(model, cfg)
        elapsed = time.time() - t0

        results[cfg.key] = {
            "N": N, "D": D, "K_hh": K_HH, "K_in": K_IN,
            "alpha_ahebb": ALPHA_AHEBB, "data_frac": 0.5,
            "label": cfg.label,
            "teacher_k": cfg.teacher_k, "teacher_ep": cfg.teacher_ep,
            "student_k": cfg.student_k, "student_ep": cfg.student_ep,
            **r,
            "effective_final_flops": student_flops,
            "effective_final_flops_M": round(student_flops / 1e6, 2),
            "n_params": n_params,
            "elapsed_s": round(elapsed, 1),
        }

        ref_best = results.get("Ref", {}).get("best_top1", 0)
        vs_ref = r["best_top1"] - ref_best if ref_best > 0 else 0
        print(f"\n  best={r['best_top1']:.4f} @ ep{r['best_epoch']} "
              f"(K_iter={r['k_at_best']})  vs_Ref={vs_ref:+.4f}  "
              f"student_K={cfg.student_k}  {elapsed/60:.1f}min")

        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # Summary
    ref_best = results.get("Ref", {}).get("best_top1", 0)
    print(f"\n{'='*70}")
    print(f"STEP 611 SUMMARY — K_iter warm transfer")
    print(f"{'='*70}")
    print(f"{'Key':12s}  {'T_K':>4}  {'T_ep':>5}  {'S_K':>4}  "
          f"{'best':>7}  {'vs_Ref':>8}  {'eff_FLOPs_M':>12}")
    print(f"{'─'*65}")
    for key, r in results.items():
        vs = r["best_top1"] - ref_best if ref_best > 0 else 0
        print(f"{key:12s}  {r['teacher_k']:>4}  {r['teacher_ep']:>5}  "
              f"{r['student_k']:>4}  {r['best_top1']:.4f}  {vs:>+.4f}  "
              f"{r['effective_final_flops_M']:>12.2f}")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
