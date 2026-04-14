"""Step 169: warm+W_proj D=32 Tier-2 validation — full data 150ep.

MOTIVATION
==========
step168-B (warm+W_proj D=32, 50% data 75ep): 93.20% — +11.24pp over Ref.
step167-B (warm+W_proj D=16, full data 150ep): 92.41% at ep80, still climbing.

Phase-exit criterion: ≥95% accuracy @ ≤6.18M FLOPs.
  D=32 FLOPs: ~6.1M ≤ 6.18M ✓
  D=32 Tier-1 (50% data): 93.20% — 1.80pp short

Full data at D=32 likely adds +1.5-2pp (based on D=16 scaling: scratch +8pp, warm+W_proj
gains less since already warm-started). Expected range: 94.7-95.5%.

CONFIGS (N=1024, D=32, K_hh=8, K_iter=8, AH=1.0, 100% data, 150ep)
=====================================================================
  B   : warm+W_proj full data 150ep ← primary phase-exit candidate

Teacher checkpoint reused from results/train_step168_teacher_d32.pt.
Ref (scratch) skipped — step168 already established scratch=81.96% (50% data);
full data scratch is less interesting than the phase-exit candidate.

To reproduce:
    python -u scripts/train_step169_warmproj_d32_fulldata.py --device mps
    python -u scripts/train_step169_warmproj_d32_fulldata.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10
D = 32; K_HH = 8; K_IN = 25; K_ITER = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

TEACHER_PATH = ROOT / "results" / "train_step168_teacher_d32.pt"
OUT_PATH     = ROOT / "results" / "train_step169_warmproj_d32_fulldata.json"


# ---------------------------------------------------------------------------
# Model (same as step168 Config B)
# ---------------------------------------------------------------------------

class SGNNET_WarmProj(nn.Module):
    def __init__(self, resonant, alpha_ahebb: float = 1.0):
        super().__init__()
        self.m           = resonant
        self.alpha_ahebb = alpha_ahebb
        D_ = resonant.base.D
        self.proj = nn.Linear(D_, D_, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return self.m.W_phase

    def warm_load(self, state: dict) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        loaded = [k for k in state if k not in unexpected]
        print(f"    warm_load: {len(loaded)} keys, {len(missing)} new, {len(unexpected)} teacher-only")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base      = self.m.base
        Z         = base._seed(x)
        Z         = self.proj(Z)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = base.conn_hh
        N_h       = base.N_hidden
        W_n       = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim   = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w    = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                    ).unsqueeze(0).unsqueeze(-1)
        Z_reflected = torch.zeros_like(Z)
        for _ in range(base.K_iter):
            Z_fwd       = F.relu(Z - theta_pos)
            Z_nb        = Z_fwd[:, conn_hh, :]
            Z_struct    = (Z_nb * supp_w).sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z           = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)
        return base._readout(Z)


# ---------------------------------------------------------------------------
# Data — FULL data
# ---------------------------------------------------------------------------

def get_loaders():
    return make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 169 — warm+W_proj D=32 Tier-2 (full data 150ep)")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=100%")
    print(f"{'='*70}")
    print(f"step168-B reference (50% data 75ep): 93.20%")
    print(f"Phase-exit target: ≥95% @ FLOPs ~6.1M\n")

    if not TEACHER_PATH.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {TEACHER_PATH}")

    teacher_state = torch.load(TEACHER_PATH, map_location=DEVICE, weights_only=True)
    print(f"  Teacher loaded from {TEACHER_PATH}")

    torch.manual_seed(SEED)
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
    model = SGNNET_WarmProj(resonant, ALPHA_AHEBB).to(DEVICE)
    print(f"  Warm-loading from D=32 teacher...")
    model.warm_load(teacher_state)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_params:,}")

    tr, va = get_loaders()
    t0      = time.time()
    kw      = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

    def _log(m):
        if str(DEVICE) == "mps":
            torch.mps.empty_cache()
        ep = m["epoch"] + 1
        if ep % 10 == 0:
            flag = " *** PHASE EXIT! ***" if m["val_top1"] >= 0.95 else ""
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}{flag}", flush=True)

    history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
    elapsed = time.time() - t0

    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    top1_best = max(top1_hist)
    best_ep   = int(np.argmax(top1_hist)) + 1
    vs_168b   = top1_best - 0.9320

    result = {
        "B": {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER, "K_in": K_IN,
            "warm_proj": True, "data_frac": 1.0,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1), "n_params": n_params,
            "label": "B warm+W_proj D=32 full data 150ep",
        }
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))

    phase_exit = "✓ PHASE EXIT ACHIEVED!" if top1_best >= 0.95 else f"({0.95 - top1_best:.3f}pp short)"
    print(f"\n{'='*70}")
    print(f"STEP 169 RESULT: {top1_best:.4f}  vs_step168B={vs_168b:+.4f}  {phase_exit}")
    print(f"FLOPs ~6.1M  params={n_params:,}  best_ep={best_ep}")
    print(f"Results saved → {OUT_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
