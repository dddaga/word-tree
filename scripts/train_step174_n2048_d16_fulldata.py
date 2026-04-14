"""Step 174: warm+W_proj N=2048 D=16 Tier-2 — full data 150ep.

MOTIVATION
==========
step170-B (warm+W_proj N=2048 D=16 50%/75ep): 93.43% (best_ep=75).
step167-B (warm+W_proj N=1024 D=16 100%/150ep): 93.17%

N=2048 Tier-1→Tier-2 extrapolation: +1.5-2pp expected → ~94.9-95.4%.
Phase-exit criterion: ≥95% @ ≤6.18M FLOPs (N=2048 D=16 K_iter=8 K_hh=8 ≈ 6.2M).

Teacher reused from results/train_step170_teacher_n2048.pt (K=12 N=2048 D=16 K_hh=8).

CONFIGS (N=2048, D=16, K_hh=8, K_iter=8, AH=1.0, 100% data, 150ep — Tier-2)
=============================================================================
  B   : warm+W_proj full data 150ep ← phase-exit candidate

To reproduce:
    python -u scripts/train_step174_n2048_d16_fulldata.py --device cpu
    python -u scripts/train_step174_n2048_d16_fulldata.py --device mps
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
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=150)
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 8; K_IN = 25; K_ITER = 8
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

TEACHER_PATH = ROOT / "results" / "train_step170_teacher_n2048.pt"
OUT_PATH     = ROOT / "results" / "train_step174_n2048_d16_fulldata.json"


class SGNNET_WarmProj(nn.Module):
    def __init__(self, resonant, alpha_ahebb: float = 1.0):
        super().__init__()
        self.m = resonant; self.alpha_ahebb = alpha_ahebb
        self.proj = nn.Linear(resonant.base.D, resonant.base.D, bias=False)
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
        base = self.m.base; Z = self.proj(base._seed(x))
        theta = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn = base.conn_hh; N_h = base.N_hidden
        W_n  = F.normalize(self.m.W_pos[:N_h], dim=-1)
        supp = (1.0 - self.alpha_ahebb * (W_n.unsqueeze(1)*W_n[conn]).sum(-1).clamp(min=0)
               ).unsqueeze(0).unsqueeze(-1)
        Zr = torch.zeros_like(Z)
        for _ in range(base.K_iter):
            Zf = F.relu(Z - theta); Zs = (Zf[:, conn, :] * supp).sum(2)
            Zr = self.m.alpha_reflect * Zr + (Zf - Z)
            Z  = F.normalize((Zs + Zr).clamp(-10, 10), dim=-1)
        return base._readout(Z)


def main():
    print(f"\n{'='*70}")
    print(f"Step 174 — warm+W_proj N=2048 D=16 Tier-2 (full data 150ep)")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  Data=100%  Epochs={EPOCHS}")
    print(f"step170-B reference (50%/75ep): 93.43%")
    print(f"Phase-exit target: ≥95% @ FLOPs ~6.2M\n{'='*70}")

    if not TEACHER_PATH.exists():
        raise FileNotFoundError(f"Teacher not found: {TEACHER_PATH}")
    teacher_state = torch.load(TEACHER_PATH, map_location=DEVICE, weights_only=True)
    print(f"  Teacher loaded from {TEACHER_PATH}")

    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    from src.sgnnet.model_resonant import SGNNET_Resonant
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    model = SGNNET_WarmProj(resonant, ALPHA_AHEBB).to(DEVICE)
    model.warm_load(teacher_state)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_p:,}")

    tr, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    t0 = time.time()
    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

    def _log(m):
        if str(DEVICE) == "mps": torch.mps.empty_cache()
        ep = m["epoch"] + 1
        if ep % 10 == 0:
            flag = " *** PHASE EXIT! ***" if m["val_top1"] >= 0.95 else ""
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}{flag}", flush=True)

    history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
    elapsed = time.time() - t0
    top1h = [round(h.get("val_top1", 0.), 4) for h in history]
    best = max(top1h); bep = int(np.argmax(top1h)) + 1
    result = {"B": {"N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
                    "warm_proj": True, "data_frac": 1.0,
                    "top1_best": best, "top1_last": top1h[-1],
                    "best_epoch": bep, "epochs_run": len(history),
                    "top1_history": top1h, "elapsed_s": round(elapsed, 1),
                    "n_params": n_p, "label": "B warm+W_proj N=2048 D=16 full data 150ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    pe = "✓ PHASE EXIT ACHIEVED!" if best >= 0.95 else f"({0.95-best:.3f}pp short)"
    print(f"\n{'='*70}\nSTEP 174: {best:.4f}  vs_step170B={best-0.9343:+.4f}  {pe}")
    print(f"best_ep={bep}  params={n_p:,}  Results → {OUT_PATH}\n{'='*70}")


if __name__ == "__main__":
    main()
