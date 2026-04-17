"""Step 854: ΔW-proj seed variance — T2 promotion of step760 step706 finding.

MOTIVATION
==========
step760 T1 (75ep, 50% data, 5 seeds) showed:
  step199 baseline: mean=93.88%  std=0.43pp
  step706 ΔW proj: mean=95.37%  std=0.24pp  — +1.49pp accuracy AND halves variance

Paper claim needs T2 numbers (150ep, 100% data) for credibility. Same 5 seeds,
same ΔW-proj mechanism, but at paper-quality tier to confirm the finding.

Expected: mean ~95.4-95.5% (matches step199 T2=95.52%), std <0.30pp.
If std narrows further at T2, the noise-reduction claim strengthens.

CONFIG: N=2048, D=16, K_hh=2, K_iter=5, ΔW mode="proj", α_AH=1.0, α_reflect=0.5
SEEDS: 42,43,44,45,46 (matches step760)
TIER: T2 (150ep, 100% data, FIXED data seed = 42)
"""
from __future__ import annotations
import argparse, json, sys, time, statistics
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.sgnnet.model_smallworld  import SGNNET_SmallWorld
from src.sgnnet.model_resonant    import SGNNET_Resonant
from src.training.trainer         import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset         import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--seeds",  default="42,43,44,45,46")
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--device", default="auto")
parser.add_argument("--data",   default="data/store.h5")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

N = 2048; D = 16; K_HH = 2; K_ITER = 5; K_IN = 25
N_IN = 25088; N_OUT = 10; BATCH = 128
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0
DATA_SEED = 42
SEEDS = [int(s) for s in args.seeds.split(",")]

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step854_dwproj_t2__{SLOT}.json"


class SGNNET_DeltaAH(nn.Module):
    """ΔW-proj: neighbor messages scaled by projection onto W_pos difference vector."""
    def __init__(self, base, mode="proj"):
        super().__init__()
        self.m = base; self.mode = mode
        self.rotation_temp = nn.Parameter(torch.tensor(0.5))
    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase
    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()
    def forward(self, x):
        Z = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh = self.m.base.conn_hh
        W_h = self.m.W_pos[:self.m.base.N_hidden]
        dw = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_nb = Z_nb * proj_coeff.abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def _build(seed):
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    sw = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                           K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                           n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(sw, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
                          mode="dynamic_z_geo")
    return SGNNET_DeltaAH(res, mode="proj")


def main():
    print(f"\n{'='*70}")
    print(f"Step 854 — ΔW-proj T2 seed variance ({args.epochs}ep, 100% data)")
    print(f"  Seeds: {SEEDS}  Device: {DEVICE}  Data seed: {DATA_SEED} (fixed)")
    print(f"  Ref: step706 T1={0.9537:.4f} std=0.24pp. Target: T2 narrows std further.")
    print(f"{'='*70}")
    tr, va = make_loaders(ROOT / args.data, batch_size=BATCH, seed=DATA_SEED)

    per_seed = []
    for seed in SEEDS:
        print(f"\n{'─'*50}\nSEED {seed}\n{'─'*50}")
        model = _build(seed).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=args.epochs)
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=args.epochs,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}",
                                   flush=True) if (m['epoch']+1) % 15 == 0 else None)
        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1
        per_seed.append({"seed": seed, "top1_best": best, "best_epoch": bep,
                         "elapsed_s": round(time.time()-t0, 1), "n_params": n_p})
        print(f"  → seed={seed}: best={best:.4f} @ep{bep}")

        # incremental save (crash-safe)
        tops = [r["top1_best"] for r in per_seed]
        m_acc = statistics.mean(tops)
        s_acc = statistics.stdev(tops) if len(tops) > 1 else 0.0
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps({
            "config": "step706_proj_t2", "epochs": args.epochs, "data_frac": 1.0,
            "per_seed": per_seed, "mean": m_acc, "std": s_acc,
            "min": min(tops), "max": max(tops), "range_pp": (max(tops)-min(tops))*100,
            "n_seeds": len(per_seed),
        }, indent=2))

    tops = [r["top1_best"] for r in per_seed]
    mean = statistics.mean(tops); std = statistics.stdev(tops) if len(tops) > 1 else 0.
    mn = min(tops); mx = max(tops)
    print(f"\n{'='*70}")
    print(f"SUMMARY — step854 ΔW-proj T2 ({len(SEEDS)} seeds, {args.epochs}ep, 100% data):")
    print(f"  mean={mean*100:.3f}%  std={std*100:.3f}pp  range={(mx-mn)*100:.3f}pp")
    print(f"  min={mn*100:.2f}%  max={mx*100:.2f}%")
    print(f"  95% CI ≈ mean ± {2*std*100:.3f}pp")
    print(f"  T1 ref: mean=95.37% std=0.24pp — T2 should match or tighten.")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
