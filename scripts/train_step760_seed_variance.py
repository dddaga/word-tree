"""Step 760: Seed-variance replication on experiment winners.

PURPOSE: Measure seed-to-seed std for key configs. Grounds the "+0.5pp threshold"
with actual noise data. Every Δ claim in the paper needs this to survive review.

Seed variance source: `torch.manual_seed(seed)` controls {conn_hh topology draw,
conn_in random draw, W_pos init, θ init, C_ho init}. Data split + subset index
are FIXED across seeds (isolates network variance, not data variance).

Configs supported:
  step199: N=2048 D=16 K_hh=2 K_iter=5 (AH-only baseline; 95.52% T2)
  step706: N=2048 D=16 K_hh=2 K_iter=5 + ΔW proj (96.87% T2)
  step729: N=4096 D=16 K_hh=2 K_iter=5 + ΔW rot (96.74% T1)
  step750: N=4096 D=16 K_hh=4 K_iter=3 AH-only (95.06% T1)

Output: per-seed top1, mean, std, min, max. 5 seeds Tier-1 (75ep, 50% data).
"""
from __future__ import annotations
import argparse, json, sys, time, statistics
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset import make_loaders

CONFIGS = {
    "step199": dict(N=2048, D=16, K_hh=2, K_iter=5, dw_mode=None),
    "step706": dict(N=2048, D=16, K_hh=2, K_iter=5, dw_mode="proj"),
    "step729": dict(N=4096, D=16, K_hh=2, K_iter=5, dw_mode="rot"),
    "step750": dict(N=4096, D=16, K_hh=4, K_iter=3, dw_mode=None),
}

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
parser.add_argument("--seeds", default="42,43,44,45,46")
parser.add_argument("--epochs", type=int, default=75)
parser.add_argument("--device", default="auto")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
CFG = CONFIGS[args.config]
SEEDS = [int(s) for s in args.seeds.split(",")]
DATA_SEED = 42  # fixed across all seeds — isolates network variance
BATCH = 128; DATA = "data/store.h5"; N_IN = 25088; N_OUT = 10
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0
K_IN = 25
OUT_PATH = ROOT / "results" / f"train_step760_seedvar_{args.config}.json"


class SGNNET_DeltaAH(nn.Module):
    def __init__(self, base, mode="proj"):
        super().__init__()
        self.m = base; self.mode = mode
        self.rotation_temp = nn.Parameter(torch.tensor(0.5))
    @property
    def W_pos(self): return self.m.W_pos
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
            if self.mode == "proj":
                Z_nb = Z_nb * proj_coeff.abs()
            elif self.mode == "rot":
                z_parallel = proj_coeff * dw
                z_perp_unit = F.normalize(Z_nb - z_parallel, dim=-1)
                theta_rot = self.rotation_temp * proj_coeff
                z_mag = Z_nb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                Z_nb = torch.cos(theta_rot) * Z_nb + torch.sin(theta_rot) * z_perp_unit * z_mag
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def _build(seed, cfg):
    torch.manual_seed(seed)
    N = cfg["N"]; D = cfg["D"]; K_HH = cfg["K_hh"]; K_ITER = cfg["K_iter"]
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    sw = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                           K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                           n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(sw, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING, mode="dynamic_z_geo")
    if cfg["dw_mode"] is None:
        return SGNNET_AntiHebbian(res, alpha_ahebb=1.0, variant="wpos")
    return SGNNET_DeltaAH(res, mode=cfg["dw_mode"])


def main():
    print(f"\n{'='*70}")
    print(f"Step 760 — seed-variance replication: {args.config}")
    print(f"Config: {CFG}  Seeds: {SEEDS}  Device: {DEVICE}")
    print(f"Data seed FIXED at {DATA_SEED} — varies only network init + topology")
    print(f"{'='*70}")
    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=DATA_SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(DATA_SEED))[:n_total // 2]
    tr = torch.utils.data.DataLoader(torch.utils.data.Subset(tr_full.dataset, idx.tolist()),
                                     batch_size=BATCH, shuffle=True, num_workers=0,
                                     generator=torch.Generator().manual_seed(DATA_SEED))

    per_seed = []
    for seed in SEEDS:
        print(f"\n{'─'*50}\nSEED {seed}\n{'─'*50}")
        model = _build(seed, CFG).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **trainer_kwargs(CFG["N"], n_epochs=args.epochs)).train(
            n_epochs=args.epochs,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
                             if (m['epoch'] + 1) % 10 == 0 else None)
        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1
        per_seed.append({"seed": seed, "top1_best": best, "best_epoch": bep,
                         "elapsed_s": round(time.time()-t0, 1), "n_params": n_p})
        print(f"  → seed={seed}: best={best:.4f} @ep{bep}")

    tops = [r["top1_best"] for r in per_seed]
    mean = statistics.mean(tops); std = statistics.stdev(tops) if len(tops) > 1 else 0.
    mn = min(tops); mx = max(tops)
    summary = {"config": args.config, "config_params": CFG, "per_seed": per_seed,
               "mean": mean, "std": std, "min": mn, "max": mx, "range_pp": (mx-mn)*100,
               "n_seeds": len(SEEDS)}
    print(f"\n{'='*70}")
    print(f"SUMMARY ({args.config}, {len(SEEDS)} seeds):")
    print(f"  mean={mean*100:.3f}%  std={std*100:.3f}pp  range={(mx-mn)*100:.3f}pp  min={mn*100:.2f}%  max={mx*100:.2f}%")
    print(f"  95% CI ≈ mean ± {2*std*100:.3f}pp  (assuming Gaussian)")
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n→ {OUT_PATH}")

if __name__ == "__main__":
    main()
