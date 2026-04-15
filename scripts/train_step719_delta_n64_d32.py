"""Step 719: ΔW projection at N=64 D=32 — testing W_pos diversity hypothesis.

QUESTION: Why does gain PEAK at N=128 (+24pp) rather than N=64 (+16pp)?
Hypothesis: at N=64 D=16, W_pos vectors on S^{D-1} are too sparse/similar → ΔW
directions are less discriminative. More dimensions (D=32) → richer W_pos diversity
→ larger ΔW gains at small N.

Prediction:
  - If W_pos diversity is the bottleneck: N=64 D=32 gain >> N=64 D=16 (+16pp)
  - If graph topology is the bottleneck: N=64 D=32 gain ≈ N=64 D=16 (+16pp)

Tier-1 only (75ep, 50% data) — diagnostic, not paper-ready.
Comparison: step717 N=64 D=16 gave Δ≈+16pp (Ref≈37%, A_proj≈53%).
"""
from __future__ import annotations
import argparse, json, sys, time
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

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75)
parser.add_argument("--configs", default="")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
EPOCHS = args.epochs
BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 64; N_IN = 25088; N_OUT = 10
D = 32; K_HH = 2; K_IN = 25; K_ITER = 5   # D=32 vs D=16 in step717
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0
FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step719_delta_n64_d32.json"


class SGNNET_DeltaAH(nn.Module):
    def __init__(self, base, alpha_ahebb=0.0):
        super().__init__()
        self.m = base; self.alpha_ahebb = alpha_ahebb
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
            Z_nb = Z_nb * (Z_nb * dw).sum(-1, keepdim=True).abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def _build(seed, delta=False):
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    sw = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                           K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                           n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(sw, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING, mode="dynamic_z_geo")
    return SGNNET_DeltaAH(res) if delta else SGNNET_AntiHebbian(res, alpha_ahebb=1.0, variant="wpos")


def main():
    import math
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else ["Ref", "A_proj"]
    print(f"\n{'='*70}")
    print(f"Step 719 — ΔW proj N=64 D=32 Tier-1 (W_pos diversity hypothesis)")
    print(f"N={N} D={D} FLOPs={FLOPS/1e6:.3f}M  Device={DEVICE}")
    print(f"Baseline: N=64 D=16 gave Ref≈37% A_proj≈53% Δ≈+16pp (step717)")
    print(f"Hypothesis: D=32 → richer W_pos → larger ΔW gain at small N")
    print(f"{'='*70}")
    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))[:n_total // 2]
    tr = torch.utils.data.DataLoader(torch.utils.data.Subset(tr_full.dataset, idx.tolist()),
                                     batch_size=BATCH, shuffle=True, num_workers=0,
                                     generator=torch.Generator().manual_seed(SEED))
    results = {}
    for key in run_keys:
        print(f"\n{'─'*50}\n{key}\n{'─'*50}")
        model = _build(SEED, delta=(key != "Ref")).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **trainer_kwargs(N, n_epochs=EPOCHS)).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
                             if (m['epoch'] + 1) % 10 == 0 else None)
        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1
        results[key] = {"top1_best": best, "best_epoch": bep, "top1_history": top1h,
                        "elapsed_s": round(time.time()-t0, 1), "n_params": n_p,
                        "N": N, "D": D, "K_hh": K_HH, "flops": FLOPS}
        print(f"  → best={best:.4f} @ ep{bep}")
    ref = results.get("Ref", {}).get("top1_best", 0.)
    proj = results.get("A_proj", {}).get("top1_best", 0.)
    delta = proj - ref if all(k in results for k in ["Ref", "A_proj"]) else float("nan")
    print(f"\n{'='*70}")
    if not math.isnan(delta):
        print(f"D=32: Ref={ref:.4f}  A_proj={proj:.4f}  Δ={delta:+.4f} ({delta*100:+.2f}pp)")
        print(f"D=16: Ref=37.0%   A_proj=53.0%   Δ=+16pp  (step717)")
        if delta * 100 > 20.0:
            print(f"  → HYPOTHESIS CONFIRMED: D=32 larger gain — W_pos diversity IS the bottleneck at N=64")
        elif delta * 100 > 16.0:
            print(f"  → PARTIAL: D=32 moderately larger — W_pos diversity is a contributing factor")
        else:
            print(f"  → HYPOTHESIS REJECTED: gain unchanged — topology/capacity is the bottleneck, not W_pos diversity")
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_PATH}")

if __name__ == "__main__":
    main()
