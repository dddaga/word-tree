"""Step 712: ΔW projection at N=256 (headroom curve, Tier-1).

Headroom curve so far:
  N=4096: −0.74pp  N=2048: +1.56pp  N=1024: +4.87pp  N=512: +10.52pp@ep30
  N=256: ? (expected >+10pp given ~35pp gap to D=16 ceiling)

CONFIGS (N=256, D=16, K_hh=2, K_iter=5, 50% data, 75ep — Tier-1)
  Ref    : AH α=1.0
  A_proj : ΔW projection, no AH
N=256 Tier-0 Ref: ~59-62% (step402a: 59.39%)
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

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser(description="Step 712: ΔW proj at N=256")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--configs", default="")
args   = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128; SEED = 42; DATA = "data/store.h5"
N = 256; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

FLOPS    = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step712_delta_n256.json"


class SGNNET_DeltaAH(nn.Module):
    def __init__(self, base, alpha_ahebb=0.0):
        super().__init__()
        self.m = base; self.alpha_ahebb = alpha_ahebb
    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase
    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()
    def forward(self, x):
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden
        W_h = self.m.W_pos[:N_h]
        supp_w = None
        if self.alpha_ahebb > 0:
            W_n = F.normalize(W_h, dim=-1)
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
            supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)).unsqueeze(0).unsqueeze(-1)
        delta_w_norm = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]
            if supp_w is not None: Z_nb = Z_nb * supp_w
            Z_nb       = Z_nb * (Z_nb * delta_w_norm).sum(dim=-1, keepdim=True).abs()
            Z_reflected = ALPHA_REFLECT * Z_reflected + (Z_fwd - Z)
            Z           = F.normalize((Z_nb.sum(2) + Z_reflected).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def _build(seed, delta=False):
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    sw  = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                            K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                            n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(sw, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING, mode="dynamic_z_geo")
    return SGNNET_DeltaAH(res, alpha_ahebb=0.0) if delta else SGNNET_AntiHebbian(res, alpha_ahebb=1.0, variant="wpos")


def main():
    import math
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else ["Ref", "A_proj"]
    print(f"\n{'='*70}")
    print(f"Step 712 — ΔW proj at N=256 (headroom curve extension)")
    print(f"N={N} D={D} K_hh={K_HH} | {EPOCHS}ep 50% | FLOPs={FLOPS/1e6:.3f}M | {DEVICE}")
    print(f"Curve: N=2048:+1.56pp N=1024:+4.87pp N=512:+10.52pp@ep30 N=256:?")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))[:n_total // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(SEED))
    print(f"Train={len(idx)}  Val={len(va.dataset)}")

    results = {}
    for key in run_keys:
        print(f"\n{'─'*50}\nConfig {key}\n{'─'*50}")
        model = _build(SEED, delta=(key != "Ref")).to(DEVICE)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")
        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        history = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
                             if (m['epoch'] + 1) % 10 == 0 else None)
        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best  = max(top1h); bep = int(np.argmax(top1h)) + 1
        results[key] = {"top1_best": best, "best_epoch": bep, "top1_last": top1h[-1],
                        "top1_history": top1h, "elapsed_s": round(time.time()-t0,1),
                        "n_params": n_p, "N": N, "D": D, "K_hh": K_HH, "flops": FLOPS}
        print(f"  → best={best:.4f} @ ep{bep}")

    ref_best  = results.get("Ref", {}).get("top1_best", 0.)
    proj_best = results.get("A_proj", {}).get("top1_best", 0.)
    delta     = proj_best - ref_best if all(k in results for k in ["Ref", "A_proj"]) else float("nan")
    print(f"\n{'='*70}\nSTEP 712 SUMMARY\n{'='*70}")
    print(f"  Ref={ref_best:.4f}  A_proj={proj_best:.4f}  Δ={delta:+.4f}" if not math.isnan(delta) else "")
    print(f"  Headroom curve: N=4096(−0.74) N=2048(+1.56) N=1024(+4.87) N=512(+10.5) N=256({delta:+.2f})"
          if not math.isnan(delta) else "")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_PATH}")

if __name__ == "__main__":
    main()
