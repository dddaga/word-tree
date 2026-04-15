"""Step 717: ΔW projection at N=64 — extending headroom curve to minimum scale.

Tier-1 curve (confirmed): N=2048(+1.56) N=1024(+4.68) N=512(+10.96@T1) N=256(+19.80) N=128(+22.03)
N=64: does the monotone gain continue? FLOPs=30,720 (0.03M) — 3% of original 0.98M record.
K_hh=2, ng=max(8,8)=8, K_local=1, K_random=1 (minimum valid topology).
Running Tier-1 (75ep, 50% data). N=64 is near the minimum viable graph size.
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
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0
FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step717_delta_n64.json"


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
    print(f"Step 717 — ΔW proj N=64 Tier-1 (headroom curve minimum)")
    print(f"N={N} FLOPs={FLOPS/1e6:.3f}M (3% of old 0.98M record)  Device={DEVICE}")
    print(f"Curve T1: N=256(+19.80) N=512(+12.82) N=1024(+4.87) N=2048(+1.56) N=128(+22.03) N=64(?)")
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
        print(f"Ref={ref:.4f}  A_proj={proj:.4f}  Δ={delta:+.4f}")
        print(f"Curve: N=128(+22.03) N=256(+19.80) N=512(+12.82) N=1024(+4.87) N=2048(+1.56) → N=64({delta*100:+.2f}pp)")
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_PATH}")

if __name__ == "__main__":
    main()
