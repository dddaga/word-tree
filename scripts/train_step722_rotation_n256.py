"""Step 722: ΔW rotation vs projection at N=256 Tier-1.

step713 Tier-2 (projection): Ref=63.4%, A_proj=83.5%, Δ=+20.02pp @ 0.12M FLOPs.
Companion to step721 (N=512). Together they map whether rotation > projection
in the high-gain regime (small N, large headroom).

Configs:
  Ref    : AH α=1.0
  A_proj : ΔW projection (+20.02pp known Tier-2)
  A_rot  : ΔW rotation (learned-angle rotate in (Z_nb, ΔW) plane)

Tier-1 (75ep, 50% data).
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
N = 256; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0
FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step722_rotation_n256.json"


class SGNNET_DeltaAH(nn.Module):
    def __init__(self, base, mode="proj"):
        super().__init__()
        self.m = base
        self.mode = mode
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
            else:  # rot
                z_parallel = proj_coeff * dw
                z_perp = Z_nb - z_parallel
                z_perp_unit = F.normalize(z_perp, dim=-1)
                theta_rot = self.rotation_temp * proj_coeff
                z_mag = Z_nb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                Z_nb = (torch.cos(theta_rot) * Z_nb +
                        torch.sin(theta_rot) * z_perp_unit * z_mag)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def _build(seed, mode="ref"):
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    sw = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                           K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                           n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(sw, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING, mode="dynamic_z_geo")
    if mode == "ref":
        return SGNNET_AntiHebbian(res, alpha_ahebb=1.0, variant="wpos")
    return SGNNET_DeltaAH(res, mode=mode)


def main():
    import math
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else ["Ref", "A_proj", "A_rot"]
    print(f"\n{'='*70}")
    print(f"Step 722 — ΔW rotation vs projection at N=256 Tier-1")
    print(f"N={N} D={D} FLOPs={FLOPS/1e6:.3f}M  Device={DEVICE}")
    print(f"Prior: proj +19.80pp T1, +20.02pp T2 (step712/713). Companion: step721 N=512.")
    print(f"{'='*70}")
    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))[:n_total // 2]
    tr = torch.utils.data.DataLoader(torch.utils.data.Subset(tr_full.dataset, idx.tolist()),
                                     batch_size=BATCH, shuffle=True, num_workers=0,
                                     generator=torch.Generator().manual_seed(SEED))
    results = {}
    for key in run_keys:
        mode = "ref" if key == "Ref" else ("rot" if key == "A_rot" else "proj")
        print(f"\n{'─'*50}\n{key}\n{'─'*50}")
        model = _build(SEED, mode=mode).to(DEVICE)
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
    rot = results.get("A_rot", {}).get("top1_best", 0.)
    d_proj = proj - ref if "Ref" in results and "A_proj" in results else float("nan")
    d_rot = rot - ref if "Ref" in results and "A_rot" in results else float("nan")
    print(f"\n{'='*70}")
    print(f"Ref={ref:.4f}  A_proj={proj:.4f}(Δ={d_proj:+.4f})  A_rot={rot:.4f}(Δ={d_rot:+.4f})")
    if not math.isnan(d_rot) and not math.isnan(d_proj):
        winner = "A_rot" if d_rot > d_proj else "A_proj"
        margin = abs(d_rot - d_proj) * 100
        print(f"  Winner: {winner} by {margin:.2f}pp at N=256")
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_PATH}")

if __name__ == "__main__":
    main()
