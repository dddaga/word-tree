"""Step 260: K_iter reduction × ΔW projection at efficiency config.

MOTIVATION
==========
Wall-clock optimization (user directive 2026-04-14): reduce K_iter if ΔW proj can compensate.

Prior (AH-only only, step196/202): K_iter=4 KILLED −2.14pp, K_iter=3 KILLED −6.3pp.
Untested: does ΔW proj (which adds +1.5pp at K_iter=5) compensate for fewer iterations?

If K_iter=4 ΔW proj ≥ K_iter=5 AH baseline → 20% wall-clock reduction at no accuracy cost.

CONFIGS (N=2048 D=16 K_hh=2, 50% data, 20ep Tier-0 scout)
  Ref : AH-only,  K_iter=5 (step199 baseline)
  A   : ΔW proj,  K_iter=3
  B   : ΔW proj,  K_iter=4
  C   : ΔW proj,  K_iter=5 (step706 baseline)
  D   : ΔW proj,  K_iter=6
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

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--configs", default="", help="Comma-separated keys. Empty = all.")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

OUT_PATH = ROOT / "results" / "train_step260_kiter_sweep_dwproj.json"

CONFIGS = [
    dict(key="Ref", dw_mode=None,   K_iter=5, label="AH K_iter=5 (step199 baseline)"),
    dict(key="A",   dw_mode="proj", K_iter=3, label="ΔW proj K_iter=3"),
    dict(key="B",   dw_mode="proj", K_iter=4, label="ΔW proj K_iter=4"),
    dict(key="C",   dw_mode="proj", K_iter=5, label="ΔW proj K_iter=5"),
    dict(key="D",   dw_mode="proj", K_iter=6, label="ΔW proj K_iter=6"),
]


# ΔW projection model (replicated from step706)
class SGNNET_DeltaAH(nn.Module):
    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float = 0.0):
        super().__init__()
        self.m = resonant
        self.alpha_ahebb = alpha_ahebb

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
        W_h       = self.m.W_pos[:N_h]
        W_n       = F.normalize(W_h, dim=-1)

        supp_w = None
        if self.alpha_ahebb > 0:
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
            supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                       ).unsqueeze(0).unsqueeze(-1)

        delta_w      = W_h.unsqueeze(1) - W_h[conn_hh]
        delta_w_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]
            if supp_w is not None:
                Z_nb = Z_nb * supp_w
            proj_coeff = (Z_nb * delta_w_norm).sum(dim=-1, keepdim=True)
            Z_nb       = Z_nb * proj_coeff.abs()
            Z_struct   = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = ALPHA_REFLECT * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z           = F.normalize(Z_new.clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def build_model(cfg):
    torch.manual_seed(SEED)
    ng = max(8, N // 8)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=cfg["K_iter"], K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(base, alpha_reflect=ALPHA_REFLECT,
                          alpha_turing=ALPHA_TURING, mode="dynamic_z_geo")
    if cfg["dw_mode"] is None:
        return SGNNET_AntiHebbian(res, alpha_ahebb=1.0, variant="wpos")
    return SGNNET_DeltaAH(res, alpha_ahebb=0.0)


def main():
    keys = [c["key"] for c in CONFIGS]
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else keys
    active = [c for c in CONFIGS if c["key"] in run_keys]

    print(f"Step 260 — K_iter × ΔW proj at efficiency config")
    print(f"  N={N} D={D} K_hh={K_HH} {EPOCHS}ep 50% Tier-0")
    print(f"  Running: {[c['key'] for c in active]}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_half = len(tr_full.dataset) // 2
    subset = torch.utils.data.Subset(tr_full.dataset, list(range(n_half)))
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for cfg in active:
        print(f"\n{'─'*60}\nConfig {cfg['key']}: {cfg['label']}\n{'─'*60}")
        t0 = time.time()
        model = build_model(cfg).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_params:,}  K_iter={cfg['K_iter']}  dw_mode={cfg['dw_mode']}")
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch'] + 1) % 5 == 0 else None
        ))
        top1h = [h.get("val_top1", 0.0) for h in history]
        top1_best = max(top1h) if top1h else 0.0
        elapsed = time.time() - t0
        print(f"  → best={top1_best:.4f}  elapsed={elapsed:.0f}s")
        results[cfg["key"]] = {
            "label": cfg["label"], "K_iter": cfg["K_iter"], "dw_mode": cfg["dw_mode"],
            "n_params": n_params, "top1_best": top1_best, "elapsed_s": elapsed,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print("\n\n========== STEP 260 SUMMARY ==========")
    print(f"{'Config':<6} {'K_iter':>7} {'dw':>6} {'params':>8} {'top1':>8}")
    for cfg in active:
        r = results.get(cfg["key"], {})
        if not r: continue
        print(f"{cfg['key']:<6} {r['K_iter']:>7} {str(r['dw_mode']):>6} "
              f"{r['n_params']:>8,} {r['top1_best']:>8.4f}")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
