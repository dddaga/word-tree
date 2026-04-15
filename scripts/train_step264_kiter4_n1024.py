"""Step 264: K_iter=4 ΔW proj at N=1024 — lower-end N-scaling check.

MOTIVATION
==========
Establishes K_iter=4 behavior across N={1024, 2048, 4096}:
  N=2048 step262 T2 3-seed: Δ=+0.02pp (equivalent)
  N=4096 step263 T1 launched (running)
  N=1024 this step (running on CPU — small enough)

If all three N scales show K=4 within noise of K=5, paper's efficiency-regime story extends.

CONFIGS (N=1024 D=16 K_hh=2 ΔW proj, seed=42, 75ep 50% T1)
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
parser.add_argument("--epochs", type=int, default=75)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--configs", default="")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

OUT_PATH = ROOT / "results" / f"train_step264_kiter4_n1024_seed{SEED}.json"

CONFIGS = [
    dict(key="Ref", K_iter=5, label="ΔW proj K_iter=5 at N=1024"),
    dict(key="A",   K_iter=4, label="ΔW proj K_iter=4 at N=1024"),
]


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
        delta_w      = W_h.unsqueeze(1) - W_h[conn_hh]
        delta_w_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)
        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]
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
    return SGNNET_DeltaAH(res, alpha_ahebb=0.0)


def main():
    keys = [c["key"] for c in CONFIGS]
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else keys
    active = [c for c in CONFIGS if c["key"] in run_keys]

    print(f"Step 264 — K_iter=4 ΔW proj at N={N} Tier-1")
    print(f"  seed={SEED}  EPOCHS={EPOCHS}  device={DEVICE}")

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
        print(f"  params={n_params:,}  K_iter={cfg['K_iter']}")
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch'] + 1) % 10 == 0 else None
        ))
        top1h = [h.get("val_top1", 0.0) for h in history]
        top1_best = max(top1h) if top1h else 0.0
        best_ep = int(np.argmax(top1h)) + 1 if top1h else 0
        elapsed = time.time() - t0
        print(f"  → best={top1_best:.4f} @ep{best_ep}  elapsed={elapsed:.0f}s")
        results[cfg["key"]] = {
            "label": cfg["label"], "K_iter": cfg["K_iter"],
            "n_params": n_params, "top1_best": top1_best, "best_epoch": best_ep,
            "elapsed_s": elapsed,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n\n========== STEP 264 SUMMARY (N={N} seed={SEED}) ==========")
    for cfg in active:
        r = results.get(cfg["key"], {})
        if not r: continue
        print(f"  {cfg['key']:<4}  K_iter={r['K_iter']}  best={r['top1_best']:.4f} @ep{r['best_epoch']}")
    if "Ref" in results and "A" in results:
        d = (results["A"]["top1_best"] - results["Ref"]["top1_best"]) * 100
        print(f"\nΔ(K=4 − K=5): {d:+.2f}pp at N={N}")


if __name__ == "__main__":
    main()
