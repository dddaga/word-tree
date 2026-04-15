"""Step 261: K_iter=4 ΔW proj — Tier-1 confirmation (75ep 50% data).

MOTIVATION
==========
step260 T0 (20ep) result: K_iter=4 ΔW proj = 94.19% > K_iter=5 ΔW proj = 93.99% (+0.20pp).
Best-ever K_iter=4 result at efficiency config. All prior K_iter=4 AH-only was KILLED (step196: −2.14pp).

Implications if T1 confirms:
  - 20% wall-clock reduction (K_iter=4 vs K_iter=5) at BETTER accuracy
  - Inference latency: 0.280ms → ~0.224ms → 7.0× faster than VGG_FC (up from 5.6×)
  - NEW paper finding: ΔW proj compensates for fewer iterations — the mechanism captures
    structural information faster than AH alone.

Protocol: 75ep 50% data Tier-1 with 3 seeds for statistical confidence.

CONFIGS (N=2048, D=16, K_hh=2)
  Ref : ΔW proj K_iter=5 (step706-equivalent, seed=42/43/44)
  A   : ΔW proj K_iter=4 (step260-B winner, seed=42/43/44)
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
parser.add_argument("--seeds", default="42,43,44")
parser.add_argument("--configs", default="", help="Comma-separated. Empty = all.")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

OUT_PATH = ROOT / "results" / "train_step261_kiter4_dwproj_tier1.json"

SEEDS = [int(s) for s in args.seeds.split(",")]

CONFIGS = [
    dict(key="Ref", K_iter=5, label="ΔW proj K_iter=5 (baseline)"),
    dict(key="A",   K_iter=4, label="ΔW proj K_iter=4 (candidate winner)"),
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


def build_model(cfg, seed):
    torch.manual_seed(seed)
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

    print(f"Step 261 — K_iter=4 ΔW proj Tier-1 confirmation")
    print(f"  N={N} D={D} K_hh={K_HH} {EPOCHS}ep 50% data  seeds={SEEDS}")
    print(f"  Running: {[c['key'] for c in active]}  device={DEVICE}")

    # Load once, reuse across seeds
    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEEDS[0])
    n_half = len(tr_full.dataset) // 2
    subset = torch.utils.data.Subset(tr_full.dataset, list(range(n_half)))
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for cfg in active:
        results[cfg["key"]] = {"label": cfg["label"], "K_iter": cfg["K_iter"], "runs": []}
        for seed in SEEDS:
            print(f"\n{'─'*60}\nConfig {cfg['key']} (seed={seed}): {cfg['label']}\n{'─'*60}")
            t0 = time.time()
            model = build_model(cfg, seed).to(DEVICE)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  params={n_params:,}  K_iter={cfg['K_iter']}  seed={seed}")
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
            results[cfg["key"]]["runs"].append({
                "seed": seed, "top1_best": top1_best, "best_epoch": best_ep,
                "elapsed_s": elapsed, "n_params": n_params,
            })
            OUT_PATH.parent.mkdir(exist_ok=True)
            with open(OUT_PATH, "w") as f:
                json.dump(results, f, indent=2)

    # Summary
    print("\n\n========== STEP 261 SUMMARY ==========")
    print(f"{'Config':<6} {'K_iter':>7} {'seeds':>6} {'mean':>7} {'σ':>6} {'range':>6}")
    for cfg in active:
        runs = results[cfg["key"]]["runs"]
        if not runs: continue
        vals = [r["top1_best"] for r in runs]
        mean_v = float(np.mean(vals)); sd = float(np.std(vals))
        rng = max(vals) - min(vals)
        print(f"{cfg['key']:<6} {cfg['K_iter']:>7} {len(vals):>6} {mean_v:>7.4f} {sd:>6.4f} {rng:>6.4f}")
    # Δ
    if "Ref" in results and "A" in results and results["Ref"]["runs"] and results["A"]["runs"]:
        ref_mean = np.mean([r["top1_best"] for r in results["Ref"]["runs"]])
        a_mean   = np.mean([r["top1_best"] for r in results["A"]["runs"]])
        delta = (a_mean - ref_mean) * 100
        print(f"\nΔ(A − Ref): {delta:+.2f}pp  (K_iter=4 vs K_iter=5)")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
