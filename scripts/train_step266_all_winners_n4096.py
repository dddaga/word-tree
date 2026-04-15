"""Step 266: All-winners combo at N=4096 Tier-2 — ΔW rotation + aug + K_iter=4.

MOTIVATION
==========
Stacking the three best mechanisms discovered so far, at the D=16 ceiling scale:
  1. ΔW rotation (step235 A_aug, no AH): +2.14pp vs AH-only at N=4096
  2. Augmentation (store_aug.h5, hflip): +0.33pp at N=2048 (97.30% vs 96.97%)
  3. K_iter=4 (step265): +0.70pp mean at N=4096 T2 vs K_iter=5

If the mechanisms compose additively (they don't necessarily — but shared-signal-path
check: rotation ≠ proj and doesn't use W_pos similarity, so orthogonal to AH), expect
~97.8-98.2% at the D=16 ceiling at K_iter=4.

Current records:
  step205 K=5 AH-only T2: 97.17%
  step265 K=4 ΔW proj T2 seed=43: 97.35% ← current best
  step235 K=5 ΔW rot+aug seed=42: 97.30% (at N=2048)
  Target this step: ≥97.4% at N=4096 K=4 with rotation + aug

CONFIGS (N=4096 D=16 K_hh=2 K_iter=4, seed=42, 150ep, aug data)
  Ref    : K=5 ΔW rotation + aug (matches step235 A_aug but at N=4096)
  A_k4   : K=4 ΔW rotation + aug (THE combo — all three wins)
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
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--data", default="data/store_aug.h5")
parser.add_argument("--configs", default="")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 4096; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

OUT_PATH = ROOT / "results" / f"train_step266_all_winners_n4096_seed{SEED}.json"

CONFIGS = [
    dict(key="Ref",  K_iter=5, label="K=5 ΔW rot + aug at N=4096 (baseline)"),
    dict(key="A_k4", K_iter=4, label="K=4 ΔW rot + aug at N=4096 (ALL WINNERS combo)"),
]


class SGNNET_DeltaRotation(nn.Module):
    """ΔW rotation mechanism (from step235) with no AH."""
    def __init__(self, N_hidden, N_out, D_, N_in, K_in, K_iter, K_local, K_random,
                 n_groups, alpha_reflect, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.base = SGNNET_SmallWorld(
            N_hidden=N_hidden, N_out=N_out, D=D_, N_in=N_in,
            K_in=K_in, K_iter=K_iter, K_local=K_local, K_random=K_random,
            n_groups=n_groups, norm_mode="l2", encoding_mode="fourier")
        self.resonant = SGNNET_Resonant(
            self.base, K_phase=8, alpha_reflect=alpha_reflect,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0)
        self.alpha_reflect = alpha_reflect
        self.rotation_temp = nn.Parameter(torch.tensor(0.5))

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase
    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"): self.resonant.tick_epoch()

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh
        N_h = self.base.N_hidden
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h = self.base.W_pos[:N_h]
        W_n = F.normalize(W_h, dim=-1)
        delta_w = W_h.unsqueeze(1) - W_h[conn_hh]
        delta_w_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)
        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            # Rotation in the (Z_nb, ΔW) plane
            proj_coeff = (Z_nb * delta_w_norm).sum(dim=-1, keepdim=True)
            z_parallel = proj_coeff * delta_w_norm
            z_perp = Z_nb - z_parallel
            z_perp_norm = F.normalize(z_perp, dim=-1)
            theta_rot = self.rotation_temp * proj_coeff.squeeze(-1)
            z_mag = Z_nb.norm(dim=-1, keepdim=True)
            cos_t = torch.cos(theta_rot).unsqueeze(-1)
            sin_t = torch.sin(theta_rot).unsqueeze(-1)
            Z_nb = cos_t * Z_nb + sin_t * z_perp_norm * z_mag
            Z_struct = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.alpha_reflect * Z_reflected + Z_remainder
            Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)
        return self.base._readout(Z)


def build_model(cfg):
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng = max(8, N // 8)
    return SGNNET_DeltaRotation(
        N_hidden=N, N_out=N_OUT, D_=D, N_in=N_IN,
        K_in=K_IN, K_iter=cfg["K_iter"], K_local=K_l, K_random=K_r,
        n_groups=ng, alpha_reflect=ALPHA_REFLECT, seed=SEED)


def main():
    keys = [c["key"] for c in CONFIGS]
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else keys
    active = [c for c in CONFIGS if c["key"] in run_keys]

    data_path = ROOT / args.data
    print(f"Step 266 — All-winners combo at N={N} T2 (ΔW rot + aug + K_iter sweep)")
    print(f"  data={args.data}  seed={SEED}  epochs={EPOCHS}  device={DEVICE}")
    print(f"  Target: ≥97.4% (beat step265 97.35%)")

    tr, va = make_loaders(data_path, batch_size=BATCH, seed=SEED)
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")

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

    print(f"\n========== STEP 266 SUMMARY (N={N} T2 seed={SEED}) ==========")
    for cfg in active:
        r = results.get(cfg["key"], {})
        if not r: continue
        print(f"  {cfg['key']:<6}  K_iter={r['K_iter']}  best={r['top1_best']:.4f}")
    if "Ref" in results and "A_k4" in results:
        d = (results["A_k4"]["top1_best"] - results["Ref"]["top1_best"]) * 100
        print(f"\nΔ(K=4 − K=5): {d:+.2f}pp at N={N} with rotation+aug")


if __name__ == "__main__":
    main()
