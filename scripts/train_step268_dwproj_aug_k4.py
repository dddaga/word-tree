"""Step 268: ΔW projection + augmentation + K=4 @ N=2048 Tier-1.

MOTIVATION (V3 Gap 2.2)
=======================
Stack two wins that have NOT been tested together at the efficiency config:
  - step262 confirmed K=4 ≈ K=5 at N=2048 (3-seed tie, Δ = +0.02pp)
  - step235 confirmed aug gives +0.33pp for ΔW rotation at N=2048 K=5

Question: does aug also give +0.33pp for ΔW **proj** at K=4?

If yes, this becomes a new efficiency Pareto point:
  N=2048 K=4 ΔW proj + aug  →  ~97.2-97.3% at 20% fewer FLOPs than step235's K=5.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=4, 75ep Tier-1, 50% data)
  Ref       : K=5 ΔW proj (no aug)           [step262 mean Tier-2 ≈ 94.6%]
  A_proj_k4 : K=4 ΔW proj (no aug, store.h5) [step262 K=4 reference]
  B_proj_aug_k4 : K=4 ΔW proj + aug (store_aug.h5)  [THE combo]
  C_proj_aug_k5 : K=5 ΔW proj + aug (comparison point for aug baseline)

Winner criterion: B (the combo) beats both A (K=4 alone) and C (aug alone)
by at least the union of their individual gains minus noise.

To run:
    python -u scripts/train_step268_dwproj_aug_k4.py
"""
from __future__ import annotations
import argparse, json, os, sys, time
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
parser.add_argument("--epochs", type=int, default=75)
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--configs", default="")
parser.add_argument("--full_data", action="store_true")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step268_dwproj_aug_k4_seed{SEED}__{SLOT}.json"

CONFIGS = [
    dict(key="Ref",            K_iter=5, data="data/store.h5",      label="K=5 ΔW proj no-aug (ref)"),
    dict(key="A_proj_k4",      K_iter=4, data="data/store.h5",      label="K=4 ΔW proj no-aug"),
    dict(key="B_proj_aug_k4",  K_iter=4, data="data/store_aug.h5",  label="K=4 ΔW proj + aug (COMBO)"),
    dict(key="C_proj_aug_k5",  K_iter=5, data="data/store_aug.h5",  label="K=5 ΔW proj + aug (baseline aug)"),
]


class SGNNET_DeltaProj(nn.Module):
    """ΔW projection routing (sign-invariant gate). No AH, no rotation."""
    def __init__(self, N_hidden, N_out, D_, N_in, K_in, K_iter, K_local, K_random,
                 n_groups, alpha_reflect, seed):
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

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase

    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh
        N_h = self.base.N_hidden
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h = self.base.W_pos[:N_h]
        delta_w = W_h.unsqueeze(1) - W_h[conn_hh]              # [N, K_hh, D]
        dw_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)    # [1, N, K_hh, D]

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]                         # [B, N, K_hh, D]
            proj = (Z_nb * dw_norm).sum(-1, keepdim=True)       # [B, N, K_hh, 1]
            Z_nb = Z_nb * proj.abs()                            # sign-invariant gate
            Z_struct = Z_nb.sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.alpha_reflect * Z_reflected + Z_remainder
            Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)
        return self.base._readout(Z)


def build_model(cfg):
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    return SGNNET_DeltaProj(
        N_hidden=N, N_out=N_OUT, D_=D, N_in=N_IN,
        K_in=K_IN, K_iter=cfg["K_iter"], K_local=K_l, K_random=K_r,
        n_groups=ng, alpha_reflect=ALPHA_REFLECT, seed=SEED)


def main():
    keys = [c["key"] for c in CONFIGS]
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else keys
    active = [c for c in CONFIGS if c["key"] in run_keys]

    print(f"Step 268 — ΔW proj + aug + K=4 compound @ N={N} Tier-1")
    print(f"  seed={SEED}  epochs={EPOCHS}  device={DEVICE}  full_data={args.full_data}")

    results = {}
    for cfg in active:
        data_path = ROOT / cfg["data"]
        tr, va = make_loaders(data_path, batch_size=BATCH, seed=SEED)
        if not args.full_data:
            n = len(tr.dataset)
            idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
            subset = torch.utils.data.Subset(tr.dataset, idx.tolist())
            tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

        print(f"\n{'─'*60}\nConfig {cfg['key']}: {cfg['label']}\n{'─'*60}")
        print(f"  data={cfg['data']}  Train={len(tr.dataset)}  Val={len(va.dataset)}")

        t0 = time.time()
        model = build_model(cfg).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_params:,}  K_iter={cfg['K_iter']}")
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 10 == 0 else None
        ))
        top1h = [h.get("val_top1", 0.0) for h in history]
        top1_best = max(top1h) if top1h else 0.0
        best_ep = int(np.argmax(top1h)) + 1 if top1h else 0
        elapsed = time.time() - t0
        print(f"  → best={top1_best:.4f} @ep{best_ep}  elapsed={elapsed:.0f}s")

        results[cfg["key"]] = {
            "label": cfg["label"], "K_iter": cfg["K_iter"], "data": cfg["data"],
            "n_params": n_params, "top1_best": top1_best, "best_epoch": best_ep,
            "elapsed_s": elapsed,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUT_PATH, "w") as f: json.dump(results, f, indent=2)

    print(f"\n========== STEP 268 SUMMARY (N={N} T1 seed={SEED}) ==========")
    ref = results.get("Ref", {}).get("top1_best", 0.0)
    for cfg in active:
        r = results.get(cfg["key"], {})
        if not r: continue
        d = (r["top1_best"] - ref) * 100 if ref else 0
        print(f"  {cfg['key']:<16}  best={r['top1_best']:.4f}  Δ_vs_Ref={d:+.2f}pp  {cfg['label']}")


if __name__ == "__main__":
    main()
