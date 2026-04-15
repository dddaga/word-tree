"""Step 524-S1: Edge-β scalar on frozen topology (learned edge attention).

MOTIVATION (LEARNINGS_design_2026_04_15 S1, V3 Gap 2.5)
=======================================================
step511-525 all KILLED discrete/stochastic topology perturbation:
  destructive rewire (step511-513), additive expansion (step514),
  alternating W_pos/edge with 1% cap (step523), teleportation (step525).

Root cause (H5 from step523 post-mortem): discrete topology edits break
credit assignment, because W_pos is co-adapted to the specific topology.

S1 is the ONE remaining hypothesis in the "dynamic element" direction:
topology is 100% FROZEN. Per-edge LEARNABLE scalar weights modulate the
structural contribution. No topology change. Gradient flows cleanly because
the parameterization is continuous and differentiable.

This is effectively "learned edge attention" — a standard GNN operation
that has NEVER been tested in SGNNET. If it wins:
  (a) first successful dynamic mechanism; opens a new research axis.
  (b) paper claim: "SGNNET with learned edge attention matches/beats ΔW proj".
If it fails: dynamic connectivity direction is definitively CLOSED — static
topology with learned W_pos is the ENTIRE story.

MECHANISM
  base:    ΔW projection (current winner at efficiency config)
  added:   β[N, K_hh] learnable scalar per edge, init=0
  gate:    Z_nb = Z_nb * (1 + tanh(β))  — multiplicative gate ∈ [0, 2]
           (β=0 → gate=1 → no-op; gradient flows)

PARAM COST: +N·K_hh = 4,096 params (2048*2). Negligible.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, ΔW proj, 20ep T0 scout + 75ep T1 winner)
  Ref     : ΔW proj (baseline)
  S1_init0 : + β init 0   (β must be learned; starts as no-op)
  S1_init1 : + β init 1   (gate starts at tanh(1)≈0.76 → scale ≈ 1.76; tests non-neutral init)

If S1 shows any positive delta at T0, advance to T1 (75ep).
If T1 confirms ≥+0.3pp, advance to T2 (150ep + full data).

To run:
    python -u scripts/train_step524_s1_edge_beta.py
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
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--data",   default="data/store.h5")
parser.add_argument("--configs", default="")
parser.add_argument("--full_data", action="store_true")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step524_s1_edge_beta_seed{SEED}__{SLOT}.json"

CONFIGS = [
    dict(key="Ref",      use_beta=False, beta_init=0.0, label="ΔW proj baseline (no β)"),
    dict(key="S1_init0", use_beta=True,  beta_init=0.0, label="ΔW proj + β init=0 (neutral)"),
    dict(key="S1_init1", use_beta=True,  beta_init=1.0, label="ΔW proj + β init=1 (non-neutral)"),
]


class SGNNET_DeltaProjBeta(nn.Module):
    """ΔW proj + optional per-edge learnable β scalar gate."""
    def __init__(self, N_hidden, N_out, D_, N_in, K_in, K_iter, K_local, K_random,
                 n_groups, alpha_reflect, use_beta, beta_init, seed):
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

        self.use_beta = use_beta
        if use_beta:
            # Per-edge learnable scalar [N_hidden, K_hh]
            self.beta = nn.Parameter(torch.full(
                (N_hidden, K_HH), float(beta_init)))
        else:
            self.beta = None

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
        delta_w = W_h.unsqueeze(1) - W_h[conn_hh]
        dw_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)

        # β gate: [N, K_hh] → [1, N, K_hh, 1], always ≥ 0 via (1 + tanh)
        if self.beta is not None:
            beta_gate = (1.0 + torch.tanh(self.beta)).unsqueeze(0).unsqueeze(-1)
        else:
            beta_gate = None

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            proj = (Z_nb * dw_norm).sum(-1, keepdim=True)
            Z_nb = Z_nb * proj.abs()
            if beta_gate is not None:
                Z_nb = Z_nb * beta_gate
            Z_struct = Z_nb.sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.alpha_reflect * Z_reflected + Z_remainder
            Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)
        return self.base._readout(Z)


def build_model(cfg):
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    return SGNNET_DeltaProjBeta(
        N_hidden=N, N_out=N_OUT, D_=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, alpha_reflect=ALPHA_REFLECT,
        use_beta=cfg["use_beta"], beta_init=cfg["beta_init"], seed=SEED)


def main():
    keys = [c["key"] for c in CONFIGS]
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else keys
    active = [c for c in CONFIGS if c["key"] in run_keys]

    tr, va = make_loaders(ROOT / args.data, batch_size=BATCH, seed=SEED)
    if not args.full_data:
        n = len(tr.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    print(f"Step 524-S1 — Edge-β scalar on frozen topology (learned edge attention)")
    print(f"  data={args.data}  seed={SEED}  epochs={EPOCHS}  device={DEVICE}")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")

    results = {}
    for cfg in active:
        print(f"\n{'─'*60}\nConfig {cfg['key']}: {cfg['label']}\n{'─'*60}")
        t0 = time.time()
        model = build_model(cfg).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_params:,}  use_beta={cfg['use_beta']}  init={cfg['beta_init']}")
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 5 == 0 else None
        ))
        top1h = [h.get("val_top1", 0.0) for h in history]
        top1_best = max(top1h) if top1h else 0.0
        best_ep = int(np.argmax(top1h)) + 1 if top1h else 0
        elapsed = time.time() - t0
        print(f"  → best={top1_best:.4f} @ep{best_ep}  elapsed={elapsed:.0f}s")

        # Extract learned β statistics (diagnostic)
        beta_stats = None
        if model.beta is not None:
            b = model.beta.detach().cpu().numpy()
            beta_stats = {
                "mean": float(b.mean()), "std": float(b.std()),
                "min":  float(b.min()),  "max": float(b.max()),
                "gate_mean": float((1 + np.tanh(b)).mean()),
            }
            print(f"  β stats: mean={beta_stats['mean']:+.3f} std={beta_stats['std']:.3f} "
                  f"gate≈{beta_stats['gate_mean']:.3f}")

        results[cfg["key"]] = {
            "label": cfg["label"], "use_beta": cfg["use_beta"],
            "beta_init": cfg["beta_init"], "n_params": n_params,
            "top1_best": top1_best, "best_epoch": best_ep, "elapsed_s": elapsed,
            "beta_stats": beta_stats,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUT_PATH, "w") as f: json.dump(results, f, indent=2)

    print(f"\n========== STEP 524-S1 SUMMARY (N={N} seed={SEED}) ==========")
    ref = results.get("Ref", {}).get("top1_best", 0.0)
    for cfg in active:
        r = results.get(cfg["key"], {})
        if not r: continue
        d = (r["top1_best"] - ref) * 100 if ref else 0
        print(f"  {cfg['key']:<10}  best={r['top1_best']:.4f}  Δ_vs_Ref={d:+.2f}pp")


if __name__ == "__main__":
    main()
