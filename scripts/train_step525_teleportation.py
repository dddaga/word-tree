"""Step 525: Teleportation — fixed topology + per-sample dynamic hop.

MOTIVATION
==========
step511-523 (6 variants) all failed because they modified conn_hh during training.
Hypothesis H5 from step523 post-mortem: discreteness of topology edits is THE
failure mode, not volume (1% cap still failed −2.68pp).

Teleportation class avoids this entirely: conn_hh FROZEN at init. Each neuron
gets 1 EXTRA "teleport edge" (conn_tel[N]) that is regenerated stochastically.
No gradient ever flows through the teleport selection → no credit-assignment
discontinuity, no optimizer-state desync, no W_pos/topology co-adaptation lock.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=4, 20ep Tier-0 scout)
  Ref  : K_hh=2 ΔW proj baseline (no teleport edge)
  T2_5 : +1 teleport edge per neuron, redrawn from randperm(N) every 5 epochs
  T2_1 : +1 teleport edge per neuron, redrawn EVERY epoch
  T3   : +1 teleport edge per neuron, redrawn every BATCH at train time, fixed at eval

Expected: if any of T2/T3 ≥ Ref, topology benefits from stochastic exploration
without gradient-based discrete selection. This reopens the dyn-conn direction
via Monte-Carlo rather than learned rewiring.

To reproduce:
    python -u scripts/train_step525_teleportation.py --device cuda
    python -u scripts/train_step525_teleportation.py --device mps --configs Ref,T2_5
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

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--data", default="data/store.h5")
parser.add_argument("--configs", default="")
parser.add_argument("--full_data", action="store_true")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 4
ALPHA_REFLECT = 0.5

OUT_PATH = ROOT / "results" / f"train_step525_teleportation_seed{SEED}.json"

CONFIGS = [
    dict(key="Ref",  mode="none",  redraw_every=0, label="ΔW proj baseline (no teleport)"),
    dict(key="T2_5", mode="epoch", redraw_every=5, label="T2: redraw teleport every 5 ep"),
    dict(key="T2_1", mode="epoch", redraw_every=1, label="T2: redraw teleport every epoch"),
    dict(key="T3",   mode="batch", redraw_every=0, label="T3: redraw per batch (train) / fixed (eval)"),
]


class SGNNET_DeltaTeleport(nn.Module):
    """ΔW proj routing with one extra stochastic teleport edge per neuron.

    The teleport edge `conn_tel[N]` is drawn from randperm(N) and is concatenated
    to the K_hh structural neighbours during routing. It is re-drawn according to
    `mode`:
      "none"  : never drawn — ablation baseline
      "epoch" : re-drawn every `redraw_every` epochs (tick_epoch)
      "batch" : re-drawn every forward pass during training; frozen at eval
    """
    def __init__(self, N_hidden, N_out, D_, N_in, K_in, K_iter, K_local, K_random,
                 n_groups, alpha_reflect, mode, redraw_every, seed):
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
        self.mode = mode
        self.redraw_every = redraw_every
        self._epoch = 0
        self._gen = torch.Generator(device="cpu").manual_seed(seed + 1)

        # Allocate teleport edge buffer [N_hidden] — initial random draw
        # (None for "none" mode → skip the teleport branch entirely)
        if mode != "none":
            self.register_buffer("conn_tel", self._fresh_perm(N_hidden), persistent=False)
        else:
            self.conn_tel = None

    def _fresh_perm(self, N_hidden):
        return torch.randperm(N_hidden, generator=self._gen)

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase

    def tick_epoch(self):
        self._epoch += 1
        if self.mode == "epoch" and self.redraw_every > 0:
            if self._epoch % self.redraw_every == 0:
                self.conn_tel.copy_(self._fresh_perm(self.base.N_hidden).to(self.conn_tel.device))
        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()

    def _current_tel(self):
        """Return the teleport index tensor for this forward pass."""
        if self.mode == "batch" and self.training:
            # Redraw a fresh teleport permutation each train step
            return self._fresh_perm(self.base.N_hidden).to(self.conn_tel.device)
        return self.conn_tel

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh
        N_h = self.base.N_hidden
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h = self.base.W_pos[:N_h]
        W_n = F.normalize(W_h, dim=-1)

        # ΔW vectors (structural edges)
        delta_w = W_h.unsqueeze(1) - W_h[conn_hh]              # [N, K_hh, D]
        dw_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)    # [1, N, K_hh, D]

        # Teleport edge: compute ΔW for tel edge (if present)
        tel = self._current_tel() if self.mode != "none" else None
        if tel is not None:
            delta_w_tel = (W_h - W_h[tel]).unsqueeze(1)        # [N, 1, D]
            dw_tel_norm = F.normalize(delta_w_tel, dim=-1).unsqueeze(0)  # [1,N,1,D]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]                         # [B,N,K_hh,D]

            # Structural edges: ΔW projection gate
            proj = (Z_nb * dw_norm).sum(-1, keepdim=True)       # [B,N,K_hh,1]
            Z_nb = Z_nb * proj.abs()
            Z_struct = Z_nb.sum(dim=2)                          # [B,N,D]

            # Teleport edge (if any): add as one more gated contribution
            if tel is not None:
                Z_tel = Z_fwd[:, tel, :].unsqueeze(2)           # [B,N,1,D]
                proj_tel = (Z_tel * dw_tel_norm).sum(-1, keepdim=True)
                Z_tel = Z_tel * proj_tel.abs()
                Z_struct = Z_struct + Z_tel.squeeze(2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.alpha_reflect * Z_reflected + Z_remainder
            Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)

        return self.base._readout(Z)


def build_model(cfg):
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng = max(8, N // 8)
    return SGNNET_DeltaTeleport(
        N_hidden=N, N_out=N_OUT, D_=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, alpha_reflect=ALPHA_REFLECT,
        mode=cfg["mode"], redraw_every=cfg["redraw_every"], seed=SEED)


def main():
    keys = [c["key"] for c in CONFIGS]
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else keys
    active = [c for c in CONFIGS if c["key"] in run_keys]

    data_path = ROOT / args.data
    print(f"Step 525 — Teleportation class (fixed conn_hh + stochastic hop)")
    print(f"  data={args.data}  seed={SEED}  epochs={EPOCHS}  device={DEVICE}")
    print(f"  N={N} D={D} K_hh={K_HH} K_iter={K_ITER}")

    tr, va = make_loaders(data_path, batch_size=BATCH, seed=SEED)
    if not args.full_data:
        n = len(tr.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")

    results = {}
    for cfg in active:
        print(f"\n{'─'*60}\nConfig {cfg['key']}: {cfg['label']}\n{'─'*60}")
        t0 = time.time()
        model = build_model(cfg).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_params:,}  mode={cfg['mode']}  redraw_every={cfg['redraw_every']}")
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch'] + 1) % 5 == 0 else None
        ))
        top1h = [h.get("val_top1", 0.0) for h in history]
        top1_best = max(top1h) if top1h else 0.0
        best_ep = int(np.argmax(top1h)) + 1 if top1h else 0
        elapsed = time.time() - t0
        print(f"  → best={top1_best:.4f} @ep{best_ep}  elapsed={elapsed:.0f}s")
        results[cfg["key"]] = {
            "label": cfg["label"], "mode": cfg["mode"],
            "redraw_every": cfg["redraw_every"],
            "n_params": n_params, "top1_best": top1_best,
            "best_epoch": best_ep, "elapsed_s": elapsed,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n========== STEP 525 SUMMARY (N={N} T0 seed={SEED}) ==========")
    ref_best = results.get("Ref", {}).get("top1_best", 0.0)
    for cfg in active:
        r = results.get(cfg["key"], {})
        if not r: continue
        d = (r["top1_best"] - ref_best) * 100 if ref_best else 0
        print(f"  {cfg['key']:<6}  best={r['top1_best']:.4f}  Δ={d:+.2f}pp  {cfg['label']}")


if __name__ == "__main__":
    main()
