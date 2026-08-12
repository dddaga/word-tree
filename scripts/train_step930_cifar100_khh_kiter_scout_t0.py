"""Step 930: CIFAR-100 K_hh × K_iter scout T0.

MOTIVATION
==========
step905 T1: N=4096=40.57% vs Linear=64.78% (−24.21pp).
step927 T0: N=8192 in progress — scaling gap ~5pp per 2×N.
K_hh and K_iter have NOT been swept on CIFAR-100.
On CIFAR-10 (step916): K_iter=5 is canonical; K_iter=3 and 7 were tested.
CIFAR-100 has 100 fine-grained classes — may need deeper propagation (K_iter=7+)
to separate visually similar classes through graph topology.

CONFIGS (T0, 20ep, 50% data, N=4096, D=16, K_in=15, seed=42)
  Linear      baseline
  K1I3        K_hh=1, K_iter=3
  K1I5        K_hh=1, K_iter=5
  K1I7        K_hh=1, K_iter=7
  K2I3        K_hh=2, K_iter=3
  K2I5        K_hh=2, K_iter=5  (canonical ref)
  K2I7        K_hh=2, K_iter=7
  K4I3        K_hh=4, K_iter=3
  K4I5        K_hh=4, K_iter=5
  K4I7        K_hh=4, K_iter=7

ADVANCE RULE
============
  Any config >42pp (>+1.5pp vs step905 N4096 T1) → advance to T1 (step932)
  Best config improves by >2pp vs canonical (K2I5) → update CIFAR-100 defaults
  All within ±2pp of canonical → K_hh/K_iter not capacity-limiting for CIFAR-100
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

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_cifar100.h5")
parser.add_argument("--configs", default="Linear,K1I3,K1I5,K1I7,K2I3,K2I5,K2I7,K4I3,K4I5,K4I7")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = args.seed
N_IN   = 25088
N_OUT  = 100
N      = 4096
D      = 16
K_IN   = 15     # K_in crossover: K_in=15 optimal at N≥4096
ALPHA_REFLECT = 0.5

STEP905_LINEAR = 0.6478
STEP905_N4096  = 0.4057

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step930_cifar100_khh_kiter_scout_t0_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Linear": None,
    "K1I3":   {"K_hh": 1, "K_iter": 3},
    "K1I5":   {"K_hh": 1, "K_iter": 5},
    "K1I7":   {"K_hh": 1, "K_iter": 7},
    "K2I3":   {"K_hh": 2, "K_iter": 3},
    "K2I5":   {"K_hh": 2, "K_iter": 5},
    "K2I7":   {"K_hh": 2, "K_iter": 7},
    "K4I3":   {"K_hh": 4, "K_iter": 3},
    "K4I5":   {"K_hh": 4, "K_iter": 5},
    "K4I7":   {"K_hh": 4, "K_iter": 7},
}


def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def make_model(cfg):
    if cfg is None:
        torch.manual_seed(SEED)
        return nn.Linear(N_IN, N_OUT)

    K_hh   = cfg["K_hh"]
    K_iter = cfg["K_iter"]
    torch.manual_seed(SEED)
    K_r = max(1, K_hh // 4); K_l = K_hh - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_iter, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )

    class DeltaW(nn.Module):
        def __init__(self):
            super().__init__()
            self.m      = resonant
            self.k_iter = K_iter

        @property
        def W_pos(self): return self.m.W_pos
        def tick_epoch(self):
            if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

        def forward(self, x):
            Z         = self.m.base._seed(x)
            theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
            conn_hh   = self.m.base.conn_hh
            dw        = _dw_proj(self.m.W_pos, conn_hh)
            Z_ref     = torch.zeros_like(Z)
            for _ in range(self.k_iter):
                Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
                Z_nb  = Z_fwd[:, conn_hh, :]
                Z_agg = _dw_agg(Z_nb, dw)
                Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
                Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
            return self.m.base._readout(Z)

    return DeltaW()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        x, y = batch[0].to(DEVICE), batch[2].to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED, pin_memory=False)
    n_full  = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[: n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=10, pin_memory=False,
    )

    print(f"\n{'='*70}")
    print(f"step930 — CIFAR-100 K_hh×K_iter Scout T0 (20ep, 50% data, N=4096)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  D={D}  K_in={K_IN}  N={N}  N_IN={N_IN}  N_OUT={N_OUT}")
    print(f"  step905 refs: Linear={STEP905_LINEAR:.4f}  N4096={STEP905_N4096:.4f}")
    print(f"  Question: does deeper/denser routing improve CIFAR-100?")
    print(f"{'='*70}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}
    linear_acc = None

    if OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text())
        results  = existing
        if "Linear" in results:
            linear_acc = results["Linear"]["best"]
        keys = [k for k in keys if k not in results]
        print(f"Resuming — {len(existing)} configs done, {len(keys)} remaining\n")

    for key in keys:
        model = make_model(CONFIGS[key]).to(DEVICE)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cfg   = CONFIGS[key]
        print(f"{'─'*60}")
        if cfg is None:
            print(f"{key}: Linear  params={n_p:,}")
        else:
            print(f"{key}: K_hh={cfg['K_hh']}  K_iter={cfg['K_iter']}  params={n_p:,}")

        opt   = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
        t0    = time.time()
        hist  = []

        for ep in range(EPOCHS):
            model.train()
            for batch in tr:
                x, y = batch[0].to(DEVICE), batch[2].to(DEVICE)
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            val = evaluate(model, va)
            hist.append(val)
            if hasattr(model, "tick_epoch"): model.tick_epoch()
            print(f"  e{ep+1:3d}  top1={val:.4f}  lr={opt.param_groups[0]['lr']:.2e}", flush=True)

        elapsed = time.time() - t0
        best    = max(hist)
        best_ep = int(np.argmax(hist)) + 1

        if key == "Linear":
            linear_acc = best
        delta = (best - linear_acc) if linear_acc is not None else None
        dstr  = f"{delta*100:+.2f}pp" if delta is not None else "(baseline)"
        print(f"  → best={best:.4f} @ep{best_ep}  {dstr}  {elapsed:.0f}s")

        results[key] = {
            "K_hh":  cfg["K_hh"]   if cfg else None,
            "K_iter": cfg["K_iter"] if cfg else None,
            "n_params": n_p, "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_linear": round(delta, 4) if delta is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 930 SUMMARY — CIFAR-100 K_hh×K_iter Scout T0")
    print(f"{'='*70}")
    print(f"  step905 T1 ref: N4096={STEP905_N4096:.4f}  Linear={STEP905_LINEAR:.4f}")
    print(f"  {'Config':<8}  {'K_hh':>5}  {'K_iter':>6}  {'best':>6}  {'Δ vs Linear':>12}  verdict")
    print(f"  {'─'*65}")
    for k, r in results.items():
        d = r["delta_vs_linear"]
        ref_delta = r["best"] - STEP905_N4096
        if d is None:
            verdict = "baseline"
        elif r["best"] > 0.42:
            verdict = f"BEST CIFAR-100 so far → T1 candidate"
        elif ref_delta > 0.02:
            verdict = f"improves vs canonical (+{ref_delta*100:.1f}pp)"
        elif abs(ref_delta) <= 0.02:
            verdict = "within ±2pp of canonical"
        else:
            verdict = "below canonical"
        khh_s  = str(r["K_hh"])   if r["K_hh"]   else "N/A"
        kiter_s = str(r["K_iter"]) if r["K_iter"] else "N/A"
        print(f"  {k:<8}  {khh_s:>5}  {kiter_s:>6}  {r['best']:.4f}  "
              f"{('N/A' if d is None else f'{d*100:+.2f}pp'):>12}  {verdict}")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
