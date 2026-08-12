"""Step 927: CIFAR-100 N-scaling extension T0 — N=8192 scout.

MOTIVATION
==========
step905 T1: SGNNET_N2048=35.40%, SGNNET_N4096=40.57% vs Linear=64.78%.
Scaling: +5.17pp per 2×N. Extrapolating: N=8192 ≈ 44-46%.
Gap at N=4096 = −24.21pp. If gap closes to ≤15pp at N=8192, worth T1.

Key question: is CIFAR-100 gap capacity-limited (closes with N) or
architectural (fixed topology can't discriminate 100 fine-grained classes)?

If gap narrows significantly (±5pp closure per 2×N): advance to T1.
If gap stagnates: CONFIRMED architectural failure mode for many-class problems.

CONFIGS (T0, 20ep, 50% data, seed=42, N=8192)
  Linear      baseline (N_in=25088 → 100)
  SGNNET_N8192  N=8192 canonical ΔW-proj

ADVANCE RULE
============
  N=8192 gap ≤ 20pp vs Linear → advance N=8192 to T1 (step929)
  N=8192 gap > 20pp and <5pp improvement vs N=4096 → CONFIRMED architectural failure
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
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_cifar100.h5")
parser.add_argument("--configs", default="Linear,SGNNET_N8192")
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
D      = 16
K_HH   = 2
K_ITER = 5
ALPHA_REFLECT = 0.5

# step905 T1 references
STEP905_LINEAR   = 0.6478
STEP905_N2048    = 0.3540
STEP905_N4096    = 0.4057

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step927_cifar100_n8192_t0_seed{SEED}__{SLOT}.json"


def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:8192]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def make_linear():
    torch.manual_seed(SEED)
    return nn.Linear(N_IN, N_OUT)


def make_sgnnet() -> nn.Module:
    N = 8192
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=15, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )

    class DeltaW(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = resonant

        @property
        def W_pos(self): return self.m.W_pos
        @property
        def W_phase(self): return getattr(self.m, "W_phase", None)
        def tick_epoch(self):
            if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

        def forward(self, x):
            Z         = self.m.base._seed(x)
            theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
            conn_hh   = self.m.base.conn_hh
            dw        = _dw_proj(self.m.W_pos, conn_hh)
            Z_ref     = torch.zeros_like(Z)
            for _ in range(K_ITER):
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

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=False)
    n_full  = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[: n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=10, pin_memory=False,
    )

    print(f"\n{'='*70}")
    print(f"step927 — CIFAR-100 N=8192 T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  step905 refs: Linear={STEP905_LINEAR:.4f}  N2048={STEP905_N2048:.4f}  N4096={STEP905_N4096:.4f}")
    print(f"  Question: does N=8192 close the −24pp CIFAR-100 gap?")
    print(f"{'='*70}\n")

    configs = {
        "Linear":      make_linear,
        "SGNNET_N8192": make_sgnnet,
    }
    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in configs]
    results = {}
    linear_acc = None

    for key in keys:
        model = configs[key]().to(DEVICE)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        is_linear = key == "Linear"
        print(f"{'─'*60}")
        print(f"{key}: params={n_p:,}")

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

        if is_linear:
            linear_acc = best
        delta = (best - linear_acc) if linear_acc is not None else None
        dstr  = f"{delta*100:+.2f}pp" if delta is not None else "(baseline)"
        print(f"  → best={best:.4f} @ep{best_ep}  {dstr}  {elapsed:.0f}s")

        results[key] = {
            "n_params": n_p, "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_linear": round(delta, 4) if delta is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 927 SUMMARY — CIFAR-100 N=8192 T0")
    print(f"{'='*70}")
    print(f"  N-scaling so far (T1 references):")
    print(f"    N=2048: {STEP905_N2048:.4f}  gap={( STEP905_N2048 - STEP905_LINEAR)*100:+.2f}pp")
    print(f"    N=4096: {STEP905_N4096:.4f}  gap={( STEP905_N4096 - STEP905_LINEAR)*100:+.2f}pp  Δ_vs_2048={( STEP905_N4096 - STEP905_N2048)*100:+.2f}pp")
    for k, r in results.items():
        d = r["delta_vs_linear"]
        if d is None:
            print(f"    {k}: {r['best']:.4f} (baseline)")
        else:
            verdict = "→ advance to T1" if d > -0.20 else "→ architectural failure confirmed"
            print(f"    N=8192: {r['best']:.4f}  gap={d*100:+.2f}pp  {verdict}")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
