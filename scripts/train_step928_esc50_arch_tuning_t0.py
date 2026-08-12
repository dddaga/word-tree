"""Step 928: ESC-50 architecture tuning T0 — D and K_in sweep.

MOTIVATION
==========
step926 T0: N2048 SGNNET=32.0% vs Linear=47.75% (−15.75pp at T0 underfit).
step406 old-arch (150ep full): SGNNET=50.5% vs Linear=64.5% (−14pp).

The gap persists with ΔW-proj. Hypothesis: geometry routing fails on audio
because Whisper features lack Euclidean structure. BUT:

Alternative hypothesis: N=2048, D=16 overparams for 1600 samples.
- Smaller D (D=4 → S^3) gives simpler geometry, less overfitting
- Larger K_in (K_in=50 → 13% of 384 features) gives richer seeding
- Combining: smaller D + larger K_in for dense audio feature space

This is a targeted ARCHITECTURE SEARCH for audio. If D=4 or K_in=50 closes
the gap to ≤5pp vs Linear, the mechanism IS cross-modal but needs geometry tuning.
If all variants stay >10pp: gap is structural, not tuning-addressable.

CONFIGS (T0, 20ep, 50% data, N=2048 fixed, seed=42)
  Linear      baseline
  Ref         D=16 K_in=25  (step926 reference)
  A_D8        D=8  K_in=25  (halve manifold dimension)
  B_D4        D=4  K_in=25  (minimal manifold — S^3)
  C_kin50     D=16 K_in=50  (13% input density)
  D_D8k50     D=8  K_in=50  (combined — compact geometry + dense seeding)

ADVANCE RULE
============
  Any config within ±5pp of Linear → advance to T1 (audio viable)
  Best config >5pp but improves on Ref → note in paper as partial
  All configs >10pp from Linear → CONFIRMED architecture-independent audio gap
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant   import SGNNET_Resonant

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/esc50/store_esc50_whisper.h5")
parser.add_argument("--configs", default="Linear,Ref,A_D8,B_D4,C_kin50,D_D8k50")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 64
SEED   = args.seed
N_IN   = 384
N_OUT  = 50
N      = 2048
K_HH   = 2
K_ITER = 5
ALPHA_REFLECT = 0.5

STEP926_LINEAR = 0.4775   # step926 T0 20ep 50% data (undertrained)
STEP926_N2048  = 0.3200   # step926 N=2048 T0

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step928_esc50_arch_tuning_t0_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Linear": None,
    "Ref":    {"D": 16, "K_in": 25},
    "A_D8":   {"D":  8, "K_in": 25},
    "B_D4":   {"D":  4, "K_in": 25},
    "C_kin50":{"D": 16, "K_in": 50},
    "D_D8k50":{"D":  8, "K_in": 50},
}


class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(self, feats, labels):
        self.feats  = torch.tensor(feats,  dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]


def load_data(h5_path: Path, seed: int):
    with h5py.File(h5_path, "r") as f:
        tr_x = f["train_features"][:]
        tr_y = f["train_labels"][:]
        va_x = f["val_features"][:]
        va_y = f["val_labels"][:]

    tr_full = ESC50Dataset(tr_x, tr_y)
    va_ds   = ESC50Dataset(va_x, va_y)

    n_full  = len(tr_full)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(seed))[: n_full // 2]
    tr_sub  = torch.utils.data.Subset(tr_full, sub_idx.tolist())

    tr = torch.utils.data.DataLoader(tr_sub, batch_size=BATCH, shuffle=True,
                                     num_workers=10, pin_memory=False)
    va = torch.utils.data.DataLoader(va_ds,  batch_size=BATCH, shuffle=False,
                                     num_workers=10, pin_memory=False)
    return tr, va


def _dw_proj(W_pos, conn_hh, n):
    W_h = W_pos[:n]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def make_model(cfg):
    if cfg is None:
        torch.manual_seed(SEED)
        return nn.Linear(N_IN, N_OUT)

    D, K_in = cfg["D"], cfg["K_in"]
    K_iter  = K_ITER
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_in, K_iter=K_iter, K_local=K_l, K_random=K_r,
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
            dw        = _dw_proj(self.m.W_pos, conn_hh, N)
            Z_ref     = torch.zeros_like(Z)
            for _ in range(K_iter):
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
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr, va = load_data(data_path, SEED)

    print(f"\n{'='*70}")
    print(f"step928 — ESC-50 Architecture Tuning T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}  N_IN={N_IN}  N_OUT={N_OUT}  K_hh={K_HH}  K_iter={K_ITER}")
    print(f"  step926 refs: Linear={STEP926_LINEAR:.4f}  Ref_N2048={STEP926_N2048:.4f}")
    print(f"  Question: does smaller D or larger K_in close the audio gap?")
    print(f"{'='*70}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}
    linear_acc = None

    for key in keys:
        model = make_model(CONFIGS[key]).to(DEVICE)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cfg   = CONFIGS[key]
        print(f"{'─'*60}")
        if cfg is None:
            print(f"{key}: Linear probe  params={n_p:,}")
        else:
            print(f"{key}: D={cfg['D']}  K_in={cfg['K_in']}  params={n_p:,}")

        opt   = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
        t0    = time.time()
        hist  = []

        for ep in range(EPOCHS):
            model.train()
            for x, y in tr:
                x, y = x.to(DEVICE), y.to(DEVICE)
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
            "D": cfg["D"] if cfg else None,
            "K_in": cfg["K_in"] if cfg else None,
            "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_linear": round(delta, 4) if delta is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 928 SUMMARY — ESC-50 Architecture Tuning T0")
    print(f"{'='*70}")
    print(f"  step926 ref: Ref_N2048={STEP926_N2048:.4f} (−{(STEP926_LINEAR-STEP926_N2048)*100:.1f}pp from Linear at T0)")
    for k, r in results.items():
        d = r["delta_vs_linear"]
        if d is None:
            verdict = "(baseline)"
        elif d >= -0.05:
            verdict = "COMPETITIVE — advance to T1"
        elif d >= -0.10:
            verdict = "MARGINAL — paper footnote"
        elif r.get("D") and r["D"] < 16 and d > (STEP926_N2048 - STEP926_LINEAR):
            verdict = f"IMPROVED vs Ref — best audio config so far"
        else:
            verdict = "AUDIO GAP PERSISTS"
        cfg_str = f"D={r['D']},K_in={r['K_in']}" if r["D"] else "Linear"
        print(f"  {k:<10} {cfg_str:<12}  best={r['best']:.4f}  "
              f"{('N/A' if d is None else f'{d*100:+.2f}pp'):>8}  {verdict}")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
