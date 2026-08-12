"""Step 945: ESC-50 K_in coverage sweep T0 (20ep, 50% data).

HYPOTHESIS
==========
At fixed N=256, increasing K_in from 6.5% to 100% of the 384-dim Whisper
embedding improves accuracy by preserving the holographic structure of the
dense audio features. Each SGNNET node is seeded from a random K_in-subset;
higher K_in → richer per-node signal → better distributed representation.

This is an isolation experiment: N is fixed (step944 handles N sweep),
K_in is the sole variable.

CONFIGS (N=256, D=16, K_hh=2, K_iter=5, α_reflect=0.5, 20ep, 50% data)
  Ref       K_in=25  (6.5%  of 384 — current global default, dense-unfriendly)
  A_kin64   K_in=64  (16.7%)
  B_kin128  K_in=128 (33.3%)
  C_kin192  K_in=192 (50.0%)
  D_kin320  K_in=320 (83.3%)
  E_kin_all K_in=384 (100%  — every node sees all features)

ADVANCE: ≥+0.5pp vs Ref → T1.
STEP_REF = 0.4775 (Linear baseline, step926 T0 20ep 50% data).
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
parser.add_argument("--configs", default="Ref,A_kin64,B_kin128,C_kin192,D_kin320,E_kin_all")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = args.seed
N_IN   = 384
N_OUT  = 50
N      = 256    # fixed for this sweep
D      = 16
K_HH   = 2
K_ITER = 5
ALPHA_REFLECT = 0.5

STEP_REF = 0.4775   # Linear baseline, step926 T0 20ep 50% data

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step945_esc50_kin_coverage_t0_seed{SEED}__{SLOT}.json"

# K_in values; coverage = K_in / N_IN * 100
CONFIGS = {
    "Ref":       {"K_in": 25},
    "A_kin64":   {"K_in": 64},
    "B_kin128":  {"K_in": 128},
    "C_kin192":  {"K_in": 192},
    "D_kin320":  {"K_in": 320},
    "E_kin_all": {"K_in": 384},
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


def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def make_model(cfg: dict) -> nn.Module:
    K_in  = cfg["K_in"]
    K_r   = max(1, K_HH // 4)
    K_l   = K_HH - K_r
    ng    = max(8, N // 8)
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_in, K_iter=K_ITER, K_local=K_l, K_random=K_r,
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
    print(f"step945 — ESC-50 K_in coverage sweep T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  α_reflect={ALPHA_REFLECT}")
    print(f"  Hypothesis: higher K_in coverage preserves holographic Whisper structure")
    print(f"  Linear baseline (step926): {STEP_REF:.4f}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}
    ref_acc = None

    for key in keys:
        cfg   = CONFIGS[key]
        model = make_model(cfg).to(DEVICE)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cov   = cfg["K_in"] / N_IN * 100
        print(f"{'─'*60}")
        print(f"{key}: K_in={cfg['K_in']}  coverage={cov:.1f}%  params={n_p:,}")

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

        if key == "Ref":
            ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP_REF)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp vs Ref  {elapsed:.0f}s")

        results[key] = {
            "K_in": cfg["K_in"],
            "coverage_pct": round(cov, 1),
            "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 945 SUMMARY — ESC-50 K_in coverage sweep T0")
    print(f"  N={N}  Linear baseline (step926): {STEP_REF:.4f}")
    print(f"{'='*70}")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        if k == "Ref":
            verdict = "(baseline)"
        elif d >= 0.005:
            verdict = "ADVANCE→T1"
        elif d >= -0.005:
            verdict = "NEUTRAL"
        else:
            verdict = "KILL"
        print(f"  {k:<12} K_in={r['K_in']:<4} cov={r['coverage_pct']:5.1f}%"
              f"  params={r['n_params']:>6,}  best={r['best']:.4f}  Δ={d*100:+.2f}pp  {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
