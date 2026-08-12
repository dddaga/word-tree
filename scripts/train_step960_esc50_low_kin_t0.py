"""Step 960: ESC-50 low-K_in sweep T0 — test whether aggressive input aggregation
destroys compressed Whisper embedding structure.

HYPOTHESIS
==========
Each Whisper embedding dimension carries distilled information from the full
transformer. K_in=25 sums 25 of 384 dimensions per node (6.5% coverage) —
this is a coarse aggregation that may collapse discriminative structure.

For VGG: K_in=25 covers 0.1% of 25,088 spatial features. Each node sees a
tiny local patch — selective and non-destructive.

For Whisper 384-d: K_in=25 covers 6.5%. Way too much aggregation.
K_in=2 covers 0.5% — much closer to the VGG coverage ratio.

Existing data supports this direction:
  K_in=25  → −14.5pp vs Linear  (step928)
  K_in=50  → −18.0pp vs Linear  (step928)  ← MORE aggregation is WORSE

This sweep goes the other way: K_in ∈ {1, 2, 4, 8, 25}.
If low K_in helps, this answers: "is the failure because K_in is too large?"

Also tests Linear as internal control (verifiable against step926 baseline).

CONFIGS (N=256, D=16, K_hh=2, K_iter=5, α_reflect=0.5, 20ep, 50% data)
  Linear     — sklearn/pytorch Linear(384, 50) — internal sanity check
  Ref        — K_in=25  (6.5% coverage — current default, known −14.5pp)
  A_kin8     — K_in=8   (2.1% coverage)
  B_kin4     — K_in=4   (1.0% coverage)
  C_kin2     — K_in=2   (0.5% coverage — closest to VGG ratio 0.1%)
  D_kin1     — K_in=1   (0.3% coverage — minimum; each node sees 1 input dim)

ADVANCE: any config ≥+1.0pp vs Ref (≥35.25%) → T1.
KILL:    all configs ≤ Ref → K_in is not the problem; direction closed.

LINEAR_BASELINE = 0.4775  (step926 T0 20ep 50% data)
REF_BASELINE    = 0.3325  (step928 K_in=25 20ep 50% data)
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
parser.add_argument("--configs", default="Linear,Ref,A_kin8,B_kin4,C_kin2,D_kin1")
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
N      = 256
D      = 16
K_HH   = 2
K_ITER = 5
ALPHA_REFLECT = 0.5

LINEAR_BASELINE = 0.4775
REF_BASELINE    = 0.3325

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step960_esc50_low_kin_t0_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Linear":  {"K_in": None},   # internal control
    "Ref":     {"K_in": 25},
    "A_kin8":  {"K_in": 8},
    "B_kin4":  {"K_in": 4},
    "C_kin2":  {"K_in": 2},
    "D_kin1":  {"K_in": 1},
}


class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(self, feats, labels):
        self.feats  = torch.tensor(feats,  dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx): return self.feats[idx], self.labels[idx]


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


def make_sgnnet(K_in: int) -> nn.Module:
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_in, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
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

        def forward(self, x):
            Z         = self.m.base._seed(x)
            theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
            conn_hh   = self.m.base.conn_hh
            W_h = self.m.W_pos[:N]
            dw  = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
            Z_ref = torch.zeros_like(Z)
            for _ in range(K_ITER):
                Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
                Z_nb  = Z_fwd[:, conn_hh, :]
                c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
                Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
                Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
                Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
            return self.m.base._readout(Z)

    return DeltaW()


def make_linear() -> nn.Module:
    torch.manual_seed(SEED)
    return nn.Linear(N_IN, N_OUT)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def train_model(model, tr, va, lr=3e-3, wd=1e-3):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
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
        print(f"  e{ep+1:3d}  top1={val:.4f}  lr={opt.param_groups[0]['lr']:.2e}",
              flush=True)
    return hist


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found"); sys.exit(1)

    tr, va = load_data(data_path, SEED)

    print(f"\n{'='*70}")
    print(f"step960 — ESC-50 low-K_in sweep T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}")
    print(f"  Hypothesis: K_in=25 is too aggressive for 384-d compressed embedding.")
    print(f"  Low K_in (1-8) ≈ VGG coverage ratio (0.1%), may preserve structure.")
    print(f"  Known: K_in=25→−14.5pp, K_in=50→−18.0pp (more agg = worse).")
    print(f"  Linear baseline (step926): {LINEAR_BASELINE:.4f}")
    print(f"  SGNNET Ref (K_in=25, step928): {REF_BASELINE:.4f}")
    print(f"{'='*70}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}
    ref_acc = None

    for key in keys:
        cfg = CONFIGS[key]
        print(f"{'─'*60}")

        if cfg["K_in"] is None:
            # Linear baseline
            model = make_linear().to(DEVICE)
            n_p   = sum(p.numel() for p in model.parameters())
            print(f"{key}: Linear({N_IN}, {N_OUT})  params={n_p:,}")
            t0   = time.time()
            hist = train_model(model, tr, va)
        else:
            K_in = cfg["K_in"]
            cov  = K_in / N_IN * 100
            model = make_sgnnet(K_in).to(DEVICE)
            n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{key}: K_in={K_in}  coverage={cov:.2f}%  params={n_p:,}")
            t0   = time.time()
            hist = train_model(model, tr, va)

        elapsed = time.time() - t0
        best    = max(hist)
        best_ep = int(np.argmax(hist)) + 1

        if key == "Ref":
            ref_acc = best
        delta_ref    = best - (ref_acc if ref_acc is not None else REF_BASELINE)
        delta_linear = best - LINEAR_BASELINE

        verdict = ("(linear control)" if key == "Linear"
                   else "(baseline)" if key == "Ref"
                   else "ADVANCE→T1" if delta_ref >= 0.010
                   else "INTERESTING" if delta_ref >= 0.005
                   else "NEUTRAL"    if delta_ref >= -0.005
                   else "KILL")

        cov_str = f"{cfg['K_in']/N_IN*100:.2f}%" if cfg["K_in"] else "100%"
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_ref={delta_ref*100:+.2f}pp  "
              f"Δ_linear={delta_linear*100:+.2f}pp  {verdict}")

        results[key] = {
            "K_in":            cfg["K_in"],
            "coverage_pct":    round(cfg["K_in"] / N_IN * 100, 2) if cfg["K_in"] else 100.0,
            "n_params":        n_p,
            "best":            round(best, 4),
            "best_ep":         best_ep,
            "delta_vs_ref":    round(delta_ref, 4),
            "delta_vs_linear": round(delta_linear, 4),
            "elapsed_s":       round(elapsed),
            "history_top1":    [round(v, 4) for v in hist],
            "verdict":         verdict,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 960 SUMMARY — ESC-50 low-K_in sweep T0")
    print(f"  Linear baseline: {LINEAR_BASELINE:.4f}  |  SGNNET Ref (K_in=25): {REF_BASELINE:.4f}")
    print(f"{'='*70}")
    for k, r in results.items():
        cov = r["coverage_pct"]
        print(f"  {k:<12} K_in={str(r['K_in']):<4}  cov={cov:5.2f}%  "
              f"best={r['best']:.4f}  Δ_ref={r['delta_vs_ref']*100:+.2f}pp  "
              f"Δ_lin={r['delta_vs_linear']*100:+.2f}pp  {r['verdict']}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
