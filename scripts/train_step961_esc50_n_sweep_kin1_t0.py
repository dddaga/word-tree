"""Step 961: ESC-50 N-sweep at K_in=1 — find the crossover point.

HYPOTHESIS
==========
K_in=1 gives 0.26% coverage of 384-d Whisper embedding (closest to VGG 0.1%).
If the failure at K_in=25 is over-aggregation, K_in=1 should help.
But SGNNET needs sufficient N to cover the input: each of 384 dims needs
at least one node assigned to read it.

Coverage analysis (K_in=1 fixed):
  N=64    → 64 assignments / 384 dims = 0.17 per dim  (under-coverage — many dims unread)
  N=128   → 128 / 384 = 0.33 per dim
  N=256   → 256 / 384 = 0.67 per dim
  N=512   → 512 / 384 = 1.33 per dim  (crossover: first N where coverage > 1.0)
  N=1024  → 1024 / 384 = 2.67 per dim
  N=2048  → 2048 / 384 = 5.33 per dim  (full coverage, redundant reads)

CROSSOVER PREDICTION: N=512 is where every input dim gets at least one reader.
Below N=512, K_in=1 will miss dims entirely → expected failure.
Above N=512, K_in=1 may start matching or beating Linear.

Existing data:
  K_in=25  → −14.5pp vs Linear  (step928, N=256)
  K_in=50  → −18.0pp vs Linear  (step928, N=256)
  K_in=1   → ? (this experiment)

CONFIGS (K_in=1 fixed, K_hh=2, K_iter=5, α_reflect=0.5, D=16, 20ep, 50% data)
  Linear     — internal control (replicates step926)
  N64        — N=64,   0.17 coverage/dim
  N128       — N=128,  0.33 coverage/dim
  N256       — N=256,  0.67 coverage/dim
  N512       — N=512,  1.33 coverage/dim  ← predicted crossover
  N1024      — N=1024, 2.67 coverage/dim
  N2048      — N=2048, 5.33 coverage/dim

ADVANCE: any N config ≥+1.0pp vs N256_ref (≥33.25%) → T1 with that N.
KILL:    all N configs ≤ N256_ref → K_in=1 is not sufficient; K_in direction closed.

LINEAR_BASELINE = 0.4775  (step926 T0 20ep 50% data)
REF_N256        = 0.3325  (step928 K_in=25 N=256, ~expected for N=256 K_in=1)
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
parser.add_argument("--configs", default="Linear,N64,N128,N256,N512,N1024,N2048")
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
K_IN   = 1          # fixed across all SGNNET configs
D      = 16
K_HH   = 2
K_ITER = 5
ALPHA_REFLECT = 0.5

LINEAR_BASELINE = 0.4775
REF_N256        = 0.3325   # K_in=25, N=256 — approximate floor

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step961_esc50_n_sweep_kin1_t0_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Linear": {"N": None},
    "N64":    {"N": 64},
    "N128":   {"N": 128},
    "N256":   {"N": 256},
    "N512":   {"N": 512},
    "N1024":  {"N": 1024},
    "N2048":  {"N": 2048},
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


def make_sgnnet(N: int) -> nn.Module:
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
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
    print(f"step961 — ESC-50 N-sweep at K_in=1 T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  K_in={K_IN} (fixed)  D={D}  K_hh={K_HH}  K_iter={K_ITER}")
    print(f"  Hypothesis: K_in=1 (0.26% coverage) preserves compressed embedding.")
    print(f"  Crossover prediction: N≥512 (coverage_per_dim ≥ 1.0).")
    print(f"  Linear baseline (step926): {LINEAR_BASELINE:.4f}")
    print(f"  SGNNET Ref (K_in=25, N=256, step928): {REF_N256:.4f}")
    print(f"{'='*70}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}
    ref_acc = None   # set from N256 if run

    for key in keys:
        cfg = CONFIGS[key]
        print(f"{'─'*60}")

        if cfg["N"] is None:
            model = make_linear().to(DEVICE)
            n_p   = sum(p.numel() for p in model.parameters())
            print(f"{key}: Linear({N_IN}, {N_OUT})  params={n_p:,}")
            t0   = time.time()
            hist = train_model(model, tr, va)
        else:
            N    = cfg["N"]
            cpd  = K_IN / N_IN          # coverage per dimension
            n_assign = N * K_IN
            print(f"{key}: N={N}  K_in={K_IN}  assignments={n_assign}  "
                  f"coverage_per_dim={cpd:.3f}")
            model = make_sgnnet(N).to(DEVICE)
            n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  params={n_p:,}")
            t0   = time.time()
            hist = train_model(model, tr, va)

        elapsed = time.time() - t0
        best    = max(hist)
        best_ep = int(np.argmax(hist)) + 1

        # use N256 run as reference if present, else fall back to known REF_N256
        if key == "N256":
            ref_acc = best
        _ref = ref_acc if ref_acc is not None else REF_N256
        delta_ref    = best - _ref
        delta_linear = best - LINEAR_BASELINE

        verdict = ("(linear control)" if key == "Linear"
                   else "(N=256 reference)" if key == "N256"
                   else "ADVANCE→T1" if delta_ref >= 0.010
                   else "INTERESTING" if delta_ref >= 0.005
                   else "NEUTRAL"    if delta_ref >= -0.005
                   else "KILL")

        cpd_str = f"{K_IN/N_IN:.3f}" if cfg["N"] else "full"
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_ref={delta_ref*100:+.2f}pp  "
              f"Δ_linear={delta_linear*100:+.2f}pp  {verdict}")

        results[key] = {
            "N":                  cfg["N"],
            "K_in":               K_IN if cfg["N"] else None,
            "coverage_per_dim":   round(K_IN / N_IN, 4) if cfg["N"] else 1.0,
            "total_assignments":  cfg["N"] * K_IN if cfg["N"] else None,
            "n_params":           n_p,
            "best":               round(best, 4),
            "best_ep":            best_ep,
            "delta_vs_ref":       round(delta_ref, 4),
            "delta_vs_linear":    round(delta_linear, 4),
            "elapsed_s":          round(elapsed),
            "history_top1":       [round(v, 4) for v in hist],
            "verdict":            verdict,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 961 SUMMARY — ESC-50 N-sweep at K_in=1 T0")
    print(f"  Linear baseline: {LINEAR_BASELINE:.4f}  |  "
          f"SGNNET Ref (K_in=25, N=256): {REF_N256:.4f}")
    print(f"{'='*70}")
    for k, r in results.items():
        cpd = r["coverage_per_dim"]
        print(f"  {k:<8} N={str(r['N']):<6}  cpd={cpd:.3f}  "
              f"best={r['best']:.4f}  Δ_ref={r['delta_vs_ref']*100:+.2f}pp  "
              f"Δ_lin={r['delta_vs_linear']*100:+.2f}pp  {r['verdict']}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
