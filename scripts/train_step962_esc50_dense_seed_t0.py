"""Step 962: ESC-50 dense projection seed — fix the seeding problem.

ROOT CAUSE DIAGNOSIS
====================
steps 960/961/961b confirmed: K_in and N scaling do not close the audio gap.
The seeding mechanism is the problem.

Standard SGNNET seeding:
  conn_in[i] assigns K_in input dimensions to node i.
  Z_seed[i] = mean(x[conn_in[i]])  — node sees K_in of 384 dims.

For VGG 25088-d: K_in=25 covers a coherent spatial patch → useful local context.
For Whisper 384-d: even K_in=384 = entire embedding → no spatial meaning.

FIX: replace sparse seeding with a LEARNED linear projection.
  W_seed ∈ ℝ^{N×384}  (initialized randomly, trained)
  Z_seed[i] = W_seed[i] @ x  — every node sees full embedding via learned weights.

After seeding, routing operates IDENTICALLY to standard SGNNET (ΔW-proj).
This isolates: "does routing over learned projections help vs just Linear(384,50)?"

CONFIGS (20ep, 50% data, seed=42)
  Linear        — Linear(384, 50)     — baseline
  A_N64_dp      — N=64,  dense proj   — tiny N, fully informed
  B_N128_dp     — N=128, dense proj
  C_N256_dp     — N=256, dense proj   — same N as step961 reference
  D_N512_dp     — N=512, dense proj
  E_N256_sparse — N=256, K_in=1       — ablation: sparse seeding (step961 Ref)

ADVANCE: any dense_proj config ≥+2pp vs E_N256_sparse (≥16.25%) → routing helps.
KILL:    dense_proj ≤ sparse → seeding is not the problem; routing itself is useless.

LINEAR_BASELINE = 0.4175
SPARSE_REF      = 0.1425  (step961 N=256, K_in=1)
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
parser.add_argument("--configs", default="Linear,A_N64_dp,B_N128_dp,C_N256_dp,D_N512_dp,E_N256_sparse")
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
D      = 16
K_HH   = 2
K_ITER = 5
ALPHA_REFLECT = 0.5

LINEAR_BASELINE = 0.4175
SPARSE_REF      = 0.1425

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step962_esc50_dense_seed_t0_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Linear":       {"N": None,  "dense": False, "K_in": None},
    "A_N64_dp":     {"N": 64,   "dense": True,  "K_in": None},
    "B_N128_dp":    {"N": 128,  "dense": True,  "K_in": None},
    "C_N256_dp":    {"N": 256,  "dense": True,  "K_in": None},
    "D_N512_dp":    {"N": 512,  "dense": True,  "K_in": None},
    "E_N256_sparse":{"N": 256,  "dense": False, "K_in": 1},
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


def make_sgnnet_dense(N: int) -> nn.Module:
    """SGNNET with full learned projection seed (W_seed ∈ ℝ^{N×N_IN})."""
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    torch.manual_seed(SEED)

    # Build base with K_in=1 (smallest valid) — we override seeding
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=1, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )

    class DenseProjectSeed(nn.Module):
        def __init__(self):
            super().__init__()
            self.m       = resonant
            # Learned projection: maps 384-d input to N seed vectors of dim D
            self.W_seed  = nn.Linear(N_IN, N * D, bias=False)
            nn.init.normal_(self.W_seed.weight, std=1.0 / (N_IN ** 0.5))

        @property
        def W_pos(self): return self.m.W_pos

        def forward(self, x):
            B = x.size(0)
            # Dense seed: each node gets a learned projection (B, N, D)
            Z = self.W_seed(x).view(B, N, D)
            Z = F.normalize(Z, dim=-1)

            # Standard ΔW-proj routing
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

    return DenseProjectSeed()


def make_sgnnet_sparse(N: int, K_in: int) -> nn.Module:
    """Standard sparse seeding (ΔW-proj) — ablation baseline."""
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
    print(f"step962 — ESC-50 dense projection seed T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Fix: W_seed ∈ ℝ^{{N×{N_IN}}} replaces sparse K_in seeding.")
    print(f"  Ablation E_N256_sparse isolates dense vs sparse contribution.")
    print(f"  Linear: {LINEAR_BASELINE:.4f}  |  Sparse ref (step961 N256): {SPARSE_REF:.4f}")
    print(f"{'='*70}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}
    sparse_ref_acc = None

    for key in keys:
        cfg = CONFIGS[key]
        print(f"{'─'*60}")

        if cfg["N"] is None:
            model = make_linear().to(DEVICE)
            n_p   = sum(p.numel() for p in model.parameters())
            print(f"{key}: Linear({N_IN}, {N_OUT})  params={n_p:,}")
        elif cfg["dense"]:
            N = cfg["N"]
            model = make_sgnnet_dense(N).to(DEVICE)
            n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{key}: N={N} dense_proj  W_seed={N_IN*N*D:,}  total_params={n_p:,}")
        else:
            N, K_in = cfg["N"], cfg["K_in"]
            model = make_sgnnet_sparse(N, K_in).to(DEVICE)
            n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{key}: N={N} sparse K_in={K_in}  params={n_p:,}")

        t0   = time.time()
        hist = train_model(model, tr, va)
        elapsed = time.time() - t0

        best    = max(hist)
        best_ep = int(np.argmax(hist)) + 1

        if key == "E_N256_sparse":
            sparse_ref_acc = best
        _sparse = sparse_ref_acc if sparse_ref_acc is not None else SPARSE_REF
        delta_sparse = best - _sparse
        delta_linear = best - LINEAR_BASELINE

        verdict = ("(linear control)"    if key == "Linear"
                   else "(sparse ablation)" if key == "E_N256_sparse"
                   else "CROSSOVER"      if delta_linear >= -0.02
                   else "APPROACHING"   if delta_linear >= -0.05
                   else "ADVANCE→T1"    if delta_sparse >= 0.020
                   else "INTERESTING"   if delta_sparse >= 0.010
                   else "NEUTRAL"       if delta_sparse >= -0.005
                   else "KILL")

        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_sparse={delta_sparse*100:+.2f}pp  "
              f"Δ_linear={delta_linear*100:+.2f}pp  {verdict}")

        results[key] = {
            "N":             cfg["N"],
            "dense":         cfg["dense"],
            "K_in":          cfg.get("K_in"),
            "n_params":      n_p,
            "best":          round(best, 4),
            "best_ep":       best_ep,
            "delta_vs_sparse": round(delta_sparse, 4),
            "delta_vs_linear": round(delta_linear, 4),
            "elapsed_s":     round(elapsed),
            "history_top1":  [round(v, 4) for v in hist],
            "verdict":       verdict,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 962 SUMMARY — ESC-50 dense projection seed T0")
    print(f"  Linear: {LINEAR_BASELINE:.4f}  |  Sparse ref: {SPARSE_REF:.4f}")
    print(f"{'='*70}")
    for k, r in results.items():
        dense_s = "dense" if r["dense"] else f"K_in={r['K_in']}"
        print(f"  {k:<16} N={str(r['N']):<5}  {dense_s:<8}  "
              f"best={r['best']:.4f}  Δ_sparse={r['delta_vs_sparse']*100:+.2f}pp  "
              f"Δ_lin={r['delta_vs_linear']*100:+.2f}pp  {r['verdict']}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
