"""Step 964: ESC-50 audio ablation — confirm routing contribution in dense-seed result.

QUESTION
========
step962 dense seed (C_N256_dp) = 0.5650 — beats Linear by +14.75pp.
BUT: W_seed ∈ ℝ^{N×N_IN×D} is a large linear layer (~1.5M params at N=256).
Is it routing that helps, or just having more parameters?

Dense seed SGNNET model:
  W_seed: 384 × (N×D)       — seed projection
  W_pos + routing params     — ~small
  readout: N × N_OUT         — classification head
  TOTAL @ N=128: ~800K params

Ablations:
  MLP_matched — 2-layer MLP with same total params: Linear(384→H) → ReLU → Linear(H→50)
  MLP_small   — MLP with same hidden width as N*D but single layer: Linear(384→N*D) → Linear(N*D→50)
  SGNNET_K0   — dense seed, K_iter=0 (no routing — just seed projection + readout)
  SGNNET_K5   — dense seed, K_iter=5 (full routing — step962 C_N256_dp)

If SGNNET_K5 >> SGNNET_K0 → routing adds value beyond projection.
If SGNNET_K0 ≈ MLP_small → seeding alone = linear projection (expected).
If SGNNET_K5 > MLP_matched → routing beats MLP at same param budget.

CONFIGS (N=256, D=16, 20ep, 50% data)
  Linear        — Linear(384, 50)                          19K params
  MLP_small     — Linear(384, 4096) + Linear(4096, 50)    1,59M params
  MLP_matched   — same total params as SGNNET_K5           ~same
  SGNNET_K0     — dense seed, no routing (K_iter=0)       ~same
  SGNNET_K5     — dense seed, full routing (step962 ref)  ~same

LINEAR_BASELINE = 0.4175
DENSE_SEED_REF  = 0.5650  (step962 C_N256_dp)
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
parser.add_argument("--configs", default="Linear,MLP_small,MLP_matched,SGNNET_K0,SGNNET_K5")
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

LINEAR_BASELINE = 0.4175
DENSE_SEED_REF  = 0.5650

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step964_esc50_audio_ablation_t0_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Linear":       {"type": "linear"},
    "MLP_small":    {"type": "mlp", "hidden": N * D},           # 1-hidden, width=N*D
    "MLP_matched":  {"type": "mlp_matched"},                    # matched params to SGNNET_K5
    "SGNNET_K0":    {"type": "sgnnet_dense", "K_iter": 0},      # no routing
    "SGNNET_K5":    {"type": "sgnnet_dense", "K_iter": K_ITER}, # full routing
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


def make_sgnnet_dense(K_iter: int) -> nn.Module:
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=1, K_iter=K_iter, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )

    class DenseProjectSeed(nn.Module):
        def __init__(self):
            super().__init__()
            self.m      = resonant
            self.W_seed = nn.Linear(N_IN, N * D, bias=False)
            nn.init.normal_(self.W_seed.weight, std=1.0 / (N_IN ** 0.5))
            self._K_iter = K_iter

        def forward(self, x):
            B_batch = x.size(0)
            Z = self.W_seed(x).view(B_batch, N, D)
            Z = F.normalize(Z, dim=-1)

            if self._K_iter == 0:
                return self.m.base._readout(Z)

            theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
            conn_hh   = self.m.base.conn_hh
            W_h = self.m.W_pos[:N]
            dw  = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
            Z_ref = torch.zeros_like(Z)
            for _ in range(self._K_iter):
                Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
                Z_nb  = Z_fwd[:, conn_hh, :]
                c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
                Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
                Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
                Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
            return self.m.base._readout(Z)

    return DenseProjectSeed()


def make_mlp(hidden: int) -> nn.Module:
    torch.manual_seed(SEED)
    return nn.Sequential(
        nn.Linear(N_IN, hidden),
        nn.ReLU(),
        nn.Linear(hidden, N_OUT),
    )


def make_mlp_matched() -> nn.Module:
    # Match total params of SGNNET_K5
    # SGNNET_K5 params = W_seed (N_IN * N * D) + W_pos (N*2, D) + theta (N) + readout (N * N_OUT)
    # ≈ 384*4096 + small = ~1.58M
    # MLP: N_IN * H + H * N_OUT = 384*H + H*50 = H*434 = 1.58M → H ≈ 3640
    H = (N_IN * N * D) // (N_IN + N_OUT)
    torch.manual_seed(SEED)
    return nn.Sequential(
        nn.Linear(N_IN, H),
        nn.ReLU(),
        nn.Linear(H, N_OUT),
    )


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
    print(f"step964 — ESC-50 audio routing ablation T0 (20ep, 50% data)")
    print(f"  Isolates: does ΔW-proj routing add value over projection alone?")
    print(f"  Linear: {LINEAR_BASELINE:.4f}  |  Dense seed ref (step962): {DENSE_SEED_REF:.4f}")
    print(f"{'='*70}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}
    k0_acc  = None  # set from SGNNET_K0

    for key in keys:
        cfg = CONFIGS[key]
        print(f"{'─'*60}")
        t = cfg["type"]

        if t == "linear":
            model = make_linear().to(DEVICE)
        elif t == "mlp":
            model = make_mlp(cfg["hidden"]).to(DEVICE)
        elif t == "mlp_matched":
            model = make_mlp_matched().to(DEVICE)
        elif t == "sgnnet_dense":
            model = make_sgnnet_dense(cfg["K_iter"]).to(DEVICE)

        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{key}: {t}  params={n_p:,}")

        t0   = time.time()
        hist = train_model(model, tr, va)
        elapsed = time.time() - t0

        best    = max(hist)
        best_ep = int(np.argmax(hist)) + 1

        if key == "SGNNET_K0":
            k0_acc = best

        delta_linear   = best - LINEAR_BASELINE
        delta_dense    = best - DENSE_SEED_REF
        routing_gain   = (best - k0_acc) if k0_acc is not None else None

        verdict = ("(linear ctrl)"  if key == "Linear"
                   else "(K0 no-routing)" if key == "SGNNET_K0"
                   else "(step962 ref)"   if key == "SGNNET_K5"
                   else "BEATS_SGNNET"    if delta_dense >= 0.005
                   else "MATCHES_SGNNET"  if delta_dense >= -0.010
                   else "BELOW_SGNNET"    if delta_linear >= 0.0
                   else "BELOW_LINEAR")

        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_linear={delta_linear*100:+.2f}pp  "
              f"Δ_dense={delta_dense*100:+.2f}pp  "
              + (f"routing_gain={routing_gain*100:+.2f}pp" if routing_gain is not None else "")
              + f"  {verdict}")

        results[key] = {
            "type":            t,
            "n_params":        n_p,
            "best":            round(best, 4),
            "best_ep":         best_ep,
            "delta_vs_linear": round(delta_linear, 4),
            "delta_vs_dense":  round(delta_dense, 4),
            "routing_gain":    round(routing_gain, 4) if routing_gain is not None else None,
            "elapsed_s":       round(elapsed),
            "history_top1":    [round(v, 4) for v in hist],
            "verdict":         verdict,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 964 SUMMARY — ESC-50 routing ablation")
    print(f"  Linear: {LINEAR_BASELINE:.4f}  |  Dense seed ref: {DENSE_SEED_REF:.4f}")
    print(f"{'='*70}")
    for k, r in results.items():
        rg = f"rg={r['routing_gain']*100:+.2f}pp" if r["routing_gain"] is not None else "          "
        print(f"  {k:<16} params={r['n_params']:<8,}  best={r['best']:.4f}  "
              f"Δ_lin={r['delta_vs_linear']*100:+.2f}pp  {rg}  {r['verdict']}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
