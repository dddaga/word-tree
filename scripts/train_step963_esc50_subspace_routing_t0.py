"""Step 963: ESC-50 subspace routing — treat embedding blocks as spatial nodes.

ROOT CAUSE HYPOTHESIS
=====================
SGNNET's routing advantage in vision: spatial patches have local correlations —
nearby patches share similar activations → routing over them is meaningful.

Whisper 384-d: the 384 dimensions correspond to attention head outputs across
transformer layers. They have SEMANTIC subspace structure, not spatial structure.
But subspace structure IS structure — if we slice the 384-d embedding into
blocks of B dims and treat each block as a "node", we create a meaningful
spatial arrangement over semantic subspaces.

Analogy:
  Vision:    25088-d → 196 spatial patches of 128-d  → route over 196 nodes
  Audio(new): 384-d  → B blocks of S-d               → route over B nodes

This is much closer to how SGNNET was designed: nodes represent regions of
a structured space. Here the "space" is the Whisper embedding's head decomposition.

CONFIGS (20ep, 50% data, seed=42)
  Linear         — baseline
  A_b24_s16      — B=24 blocks × S=16-d  (exactly 24 Whisper-head-sized blocks)
  B_b16_s24      — B=16 blocks × S=24-d
  C_b48_s8       — B=48 blocks × S=8-d   (more nodes, smaller subspaces)
  D_b12_s32      — B=12 blocks × S=32-d  (fewer nodes, richer subspaces)
  E_b96_s4       — B=96 blocks × S=4-d   (many tiny nodes)

For each: N_hidden = B (nodes = embedding blocks).
Routing: K_hh=2 between blocks, K_iter=5.
Seeding: block i gets x[i*S:(i+1)*S] padded/projected to D=16 via a small linear.

ADVANCE: any config ≥+5pp vs Linear (≥46.75%) → subspace routing works.
         OR ≥+10pp vs sparse baseline (≥25.25%) → worth pursuing.

LINEAR_BASELINE = 0.4175
SPARSE_REF      = 0.1425  (step961 N=256 K_in=1)
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

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/esc50/store_esc50_whisper.h5")
parser.add_argument("--configs", default="Linear,A_b24_s16,B_b16_s24,C_b48_s8,D_b12_s32,E_b96_s4")
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
K_HH   = 2
K_ITER = 5
ALPHA_REFLECT = 0.5
D_NODE = 16    # hidden dim per node (routing operates in D_NODE-space)

LINEAR_BASELINE = 0.4175
SPARSE_REF      = 0.1425

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step963_esc50_subspace_routing_t0_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Linear":    {"B": None, "S": None},
    "A_b24_s16": {"B": 24,  "S": 16},
    "B_b16_s24": {"B": 16,  "S": 24},
    "C_b48_s8":  {"B": 48,  "S": 8},
    "D_b12_s32": {"B": 12,  "S": 32},
    "E_b96_s4":  {"B": 96,  "S": 4},
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


class SubspaceRoutingModel(nn.Module):
    """Route over B semantic subspace nodes, each initialized from S-d block of embedding."""

    def __init__(self, B: int, S: int):
        super().__init__()
        assert B * S == N_IN, f"B*S must equal {N_IN}, got {B}*{S}={B*S}"
        self.B = B
        self.S = S

        torch.manual_seed(SEED)

        # Project each S-d block to D_NODE-d for routing
        self.block_proj = nn.Linear(S, D_NODE, bias=False)
        nn.init.normal_(self.block_proj.weight, std=1.0 / (S ** 0.5))

        # Graph topology: small-world over B nodes
        # Build K_hh=2 connectivity manually (ring + random)
        K_local  = max(1, K_HH // 2)
        K_random = K_HH - K_local
        conn = []
        for i in range(B):
            # local ring neighbors
            local = [(i + j) % B for j in range(1, K_local + 1)]
            # random (seeded)
            g = torch.Generator(); g.manual_seed(SEED + i)
            rng = torch.randperm(B, generator=g)
            rng = [r.item() for r in rng if r.item() != i][:K_random]
            conn.append(local + rng)
        # conn: [B, K_hh]
        self.register_buffer("conn_hh", torch.tensor(conn, dtype=torch.long))

        # W_pos: Fourier features on unit circle for B nodes
        angles = torch.linspace(0, 2 * 3.14159, B + 1)[:-1]
        freqs  = torch.arange(1, D_NODE // 2 + 1).float()
        W = torch.cat([
            torch.sin(angles.unsqueeze(1) * freqs.unsqueeze(0)),
            torch.cos(angles.unsqueeze(1) * freqs.unsqueeze(0)),
        ], dim=1)  # (B, D_NODE)
        self.W_pos = nn.Parameter(F.normalize(W, dim=-1))

        # Threshold
        self.theta = nn.Parameter(torch.zeros(B))

        # Readout: linear over all node states
        self.readout = nn.Linear(B * D_NODE, N_OUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_batch = x.size(0)
        B = self.B

        # Seed: reshape x into (batch, B, S) blocks, project to D_NODE
        blocks = x.view(B_batch, B, self.S)               # (batch, B, S)
        Z = self.block_proj(blocks)                        # (batch, B, D_NODE)
        Z = F.normalize(Z, dim=-1)

        # ΔW-proj routing over B nodes
        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)  # (1, B, 1)
        conn_hh   = self.conn_hh                                  # (B, K_hh)
        W_h = self.W_pos                                          # (B, D_NODE)
        dw  = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
        # dw: (1, B, K_hh, D_NODE)

        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]                    # (batch, B, K_hh, D_NODE)
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)   # (batch, B, K_hh, 1)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)           # (batch, B, D_NODE)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)

        return self.readout(Z.view(B_batch, -1))


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
    print(f"step963 — ESC-50 subspace routing T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Route over B semantic subspace nodes (each sees S-d block of embedding).")
    print(f"  Linear: {LINEAR_BASELINE:.4f}  |  Sparse ref: {SPARSE_REF:.4f}")
    print(f"{'='*70}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}

    for key in keys:
        cfg = CONFIGS[key]
        print(f"{'─'*60}")

        if cfg["B"] is None:
            model = make_linear().to(DEVICE)
            n_p   = sum(p.numel() for p in model.parameters())
            print(f"{key}: Linear({N_IN}, {N_OUT})  params={n_p:,}")
        else:
            B, S  = cfg["B"], cfg["S"]
            model = SubspaceRoutingModel(B, S).to(DEVICE)
            n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{key}: B={B} blocks × S={S}-d → {B} nodes  D_node={D_NODE}  params={n_p:,}")

        t0   = time.time()
        hist = train_model(model, tr, va)
        elapsed = time.time() - t0

        best    = max(hist)
        best_ep = int(np.argmax(hist)) + 1

        delta_sparse = best - SPARSE_REF
        delta_linear = best - LINEAR_BASELINE

        verdict = ("(linear control)" if key == "Linear"
                   else "EXCEEDS_LINEAR"  if delta_linear >= 0.0
                   else "APPROACHING"     if delta_linear >= -0.05
                   else "ADVANCE→T1"      if delta_sparse >= 0.100
                   else "INTERESTING"     if delta_sparse >= 0.050
                   else "NEUTRAL"         if delta_sparse >= 0.010
                   else "KILL")

        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_sparse={delta_sparse*100:+.2f}pp  "
              f"Δ_linear={delta_linear*100:+.2f}pp  {verdict}")

        results[key] = {
            "B":               cfg["B"],
            "S":               cfg["S"],
            "n_params":        n_p,
            "best":            round(best, 4),
            "best_ep":         best_ep,
            "delta_vs_sparse": round(delta_sparse, 4),
            "delta_vs_linear": round(delta_linear, 4),
            "elapsed_s":       round(elapsed),
            "history_top1":    [round(v, 4) for v in hist],
            "verdict":         verdict,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 963 SUMMARY — ESC-50 subspace routing T0")
    print(f"  Linear: {LINEAR_BASELINE:.4f}  |  Sparse ref: {SPARSE_REF:.4f}")
    print(f"{'='*70}")
    for k, r in results.items():
        bs = f"B={r['B']},S={r['S']}" if r["B"] else "full"
        print(f"  {k:<14} {bs:<10}  best={r['best']:.4f}  "
              f"Δ_sparse={r['delta_vs_sparse']*100:+.2f}pp  "
              f"Δ_lin={r['delta_vs_linear']*100:+.2f}pp  {r['verdict']}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
