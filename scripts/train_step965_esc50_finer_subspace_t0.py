"""Step 965: ESC-50 finer subspace routing — follow step963 trend.

step963 showed monotonic improvement with more, smaller blocks:
  B=12  (S=32): 0.1700  (−24.75pp vs Linear)
  B=16  (S=24): 0.1950  (−22.25pp)
  B=24  (S=16): 0.2250  (−19.25pp)
  B=48  (S=8):  0.2650  (−15.25pp)
  B=96  (S=4):  0.3125  (−10.50pp)  ← best, still trending up

The trend is clear: smaller S (larger B) = better routing.
At B=96, S=4: each block is a 4-d subspace. The routing is finding
structure across 96 Whisper head-scale subspace nodes.

This sweep pushes further: B=192 (S=2), B=384 (S=1 — one node per dim).
B=384 is pure per-dimension routing: each dim becomes its own node with D=16
projection. This maximally respects the individual semantic role of each dim.

Also tests: B=96 + higher D_node (32 instead of 16) — richer per-node rep.

CONFIGS (20ep, 50% data, seed=42)
  Linear           — baseline
  Ref_b96_s4       — step963 best (B=96, S=4, D_node=16)
  A_b192_s2        — B=192, S=2,  D_node=16
  B_b384_s1        — B=384, S=1,  D_node=16  (one node per input dim)
  C_b96_d32        — B=96,  S=4,  D_node=32  (richer node representation)
  D_b192_d32       — B=192, S=2,  D_node=32

ADVANCE: any config > 0.3125 (step963 best) → finer is better, push to T1.

LINEAR_BASELINE = 0.4175
SUBSPACE_REF    = 0.3125   (step963 E_b96_s4)
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
parser.add_argument("--configs", default="Linear,Ref_b96_s4,A_b192_s2,B_b384_s1,C_b96_d32,D_b192_d32")
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

LINEAR_BASELINE = 0.4175
SUBSPACE_REF    = 0.3125

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step965_esc50_finer_subspace_t0_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Linear":      {"B": None, "S": None, "D_node": None},
    "Ref_b96_s4":  {"B": 96,  "S": 4,    "D_node": 16},
    "A_b192_s2":   {"B": 192, "S": 2,    "D_node": 16},
    "B_b384_s1":   {"B": 384, "S": 1,    "D_node": 16},
    "C_b96_d32":   {"B": 96,  "S": 4,    "D_node": 32},
    "D_b192_d32":  {"B": 192, "S": 2,    "D_node": 32},
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
    def __init__(self, B: int, S: int, D_node: int):
        super().__init__()
        assert B * S == N_IN, f"B*S must equal {N_IN}"
        self.B = B; self.S = S; self.D_node = D_node

        torch.manual_seed(SEED)

        # Project S-d block → D_node (or identity if S == D_node)
        if S == D_node:
            self.block_proj = nn.Identity()
        else:
            self.block_proj = nn.Linear(S, D_node, bias=False)
            nn.init.normal_(self.block_proj.weight, std=1.0 / (S ** 0.5))

        # Small-world connectivity over B nodes
        K_local = max(1, K_HH // 2); K_rand = K_HH - K_local
        conn = []
        for i in range(B):
            local = [(i + j) % B for j in range(1, K_local + 1)]
            g = torch.Generator(); g.manual_seed(SEED + i)
            rng = torch.randperm(B, generator=g)
            rng = [r.item() for r in rng if r.item() != i][:K_rand]
            conn.append(local + rng)
        self.register_buffer("conn_hh", torch.tensor(conn, dtype=torch.long))

        # Fourier W_pos on unit circle
        angles = torch.linspace(0, 2 * 3.14159, B + 1)[:-1]
        freqs  = torch.arange(1, D_node // 2 + 1).float()
        W = torch.cat([
            torch.sin(angles.unsqueeze(1) * freqs.unsqueeze(0)),
            torch.cos(angles.unsqueeze(1) * freqs.unsqueeze(0)),
        ], dim=1)[:, :D_node]
        self.W_pos = nn.Parameter(F.normalize(W, dim=-1))
        self.theta = nn.Parameter(torch.zeros(B))
        self.readout = nn.Linear(B * D_node, N_OUT)

    def forward(self, x):
        B_batch = x.size(0)
        B, D_node = self.B, self.D_node

        if self.S == 1:
            # B=384: each dim becomes its own node — project scalar to D_node
            blocks = x.unsqueeze(-1)  # (batch, 384, 1)
        else:
            blocks = x.view(B_batch, B, self.S)

        Z = self.block_proj(blocks)   # (batch, B, D_node)
        Z = F.normalize(Z, dim=-1)

        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.conn_hh
        W_h = self.W_pos
        dw  = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)

        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)

        return self.readout(Z.view(B_batch, -1))


def make_linear():
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
    print(f"step965 — ESC-50 finer subspace routing T0 (20ep, 50% data)")
    print(f"  Following step963 trend: more blocks = better.")
    print(f"  Linear: {LINEAR_BASELINE:.4f}  |  Subspace ref (step963 B96): {SUBSPACE_REF:.4f}")
    print(f"{'='*70}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}

    for key in keys:
        cfg = CONFIGS[key]
        print(f"{'─'*60}")

        if cfg["B"] is None:
            model = make_linear().to(DEVICE)
            n_p   = sum(p.numel() for p in model.parameters())
            print(f"{key}: Linear({N_IN},{N_OUT})  params={n_p:,}")
        else:
            B, S, D_node = cfg["B"], cfg["S"], cfg["D_node"]
            model = SubspaceRoutingModel(B, S, D_node).to(DEVICE)
            n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{key}: B={B} S={S} D_node={D_node}  params={n_p:,}")

        t0   = time.time()
        hist = train_model(model, tr, va)
        elapsed = time.time() - t0

        best    = max(hist)
        best_ep = int(np.argmax(hist)) + 1
        delta_ref    = best - SUBSPACE_REF
        delta_linear = best - LINEAR_BASELINE

        verdict = ("(linear control)" if key == "Linear"
                   else "(step963 ref)"   if key == "Ref_b96_s4"
                   else "EXCEEDS_LINEAR"  if delta_linear >= 0.0
                   else "APPROACHING"     if delta_linear >= -0.05
                   else "ADVANCE→T1"      if delta_ref >= 0.010
                   else "INTERESTING"     if delta_ref >= 0.005
                   else "NEUTRAL"         if delta_ref >= -0.005
                   else "KILL")

        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_ref={delta_ref*100:+.2f}pp  "
              f"Δ_linear={delta_linear*100:+.2f}pp  {verdict}")

        results[key] = {
            "B": cfg["B"], "S": cfg["S"], "D_node": cfg["D_node"],
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
    print(f"STEP 965 SUMMARY — ESC-50 finer subspace routing T0")
    print(f"  Linear: {LINEAR_BASELINE:.4f}  |  Subspace ref: {SUBSPACE_REF:.4f}")
    print(f"{'='*70}")
    for k, r in results.items():
        bs = f"B={r['B']},S={r['S']},D={r['D_node']}" if r["B"] else "full"
        print(f"  {k:<14} {bs:<16}  best={r['best']:.4f}  "
              f"Δ_ref={r['delta_vs_ref']*100:+.2f}pp  "
              f"Δ_lin={r['delta_vs_linear']*100:+.2f}pp  {r['verdict']}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
