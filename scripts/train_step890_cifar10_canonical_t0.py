"""Step 890: CIFAR-10 cross-dataset T0 with canonical ΔW-proj config.

MOTIVATION
==========
step862 failed: script auto-scaled to N=1024/D=8 (9,296 params) because it
assumed 512-dim features. Actual CIFAR-10 store has 25088-dim VGG16 pool5
features (same as Imagenette). SGNNET needs canonical N=2048/D=16 config.

This test validates paper cross-dataset claim:
  "SGNNET ΔW-proj (34,976 params) generalizes beyond Imagenette to CIFAR-10"
  Expected: SGNNET competitive with Linear/MLP at matched FLOPs.

DATA: data/store_cifar10.h5 — VGG16 pool5 features, N_in=25088, 10 classes.

CONFIGS (T0: 20ep, 50% data, seed=42)
  Linear     : 25088 → 10 (250,890 params — for accuracy reference)
  Ref_SGNNET : canonical ΔW-proj N=2048 D=16 K_in=25 (34,976 params)

SUCCESS: SGNNET within -5pp of Linear → advance to T1/T2 (paper cross-dataset)
KILL:    SGNNET > -10pp → architecture fails to generalize
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
import torch.utils.data

from src.sgnnet.model_smallworld  import SGNNET_SmallWorld
from src.sgnnet.model_resonant    import SGNNET_Resonant

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_cifar10.h5")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_OUT = 10; D = 16; K_IN = 25; K_ITER = 5; K_HH = 2
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step890_cifar10_canonical_t0_seed{SEED}__{SLOT}.json"


class H5FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, path, split):
        with h5py.File(path, "r") as f:
            keys = list(f[split].keys())
            feat_key = "features" if "features" in keys else keys[0]
            lbl_key  = "labels"   if "labels"   in keys else keys[1]
            self.X = torch.from_numpy(f[f"{split}/{feat_key}"][:]).float()
            self.y = torch.from_numpy(f[f"{split}/{lbl_key}"][:]).long()
            if self.X.dim() > 2:
                self.X = self.X.view(len(self.X), -1)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class SGNNET_DeltaW(nn.Module):
    def __init__(self, resonant):
        super().__init__()
        self.m = resonant

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x):
        Z = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh = self.m.base.conn_hh
        W_h = self.m.W_pos[:self.m.base.N_hidden]
        dw = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_nb = Z_nb * proj_coeff.abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def make_sgnnet(n_in):
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=n_in,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    )
    res = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
                           beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_DeltaW(res)


def train_eval(model, tr, va, epochs, is_sgnnet=False):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    history = []
    for epoch in range(epochs):
        model.train()
        if is_sgnnet and hasattr(model, "tick_epoch"):
            model.tick_epoch()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(-1) == y).sum().item()
                total += y.numel()
        val_top1 = correct / max(total, 1)
        history.append(val_top1)
        if (epoch + 1) % 5 == 0:
            print(f"  ep{epoch+1:3d}  val={val_top1:.4f}", flush=True)
    return history


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found"); sys.exit(1)

    tr_full = H5FeatureDataset(str(data_path), "train")
    va_ds   = H5FeatureDataset(str(data_path), "val")
    N_IN = tr_full.X.shape[-1]
    print(f"CIFAR-10 features: N_in={N_IN}  train={len(tr_full)}  val={len(va_ds)}")
    assert N_IN == 25088, f"Expected N_in=25088, got {N_IN}"

    n_full = len(tr_full)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0)
    va = torch.utils.data.DataLoader(va_ds, batch_size=BATCH, shuffle=False)

    sgnnet = make_sgnnet(N_IN)
    n_p_sgnnet = sum(p.numel() for p in sgnnet.parameters() if p.requires_grad)
    canonical = n_p_sgnnet == 34976
    print(f"SGNNET params={n_p_sgnnet:,} {'(CANONICAL ✓)' if canonical else '(NON-CANONICAL)'}")

    linear = nn.Linear(N_IN, N_OUT)
    n_p_linear = sum(p.numel() for p in linear.parameters() if p.requires_grad)

    CONFIGS = [
        ("Linear",     linear,  n_p_linear,  False),
        ("Ref_SGNNET", sgnnet,  n_p_sgnnet,  True),
    ]

    print(f"\n{'='*70}")
    print(f"step890 — CIFAR-10 cross-dataset T0 (canonical ΔW-proj, 20ep, 50% data)")
    print(f"  device={DEVICE}  N_in={N_IN}  SGNNET N={N} D={D} K_in={K_IN}")
    print(f"{'='*70}\n")

    results = {}; linear_acc = None

    for key, model, n_p, is_sg in CONFIGS:
        print(f"{'─'*60}\n{key}: params={n_p:,}")
        t0 = time.time()
        history = train_eval(model, tr, va, EPOCHS, is_sgnnet=is_sg)
        elapsed = time.time() - t0
        best, best_ep = max(history), int(np.argmax(history)) + 1
        if key == "Linear": linear_acc = best
        delta = best - (linear_acc or 0.86)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Linear={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {"n_params": n_p, "best": best, "best_ep": best_ep,
                        "delta_vs_linear": round(delta, 4), "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    sgnnet_r = results.get("Ref_SGNNET", {})
    print(f"\n{'='*70}")
    print(f"STEP 890 SUMMARY — CIFAR-10 cross-dataset T0 (canonical)")
    print(f"{'='*70}")
    for k, r in results.items():
        print(f"  {k:<14} {r['n_params']:>8,}  {r['best']:>7.4f}  {r['delta_vs_linear']*100:>+9.2f}pp")
    if sgnnet_r:
        gap = sgnnet_r["delta_vs_linear"] * 100
        verdict = "VIABLE — advance to T1" if gap >= -5 else ("MARGINAL — T1 needed" if gap >= -10 else "KILLED — architecture fails cross-dataset")
        print(f"\n  SGNNET vs Linear: {gap:+.2f}pp → {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
