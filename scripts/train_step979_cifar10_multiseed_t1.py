"""Step 979: CIFAR-10 multi-seed T1 — paper-grade variance estimate.

# CUDA-5060ti-validated

MOTIVATION
==========
step882 (T2, single seed42): SGNNET=80.69%, Linear=86.24%, gap=−5.55pp.
That's a single-seed T2 result. For the paper, we need:
  (a) Variance estimate for the gap (±pp)
  (b) Confirmation gap is reproducible across seeds
  (c) Match step887 protocol: multi-seed T1 → report mean ± std

This gives a paper-grade "SGNNET achieves X.XX% ± Y.YYpp on CIFAR-10 vs
Linear Z.ZZ% (gap −W.WWpp)" analogous to step887 for Imagenette.

PROTOCOL
========
  5 seeds × 1 config: canonical SGNNET ΔW-proj (same arch as step882)
  75ep T1, 50% data (25K samples), BATCH=512, 5060ti CUDA
  Data: data/store_cifar10.h5 (50K train, 10K val, 25088-dim VGG16 features)

REFERENCE
=========
  step882 T2 seed42: SGNNET=80.69%, Linear=86.24%, gap=−5.55pp
  step887 Imagenette multi-seed T2: SGNNET=96.38% ± 0.18pp
  step909 N=4096 T2: 82.53%
  step922 N=8192 T2 multi-seed: 83.55% ± 0.014pp
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import h5py

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant   import SGNNET_Resonant
# SGNNET_Resonant_CUDA / SGNNET_AntiHebbian_CUDA: DeltaW loop compiled via torch.compile below

DEVICE    = torch.device("cuda")
EPOCHS    = 75
BATCH     = 512   # pin_memory=True, non_blocking=True — CUDA throughput pattern
DATA_FRAC = 0.5
SEEDS     = [0, 1, 42, 123, 2024]

# Canonical config — matches step882 / step916
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

DATA_PATH = ROOT / "data" / "store_cifar10.h5"
OUT_PATH  = ROOT / "results" / "train_step979_cifar10_multiseed_t1__5060ti_cuda.json"

STEP882_SGNNET = 0.8069   # single-seed T2 reference
STEP882_LINEAR = 0.8624


# ── DeltaW model (exact match to step882 / step916) ──────────────────────────

def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)

def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


class DeltaW(torch.nn.Module):
    def __init__(self, resonant):
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


def build_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    model = DeltaW(resonant).to(DEVICE)
    return torch.compile(model)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(seed: int):
    with h5py.File(DATA_PATH, "r") as f:
        tr_x  = torch.tensor(f["train/features"][:],    dtype=torch.float32)
        tr_y  = torch.tensor(f["train/labels"][:],      dtype=torch.long)
        tr_sf = torch.tensor(f["train/soft_labels"][:], dtype=torch.float32)
        va_x  = torch.tensor(f["val/features"][:],      dtype=torch.float32).to(DEVICE)
        va_y  = torch.tensor(f["val/labels"][:],        dtype=torch.long).to(DEVICE)
    n = tr_x.shape[0]
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:int(n * DATA_FRAC)]
    tr_x = tr_x[idx]; tr_y = tr_y[idx]; tr_sf = tr_sf[idx]
    ds = TensorDataset(tr_x, tr_y, tr_sf)
    tr = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0,
                    pin_memory=True, generator=torch.Generator().manual_seed(seed))
    return tr, va_x, va_y


# ── Training ──────────────────────────────────────────────────────────────────

def train_one(seed: int) -> dict:
    tr, va_x, va_y = load_data(seed)
    model = build_model(seed)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=EPOCHS * len(tr),
        pct_start=0.1, anneal_strategy="cos",
    )
    best = 0.0; best_ep = 0
    t0   = time.time()

    for ep in range(EPOCHS):
        model.train()
        for bx, by, _bsf in tr:
            bx = bx.to(DEVICE, non_blocking=True)
            by = by.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            correct = total = 0
            for i in range(0, va_x.shape[0], BATCH):
                s = model(va_x[i:i+BATCH])
                correct += (s.argmax(1) == va_y[i:i+BATCH]).sum().item()
                total   += s.shape[0]
        acc = correct / total
        if acc > best:
            best = acc; best_ep = ep + 1
        if (ep + 1) % 15 == 0 or ep == 0:
            print(f"    seed={seed} ep{ep+1:2d}/{EPOCHS} val={acc:.4f} best={best:.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    elapsed = time.time() - t0
    print(f"  DONE seed={seed}: best={best:.4f} @ep{best_ep}  {elapsed:.0f}s  params={n_params}", flush=True)
    return {"seed": seed, "top1_best": best, "best_epoch": best_ep,
            "n_params": n_params, "elapsed_s": round(elapsed)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found."); sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Step 979 — CIFAR-10 multi-seed T1 (paper-grade variance)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seeds={SEEDS}  data={DATA_FRAC:.0%}")
    print(f"  N={N} D={D} K_hh={K_HH} K_iter={K_ITER} K_in={K_IN} BATCH={BATCH}")
    print(f"  Ref (step882): SGNNET={STEP882_SGNNET:.4f}  Linear={STEP882_LINEAR:.4f}")
    print(f"{'='*70}\n")

    results = []
    for seed in SEEDS:
        print(f"\n── seed={seed} ──")
        r = train_one(seed)
        results.append(r)
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    accs = [r["top1_best"] for r in results]
    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))
    gap      = mean_acc - STEP882_LINEAR

    print(f"\n{'='*70}")
    print(f"STEP 979 SUMMARY — CIFAR-10 multi-seed T1")
    print(f"{'='*70}")
    print(f"{'seed':>6} | {'top1':>8} | {'gap vs Linear':>14}")
    for r in results:
        g = (r['top1_best'] - STEP882_LINEAR) * 100
        print(f"{r['seed']:>6} | {r['top1_best']:>8.4f} | {g:>+13.2f}pp")
    print(f"\n  SGNNET CIFAR-10 T1: {mean_acc:.4f} ± {std_acc:.4f} ({std_acc*100:.2f}pp)")
    print(f"  Linear (step882):   {STEP882_LINEAR:.4f}")
    print(f"  Gap: {gap*100:+.2f}pp")
    print(f"  Ref single-seed (step882 T2 seed42): {STEP882_SGNNET:.4f}")
    print(f"\n→ {OUT_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
