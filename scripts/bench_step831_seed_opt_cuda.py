"""bench_step831: Validate spatial-precomputation speedup on CUDA.

MOTIVATION
==========
Spatial precomputation (merged into model_smallworld.py) showed:
  - CPU: 8.58× speedup
  - MPS: 8×–10× speedup (B=8–128)
  - MPS B=1: 0.87× (slight overhead from kernel launch)

This script validates the CUDA speedup on RTX 5060 Ti and measures
per-sample inference time for comparison against prior benchmarks.

Tests:
  1. Correctness: max_diff(old, new) < 1e-4
  2. Speed: old vs new seed at B=1,8,32,128 (CUDA synchronize timing)
  3. Full forward: SGNNET K=5 full forward at B=1,32,128

Expected result: >8× speedup at B=32+ on CUDA.

To run:
    python -u scripts/bench_step831_seed_opt_cuda.py --device cuda
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
DEVICE = torch.device(args.device)

N_IN, N_HIDDEN, D, K_IN = 25088, 2048, 16, 25
K_HH, K_ITER = 2, 5

def sync():
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    elif DEVICE.type == "mps": torch.mps.synchronize()

# ─────────────────────────────────────────────────────────────────────────────
# Isolated seed benchmark (old vs new)
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
rng = np.random.default_rng(42)
conn_in     = torch.from_numpy(rng.integers(0, N_IN, (N_HIDDEN, K_IN))).to(DEVICE)
spatial_coords = torch.randn(N_IN, D-1, device=DEVICE)
spatial_sum    = spatial_coords[conn_in].sum(dim=1)  # [N_HIDDEN, D-1]

def seed_old(x):
    B = x.shape[0]
    sp = spatial_coords.unsqueeze(0).expand(B, -1, -1)
    A = torch.cat([x.unsqueeze(-1), sp], dim=-1)
    return F.normalize(A[:, conn_in, :].sum(dim=2), dim=-1)

def seed_new(x):
    B = x.shape[0]
    xs = x[:, conn_in].sum(dim=2, keepdim=True)
    Z = torch.cat([xs, spatial_sum.unsqueeze(0).expand(B, -1, -1)], dim=-1)
    return F.normalize(Z, dim=-1)

# Correctness
x_test = torch.randn(4, N_IN, device=DEVICE)
diff = (seed_old(x_test) - seed_new(x_test)).abs().max().item()
print(f"Correctness check: max_diff = {diff:.2e}  {'✓ PASS' if diff < 1e-4 else '✗ FAIL'}")

# Speed
n_warmup, n_iter = 50, 200
print(f"\n{'B':>5}  {'old(ms)':>9}  {'new(ms)':>9}  {'speedup':>9}  {'mem_old(MB)':>12}  {'mem_new(MB)':>12}")
for B in [1, 8, 32, 128]:
    x = torch.randn(B, N_IN, device=DEVICE)
    # Warmup
    for _ in range(n_warmup): seed_old(x); sync()
    sync(); t0 = time.perf_counter()
    for _ in range(n_iter): seed_old(x); sync()
    sync(); t_old = (time.perf_counter()-t0)/n_iter*1000

    for _ in range(n_warmup): seed_new(x); sync()
    sync(); t0 = time.perf_counter()
    for _ in range(n_iter): seed_new(x); sync()
    sync(); t_new = (time.perf_counter()-t0)/n_iter*1000

    mem_old = (B*N_IN*D + B*N_HIDDEN*K_IN*D)*4/1e6
    mem_new = (B*N_HIDDEN*K_IN + B*N_HIDDEN*D)*4/1e6
    print(f"{B:>5}  {t_old:>9.3f}  {t_new:>9.3f}  {t_old/t_new:>9.2f}×  {mem_old:>12.1f}  {mem_new:>12.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# Full SGNNET forward benchmark
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n--- Full SGNNET forward (with optimized _seed) ---")
from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian

torch.manual_seed(42)
K_r = max(1, K_HH//4); K_l = K_HH - K_r
base = SGNNET_SmallWorld(N_hidden=N_HIDDEN, N_out=10, D=D, N_in=N_IN,
    K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
    n_groups=256, norm_mode="l2", encoding_mode="fourier")
resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=0.5, alpha_turing=0.0,
    beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
model = SGNNET_AntiHebbian(resonant, alpha_ahebb=1.0, variant="wpos").to(DEVICE).eval()

print(f"{'B':>5}  {'fwd(ms)':>9}  {'per_sample(ms)':>16}")
for B in [1, 8, 32, 128]:
    x = torch.randn(B, N_IN, device=DEVICE)
    for _ in range(n_warmup):
        with torch.no_grad(): model(x); sync()
    sync(); t0 = time.perf_counter()
    for _ in range(n_iter):
        with torch.no_grad(): model(x); sync()
    sync()
    t = (time.perf_counter()-t0)/n_iter*1000
    print(f"{B:>5}  {t:>9.3f}  {t/B:>16.4f}")

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nParams: {n_params:,}  K_iter={K_ITER}  K_in={K_IN}  device={DEVICE}")
