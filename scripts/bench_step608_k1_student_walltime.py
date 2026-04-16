"""bench_step608: K=1 student wall-time + memory footprint benchmark.

MOTIVATION
==========
step605/607 validated K=1 KD student accuracy (~95.9% at -0.76pp vs K=5 teacher).
Now measure the compound efficiency story:

| Dimension         | VGG FC   | SGNNET K=5 | SGNNET K=1 student | MLP_64  | Linear  |
|-------------------|----------|------------|--------------------|---------|---------|
| Accuracy          | 95.00%   | 95.52%     | 95.95%             | 94.00%* | 92.00%* |
| Params            | 123M     | 34,976     | 34,976             | 160K    | 250K    |
| FLOPs per sample  | 123M     | 0.98M      | ~0.20M             | 160K    | 250K    |
| Wall-time B=32    | ?        | ?          | ?                  | ?       | ?       |
| Peak GPU mem      | ?        | ?          | ?                  | ?       | ?       |

This script measures Wall-time + Peak memory at multiple batch sizes (B=1, 32, 128).
FLOPs are analytical; accuracy comes from prior training steps.

Usage: bash scripts/launch_slot.sh 5060ti_cuda scripts/bench_step608_k1_student_walltime.py
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--repeats", type=int, default=100)
parser.add_argument("--warmup",  type=int, default=20)
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"bench_step608_walltime__{SLOT}.json"


def build_sgnnet_base(k_iter):
    """Build base SGNNET with AntiHebb wrapper — used for K=5 teacher-style inference."""
    torch.manual_seed(42)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=k_iter, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


class SGNNET_K1_DeltaProj(nn.Module):
    """K=1 ΔW projection student matching step605/607 architecture."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        K_r = max(1, K_HH // 4); K_l = K_HH - K_r
        self.base = SGNNET_SmallWorld(
            N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
            K_in=K_IN, K_iter=1, K_local=K_l, K_random=K_r,
            n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
        self.resonant = SGNNET_Resonant(
            self.base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
            beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h = self.base.W_pos[:self.base.N_hidden]
        delta_w = W_h.unsqueeze(1) - W_h[conn_hh]
        dw_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)
        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            proj = (Z_nb * dw_norm).sum(-1, keepdim=True)
            Z_nb = Z_nb * proj.abs()
            Z_struct = Z_nb.sum(dim=2)
            Z_reflected = ALPHA_REFLECT * Z_reflected + (Z_fwd - Z)
            Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)
        return self.base._readout(Z)


def build_mlp_64():
    return nn.Sequential(nn.Linear(N_IN, 64), nn.ReLU(), nn.Linear(64, N_OUT)).eval()


def build_linear():
    return nn.Linear(N_IN, N_OUT).eval()


def build_vgg_fc_ref():
    """VGG16 FC reference: 3 FC layers matching the original VGG16 head."""
    return nn.Sequential(
        nn.Linear(N_IN, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, N_OUT)).eval()


def measure(name, model, batch_sizes):
    model = model.to(DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{name}: {n_params:,} params")
    results = {"name": name, "n_params": n_params, "batch_results": {}}
    for B in batch_sizes:
        x = torch.randn(B, N_IN, device=DEVICE)
        # Warmup
        for _ in range(args.warmup):
            with torch.no_grad(): _ = model(x)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        # Peak memory tracker
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()
        # Timing
        times = []
        for _ in range(args.repeats):
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(): _ = model(x)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        median_ms = float(np.median(times) * 1000)
        p95_ms = float(np.percentile(times, 95) * 1000)
        per_sample_us = median_ms * 1000 / B
        mem_mb = None
        if DEVICE.type == "cuda":
            mem_mb = (torch.cuda.max_memory_allocated() - mem_before) / 1e6
        print(f"  B={B:4d}  median={median_ms:7.3f}ms  per-sample={per_sample_us:7.2f}us  p95={p95_ms:7.3f}ms" +
              (f"  peak_mem={mem_mb:.1f}MB" if mem_mb is not None else ""))
        results["batch_results"][str(B)] = {
            "median_ms": median_ms, "p95_ms": p95_ms,
            "per_sample_us": per_sample_us, "peak_mem_mb": mem_mb,
        }
    return results


def main():
    print(f"bench_step608: K=1 student wall-time + memory benchmark")
    print(f"  device={DEVICE}  repeats={args.repeats}  warmup={args.warmup}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    batch_sizes = [1, 32, 128]

    models = [
        ("Linear",           build_linear()),
        ("MLP_64",           build_mlp_64()),
        ("VGG_FC (ref)",     build_vgg_fc_ref()),
        ("SGNNET K=5",       build_sgnnet_base(k_iter=5)),
        ("SGNNET K=1 student", SGNNET_K1_DeltaProj()),
    ]

    all_results = []
    for name, m in models:
        try:
            r = measure(name, m, batch_sizes)
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR on {name}: {e}")
            all_results.append({"name": name, "error": str(e)})

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(all_results, indent=2))

    # Summary table
    print(f"\n{'='*80}\nSUMMARY (per-sample wall-time)")
    print(f"{'Model':<22}  {'Params':>10}  " + "  ".join(f"B={B}".rjust(12) for B in batch_sizes))
    for r in all_results:
        if "error" in r:
            print(f"{r['name']:<22}  ERROR: {r['error'][:40]}")
            continue
        row = f"{r['name']:<22}  {r['n_params']:>10,}  "
        for B in batch_sizes:
            us = r["batch_results"][str(B)]["per_sample_us"]
            row += f"{us:8.1f}us  "
        print(row)
    print(f"\nWritten: {OUT_PATH}")


if __name__ == "__main__":
    main()
