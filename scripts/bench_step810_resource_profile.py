"""Step 810: Full resource profile — wall-clock, GPU util, memory, across all promising variants.

MOTIVATION
==========
Gap: SGNNET has 116× FLOPs advantage but 4.3× wall-clock SLOWER than VGG FC. Memory-bandwidth bound.
User directive (2026-04-14): find a dimension where SGNNET clearly wins on wall-clock + resource usage.

Measures, for each variant, on 5060ti CUDA:
  - Inference latency: median, p95, p99 at bs=32
  - Training step time (fwd+bwd+opt)
  - GPU peak memory (torch.cuda.max_memory_allocated)
  - GPU memory reserved (torch.cuda.max_memory_reserved)
  - Parameter count
  - FLOPs (analytical, not ncu)
  - Throughput (samples/sec)
  - Works both with and without torch.compile(reduce-overhead)

VARIANTS PROFILED (all at bs=32 inference, bs=128 training):
  1. VGG_FC          : Standard FC head 25088→4096→4096→10 (123M params, 123M MACs baseline)
  2. VGG_FC_small    : Pruned FC 25088→512→10 (~12.9M params, ~12.9M MACs)
  3. Linear          : Single 25088→10 (250K params, 250K MACs)
  4. MLP_64          : 25088→64→10 (~1.6M params, 1.6M MACs)
  5. SGNNET_AH       : step199 efficiency config (67K params, 0.98M routing MACs)
  6. SGNNET_DeltaAH  : step706 ΔW proj (67K + 1 param, 0.98M routing MACs)

OUTPUT
======
results/bench_step810_resource_profile.json — structured comparison
Summary table printed to stdout with all dimensions for paper table.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--iters",  type=int, default=200, help="timed iterations per config")
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--bs_inf", type=int, default=32)
parser.add_argument("--bs_train", type=int, default=128)
parser.add_argument("--configs", default="", help="Comma-separated config keys. Empty = all.")
parser.add_argument("--compile", action="store_true", help="run with and without torch.compile")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}  SM={torch.cuda.get_device_capability(0)}")

N_IN = 25088
N_OUT = 10
SEED = 42

OUT_PATH = ROOT / "results" / "bench_step810_resource_profile.json"


# ─────────────────────────────────────────────────────────────────────────────
# Model factories
# ─────────────────────────────────────────────────────────────────────────────

class VGG_FC(nn.Module):
    """Standard VGG16 FC head: 25088→4096→4096→N_OUT. Matches original VGG16 classifier."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, N_OUT),
        )
    def forward(self, x): return self.net(x)


class VGG_FC_small(nn.Module):
    def __init__(self, h=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN, h), nn.ReLU(inplace=True),
            nn.Linear(h, N_OUT),
        )
    def forward(self, x): return self.net(x)


class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(N_IN, N_OUT)
    def forward(self, x): return self.fc(x)


class MLP_64(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN, h), nn.ReLU(inplace=True),
            nn.Linear(h, N_OUT),
        )
    def forward(self, x): return self.net(x)


def make_sgnnet_ah():
    torch.manual_seed(SEED)
    N = 2048; D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(base, K_phase=8, alpha_reflect=0.5,
                          alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
                          mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(res, alpha_ahebb=1.0, variant="wpos")


MODEL_REGISTRY = {
    "VGG_FC":       lambda: VGG_FC(),
    "VGG_FC_small": lambda: VGG_FC_small(h=512),
    "Linear":       lambda: LinearProbe(),
    "MLP_64":       lambda: MLP_64(h=64),
    "SGNNET_AH":    lambda: make_sgnnet_ah(),
}


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark core
# ─────────────────────────────────────────────────────────────────────────────

def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def bench_inference(model: nn.Module, bs: int, warmup: int, iters: int) -> dict:
    """Returns {median_ms, p95_ms, p99_ms, mean_ms, throughput_sps}."""
    model.eval()
    x = torch.randn(bs, N_IN, device=DEVICE)
    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        _sync()
    # timed
    times = []
    with torch.no_grad():
        for _ in range(iters):
            _sync()
            t0 = time.perf_counter()
            _ = model(x)
            _sync()
            times.append((time.perf_counter() - t0) * 1000.0)
    times = np.asarray(times)
    return {
        "median_ms":      float(np.median(times)),
        "p95_ms":         float(np.percentile(times, 95)),
        "p99_ms":         float(np.percentile(times, 99)),
        "mean_ms":        float(np.mean(times)),
        "throughput_sps": float(bs / (np.median(times) / 1000.0)),
    }


def bench_training_step(model: nn.Module, bs: int, warmup: int, iters: int) -> dict:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(bs, N_IN, device=DEVICE)
    y = torch.randint(0, N_OUT, (bs,), device=DEVICE)
    # warmup
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()
    _sync()
    # timed
    times = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()
        _sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.asarray(times)
    return {
        "median_ms":      float(np.median(times)),
        "throughput_sps": float(bs / (np.median(times) / 1000.0)),
    }


def measure_memory(model: nn.Module, bs: int) -> dict:
    if DEVICE.type != "cuda":
        return {"peak_mib": -1.0, "reserved_mib": -1.0}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(bs, N_IN, device=DEVICE)
    y = torch.randint(0, N_OUT, (bs,), device=DEVICE)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()
    _sync()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    reserved = torch.cuda.max_memory_reserved() / 1024**2
    return {"peak_mib": float(peak), "reserved_mib": float(reserved)}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_all(keys, compile_on: bool) -> dict:
    out = {}
    for key in keys:
        print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  {key}   compile={compile_on}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        model = MODEL_REGISTRY[key]().to(DEVICE)
        n_params = count_params(model)
        print(f"  params: {n_params:,}")

        if compile_on and DEVICE.type == "cuda":
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
            print("  torch.compile(reduce-overhead) applied")

        # Memory (reset first)
        mem = measure_memory(model, bs=args.bs_train)
        print(f"  peak mem (training bs={args.bs_train}): {mem['peak_mib']:.1f} MiB (reserved {mem['reserved_mib']:.1f})")

        # Inference
        inf = bench_inference(model, bs=args.bs_inf, warmup=args.warmup, iters=args.iters)
        print(f"  INF bs={args.bs_inf}: {inf['median_ms']:.3f}ms  p95={inf['p95_ms']:.3f}  p99={inf['p99_ms']:.3f}  → {inf['throughput_sps']:.0f} sps")

        # Training
        tr = bench_training_step(model, bs=args.bs_train, warmup=args.warmup // 2,
                                 iters=max(30, args.iters // 4))
        print(f"  TRAIN bs={args.bs_train}: {tr['median_ms']:.3f}ms → {tr['throughput_sps']:.0f} sps")

        out[key] = {
            "n_params":  n_params,
            "inference": inf,
            "training":  tr,
            "memory":    mem,
            "compile":   compile_on,
        }

        # Free memory
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return out


def main():
    all_keys = list(MODEL_REGISTRY.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",") if k.strip() in all_keys]
    else:
        run_keys = all_keys

    print("Variants:", run_keys)
    print(f"iters={args.iters}  warmup={args.warmup}  bs_inf={args.bs_inf}  bs_train={args.bs_train}")

    results = {"device": str(DEVICE), "bs_inf": args.bs_inf, "bs_train": args.bs_train}

    # Pass 1: eager
    print("\n\n========== PASS 1: EAGER ==========")
    results["eager"] = run_all(run_keys, compile_on=False)

    # Pass 2: compiled (CUDA only)
    if args.compile and DEVICE.type == "cuda":
        print("\n\n========== PASS 2: torch.compile ==========")
        results["compiled"] = run_all(run_keys, compile_on=True)

    # Summary table
    print("\n\n========== SUMMARY ==========")
    print(f"{'Config':<18} {'params':>10} {'inf_ms':>8} {'tr_ms':>8} {'mem_MiB':>9} {'sps_inf':>9}")
    mode = "compiled" if (args.compile and DEVICE.type == "cuda") else "eager"
    for k in run_keys:
        r = results[mode][k]
        print(f"{k:<18} {r['n_params']:>10,} {r['inference']['median_ms']:>8.3f} "
              f"{r['training']['median_ms']:>8.3f} {r['memory']['peak_mib']:>9.1f} "
              f"{r['inference']['throughput_sps']:>9.0f}")

    # vs VGG_FC ratios
    if "VGG_FC" in run_keys:
        vgg_inf = results[mode]["VGG_FC"]["inference"]["median_ms"]
        vgg_mem = results[mode]["VGG_FC"]["memory"]["peak_mib"]
        vgg_params = results[mode]["VGG_FC"]["n_params"]
        print(f"\n{'Config':<18} {'vs_VGG_inf':>12} {'vs_VGG_params':>14} {'vs_VGG_mem':>12}")
        for k in run_keys:
            r = results[mode][k]
            inf_ratio = vgg_inf / r["inference"]["median_ms"]
            p_ratio   = vgg_params / max(r["n_params"], 1)
            m_ratio   = vgg_mem / max(r["memory"]["peak_mib"], 0.01)
            print(f"{k:<18} {inf_ratio:>11.2f}× {p_ratio:>13.1f}× {m_ratio:>11.2f}×")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
