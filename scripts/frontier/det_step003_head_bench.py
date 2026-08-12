"""det_step003: does the compact detection head's PARAM win convert to WALL-TIME + MACs?

BRANCH B Pareto completion (DETECTION_TRAJECTORY B3). det_step002 proved a SGNNET-style
head reproduces FasterRCNN's box head at 11.7% params / 94.2% fg-agreement — but CLAUDE.md
mandates the FULL Pareto (accuracy + params + FLOPs + wall-time + memory), never params
alone. This project's KNOWN failure mode: tiny-param heads go dispatch-bound and LOSE on
wall-time. So the 88% param cut is meaningless for the drone goal until it is measured here.

HONEST CAVEAT baked into the metric: hard top-k masking does NOT shrink a dense matmul —
the up-proj still multiplies the full rank vector after masking. So MACs are governed by
RANK, not k; sgn_r128k32 MACs ~= a dense rank-128 head. The param win (fewer stored
weights) and the MAC/wall-time win (down-proj 12544xrank) are DIFFERENT axes — this bench
separates them. The box head runs once per ROI, ~1000 proposals/image at inference, so we
bench at ROI batch B in {64, 256, 1000}: small B exposes dispatch-bound flattening, B=1000
is the realistic per-image head workload.

Heads mirror det_step002 exactly (dense teacher-arch | lowrank_r256 | sgn_r128k32 |
sgn_r64k16). Output: results/frontier/det_step003_head_bench__{SLOT}.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--repeats", type=int, default=100)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--batches", default="64,256,1000")   # ROIs/forward; 1000 = 1 img
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SLOT = os.environ.get("SGN_SLOT", "local")
OUT = ROOT / "results" / "frontier" / f"det_step003_head_bench__{SLOT}.json"
D_IN, N_CLS, N_BB = 12544, 91, 364
CONFIGS = ["dense", "lowrank_r256", "sgn_r128k32", "sgn_r64k16"]


class DenseHead(nn.Module):        # teacher arch (TwoMLPHead + predictor), 12544->1024->1024
    def __init__(self, d_in, n_cls, n_bb, hid=1024):
        super().__init__()
        self.fc6 = nn.Linear(d_in, hid); self.fc7 = nn.Linear(hid, hid)
        self.cls = nn.Linear(hid, n_cls); self.bb = nn.Linear(hid, n_bb)

    def forward(self, x):
        h = F.relu(self.fc7(F.relu(self.fc6(x))))
        return self.cls(h), self.bb(h)


class LowRankHead(nn.Module):      # dense low-rank, no sparsity
    def __init__(self, d_in, n_cls, n_bb, rank):
        super().__init__()
        self.down = nn.Linear(d_in, rank); self.up = nn.Linear(rank, rank)
        self.cls = nn.Linear(rank, n_cls); self.bb = nn.Linear(rank, n_bb)

    def forward(self, x):
        h = F.gelu(self.up(F.gelu(self.down(x))))
        return self.cls(h), self.bb(h)


class SGNNETHead(nn.Module):       # low-rank + hard top-k (dense matmul; k cuts params not MACs)
    def __init__(self, d_in, n_cls, n_bb, rank, k):
        super().__init__()
        self.down = nn.Linear(d_in, rank); self.up = nn.Linear(rank, rank)
        self.cls = nn.Linear(rank, n_cls); self.bb = nn.Linear(rank, n_bb); self.k = k

    def forward(self, x):
        h = F.gelu(self.down(x))
        kth = h.topk(self.k, dim=-1).values[..., -1:]
        h = F.gelu(self.up(h * (h >= kth)))
        return self.cls(h), self.bb(h)


def build(name):
    if name == "dense":        return DenseHead(D_IN, N_CLS, N_BB)
    if name == "lowrank_r256": return LowRankHead(D_IN, N_CLS, N_BB, 256)
    if name == "sgn_r128k32":  return SGNNETHead(D_IN, N_CLS, N_BB, 128, 32)
    if name == "sgn_r64k16":   return SGNNETHead(D_IN, N_CLS, N_BB, 64, 16)
    raise ValueError(name)


def macs_per_roi(head):
    """Analytical MAC count = sum(in*out) over Linear layers (dense matmul; top-k unmasked)."""
    return sum(m.in_features * m.out_features for m in head.modules() if isinstance(m, nn.Linear))


def sync():
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    elif DEVICE.type == "mps": torch.mps.synchronize()


def peak_mem_mb():
    try:
        if DEVICE.type == "cuda": return torch.cuda.max_memory_allocated() / 1e6
        if DEVICE.type == "mps":  return torch.mps.current_allocated_memory() / 1e6
    except Exception:
        return None
    return None


def measure(name, head, batches):
    head = head.to(DEVICE).eval()
    n_p = sum(p.numel() for p in head.parameters()); macs = macs_per_roi(head)
    out = {"config": name, "params": n_p, "macs_per_roi": macs, "batch": {}}
    for B in batches:
        x = torch.randn(B, D_IN, device=DEVICE)
        with torch.no_grad():
            for _ in range(args.warmup): head(x)
        sync()
        if DEVICE.type == "cuda": torch.cuda.reset_peak_memory_stats()
        times = []
        with torch.no_grad():
            for _ in range(args.repeats):
                sync(); t0 = time.perf_counter(); head(x); sync()
                times.append(time.perf_counter() - t0)
        med_ms = float(np.median(times) * 1000)
        out["batch"][str(B)] = {"median_ms": round(med_ms, 4),
                                "per_roi_us": round(med_ms * 1000 / B, 3),
                                "peak_mem_mb": peak_mem_mb()}
        print(f"    B={B:>4} med={med_ms:8.4f}ms  per_roi={med_ms*1000/B:7.3f}us", flush=True)
    return out


def main():
    batches = [int(b) for b in args.batches.split(",")]
    if args.smoke_test:
        for nm in CONFIGS:
            h = build(nm); c, b = h(torch.randn(4, D_IN))
            print(f"  {nm:<13} {sum(p.numel() for p in h.parameters()):>9,}p  "
                  f"macs/roi={macs_per_roi(h):>12,}  cls={tuple(c.shape)} bb={tuple(b.shape)}")
        sys.exit(0)

    print(f"{'='*66}\ndet_step003 head bench  device={DEVICE}  batches={batches}")
    res = {"step": "det_step003", "device": str(DEVICE), "repeats": args.repeats,
           "batches": batches, "configs": []}
    for nm in CONFIGS:
        print(f"  {nm}")
        res["configs"].append(measure(nm, build(nm), batches))
        OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))

    dense = res["configs"][0]
    print(f"\n  {'config':<13} {'params':>11} {'macs/roi':>13} {'B1000 us/roi':>13} "
          f"{'wall×':>7} {'param×':>7} {'mac×':>7}")
    for r in res["configs"]:
        w1k = r["batch"][str(batches[-1])]["per_roi_us"]
        dw = dense["batch"][str(batches[-1])]["per_roi_us"]
        print(f"  {r['config']:<13} {r['params']:>11,} {r['macs_per_roi']:>13,} "
              f"{w1k:>13.3f} {dw/w1k:>6.2f}x {dense['params']/r['params']:>6.2f}x "
              f"{dense['macs_per_roi']/r['macs_per_roi']:>6.2f}x")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
