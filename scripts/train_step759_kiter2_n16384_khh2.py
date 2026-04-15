"""Step 759: K_iter=2 at N=16384 K_hh=2 T1 + inline inference latency benchmark.

K_hh=2 FIXED (vs step754 which uses K_hh=round(N/1000)=16).
PURPOSE: test whether K_hh=2 is universally good (efficiency config) or needs to
scale with N. Paired comparison with step754 (same N, K_iter; only K_hh differs).

At N=16384, K_hh=2 is a very sparse graph (degree 2 vs 16). If accuracy holds,
this is the strongest evidence that K_hh=2 is universal — the topology density
doesn't need to scale with graph size.

Existing curve:
  N=16384 K_hh=16 K_iter=2: step754 (K_hh=round(N/1000) rule)
  N=16384 K_hh=2  K_iter=2: THIS STEP

Note: batch=64 for N=16384 (VRAM headroom; matches step754).

Inline benchmark (after training):
  - torch.compile V1 (mode="reduce-overhead")
  - batch=32 inference, 20 warmup + 100 timed iters, report median ms
  - Compare to VGG16 FC reference (~0.07ms @ bs=32 on 5060ti, step500)

Tier-1 (75ep, 50% data). Device: 5060ti_cuda.
Threshold: ≥96%.
"""
from __future__ import annotations
import argparse, json, sys, time, statistics
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset import make_loaders

parser = argparse.ArgumentParser(
    description="Step 759: K_iter=2 @ N=16384 K_hh=2 T1 + inline latency bench"
)
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75)
parser.add_argument("--skip_bench", action="store_true")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
EPOCHS = args.epochs
BATCH = 64; SEED = 42; DATA = "data/store.h5"  # batch=64: N=16384 is large
N = 16384; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 2   # K_hh=2 FIXED (efficiency config value)
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0
FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step759_kiter2_n16384_khh2.json"


def _build(seed):
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    sw = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                           K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                           n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(sw, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING, mode="dynamic_z_geo")
    return SGNNET_AntiHebbian(res, alpha_ahebb=1.0, variant="wpos")


def bench_inference(model, device, bench_bs=32, n_warm=20, n_iter=100):
    """Measure median inference latency with torch.compile V1 (reduce-overhead)."""
    model.eval()
    try:
        model_c = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print(f"  compile failed: {e}; falling back to eager")
        model_c = model
    x = torch.randn(bench_bs, N_IN, device=device)
    with torch.no_grad():
        for _ in range(n_warm):
            _ = model_c(x)
        if device.type == "cuda": torch.cuda.synchronize()
        times_ms = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = model_c(x)
            if device.type == "cuda": torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": statistics.median(times_ms),
            "p90_ms": sorted(times_ms)[int(n_iter*0.9)],
            "min_ms": min(times_ms), "n_iter": n_iter, "batch": bench_bs}


def main():
    print(f"\n{'='*70}")
    print(f"Step 759 — K_iter=2 @ N=16384 K_hh=2 (fixed) T1 + inline latency")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} FLOPs={FLOPS/1e6:.3f}M  Device={DEVICE}")
    print(f"K_hh=2 FIXED. Paired with step754 (K_hh=16). Threshold: ≥96%")
    print(f"{'='*70}")
    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))[:n_total // 2]
    tr = torch.utils.data.DataLoader(torch.utils.data.Subset(tr_full.dataset, idx.tolist()),
                                     batch_size=BATCH, shuffle=True, num_workers=0,
                                     generator=torch.Generator().manual_seed(SEED))
    model = _build(SEED).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params={n_p:,}")
    t0 = time.time()
    history = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **trainer_kwargs(N, n_epochs=EPOCHS)).train(
        n_epochs=EPOCHS,
        log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
                         if (m['epoch'] + 1) % 5 == 0 else None)
    top1h = [round(h.get("val_top1", 0.), 4) for h in history]
    best = max(top1h); bep = int(np.argmax(top1h)) + 1
    elapsed = round(time.time()-t0, 1)
    print(f"\nbest={best:.4f} @ep{bep}  train_elapsed={elapsed}s")

    bench = None
    if not args.skip_bench and DEVICE.type == "cuda":
        print(f"\n{'─'*50}\nInference latency bench (torch.compile V1, bs=32)\n{'─'*50}")
        try:
            bench = bench_inference(model, DEVICE)
            print(f"  median={bench['median_ms']:.3f}ms  p90={bench['p90_ms']:.3f}ms  min={bench['min_ms']:.3f}ms")
            print(f"  VGG16 FC ref: ~0.07ms @bs=32 on 5060ti (step500)")
        except Exception as e:
            print(f"  bench failed: {e}")
            bench = {"error": str(e)}

    results = {"best": {"top1_best": best, "best_epoch": bep, "top1_history": top1h,
                        "elapsed_s": elapsed, "n_params": n_p,
                        "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER, "flops": FLOPS},
               "latency": bench}
    verdict = "≥96% VIABLE" if best >= 0.96 else ("borderline 94-96%" if best >= 0.94 else "<94% fail")
    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}  ({best:.4f})")
    if best >= 0.96:
        print(f"  → K_hh=2 viable at N=16384 K_iter=2. Strongest K_hh=2-universal evidence.")
        print(f"  → Compare latency to step754 (K_hh=16) for FLOPs-accuracy Pareto plot.")
    else:
        print(f"  → Below threshold. Delta vs step754 determines K_hh sensitivity at large N.")
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_PATH}")

if __name__ == "__main__":
    main()
