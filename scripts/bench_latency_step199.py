"""bench_latency_step199: Standalone inference-latency benchmarker for step199 config.

Back-fills the latency number for step199 (N=2048 K_hh=2 K_iter=5 D=16, AH-only)
for the Pareto plot.

Protocol:
  - Build fresh step199 config (AH-only, no ΔW)
  - Train briefly (default 5 epochs) to warm the model's learned state
  - torch.compile V1 (mode="reduce-overhead")
  - Inference: bs=32, 20 warmup + 100 timed iters
  - Report median / p90 / min ms

Device: 5060ti_cuda preferred (to compare against VGG16 FC ~0.07ms ref).
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
    description="bench_latency_step199: inference latency for step199 config (AH-only, N=2048)"
)
parser.add_argument("--device",       default="auto")
parser.add_argument("--train_epochs", type=int, default=5,
                    help="Short warm-up training before benchmark (default: 5)")
parser.add_argument("--bench_bs",     type=int, default=32,
                    help="Batch size for latency benchmark (default: 32)")
parser.add_argument("--n_warm",       type=int, default=20,
                    help="Warmup iterations (default: 20)")
parser.add_argument("--n_iter",       type=int, default=100,
                    help="Timed iterations (default: 100)")
parser.add_argument("--no_compile",   action="store_true",
                    help="Skip torch.compile, use eager mode")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

# step199 config
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
SEED = 42; BATCH = 128; DATA = "data/store.h5"
FLOPS = 3 * N * K_HH * D * K_ITER
STEP_NAME = "step199_ah_only"
OUT_PATH = ROOT / "results" / "bench_latency_step199.json"


def _build(seed):
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    sw = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                           K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                           n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(sw, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
                          mode="dynamic_z_geo")
    return SGNNET_AntiHebbian(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def bench_inference(model, device, bench_bs, n_warm, n_iter, no_compile):
    """Measure inference latency with optional torch.compile V1."""
    model.eval()
    compile_mode = "eager" if no_compile else "compiled (reduce-overhead)"
    if not no_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"  torch.compile failed: {e}; falling back to eager")
            compile_mode = "eager (fallback)"
    x = torch.randn(bench_bs, N_IN, device=device)
    with torch.no_grad():
        for _ in range(n_warm):
            _ = model(x)
        if device.type == "cuda": torch.cuda.synchronize()
        times_ms = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda": torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": statistics.median(times_ms),
            "p90_ms": sorted(times_ms)[int(n_iter * 0.9)],
            "min_ms": min(times_ms), "n_iter": n_iter, "batch": bench_bs,
            "compile_mode": compile_mode}


def main():
    print(f"\n{'='*70}")
    print(f"bench_latency_step199 — AH-only N=2048 K_hh=2 K_iter=5 D=16")
    print(f"FLOPs={FLOPS/1e6:.3f}M  Device={DEVICE}")
    print(f"Warm-up epochs={args.train_epochs}  bench_bs={args.bench_bs}  "
          f"n_warm={args.n_warm}  n_iter={args.n_iter}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    # 50% subset for warm-up (Tier-1 convention; fast)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))[:n_total // 2]
    tr = torch.utils.data.DataLoader(torch.utils.data.Subset(tr_full.dataset, idx.tolist()),
                                     batch_size=BATCH, shuffle=True, num_workers=0,
                                     generator=torch.Generator().manual_seed(SEED))

    model = _build(SEED).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params={n_p:,}")

    if args.train_epochs > 0:
        print(f"\nWarm-up training ({args.train_epochs} epochs)...")
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **trainer_kwargs(N, n_epochs=args.train_epochs)).train(
            n_epochs=args.train_epochs,
            log_fn=lambda m: print(f"  ep{m['epoch']+1}  val={m['val_top1']:.4f}", flush=True))
        top1h = [h.get("val_top1", 0.) for h in history]
        print(f"  warm-up done  best={max(top1h):.4f}  elapsed={time.time()-t0:.1f}s")
    else:
        print("  Skipping warm-up (train_epochs=0); benchmarking untrained model")
        top1h = []

    print(f"\n{'─'*50}")
    print(f"Inference latency benchmark")
    print(f"{'─'*50}")
    bench = bench_inference(model, DEVICE, args.bench_bs, args.n_warm, args.n_iter,
                            no_compile=args.no_compile)
    print(f"  median={bench['median_ms']:.3f}ms  p90={bench['p90_ms']:.3f}ms  "
          f"min={bench['min_ms']:.3f}ms")
    print(f"  compile_mode: {bench['compile_mode']}")
    print(f"  VGG16 FC ref: ~0.07ms @bs=32 on 5060ti (step500)")
    if bench["median_ms"] > 0:
        print(f"  ratio vs VGG16 FC: {bench['median_ms']/0.07:.1f}× slower / "
              f"{0.07/bench['median_ms']:.1f}× faster")

    results = {"config": {"step": STEP_NAME, "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
                          "flops": FLOPS, "n_params": n_p, "mechanism": "AH-only",
                          "device": str(DEVICE)},
               "latency": bench,
               "warmup_best_top1": round(max(top1h), 4) if top1h else None}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
