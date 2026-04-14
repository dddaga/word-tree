"""Step 402: N-scaling law — efficiency config across N={256..8192}.

MOTIVATION
==========
Known data points (D=16, K_hh=2, K_iter=5, AH=1.0):
  N=1024  → ~89%  (approx)
  N=2048  → 95.52% (step199, efficiency milestone)
  N=4096  → 97.17% (step205/209)
  N=8192  → 97.17% (step209, ceiling)

Need the full curve (N=256, 512, 1024, 2048, 4096, 8192) to:
  1. Fit a power-law scaling curve for the paper
  2. Confirm N-scaling law: accuracy ~ f(N) at fixed compute budget
  3. Identify the knee of the curve (where marginal gain per ΔN drops off)

FLOPs formula (same as eval_efficiency_config.py):
  FLOPS = 3 * N * K_hh * D * K_iter

CONFIGS (D=16, K_hh=2, K_iter=5, AH=1.0, reflect=0.5, 75ep, 50% data — Tier-1)
  N=256, N=512, N=1024, N=2048, N=4096, N=8192
  SEED=42 for all
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--configs", default="",
                    help="Comma-separated N values to run, e.g. '256,512'")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

# FLOPs formula (matches eval_efficiency_config.py)
VGG16_FC_FLOPS = 123_600_000

def compute_flops(n: int) -> int:
    """Approximate FLOPs: 3 * N * K_hh * D * K_iter (hidden-to-hidden pass)."""
    return 3 * n * K_HH * D * K_ITER

OUT_PATH = ROOT / "results" / "train_step402_n_scaling_imagenette.json"

ALL_N = [256, 512, 1024, 2048, 4096, 8192]


def build_model(n: int) -> torch.nn.Module:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4)
    K_l = K_HH - K_r
    ng  = max(8, n // 8)
    base = SGNNET_SmallWorld(
        N_hidden=n, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def main():
    run_ns = ALL_N
    if args.configs:
        run_ns = [int(v.strip()) for v in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"Step 402 — N-scaling law (D=16, K_hh=2, K_iter=5, AH=1.0, Tier-1)")
    print(f"Fitting power-law curve for paper: accuracy ~ f(N)")
    print(f"Running N: {run_ns}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))[:n_total // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for n in run_ns:
        flops = compute_flops(n)
        flops_pct = 100.0 * flops / VGG16_FC_FLOPS
        print(f"\n{'─'*60}\nN={n}  FLOPs={flops:,} ({flops/1e6:.3f}M, {flops_pct:.2f}% VGG16-FC)\n{'─'*60}")

        try:
            model = build_model(n).to(DEVICE)
        except RuntimeError as e:
            print(f"  SKIP N={n} — build failed: {e}")
            results[str(n)] = {"error": str(e), "N": n}
            continue

        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}  n_groups={max(8, n//8)}")

        kw = trainer_kwargs(n, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)
        t0 = time.time()
        def _log(m, _n=n):
            ep = m["epoch"] + 1
            if ep % 10 == 0 or ep == 1 or ep == EPOCHS:
                print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)
        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best  = max(top1h); bep = int(np.argmax(top1h)) + 1
        results[str(n)] = {
            "N":            n,
            "top1_best":    best,
            "top1_last":    top1h[-1],
            "best_epoch":   bep,
            "top1_history": top1h,
            "elapsed_s":    round(elapsed, 1),
            "n_params":     n_p,
            "flops":        flops,
            "flops_M":      round(flops / 1e6, 4),
            "flops_pct_vgg16fc": round(flops_pct, 4),
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

        # Persist after each N in case of OOM on large N
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))
        print(f"  (checkpoint saved → {OUT_PATH})")

        # Free memory before next config
        del model, trainer
        if str(DEVICE) == "mps":
            torch.mps.empty_cache()

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}\nSTEP 402 SUMMARY — N-scaling law\n{'='*70}")
    print(f"  {'N':<6} {'FLOPs(M)':<10} {'FLOPs%':<9} {'Acc%':<8} {'Params':<10} {'best_ep'}")
    print(f"  {'-'*55}")
    for n in run_ns:
        key = str(n)
        if key not in results or "error" in results[key]:
            err = results.get(key, {}).get("error", "not run")
            print(f"  N={n:<5}  FAILED: {err}")
            continue
        r = results[key]
        print(f"  {r['N']:<6} {r['flops_M']:<10.3f} {r['flops_pct_vgg16fc']:<9.2f} "
              f"{r['top1_best']*100:<8.2f} {r['n_params']:<10,} {r['best_epoch']}")

    print(f"\nPrior known points (for curve fitting):")
    print(f"  N=2048 → 95.52% @ 0.98M FLOPs  (step199, efficiency milestone)")
    print(f"  N=4096 → 97.17% @ 1.97M FLOPs  (step205/209)")
    print(f"  N=8192 → 97.17% @ 3.93M FLOPs  (step209, D=16 ceiling)")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
