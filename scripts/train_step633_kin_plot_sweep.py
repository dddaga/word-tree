"""Step 633: K_in plot sweep — full curve from K_in=1 to K_in=25.

MOTIVATION
==========
step631 gave T1 data at K_in=5,10,15,25 (advance threshold ≥-0.5pp → K_in=15 won).
For publication, we need a smooth curve showing how accuracy degrades as fan-in
shrinks, to visually justify the K_in=15 choice and quantify the capacity cliff.

K_in values: 1, 2, 3, 5, 7, 10, 15, 20, 25

Note: spatial precomputation makes seed cost = N×K_in MACs (not N×K_in×D).
The curve plots accuracy vs seed MACs to show the accuracy-efficiency tradeoff.

Tier: T1 (75ep, 50% data) — sufficient for relative comparison and plotting.

Existing step631 data (reusable):
  K_in=5:  91.26% (T1)
  K_in=10: 92.84% (T1)
  K_in=15: 93.66% (T1)
  K_in=25: 94.01% (T1)

New values needed: K_in=1,2,3,7,20 (+ re-run all for consistency).

To run (CUDA recommended — 9 configs × 75ep):
    python -u scripts/train_step633_kin_plot_sweep.py --device cuda
    python -u scripts/train_step633_kin_plot_sweep.py --device mps
    python -u scripts/train_step633_kin_plot_sweep.py --kin_values 1,2,3,7,20 --device cuda
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",     default="auto")
parser.add_argument("--epochs",     type=int, default=75)
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--data",       default="data/store.h5")
parser.add_argument("--kin_values", default="1,2,3,5,7,10,15,20,25",
                    help="Comma-separated K_in values to sweep")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step633_kin_plot_sweep_seed{SEED}__{SLOT}.json"

KIN_VALUES = [int(x.strip()) for x in args.kin_values.split(",") if x.strip()]


def make_model(K_in: int) -> nn.Module:
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_in, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def main():
    torch.manual_seed(SEED)
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))[:n_total//2].tolist()
    tr = torch.utils.data.DataLoader(
        Subset(tr_full.dataset, idx), batch_size=BATCH, shuffle=True, drop_last=False)

    print(f"Step 633 — K_in plot sweep (T1: {EPOCHS}ep, 50% data)")
    print(f"  device={DEVICE}  seed={SEED}  K_in values={KIN_VALUES}")
    print(f"  N={N}  D={D}  K_iter={K_ITER}  Train={len(tr.dataset)}  Val={len(va.dataset)}\n")

    results = {}
    # Load existing results to allow resuming
    if OUT_PATH.exists():
        try:
            results = json.loads(OUT_PATH.read_text())
            print(f"  Resuming: {list(results.keys())} already done\n")
        except Exception:
            pass

    ref_acc = results.get("25", {}).get("best") or results.get(25, {}).get("best")

    for K_in in KIN_VALUES:
        key = str(K_in)
        if key in results:
            print(f"  skip K_in={K_in} (already done: {results[key]['best']:.4f})")
            if K_in == 25: ref_acc = results[key]["best"]
            continue

        model = make_model(K_in)
        seed_macs = N * K_in          # after spatial precomputation
        seed_macs_orig = N * K_in * D  # before optimization (for reference)
        print(f"{'─'*55}")
        print(f"K_in={K_in:2d}  seed_MACs={seed_macs:,}  (orig={seed_macs_orig:,})")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 25 == 0 else None))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1

        if K_in == 25: ref_acc = best
        delta = best - (ref_acc or 0)
        print(f"  → K_in={K_in}  best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "K_in": K_in,
            "seed_macs_optimized": seed_macs,
            "seed_macs_original": seed_macs_orig,
            "best": best, "best_ep": best_ep,
            "delta_vs_k25": round(delta, 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # Summary table
    print(f"\n{'='*55}")
    print(f"STEP 633 — K_in sweep plot data")
    print(f"{'K_in':>6}  {'seed_MACs':>10}  {'best':>7}  {'Δ_k25':>8}")
    ref_b = results.get("25", {}).get("best", 0)
    for k_in in sorted(KIN_VALUES):
        r = results.get(str(k_in), {})
        if not r: continue
        d = r["best"] - ref_b
        print(f"{k_in:>6}  {r['seed_macs_optimized']:>10,}  {r['best']:>7.4f}  {d*100:>+7.2f}pp")


if __name__ == "__main__":
    main()
