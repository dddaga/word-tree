"""Step 321: Critical ablation — AH alpha_ahebb sweep at N=2048 D=16.

CLAIM UNDER TEST
================
"AH is load-bearing" — prior evidence (step218 at N=2048) showed
alpha_ahebb=0 collapses training to 18.8%. But step234/235 show that
ΔW projection REPLACES AH entirely. So is AH necessary, or just one
of multiple sufficient mechanisms?

This sweep nails down the dependence structure at the efficiency config:
- α=0.0 alone (no AH, no ΔW, no polarizer)  → tests AH-necessity claim
- α={0.25, 0.5, 0.75, 1.0, 1.25, 1.5}      → calibration curve

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep Tier-0)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.sgnnet.model_smallworld     import SGNNET_SmallWorld
from src.sgnnet.model_resonant       import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer            import Trainer
from src.training.experiment_config  import trainer_kwargs
from src.training.dataset            import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--configs", default="")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

OUT_PATH = ROOT / "results" / "train_step321_ah_alpha_sweep.json"

ALPHA_SWEEP = {
    "A0":    0.00,
    "A025":  0.25,
    "A050":  0.50,
    "A075":  0.75,
    "Ref":   1.00,   # current default (step199)
    "A125":  1.25,
    "A150":  1.50,
}


def build(alpha_ahebb):
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=alpha_ahebb, variant="wpos")


def main():
    run_keys = list(ALPHA_SWEEP.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"Step 321 — AH alpha_ahebb sweep at N=2048 D=16")
    print(f"Testing 'AH is load-bearing' claim; sweeping α")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        alpha = ALPHA_SWEEP[key]
        print(f"\n{'─'*60}\nConfig {key}: alpha_ahebb={alpha}\n{'─'*60}")
        model = build(alpha).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)
        t0 = time.time()
        def _log(m):
            ep = m["epoch"] + 1
            if ep % 5 == 0 or ep == 1:
                print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)
        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1
        results[key] = {
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "top1_history": top1h, "elapsed_s": round(elapsed, 1),
            "n_params": n_p, "alpha_ahebb": alpha,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"\n{'='*70}\nSTEP 321 SUMMARY — AH α sweep\n{'='*70}")
    print(f"  {'α':>8}  {'acc':>8}  {'Δvs α=1.0':>10}")
    for key in run_keys:
        r = results[key]
        delta = f"  {r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        print(f"  {r['alpha_ahebb']:>8.2f}  {r['top1_best']:>8.4f}{delta}")

    a0 = results.get("A0", {}).get("top1_best", None)
    if a0 is not None:
        print(f"\nInterpretation:")
        print(f"  α=0 = {a0:.4f}")
        print(f"  α=0 collapses (<0.30) → AH IS load-bearing (confirms step218)")
        print(f"  α=0 retains (>0.85) → AH NOT strictly load-bearing at this config")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
