"""Step XXX: <one-line description of the experiment>.

MOTIVATION
==========
Why are we running this? What claim does it test?
- Prior evidence: (cite step numbers that motivate this)
- Hypothesis: (what outcome would confirm vs disconfirm)
- Tier: (Tier-0 scout / Tier-1 calibration / Tier-2 validation)

CONFIGS (state exact scale and what changes)
============================================
  Ref  : baseline reference (cite step199 or whatever the relevant Ref is)
  A_X  : variation A — describe the single variable change
  B_Y  : variation B — ...

CUDA-5060ti-validated
=====================
This template uses the CUDA-optimized code paths enforced by
`.claude/skills/sgnnet-research/CUDA_CHECKLIST.md` and audited by
`scripts/launch_slot.sh` for the `5060ti_cuda` slot. Keep the markers
(SGNNET_Resonant_CUDA / pin_memory=True / non_blocking=True) intact unless
you have a documented reason to diverge.
"""
# CUDA-5060ti-validated — this template meets the 5060ti checklist
# (see .claude/skills/sgnnet-research/CUDA_CHECKLIST.md).
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld     import SGNNET_SmallWorld
# CUDA-optimized model path (torch.compile fused routing loop, step500: 2.6%→99.6% util, 4× speedup).
# For CPU/MPS the eager SGNNET_Resonant still works but is 4× slower on CUDA.
from src.sgnnet.model_resonant_cuda  import SGNNET_Resonant_CUDA, SGNNET_AntiHebbian_CUDA
from src.training.trainer            import Trainer
from src.training.experiment_config  import trainer_kwargs
from src.training.dataset            import make_loaders

# --- CLI --------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--configs", default="", help="Comma-separated config keys. Empty = all.")
args   = parser.parse_args()

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
          ) if args.device == "auto" else torch.device(args.device)

# --- Experiment constants ---------------------------------------------------
# Efficiency config (step199): N=2048, D=16, K_hh=2, K_iter=5, AH=1.0
EPOCHS = args.epochs
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"

N      = 2048;   N_IN   = 25088;   N_OUT = 10
D      = 16;     K_HH   = 2;       K_IN  = 25;    K_ITER = 5
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
ALPHA_AHEBB   = 1.0

STEP_NAME = Path(__file__).stem
# Slot suffix set by scripts/launch_slot.sh (SGN_SLOT env var).
# "local" fallback when run manually without the wrapper.
SLOT      = os.environ.get("SGN_SLOT", "local")
OUT_PATH  = ROOT / "results" / f"{STEP_NAME}__{SLOT}.json"


# --- Config definitions -----------------------------------------------------
# Each key maps to whatever parameters your experiment varies.
# Replace this with your actual variation axis.
CONFIGS = {
    "Ref":   ("standard",),
    # "A_X": ("variant_a", ...),
}


def build_model(variant: str) -> nn.Module:
    """Build model for a given variant key. Uses CUDA-optimized path."""
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    # torch.compile enabled by default (step500 evidence). Compilation fires on first
    # forward call (~30s one-time cost), then the K_iter routing loop fuses to one kernel.
    # NOTE: SGNNET_Resonant_CUDA requires alpha_turing=0 (efficiency config hardcodes it).
    resonant = SGNNET_Resonant_CUDA(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
        compile=True,
    )
    model = SGNNET_AntiHebbian_CUDA(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")

    # --- Apply variant-specific modifications here ---
    # Example:
    # if variant == "variant_a":
    #     model.some_attribute = 0.5

    return model


def main():
    run_keys = list(CONFIGS.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"{STEP_NAME}")
    print(f"Running: {run_keys}  on {DEVICE}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    # 50% subset for Tier-0/Tier-1 (comment out for Tier-2)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    # pin_memory=True + non_blocking=True in transfer: DMA overlap with compute on CUDA.
    # num_workers=0 is optimal for in-RAM H5 datasets (no I/O to parallelize).
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True,
                                     num_workers=0, pin_memory=True)

    results = {}
    for key in run_keys:
        variant = CONFIGS[key][0]
        print(f"\n{'─'*60}\nConfig {key}: variant={variant}\n{'─'*60}")

        model = build_model(variant).to(DEVICE)
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
        best  = max(top1h); bep = int(np.argmax(top1h)) + 1
        results[key] = {
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "top1_history": top1h, "elapsed_s": round(elapsed, 1),
            "n_params": n_p, "variant": variant,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"\n{'='*70}\n{STEP_NAME} SUMMARY\n{'='*70}")
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        print(f"  {key:14s}: {r['top1_best']:.4f}{delta}")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
