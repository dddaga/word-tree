"""Step 531: T0 training smoke-test after Triton _route() integration.

# CUDA-5060ti-validated

MOTIVATION
==========
step530 bench confirmed Triton v2 1.17-1.18× over torch.compile at inference.
Kernel integrated into _route() with guard: CUDA + l2 + power-of-2 D + no_grad.
Triton does NOT run during training (grad required through _route() → W_pos).

This T0 (20ep, 50% data) confirms:
  - Training still works after model_smallworld.py patch
  - Accuracy within expected T0 range (step199 ref: 0.87-0.93)

Correctness (Triton vs eager forward): already validated in step530 (max_diff=2.38e-07).
"""
# CUDA-5060ti-validated
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.sgnnet.model_smallworld     import SGNNET_SmallWorld
from src.sgnnet.model_resonant_cuda  import SGNNET_Resonant_CUDA, SGNNET_AntiHebbian_CUDA
from src.training.trainer            import Trainer
from src.training.experiment_config  import trainer_kwargs
from src.training.dataset            import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
          ) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"

N      = 2048;   N_IN   = 25088;   N_OUT = 10
D      = 16;     K_HH   = 2;       K_IN  = 25;    K_ITER = 5

STEP_NAME = Path(__file__).stem
SLOT      = os.environ.get("SGN_SLOT", "local")
OUT_PATH  = ROOT / "results" / f"{STEP_NAME}__{SLOT}.json"

T0_MIN = 0.87


def main():
    print(f"\n{'='*70}")
    print(f"{STEP_NAME}  —  T0 training smoke-test (Triton integration)")
    print(f"Device: {DEVICE}  epochs={EPOCHS}")
    print(f"Triton active at inference, eager during training (no_grad guard)")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant_CUDA(
        base, K_phase=8, alpha_reflect=0.5, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
        resonance_threshold=0.0, compile=False,
    )
    model = SGNNET_AntiHebbian_CUDA(resonant, alpha_ahebb=1.0, variant="wpos").to(DEVICE)

    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_p:,}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"))

    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

    t0 = time.time()
    def _log(m):
        ep = m["epoch"] + 1
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)

    history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
    elapsed = time.time() - t0

    top1h = [round(h.get("val_top1", 0.), 4) for h in history]
    best  = max(top1h); bep = int(np.argmax(top1h)) + 1
    verdict = "PASS" if best >= T0_MIN else f"FAIL (< {T0_MIN})"

    print(f"\n  best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)  [{verdict}]")
    if best >= T0_MIN:
        print(f"  → Training intact. Triton integration production-confirmed.")
        print(f"    Inference speedup: 1.17-1.18× (CUDA eval/no_grad only)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
        "top1_history": top1h, "elapsed_s": round(elapsed, 1),
        "n_params": n_p, "verdict": verdict, "t0_min": T0_MIN,
    }, indent=2))
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
