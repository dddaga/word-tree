"""Evaluate (or reproduce) the SGNNET final efficiency config (step199).

N=2048, D=16, K_hh=2, K_iter=5 — 95.52% @ 0.98M FLOPs (0.79% of VGG16 FC)

Usage
-----
  # Report metrics from the saved result JSON (no training):
  python scripts/eval_efficiency_config.py

  # Retrain from scratch and report:
  python scripts/eval_efficiency_config.py --train [--device mps|cpu]

  # Retrain for fewer epochs (quick smoke test):
  python scripts/eval_efficiency_config.py --train --epochs 20 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

# ── Config ────────────────────────────────────────────────────────────────────

N        = 2048
D        = 16
K_HH     = 2       # total hidden-to-hidden edges (K_local=1, K_random=1)
K_ITER   = 5       # routing iterations
K_IN     = 25      # input fan-in per neuron
N_IN     = 25088   # VGG16 feature extractor output dim
N_OUT    = 10      # FashionMNIST classes
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
ALPHA_AHEBB   = 1.0

FLOPS = 3 * N * K_HH * D * K_ITER   # 983,040 ≈ 0.98M

VGG16_FC_FLOPS  = 123_600_000   # VGG16 FC layers (123.6M)
VGG16_FC_PARAMS = 123_600_000   # same order

RESULT_PATH = ROOT / "results" / "train_step199_n2048_d16_khh2_kiter5_tier2.json"
DATA_PATH   = ROOT / "data" / "store.h5"

SEED  = 42
BATCH = 128

# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> torch.nn.Module:
    from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
    from src.sgnnet.model_resonant        import SGNNET_Resonant
    from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian

    K_r = max(1, K_HH // 4)   # K_random = 1
    K_l = K_HH - K_r          # K_local  = 1
    ng  = max(8, N // 8)       # n_groups = 256

    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    model = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return model.to(device)


# ── Training ──────────────────────────────────────────────────────────────────

def train(device: torch.device, epochs: int) -> dict:
    from src.training.trainer           import Trainer
    from src.training.experiment_config import trainer_kwargs
    from src.training.dataset           import make_loaders

    torch.manual_seed(SEED)
    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*68}")
    print(f"SGNNET final efficiency config — train from scratch")
    print(f"  N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  seed={SEED}")
    print(f"  params={n_params:,}  FLOPs={FLOPS:,} ({FLOPS/1e6:.3f}M)")
    print(f"  device={device}  epochs={epochs}")
    print(f"{'='*68}")

    tr, va = make_loaders(DATA_PATH, batch_size=BATCH, seed=SEED)
    kw      = trainer_kwargs(N, n_epochs=epochs)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=device, **kw)

    t0 = time.time()

    def _log(m):
        if str(device) == "mps":
            torch.mps.empty_cache()
        ep = m["epoch"] + 1
        if ep % 10 == 0 or ep <= 5:
            flag = "  *** PHASE EXIT ***" if m["val_top1"] >= 0.95 else ""
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}{flag}", flush=True)

    history  = trainer.train(n_epochs=epochs, log_fn=_log)
    elapsed  = time.time() - t0
    top1h    = [round(h.get("val_top1", 0.0), 4) for h in history]
    best     = max(top1h)
    best_ep  = int(np.argmax(top1h)) + 1

    return {
        "top1_best":    best,
        "top1_last":    top1h[-1],
        "best_epoch":   best_ep,
        "epochs_run":   len(history),
        "n_params":     n_params,
        "flops":        FLOPS,
        "elapsed_s":    round(elapsed, 1),
        "top1_history": top1h,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def report(res: dict) -> None:
    n_params = res["n_params"]
    flops    = res["flops"]
    top1     = res["top1_best"]

    flops_pct  = 100.0 * flops    / VGG16_FC_FLOPS
    params_pct = 100.0 * n_params / VGG16_FC_PARAMS

    print(f"\n{'='*68}")
    print(f"SGNNET FINAL EFFICIENCY CONFIG — step199")
    print(f"{'='*68}")
    print(f"  Accuracy:    {top1:.4f} ({top1*100:.2f}%)   best_ep={res.get('best_epoch', '?')}")
    print(f"  FLOPs:       {flops:,}  ({flops/1e6:.3f}M)  =  {flops_pct:.2f}% of VGG16 FC")
    print(f"  Params:      {n_params:,}  ({n_params/1e3:.1f}K)    =  {params_pct:.3f}% of VGG16 FC")
    print(f"")
    print(f"  ≤1% FLOPs criterion:  {'✓ MET' if flops_pct <= 1.0 else '✗ NOT MET'}  ({flops_pct:.2f}%)")
    print(f"  ≤1% Params criterion: {'✓ MET' if params_pct <= 1.0 else '✗ NOT MET'}  ({params_pct:.3f}%)")
    print(f"  ≥95% Accuracy:        {'✓ MET' if top1 >= 0.95 else '✗ NOT MET'}  ({top1*100:.2f}%)")
    print(f"")

    if "elapsed_s" in res:
        print(f"  Training time: {res['elapsed_s']:.0f}s ({res.get('epochs_run', '?')} epochs)")

    print(f"\n  Config: N={N} D={D} K_hh={K_HH} K_iter={K_ITER}")
    print(f"         alpha_ahebb={ALPHA_AHEBB}  alpha_reflect={ALPHA_REFLECT}  alpha_turing={ALPHA_TURING}")
    print(f"         K_in={K_IN}  n_groups={max(8, N//8)}  norm=l2  encoding=fourier")
    print(f"         mode=dynamic_z_geo  beam_size=16  geo_gamma=0.5  K_phase=8")
    print(f"{'='*68}\n")

    print(f"  Efficiency frontier (D=16, K_hh=2 family):")
    frontier = [
        ("step199", 2048, 5, 0.98,  95.52, "final efficiency config"),
        ("step195", 2048, 6, 1.18,  96.08, "first ≤1% FLOPs"),
        ("step205", 4096, 5, 1.97,  97.17, "D=16 record"),
        ("step209", 8192, 5, 3.93,  97.17, "D=16 ceiling"),
        ("step89 ", 4096, 12, 38.8, 97.86, "project best (D=64)"),
        ("VGG16FC", None, None, 123.6, 95.0, "baseline"),
    ]
    print(f"  {'Step':<10} {'N':<6} {'K_iter':<8} {'FLOPs(M)':<10} {'FLOPs%':<9} {'Acc%':<8} Note")
    print(f"  {'-'*70}")
    for step, n, ki, fl, acc, note in frontier:
        n_str  = str(n)  if n  is not None else "—"
        ki_str = str(ki) if ki is not None else "—"
        pct    = 100.0 * fl * 1e6 / VGG16_FC_FLOPS
        marker = " <-- THIS" if step.strip() == "step199" else ""
        print(f"  {step:<10} {n_str:<6} {ki_str:<8} {fl:<10.2f} {pct:<9.2f} {acc:<8.2f} {note}{marker}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",   action="store_true",
                        help="Retrain from scratch instead of reading saved result")
    parser.add_argument("--device",  default="auto",
                        help="mps | cpu | cuda | auto (default: auto)")
    parser.add_argument("--epochs",  type=int, default=150,
                        help="Training epochs (only used with --train, default: 150)")
    parser.add_argument("--save",    action="store_true",
                        help="Save retrained result to RESULT_PATH (only with --train)")
    args = parser.parse_args()

    if args.device == "auto":
        device = (torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))
    else:
        device = torch.device(args.device)

    if args.train:
        res = train(device, args.epochs)
        if args.save:
            RESULT_PATH.parent.mkdir(exist_ok=True)
            RESULT_PATH.write_text(json.dumps({"A": res}, indent=2))
            print(f"  Result saved → {RESULT_PATH}")
    else:
        if not RESULT_PATH.exists():
            print(f"ERROR: result file not found: {RESULT_PATH}")
            print("Run with --train to reproduce from scratch.")
            sys.exit(1)
        data = json.loads(RESULT_PATH.read_text())
        # Result JSON has a top-level key "A"
        res  = data.get("A", data)

    report(res)


if __name__ == "__main__":
    main()
