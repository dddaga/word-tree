"""Step 863: D=8 efficiency probe T0.

MOTIVATION
==========
Current default: D=16. W_pos = [N+N_out, 16] = [2058, 16] = 32,928 params.
At D=8: W_pos = [2058, 8] = 16,464 params → total drops from 34,976 to ~18,512.
Better params headline: 18K vs 35K. Halves the dominant parameter block.

Hypothesis: D=8 has sufficient representational capacity for N=2048 routing.
Fourier encoding on S^{D-1}: D=8 allows 4 complex frequency components.
Risk: lower-dimensional manifold may collapse routing diversity → accuracy loss.

CONFIGS (T0: 20ep, 50% data, seed=42)
  Ref_D16  : D=16 standard (control)
  A_D8     : D=8 with default K_iter=5
  B_D8_K10 : D=8 + K_iter=10 (compensate for lower capacity with more iterations)
  C_D12    : D=12 intermediate probe (3 complex components)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.model_resonant_cuda   import SGNNET_Resonant_CUDA, SGNNET_AntiHebbian_CUDA
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser(description="Step 863: D=8 efficiency probe T0")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref_D16,A_D8,B_D8_K10,C_D12")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
K_HH = 2; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step863_d8_efficiency_seed{SEED}__{SLOT}.json"

REF_D16_PARAMS = (N + N_OUT) * 16 + N * 25 + N * 2 + int(N * 0.02) * N_OUT
D8_PARAMS      = (N + N_OUT) * 8  + N * 25 + N * 2 + int(N * 0.02) * N_OUT
D12_PARAMS     = (N + N_OUT) * 12 + N * 25 + N * 2 + int(N * 0.02) * N_OUT


def make_model(D: int, K_iter: int):
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_iter, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier", sparsity=0.90)
    if DEVICE.type == "cuda":
        res = SGNNET_Resonant_CUDA(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                                    alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
                                    mode="dynamic_z_geo", resonance_threshold=0.0, compile=False)
        return SGNNET_AntiHebbian_CUDA(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos", compile=False)
    else:
        res = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
                               beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
        return SGNNET_AntiHebbian(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


CONFIGS = {
    "Ref_D16":  (16, 5,  "D=16 K=5 standard — control"),
    "A_D8":     (8,  5,  "D=8  K=5 — half position params"),
    "B_D8_K10": (8,  10, "D=8  K=10 — more iters compensate D"),
    "C_D12":    (12, 5,  "D=12 K=5 — intermediate probe"),
}


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=(DEVICE.type == "cuda"))
    n_full = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    print(f"\n{'='*70}")
    print(f"step863 — D=8 efficiency probe T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Param estimate: D=16={REF_D16_PARAMS:,}  D=8={D8_PARAMS:,}  D=12={D12_PARAMS:,}")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        D, K_iter, desc = CONFIGS[key]
        model = make_model(D, K_iter)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)

        def make_log_fn(k):
            def log_fn(m):
                ep = m['epoch'] + 1
                if ep % 5 == 0:
                    print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)
            return log_fn

        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(n_epochs=EPOCHS, log_fn=make_log_fn(key))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_D16": ref_acc = best
        delta = best - (ref_acc or 0.91)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "D": D, "K_iter": K_iter, "label": desc,
            "n_params": n_p, "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 863 SUMMARY — D=8 efficiency probe T0")
    print(f"{'='*70}")
    print(f"  {'config':<14} {'D':>3} {'K':>3} {'params':>8} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        print(f"  {k:<14} {r['D']:>3} {r['K_iter']:>3} {r['n_params']:>8,} {r['best']:>7.4f} "
              f"{r['delta_vs_ref']*100:>+9.2f}pp")
    d8 = results.get("A_D8", {})
    if d8:
        thresh = -0.005  # tolerate up to -0.5pp for a 50% param reduction
        verdict = "ADVANCES to T1 (D=8 viable)" if d8["delta_vs_ref"] >= thresh else \
                  f"D=8 drops {d8['delta_vs_ref']*100:.2f}pp — D=12 intermediate may help"
        print(f"\n  D=8 verdict: {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
