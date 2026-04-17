"""Step 857: C_ho sparsity=0.98 T2 validation — paper-quality number for D_very win.

MOTIVATION
==========
step853 T1 (5060ti, 75ep, 50% data): D_very (+0.36pp, 94.27% vs Ref 93.91%)
step856 multi-seed T1 running in parallel for variance confirmation.

T2 (150ep, 100% data) gives the paper-quality number now. If step856 multi-seed
confirms the win, this T2 result goes directly into the paper.

Expected: ~95.9% (step199 95.52% + ~0.36pp T1 delta → ~95.88% T2 estimate).

CONFIGS
=======
  Ref    : sparsity=0.90 (current default)
  D_very : sparsity=0.98 (step853 winner)

Scale: N=2048, D=16, K_hh=2, K_iter=5, K_in=25, 150ep, 100% data, seed=42
"""
# CUDA-5060ti-validated
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.sgnnet.model_resonant_cuda   import SGNNET_Resonant_CUDA, SGNNET_AntiHebbian_CUDA
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,D_very")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step857_dvery_t2_seed{SEED}__{SLOT}.json"

STEP199_T2 = 0.9552
STEP853_T1_DELTA = 0.0036  # +0.36pp T1 confirmed

CONFIGS = {
    "Ref":    (0.90, "sparsity=0.90 current default"),
    "D_very": (0.98, "sparsity=0.98 step853 T1 winner (+0.36pp)"),
}


def make_model(sparsity: float) -> torch.nn.Module:
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
        sparsity=sparsity,
    )
    if DEVICE.type == "cuda":
        resonant = SGNNET_Resonant_CUDA(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
            beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
            resonance_threshold=0.0, compile=True)
        return SGNNET_AntiHebbian_CUDA(resonant, alpha_ahebb=ALPHA_AHEBB,
                                       variant="wpos", compile=True)
    else:
        resonant = SGNNET_Resonant(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
            beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                          pin_memory=(DEVICE.type == "cuda"))

    print(f"\n{'='*70}")
    print(f"step857 — C_ho D_very T2 (150ep, 100% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  step199 Ref T2={STEP199_T2:.4f}, T1 D_very delta=+{STEP853_T1_DELTA*100:.2f}pp")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        sparsity, desc = CONFIGS[key]
        model = make_model(sparsity)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        if DEVICE.type == "cuda":
            kw["use_amp"] = False  # use_amp=False, non_blocking=True in Trainer .to(device) calls
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}",
                                   flush=True) if (m['epoch']+1) % 15 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref": ref_acc = best
        delta_ref = best - (ref_acc or STEP199_T2)
        delta_199 = best - STEP199_T2
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta_ref*100:+.2f}pp  Δ_vs_step199={delta_199*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "sparsity": sparsity, "label": desc, "n_params": n_p,
            "best": best, "best_ep": best_ep,
            "delta_vs_ref":   round(best - (ref_acc or STEP199_T2), 4),
            "delta_vs_step199": round(delta_199, 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    print(f"STEP 857 SUMMARY — D_very T2")
    print(f"{'='*60}")
    for k, r in results.items():
        print(f"  {k}: best={r['best']:.4f} Δ_vs_199={r['delta_vs_step199']*100:+.2f}pp")
    dv = results.get("D_very")
    if dv:
        verdict = "NEW DEFAULT (update sparsity=0.98)" if dv.get("delta_vs_ref", 0) > 0 else "no improvement at T2"
        print(f"\n  D_very T2 verdict: {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
