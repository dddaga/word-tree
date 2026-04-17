"""Step 856: C_ho sparsity=0.98 multi-seed T1 — confirm step853 D_very win.

MOTIVATION
==========
step853 T1 (5060ti, 75ep, 50% data, seed=42):
  Ref  (sparsity=0.90): 93.91%
  D_very (sparsity=0.98): 94.27%  (+0.36pp)

Single-seed result needs multi-seed confirmation before updating the default
and claiming this as a paper finding. step760 showed seed variance at
N=2048 is 0.43pp (step199 Ref) — so +0.36pp could be noise.

5 seeds × 2 configs (Ref + D_very) = 10 runs at T1 (75ep, 50% data).
If D_very mean > Ref mean across all 5 seeds, update default to sparsity=0.98.

CONFIGS
=======
  Ref    : sparsity=0.90 (current default, ~274 connections/class)
  D_very : sparsity=0.98 (step853 T1 winner, ~208 connections/class)

Scale: N=2048, D=16, K_hh=2, K_iter=5, K_in=25, 75ep, 50% data
SEEDS: 42,43,44,45,46 (matches step760)
"""
# CUDA-5060ti-validated
from __future__ import annotations
import argparse, json, os, sys, time, statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.utils.data

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.sgnnet.model_resonant_cuda   import SGNNET_Resonant_CUDA, SGNNET_AntiHebbian_CUDA
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--seeds",   default="42,43,44,45,46")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--device",  default="auto")
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,D_very")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r
DATA_SEED = 42

SEEDS = [int(s) for s in args.seeds.split(",")]

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step856_cho_sparsity_multiseed_t1__{SLOT}.json"

CONFIGS = {
    "Ref":    (0.90, "sparsity=0.90 current default (~274/class)"),
    "D_very": (0.98, "sparsity=0.98 step853 T1 winner (~208/class)"),
}

# step853 T1 single-seed references
STEP853_REF   = 0.9391
STEP853_DVERY = 0.9427


def make_model(sparsity: float, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
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

    tr_full, va = make_loaders(
        str(data_path), batch_size=BATCH, seed=DATA_SEED,
        pin_memory=(DEVICE.type == "cuda"),
    )
    n_full = len(tr_full.dataset)
    sub_idx = torch.randperm(
        n_full, generator=torch.Generator().manual_seed(DATA_SEED)
    )[: n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"\n{'='*70}")
    print(f"step856 — C_ho sparsity multi-seed T1 (75ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seeds={SEEDS}")
    print(f"  Ref={STEP853_REF:.4f}  D_very={STEP853_DVERY:.4f} (+0.36pp step853 T1 seed42)")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {k: {"sparsity": CONFIGS[k][0], "label": CONFIGS[k][1], "per_seed": []} for k in keys if k in CONFIGS}

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown config: {key}"); continue
        sparsity, desc = CONFIGS[key]
        print(f"\n{'='*60}\n{key}: {desc}\n{'='*60}")

        for seed in SEEDS:
            print(f"\n  SEED {seed}")
            model = make_model(sparsity, seed).to(DEVICE)
            n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            kw = trainer_kwargs(N, n_epochs=EPOCHS)
            if DEVICE.type == "cuda":
                kw["use_amp"] = False  # use_amp=False, non_blocking=True in Trainer .to(device) calls
            t0 = time.time()
            history = Trainer(model=model, train_loader=tr, val_loader=va,
                              device=DEVICE, **kw).train(
                n_epochs=EPOCHS,
                log_fn=lambda m: print(f"    ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}",
                                       flush=True) if (m['epoch']+1) % 15 == 0 else None)
            top1h = [round(h.get("val_top1", 0.0) if isinstance(h, dict) else float(h), 4)
                     for h in history]
            best = max(top1h); bep = int(np.argmax(top1h)) + 1
            elapsed = round(time.time() - t0, 1)
            print(f"  -> seed={seed}: best={best:.4f} @ep{bep}  {elapsed:.0f}s")
            results[key]["per_seed"].append({
                "seed": seed, "best": best, "best_ep": bep,
                "n_params": n_p, "elapsed_s": elapsed,
            })

            # incremental save
            tops = [r["best"] for r in results[key]["per_seed"]]
            results[key]["mean"] = statistics.mean(tops)
            results[key]["std"]  = statistics.stdev(tops) if len(tops) > 1 else 0.0
            results[key]["min"]  = min(tops); results[key]["max"] = max(tops)
            OUT_PATH.parent.mkdir(exist_ok=True)
            OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 856 SUMMARY — C_ho sparsity multi-seed T1")
    print(f"{'='*70}")
    print(f"  {'config':<10} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} {'Δ_mean':>9}")
    ref_mean = results.get("Ref", {}).get("mean", STEP853_REF)
    for k, r in results.items():
        if "mean" not in r: continue
        delta = r["mean"] - ref_mean if k != "Ref" else 0.0
        print(f"  {k:<10} {r['mean']*100:>6.3f}% {r['std']*100:>6.3f}pp "
              f"{r['min']*100:>6.3f}% {r['max']*100:>6.3f}%  {delta*100:>+7.3f}pp")
    if "Ref" in results and "D_very" in results and "mean" in results["Ref"] and "mean" in results["D_very"]:
        win = results["D_very"]["mean"] > results["Ref"]["mean"]
        print(f"\n  Verdict: D_very {'BEATS' if win else 'TRAILS'} Ref across {len(SEEDS)} seeds.")
        if win:
            print(f"  -> UPDATE DEFAULT: change sparsity=0.90 to sparsity=0.98 in model_smallworld.py")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
