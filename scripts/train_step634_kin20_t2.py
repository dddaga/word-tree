"""Step 634: K_in=20 T2 validation — confirm step633 knee at paper quality.

MOTIVATION
==========
step633 K_in plot sweep T1 (75ep, 50% data, N=2048) — cross-device consensus:
  K_in=20: 94.50% (mini_cpu) / 94.24% (5060ti) — beats K_in=15 AND K_in=25
  K_in=15: 93.89% / 94.11% — −0.6pp vs K_in=20
  K_in=25: 94.01% / 93.81% — trails K_in=20, more seed connections

K_in=20 is the true optimum at N=2048. K_in=25 (current default) slightly
over-connects. K_in=15 is a valid efficiency tradeoff (25% fewer connections,
−0.6pp T1 cost). This T2 run gives paper-quality validation of the crossover.

Compare to:
  step632 K_in=15 T2 = 95.13% (Ref_k25=95.46%, Δ=−0.33pp)
  step199 Ref K_in=25 T2 = 95.52%

If K_in=20 T2 > 95.52% (step199) → K_in=20 is the new optimal default.
If K_in=20 T2 ≈ 95.52% → K_in=20 confirms efficiency at no accuracy cost.

CONFIGS
=======
  Ref_k25 : K_in=25 (step199 reference, retrain for fair comparison)
  A_k20   : K_in=20 (step633 T1 winner)
  B_k15   : K_in=15 (step632 T2 confirmed, included for 3-way comparison)

Scale: N=2048, D=16, K_hh=2, K_iter=5
Tier: T2 (150ep, 100% data)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

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
parser.add_argument("--configs", default="Ref_k25,A_k20,B_k15")
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
OUT_PATH = ROOT / "results" / f"train_step634_kin20_t2_seed{SEED}__{SLOT}.json"

# Prior references
STEP199_REF  = 0.9552   # K_in=25 T2
STEP632_K15  = 0.9513   # K_in=15 T2

CONFIGS = {
    "Ref_k25": (25, "K_in=25 reference (step199 config)"),
    "A_k20":   (20, "K_in=20 T1 knee — step633 cross-device winner"),
    "B_k15":   (15, "K_in=15 T2 ref — step632 confirmed"),
}


def make_model(K_in: int) -> nn.Module:
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_in, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
    if DEVICE.type == "cuda":
        # CUDA path: use_amp=False (Blackwell fp16 4.4× slower, step801)
        # non_blocking=True transfers handled in Trainer; pin_memory=True on loader
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
                          pin_memory=(DEVICE.type == "cuda"))  # pin_memory=True on CUDA
    print(f"Step 634 — K_in=20 T2 validation (150ep, 100% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Refs: step199(K_in=25)={STEP199_REF:.4f}  step632(K_in=15)={STEP632_K15:.4f}")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown config: {key}"); continue
        K_in, desc = CONFIGS[key]
        model = make_model(K_in)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        seed_macs = N * K_in
        routing_macs = K_ITER * N * K_HH * D * 2
        print(f"{'─'*60}\n{key}: {desc}")
        print(f"  K_in={K_in}  params={n_p:,}  seed_MACs={seed_macs/1e3:.0f}K  routing={routing_macs/1e3:.0f}K")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        if DEVICE.type == "cuda":
            kw["use_amp"] = False  # use_amp=False: Blackwell fp16 4.4× slower (step801)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 15 == 0 else None))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_k25": ref_acc = best
        delta_ref  = best - (ref_acc or STEP199_REF)
        delta_199  = best - STEP199_REF
        delta_632  = best - STEP632_K15
        print(f"  → best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta_ref*100:+.2f}pp  Δ_vs_step199={delta_199*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "K_in": K_in, "label": desc, "n_params": n_p,
            "seed_macs": seed_macs, "routing_macs": routing_macs,
            "best": best, "best_ep": best_ep,
            "delta_vs_ref":   round(best - (ref_acc or STEP199_REF), 4),
            "delta_vs_step199": round(delta_199, 4),
            "delta_vs_step632": round(delta_632, 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    print(f"STEP 634 SUMMARY — K_in=20 T2 (150ep 100% data)")
    print(f"{'='*60}")
    print(f"  {'Config':<10} {'K_in':>5} {'best':>7}  {'Δ_vs_199':>10}  {'Δ_vs_k15':>10}")
    for k, r in results.items():
        print(f"  {k:<10} {r['K_in']:>5} {r['best']:>7.4f}  {r['delta_vs_step199']*100:>+9.2f}pp  {r['delta_vs_step632']*100:>+9.2f}pp")
    k20 = results.get("A_k20")
    if k20:
        verdict = "BEATS step199" if k20["delta_vs_step199"] > 0 else f"trails by {-k20['delta_vs_step199']*100:.2f}pp"
        print(f"\n  K_in=20 T2: {verdict} vs K_in=25 (step199). Update default? {'YES' if k20['delta_vs_step199'] >= 0 else 'NO'}")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
