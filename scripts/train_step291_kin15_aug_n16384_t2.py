"""Step 291: K_in=15+aug compound T2 @ N=16384 — validate step288 gains.

MOTIVATION
==========
step288 T1 results (N=16384):
  C_k15_naug = 94.70% (+1.27pp vs Ref=93.43%) — K_in=15 ANOMALY (helps at N=16384)
  D_k15_aug  = 95.92% (+2.50pp vs Ref=93.43%) — K_in=15+aug compound STRONGEST

This is the K_in=15+aug compound T2 validation at N=16384, completing the compound curve:
  N=1024: +0.79pp T2 (step284)
  N=2048: +0.18pp T2 (step274)
  N=4096: +0.18pp T2 (step279)
  N=8192: +0.36pp T2 (step282)
  N=16384: PENDING (this step)

T1 Ref for step288 comparisons: step286 = 93.43%.
T2 Ref: step287 (running on 5060ti). Script compares vs T1 Ref hardcoded; update after step287.

CONFIGS
=======
  C_k15_naug : K_in=15 no-aug T2 (validates +1.27pp T1 anomaly — regularization hypothesis)
  D_k15_aug  : K_in=15+aug T2   (validates +2.50pp T1 compound — paper primary claim)

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
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",   default="auto")
parser.add_argument("--epochs",   type=int, default=150)
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data",     default="data/store.h5")
parser.add_argument("--data_aug", default="data/store_aug.h5")
parser.add_argument("--configs",  default="C_k15_naug,D_k15_aug")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 64; SEED = args.seed
N = 16384; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN_EFF = 15
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step291_kin15_aug_n16384_t2_seed{SEED}__{SLOT}.json"

# T1 Ref from step286: 93.43%. T2 Ref from step287: 95.87% (Ref_n16384 done 2026-04-16).
REF_ACC_T1 = 0.9587   # step287 T2 Ref — use this for paper-quality T2 delta
CONFIGS = {
    "C_k15_naug": (K_IN_EFF, False, "N=16384 K_in=15 no-aug T2 (validates +1.27pp T1 anomaly)"),
    "D_k15_aug":  (K_IN_EFF, True,  "N=16384 K_in=15+aug T2 (validates +2.50pp T1 compound)"),
}


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
    for path in [args.data, args.data_aug]:
        if not (ROOT / path).exists():
            print(f"ERROR: {ROOT / path} not found."); sys.exit(1)

    tr_clean, va = make_loaders(str(ROOT / args.data), batch_size=BATCH, seed=SEED)
    tr_aug, _    = make_loaders(str(ROOT / args.data_aug), batch_size=BATCH, seed=SEED)

    print(f"Step 291 — K_in=15 compound T2 @ N=16384 ({EPOCHS}ep, 100% data)")
    print(f"  device={DEVICE}  seed={SEED}  N={N}")
    print(f"  T1 Ref=93.43% (step286), C_k15_naug=94.70% (+1.27pp), D_k15_aug=95.92% (+2.50pp)\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    if OUT_PATH.exists():
        try:
            results = json.loads(OUT_PATH.read_text())
            print(f"  Resuming: {list(results.keys())} already done\n")
        except Exception:
            pass

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        if key in results:
            r = results[key]
            print(f"  skip {key} (done: {r['top1_best']:.4f}  Δ={( r['top1_best']-REF_ACC_T1)*100:+.2f}pp vs T1-Ref)")
            continue

        K_in, use_aug, desc = CONFIGS[key]
        tr = tr_aug if use_aug else tr_clean
        model = make_model(K_in)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}\n  K_in={K_in}  aug={'yes' if use_aug else 'no'}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 25 == 0 else None))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        delta_t1 = best - REF_ACC_T1
        print(f"  → best={best:.4f} @ep{best_ep}  Δ_T1={delta_t1*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "label": desc, "N": N, "K_in": K_in, "use_aug": use_aug,
            "top1_best": best, "best_epoch": best_ep,
            "delta_vs_ref_t1": round(delta_t1, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}\nSTEP 291 — K_in=15 compound T2 @ N=16384 summary")
    print(f"  T1: Ref=93.43% | C_k15_naug=94.70% (+1.27pp) | D_k15_aug=95.92% (+2.50pp)")
    for k in keys:
        r = results.get(k, {})
        if r: print(f"  {k:<14}  aug={'y' if r['use_aug'] else 'n'}  best={r['top1_best']:.4f}  Δ_T1={r['delta_vs_ref_t1']*100:+.2f}pp")


if __name__ == "__main__":
    main()
