"""Step 287: Aug @ N=16384 T2 — validating step286 +2.11pp T1 result.

MOTIVATION
==========
step286 T1: Ref=93.43%, A_n16384_aug=95.54% (+2.11pp). ADVANCES.
Aug N-scaling T1 curve:
  N=1024: +1.30pp, N=2048: +0.54pp (T1 from step278/step269 context)
  N=4096: +0.87pp, N=8192: +1.66pp, N=16384: +2.11pp — non-monotonic, large gain.

T2 validates whether the gain holds at full training.
Expected ~+0.4pp T2 (consistent with smaller N T2 deltas of +0.43–0.66pp).

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
parser.add_argument("--configs",  default="Ref_n16384,A_n16384_aug")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 64; SEED = args.seed
N = 16384; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step287_aug_n16384_t2_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Ref_n16384":   (False, "N=16384 no-aug T2 ref"),
    "A_n16384_aug": (True,  "N=16384 + aug T2 (validates +2.11pp T1)"),
}


def make_model() -> nn.Module:
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
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

    routing_macs = K_ITER * N * K_HH * D * 2
    print(f"Step 287 — Aug @ N=16384 T2 ({EPOCHS}ep, 100% data)")
    print(f"  device={DEVICE}  seed={SEED}  N={N}  routing_MACs={routing_macs/1e6:.2f}M")
    print(f"  T1 result: Ref=93.43%, Aug=95.54% (+2.11pp)\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    if OUT_PATH.exists():
        try:
            results = json.loads(OUT_PATH.read_text())
            print(f"  Resuming: {list(results.keys())} already done\n")
        except Exception:
            pass

    ref_acc = results.get("Ref_n16384", {}).get("top1_best")

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        if key in results:
            r = results[key]
            print(f"  skip {key} (done: {r['top1_best']:.4f})")
            if key == "Ref_n16384": ref_acc = r["top1_best"]
            continue

        use_aug, desc = CONFIGS[key]
        tr = tr_aug if use_aug else tr_clean
        model = make_model()
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}  aug={'yes' if use_aug else 'no'}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 25 == 0 else None))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_n16384": ref_acc = best
        delta = best - (ref_acc or 0)
        print(f"  → {key}  best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "label": desc, "N": N, "use_aug": use_aug,
            "routing_macs": routing_macs, "n_params": n_p,
            "top1_best": best, "best_epoch": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}\nSTEP 287 SUMMARY — Aug @ N=16384 T2")
    ref_b = results.get("Ref_n16384", {}).get("top1_best", 0)
    for k in keys:
        r = results.get(k, {})
        if not r: continue
        print(f"  {k:<18} aug={'y' if r['use_aug'] else 'n'}  best={r['top1_best']:.4f}  Δ={( r['top1_best']-ref_b)*100:+.2f}pp")


if __name__ == "__main__":
    main()
