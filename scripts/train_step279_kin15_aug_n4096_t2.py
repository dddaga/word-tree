"""Step 279: K_in=15 + aug compound @ N=4096 T2.

MOTIVATION
==========
step277 T1: C_k15_aug_n4096 = 96.56% (+0.33pp vs 96.23% ref). Advances.
T2 validates whether compound (K_in=15+aug) holds at N=4096 at full training.
Expected: ~97.4% (step273 N=4096+aug T2=97.68%, minus ~0.3pp K_in penalty).

Paper claim: K_in=15+aug@N=4096 matches clean N=4096 accuracy with 26.7× seed reduction.

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
parser.add_argument("--configs",  default="Ref_n4096,C_k15_aug_n4096")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 4096; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step279_kin15_aug_n4096_t2_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Ref_n4096":       (25, False, "N=4096 K_in=25 no-aug (step205 ref)"),
    "C_k15_aug_n4096": (15, True,  "N=4096 K_in=15+aug (compound T2)"),
}


def make_model(K_in):
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_in, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N//8), norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def main():
    torch.manual_seed(SEED)
    tr_clean, va = make_loaders(str(ROOT/args.data), batch_size=BATCH, seed=SEED)
    tr_aug, _    = make_loaders(str(ROOT/args.data_aug), batch_size=BATCH, seed=SEED)
    print(f"Step 279 — K_in=15+aug@N=4096 T2 ({EPOCHS}ep)  device={DEVICE}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    if OUT_PATH.exists():
        try: results = json.loads(OUT_PATH.read_text())
        except: pass

    ref_acc = results.get("Ref_n4096", {}).get("top1_best")
    for key in keys:
        if key not in CONFIGS: continue
        if key in results:
            r = results[key]; print(f"  skip {key} ({r['top1_best']:.4f})")
            if key == "Ref_n4096": ref_acc = r["top1_best"]
            continue
        K_in, use_aug, desc = CONFIGS[key]
        tr = tr_aug if use_aug else tr_clean
        model = make_model(K_in)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*55}\n{key}: {desc}  K_in={K_in}  params={n_p:,}")
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 15 == 0 else None))
        elapsed = time.time() - t0
        top1h = [h.get("val_top1",0.0) if isinstance(h,dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h))+1
        if key == "Ref_n4096": ref_acc = best
        delta = best - (ref_acc or 0)
        print(f"  → {key} best={best:.4f} @ep{best_ep} Δ={delta*100:+.2f}pp {elapsed:.0f}s")
        results[key] = {"label": desc, "K_in": K_in, "use_aug": use_aug,
            "top1_best": best, "best_epoch": best_ep,
            "delta_vs_ref": round(delta,4), "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*55}\nSTEP 279 SUMMARY")
    ref_b = results.get("Ref_n4096",{}).get("top1_best",0)
    for k in keys:
        r = results.get(k,{})
        if r: print(f"  {k:<22} best={r['top1_best']:.4f} Δ={( r['top1_best']-ref_b)*100:+.2f}pp")

if __name__ == "__main__": main()
