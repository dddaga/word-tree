"""Step 290: K_in=10,20 T1 @ N=16384 — validate K_in anomaly direction.

MOTIVATION
==========
step289 T0 results at N=16384 (vs T1 Ref=93.43% — directional only):
  K_in=20: 91.01% (-2.42pp T0 delta) — likely dead
  K_in=10: 92.43% (-0.99pp T0 delta) — borderline, advance per rejection-filter rule

step288 confirmed K_in=15 HELPS at N=16384: +1.27pp T1 (anomaly vs N≤8192 where K_in=15 costs -0.33pp).
Hypothesis: at large N, K_in reduction acts as regularization (fewer spurious connections).
K_in curve at N=16384 T1: K_in=10 and K_in=20 complete the picture.

Expected:
  K_in=20: likely -1pp (between K_in=25 and K_in=15)
  K_in=10: unclear — either regularization continues (~94%) or too sparse (~92%)

Tier: T1 (75ep, 50% data)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="A_k20,B_k10")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 64; SEED = args.seed
N = 16384; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step290_kin_t1_n16384_seed{SEED}__{SLOT}.json"

# Ref from step286: 93.43% (T1). K_in=15 T1 from step288: 94.70%.
REF_ACC = 0.934267520904541
K15_ACC = 0.9469780325889587
CONFIGS = {
    "A_k20": (20, "N=16384 K_in=20 T1 (directional: T0=-2.42pp)"),
    "B_k10": (10, "N=16384 K_in=10 T1 (directional: T0=-0.99pp)"),
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
    if not (ROOT / args.data).exists():
        print(f"ERROR: {ROOT / args.data} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(ROOT / args.data), batch_size=BATCH, seed=SEED)
    ds = tr_full.dataset
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(SEED))[:len(ds)//2].tolist()
    tr = torch.utils.data.DataLoader(Subset(ds, idx), batch_size=BATCH, shuffle=True)

    print(f"Step 290 — K_in T1 @ N=16384 ({EPOCHS}ep, 50% data)")
    print(f"  device={DEVICE}  Ref=93.43% (step286)  K_in=15=94.70% (step288 C)")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    if OUT_PATH.exists():
        try: results = json.loads(OUT_PATH.read_text())
        except Exception: pass

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        if key in results:
            r = results[key]
            print(f"  skip {key} (done: {r['top1_best']:.4f}  Δ={( r['top1_best']-REF_ACC)*100:+.2f}pp)")
            continue

        K_in, desc = CONFIGS[key]
        model = make_model(K_in)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{key}: {desc}  K_in={K_in}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 15 == 0 else None))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        delta = best - REF_ACC
        delta_k15 = best - K15_ACC
        print(f"  → best={best:.4f} @ep{best_ep}  Δ_Ref={delta*100:+.2f}pp  Δ_K15={delta_k15*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {"label": desc, "N": N, "K_in": K_in,
                        "top1_best": best, "best_epoch": best_ep,
                        "delta_vs_ref": round(delta, 4),
                        "delta_vs_k15": round(delta_k15, 4),
                        "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}\nSTEP 290 — K_in T1 @ N=16384 summary")
    print(f"  Ref_k25=93.43% (step286)  K_in=15=94.70% (step288 C)")
    for k in keys:
        r = results.get(k, {})
        if r: print(f"  {k}  K_in={r['K_in']}  best={r['top1_best']:.4f}  Δ_Ref={r['delta_vs_ref']*100:+.2f}pp  Δ_K15={r['delta_vs_k15']*100:+.2f}pp")


if __name__ == "__main__":
    main()
