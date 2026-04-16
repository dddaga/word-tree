"""Step 293: K_in=15 no-aug T1 @ N=4096, N=8192 — crossover characterization.

MOTIVATION
==========
K_in=15 isolation (no aug) data so far:
  N=2048 T2: Ref=95.46%, K_in=15=95.13% (-0.33pp) — COSTS at small N (step632)
  N=16384 T1: Ref=93.43%, K_in=15=94.70% (+1.27pp) — HELPS at large N (step288 C)

The crossover N is unknown. Where does K_in=15 switch from hurting to helping?

Paper hypothesis: at large N, K_in=25 creates too many weak/spurious seed connections.
K_in=15 acts as input regularization — fewer, stronger initial activations.
Crossover N tells us when the feature space is dense enough to benefit from sparser seeding.

Configs (one variable changed vs Ref: K_in only, no aug):
  N=4096: Ref_k25=96.23% (step205 T2, use as directional; T1 Ref ~94%). K_in=15 vs K_in=25.
  N=8192: Ref_k25=94.90% (step209 T2, use as directional; T1 Ref ~93%). K_in=15 vs K_in=25.

Tier: T1 (75ep, 50% data). One run per N (K_in=15 only; compare against known T1 Refs from prior steps).

T1 Refs from queue:
  N=4096: step272 Ref T1 = 95.85%
  N=8192: step275 Ref T1 = 94.80%
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
parser.add_argument("--configs", default="A_n4096_k15,B_n8192_k15")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 64; SEED = args.seed
N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 15
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step293_kin15_crossover_t1_seed{SEED}__{SLOT}.json"

# T1 Refs from prior steps (50% data, 75ep comparable runs)
REFS = {
    4096: 0.9585,   # step272 N=4096+aug T1: Ref=95.85%
    8192: 0.9480,   # step275 N=8192+aug T1: Ref=94.80%
}
CONFIGS = {
    "A_n4096_k15": (4096, "N=4096 K_in=15 no-aug T1 (crossover test)"),
    "B_n8192_k15": (8192, "N=8192 K_in=15 no-aug T1 (crossover test)"),
}


def make_model(N: int) -> nn.Module:
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
    if not (ROOT / args.data).exists():
        print(f"ERROR: {ROOT / args.data} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(ROOT / args.data), batch_size=BATCH, seed=SEED)
    ds = tr_full.dataset
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(SEED))[:len(ds)//2].tolist()
    tr = torch.utils.data.DataLoader(Subset(ds, idx), batch_size=BATCH, shuffle=True)

    print(f"Step 293 — K_in=15 crossover T1 ({EPOCHS}ep, 50% data, no aug)")
    print(f"  device={DEVICE}  K_in=15 vs Refs: N=4096~95.85%, N=8192~94.80%\n")

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
            print(f"  skip {key} (done: {r['top1_best']:.4f}  Δ={r['delta_vs_ref']*100:+.2f}pp)")
            continue

        N, desc = CONFIGS[key]
        ref = REFS[N]
        model = make_model(N)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{key}: {desc}  params={n_p:,}  Ref~{ref*100:.2f}%")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 15 == 0 else None))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        delta = best - ref
        print(f"  → best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp vs Ref  {elapsed:.0f}s")

        results[key] = {"label": desc, "N": N, "K_in": K_IN,
                        "top1_best": best, "best_epoch": best_ep,
                        "ref_acc": ref, "delta_vs_ref": round(delta, 4),
                        "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}\nSTEP 293 — K_in=15 crossover summary")
    print(f"  N=2048 K_in=15 T2: -0.33pp (step632) — COSTS")
    print(f"  N=16384 K_in=15 T1: +1.27pp (step288) — HELPS")
    for k in keys:
        r = results.get(k, {})
        if r: print(f"  {k}  N={r['N']}  best={r['top1_best']:.4f}  Δ={r['delta_vs_ref']*100:+.2f}pp")


if __name__ == "__main__":
    main()
