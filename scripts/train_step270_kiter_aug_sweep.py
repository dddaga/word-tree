"""Step 270: K_iter reduction + augmentation sweep.

MOTIVATION
==========
step268 T1 showed aug alone gives +1.33pp (C_proj_aug_k5=96.66% vs Ref=95.34%).
K=4+aug (B) underperforms K=5+aug (C) — aug alone is the gain.

Question: can we reduce K_iter (routing FLOPs) while keeping aug gain intact?
  - K_iter=3+aug vs K_iter=5+aug: does fewer routing steps hurt accuracy under aug?
  - K_iter=2+aug: floor of routing FLOPs?
  - K_iter=3 no-aug: isolate K_iter=3 effect alone (should match step630 ~?)

FLOPs at inference (routing only):
  K_iter=5: N×K_hh×D×2 = 2048×2×16×2 = 131,072 MACs per step → 655K total
  K_iter=3: 393K MACs  (40% reduction)
  K_iter=2: 262K MACs  (60% reduction)

Advance rule: K_iter=X + aug ≥ Ref (95.52%) + 0.5pp = 96.02%

CONFIGS
=======
  Ref        : K_iter=5 no-aug  (matches step268 Ref=95.34%)
  A_k3_naug  : K_iter=3 no-aug  (FLOPs reduction baseline)
  B_k3_aug   : K_iter=3 + aug   (COMBO: fewer steps + aug)
  C_k2_aug   : K_iter=2 + aug   (aggressive reduction)

Tier: T1 (75ep, 50% data)

To run:
    python -u scripts/train_step270_kiter_aug_sweep.py --device mps
    python -u scripts/train_step270_kiter_aug_sweep.py --device cuda
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
parser.add_argument("--data_aug", default="data/store_aug.h5")
parser.add_argument("--configs", default="Ref,A_k3_naug,B_k3_aug,C_k2_aug")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step270_kiter_aug_sweep_seed{SEED}__{SLOT}.json"

# (K_iter, use_aug, description, routing_macs)
CONFIGS = {
    "Ref":       (5, False, "K_iter=5 no-aug (step268 ref)"),
    "A_k3_naug": (3, False, "K_iter=3 no-aug (FLOPs baseline)"),
    "B_k3_aug":  (3, True,  "K_iter=3 + aug (40% FLOPs reduction)"),
    "C_k2_aug":  (2, True,  "K_iter=2 + aug (60% FLOPs reduction)"),
}


def make_model(K_iter: int) -> nn.Module:
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_iter, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def main():
    torch.manual_seed(SEED)
    for attr, path in [("data", args.data), ("data_aug", args.data_aug)]:
        if not (ROOT / path).exists():
            print(f"ERROR: {ROOT / path} not found."); sys.exit(1)

    tr_clean_full, va = make_loaders(str(ROOT / args.data), batch_size=BATCH, seed=SEED)
    tr_aug_full, _    = make_loaders(str(ROOT / args.data_aug), batch_size=BATCH, seed=SEED)

    def subset_50pct(loader):
        ds = loader.dataset
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(SEED))[:len(ds)//2].tolist()
        return torch.utils.data.DataLoader(
            Subset(ds, idx), batch_size=BATCH, shuffle=True, drop_last=False)

    tr_clean = subset_50pct(tr_clean_full)
    tr_aug   = subset_50pct(tr_aug_full)

    print(f"Step 270 — K_iter + aug sweep (T1: {EPOCHS}ep, 50% data)")
    print(f"  device={DEVICE}  seed={SEED}  N={N}  D={D}  K_hh={K_HH}")
    print(f"  Train={len(tr_clean.dataset)}  Val={len(va.dataset)}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    if OUT_PATH.exists():
        try:
            results = json.loads(OUT_PATH.read_text())
            print(f"  Resuming: {list(results.keys())} already done\n")
        except Exception:
            pass

    ref_acc = results.get("Ref", {}).get("top1_best")

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        if key in results:
            print(f"  skip {key} (done: {results[key]['top1_best']:.4f})")
            if key == "Ref": ref_acc = results[key]["top1_best"]
            continue

        K_iter, use_aug, desc = CONFIGS[key]
        tr = tr_aug if use_aug else tr_clean
        routing_macs = K_iter * N * K_HH * D * 2
        model = make_model(K_iter)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: {desc}")
        print(f"  K_iter={K_iter}  aug={'yes' if use_aug else 'no'}  routing={routing_macs/1e3:.0f}K MACs  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 25 == 0 else None))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1

        if key == "Ref": ref_acc = best
        delta = best - (ref_acc or 0)
        print(f"  → {key}  best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "label": desc, "K_iter": K_iter, "use_aug": use_aug,
            "routing_macs": routing_macs, "n_params": n_p,
            "top1_best": best, "best_epoch": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    STEP199_REF = 0.9552
    print(f"\n{'='*60}")
    print(f"STEP 270 SUMMARY — K_iter + aug sweep (T1)")
    print(f"{'='*60}")
    ref_b = results.get("Ref", {}).get("top1_best", 0)
    for k in keys:
        r = results.get(k, {})
        if not r: continue
        d = r["top1_best"] - ref_b
        d199 = r["top1_best"] - STEP199_REF
        adv = "ADVANCE" if d >= 0.005 else ("MARGINAL" if d >= 0 else "KILLED")
        print(f"  {k:<14} K={r['K_iter']} aug={'y' if r['use_aug'] else 'n'}  "
              f"best={r['top1_best']:.4f}  Δ_ref={d*100:+.2f}pp  Δ_199={d199*100:+.2f}pp  {adv}")


if __name__ == "__main__":
    main()
