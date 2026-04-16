"""Step 288: K_in=15+aug compound @ N=16384 T1.

MOTIVATION
==========
K_in=15+aug compound T2 curve COMPLETE:
  N=1024: +0.79pp (step284)
  N=2048: +0.18pp (step274)
  N=4096: +0.18pp (step279)
  N=8192: +0.36pp (step282)

Ref_n16384=93.43% from step286. step286 A_n16384_aug in progress.
This step isolates K_in=15 only (no-aug) and K_in=15+aug compound at N=16384.
With Ref and aug-only from step286, we get the full decomposition at new largest N.

Expected: K_in=15 costs ~-0.33pp (consistent across all N).
Expected: Compound ~= aug_delta - 0.33pp

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
parser.add_argument("--device",   default="auto")
parser.add_argument("--epochs",   type=int, default=75)
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data",     default="data/store.h5")
parser.add_argument("--data_aug", default="data/store_aug.h5")
parser.add_argument("--configs",  default="C_k15_naug,D_k15_aug")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 64; SEED = args.seed  # B=64 for N=16384 memory
N = 16384; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN_REF = 25; K_IN_EFF = 15
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step288_kin15_aug_n16384_t1_seed{SEED}__{SLOT}.json"

# Ref_n16384=93.43% from step286 (K_in=25, no-aug). Reused, not re-run.
# A_n16384_aug=??? from step286 (K_in=25, aug). Reused when done.
CONFIGS = {
    "C_k15_naug": (K_IN_EFF, False, "N=16384 K_in=15 no-aug (K_in isolation at new N)"),
    "D_k15_aug":  (K_IN_EFF, True,  "N=16384 K_in=15 + aug (compound at new N)"),
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

    tr_clean_full, va = make_loaders(str(ROOT / args.data), batch_size=BATCH, seed=SEED)
    tr_aug_full, _    = make_loaders(str(ROOT / args.data_aug), batch_size=BATCH, seed=SEED)

    def subset_50pct(loader):
        ds = loader.dataset
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(SEED))[:len(ds)//2].tolist()
        return torch.utils.data.DataLoader(
            Subset(ds, idx), batch_size=BATCH, shuffle=True, drop_last=False)

    tr_clean = subset_50pct(tr_clean_full)
    tr_aug   = subset_50pct(tr_aug_full)

    routing_macs = K_ITER * N * K_HH * D * 2
    print(f"Step 288 — K_in=15+aug compound @ N=16384 T1 ({EPOCHS}ep, 50% data)")
    print(f"  device={DEVICE}  seed={SEED}  N={N}  D={D}  K_iter={K_ITER}")
    print(f"  routing_MACs={routing_macs/1e6:.2f}M  Ref_n16384=93.43% (step286)")
    print(f"  Train={len(tr_clean.dataset)}  Val={len(va.dataset)}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    if OUT_PATH.exists():
        try:
            results = json.loads(OUT_PATH.read_text())
            print(f"  Resuming: {list(results.keys())} already done\n")
        except Exception:
            pass

    # Ref from step286: 93.43%
    ref_acc = 0.934267520904541

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        if key in results:
            r = results[key]
            delta = r["top1_best"] - ref_acc
            print(f"  skip {key} (done: {r['top1_best']:.4f}  Δ={delta*100:+.2f}pp)")
            continue

        K_in, use_aug, desc = CONFIGS[key]
        tr = tr_aug if use_aug else tr_clean
        seed_macs = N * K_in
        model = make_model(K_in)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")
        print(f"  K_in={K_in}  aug={'yes' if use_aug else 'no'}  seed_MACs={seed_macs/1e6:.2f}M")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 25 == 0 else None))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        delta = best - ref_acc
        print(f"  → {key}  best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp vs Ref  {elapsed:.0f}s")

        results[key] = {
            "label": desc, "N": N, "K_in": K_in, "use_aug": use_aug,
            "routing_macs": routing_macs, "seed_macs": seed_macs, "n_params": n_p,
            "top1_best": best, "best_epoch": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}\nSTEP 288 SUMMARY — K_in=15+aug compound @ N=16384 T1")
    print(f"  Ref_n16384=93.43% (step286 K_in=25 no-aug)")
    for k in keys:
        r = results.get(k, {})
        if not r: continue
        d = r["top1_best"] - ref_acc
        seed_ratio = (N * K_IN_REF) / r["seed_macs"] if r["seed_macs"] > 0 else 1
        print(f"  {k:<14} K_in={r['K_in']} aug={'y' if r['use_aug'] else 'n'}  "
              f"best={r['top1_best']:.4f}  Δ={d*100:+.2f}pp  seed={seed_ratio:.1f}×")


if __name__ == "__main__":
    main()
