"""Step 271: K_in=15 + aug compound validation (T1).

MOTIVATION
==========
Two confirmed gains from prior steps:
  step632: K_in=15 saves 26.7× seed FLOPs at −0.33pp cost (T2 confirmed)
  step269: aug alone gives +0.66pp (T2 confirmed)

Hypothesis: combining K_in=15 + aug nets ~+0.33pp vs clean K_in=25,
while also having 26.7× seed FLOP reduction. If confirmed, this is the
new efficiency default: same accuracy headroom + dramatically cheaper seeds.

CONFIGS
=======
  Ref       : K_in=25, no-aug   (95.52% step199 standard)
  A_k15     : K_in=15, no-aug   (step632: 95.13%, −0.33pp)
  B_aug     : K_in=25, aug      (step269 C: 96.13% T2, ~96.7% T1)
  C_k15_aug : K_in=15, aug      (COMPOUND — hypothesis: ≥95.8%)

Advance rule: C_k15_aug ≥ Ref + 0.0pp (nets no accuracy loss vs clean)

Tier: T1 (75ep, 50% data)

To run:
    python -u scripts/train_step271_kin15_aug_compound.py --device mps
    python -u scripts/train_step271_kin15_aug_compound.py --device cuda
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
parser.add_argument("--configs",  default="Ref,A_k15,B_aug,C_k15_aug")
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
OUT_PATH = ROOT / "results" / f"train_step271_kin15_aug_compound_seed{SEED}__{SLOT}.json"

# (K_in, use_aug, description)
CONFIGS = {
    "Ref":       (25, False, "K_in=25 no-aug (step199 ref)"),
    "A_k15":     (15, False, "K_in=15 no-aug (step632: −0.33pp)"),
    "B_aug":     (25, True,  "K_in=25 + aug (step269 C: +0.66pp T2)"),
    "C_k15_aug": (15, True,  "K_in=15 + aug (compound: efficiency + accuracy)"),
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

    print(f"Step 271 — K_in=15 + aug compound (T1: {EPOCHS}ep, 50% data)")
    print(f"  device={DEVICE}  seed={SEED}  N={N}  D={D}  K_iter={K_ITER}")
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
            r = results[key]
            print(f"  skip {key} (done: {r['top1_best']:.4f})")
            if key == "Ref": ref_acc = r["top1_best"]
            continue

        K_in, use_aug, desc = CONFIGS[key]
        tr = tr_aug if use_aug else tr_clean
        seed_macs = N * K_in
        model = make_model(K_in)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: {desc}")
        print(f"  K_in={K_in}  aug={'yes' if use_aug else 'no'}  seed_MACs={seed_macs:,}  params={n_p:,}")

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
            "label": desc, "K_in": K_in, "use_aug": use_aug,
            "seed_macs": seed_macs, "n_params": n_p,
            "top1_best": best, "best_epoch": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    STEP199_REF = 0.9552
    print(f"\n{'='*60}")
    print(f"STEP 271 SUMMARY — K_in=15 + aug compound (T1)")
    print(f"{'─'*60}")
    ref_b = results.get("Ref", {}).get("top1_best", 0)
    for k in keys:
        r = results.get(k, {})
        if not r: continue
        d = r["top1_best"] - ref_b
        d199 = r["top1_best"] - STEP199_REF
        adv = "ADVANCE" if d >= 0.0 else "KILLED"
        print(f"  {k:<14} K_in={r['K_in']} aug={'y' if r['use_aug'] else 'n'}  "
              f"best={r['top1_best']:.4f}  Δ_ref={d*100:+.2f}pp  Δ_199={d199*100:+.2f}pp  {adv}")
    if "C_k15_aug" in results and "Ref" in results:
        c = results["C_k15_aug"]
        compound_verdict = "COMPOUND ADVANCE" if c["top1_best"] >= ref_b else "NET LOSS"
        seed_ratio = results["Ref"]["seed_macs"] / c["seed_macs"]
        print(f"\n  C_k15_aug verdict: {compound_verdict}  ({seed_ratio:.1f}× seed FLOP reduction)")


if __name__ == "__main__":
    main()
