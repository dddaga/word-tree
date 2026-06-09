"""Step 993: Additive dynamic connectivity — T1 calibration (75ep, 50% Imagenette).

T0 ADVANCED: step990 T0 v2 all 3 configs advance.
  Ref=61.27%  A_additive_10=+0.67pp  B_additive_05=+1.51pp  C_learned_alpha=+1.15pp

T1 at N=512 (NOT N=2048): O(N²) cdist makes N=2048 ~16× slower (~6 hrs/config).
N=512 calibration sufficient to confirm real signal vs T0 noise.
Advance criterion: any config >= Ref + 0.5pp → defaults update / T2.

Configs (all T0 advancers):
  Ref              — SGNNET_AH_AdditiveDynamic alpha_dyn=0.0 (static baseline N=512)
  A_additive_10    — alpha_dyn=1.0
  B_additive_05    — alpha_dyn=0.5 (T0 best, +1.51pp)
  C_learned_alpha  — alpha_dyn=1.0, learn_alpha=True

CUDA AUDIT:
  Model: SGNNET_AH_AdditiveDynamic wraps SmallWorld — does NOT use SGNNET_Resonant_CUDA
  or SGNNET_AntiHebbian_CUDA (mechanism isolation). pin_memory=True. AMP via trainer.

Usage:
    python scripts/train_step993_additive_t1.py [--device cuda] [--smoke_test]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders, make_subset_loader
from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_additive_dynamic import SGNNET_AH_AdditiveDynamic

# ── Constants ─────────────────────────────────────────────────────────────────
N      = 512
D      = 16
K_ITER = 5
K_HH   = 2
K_IN   = 25
N_OUT  = 10
N_IN   = 25088
BATCH  = 512

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device",     default="auto")
parser.add_argument("--epochs",     type=int, default=75)
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--data",       default="data/store.h5")
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS   = args.epochs
SEED     = args.seed
SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step993_additive_t1_seed{SEED}__{SLOT}.json"

T0_REF = 0.6127
T0_A   = 0.6194
T0_B   = 0.6278
T0_C   = 0.6242


# ── Model builders ────────────────────────────────────────────────────────────

def _base(seed: int = 42) -> SGNNET_SmallWorld:
    torch.manual_seed(seed)
    n_groups = max(8, N // 8)
    K_l = max(1, K_HH // 2); K_r = K_HH - K_l
    return SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_local=K_l, K_random=K_r,
        n_groups=n_groups, K_iter=K_ITER,
        norm_mode="l2", encoding_mode="fourier",
    )


def make_configs(seed: int = 42) -> dict:
    torch.manual_seed(seed)
    return {
        "Ref":             lambda: SGNNET_AH_AdditiveDynamic(_base(seed), alpha_dyn=0.0),
        "A_additive_10":   lambda: SGNNET_AH_AdditiveDynamic(_base(seed), alpha_dyn=1.0),
        "B_additive_05":   lambda: SGNNET_AH_AdditiveDynamic(_base(seed), alpha_dyn=0.5),
        "C_learned_alpha": lambda: SGNNET_AH_AdditiveDynamic(_base(seed), alpha_dyn=1.0,
                                                             learn_alpha=True),
    }


# ── Train one config ──────────────────────────────────────────────────────────

def train_config(name: str, builder, tr, va) -> dict:
    model = builder().to(DEVICE)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    kw    = trainer_kwargs(N, n_epochs=EPOCHS)
    print(f"\n{'─'*60}")
    print(f"Config: {name}  params={n_p:,}  device={DEVICE}", flush=True)
    t0 = time.time()
    history = Trainer(
        model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw,
    ).train(
        n_epochs=EPOCHS,
        log_fn=lambda m: print(
            f"  e{m['epoch']+1:3d}  loss={m['train_loss']:.4f}  "
            f"top1={m['val_top1']:.4f}  lr={m['lr']:.2e}", flush=True,
        ),
    )
    elapsed = time.time() - t0
    top1h   = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
    best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
    print(f"  DONE: best={best:.4f} @ep{best_ep}  {elapsed:.0f}s")
    return {"best": round(best, 4), "best_ep": best_ep, "n_params": n_p,
            "elapsed_s": round(elapsed, 1), "history": [round(v, 4) for v in top1h]}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if args.smoke_test:
        print("=== SMOKE TEST ===")
        CONFIGS = make_configs(SEED)
        dummy   = torch.randn(4, N_IN)
        all_ok  = True
        for name, builder in CONFIGS.items():
            m = builder()
            with torch.no_grad():
                out = m(dummy)
            ok = (out.shape == (4, N_OUT)) and (not torch.isnan(out).any())
            n_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"  {name:<30}  params={n_p:,}  shape={tuple(out.shape)}  {'OK' if ok else 'FAIL'}")
            all_ok = all_ok and ok
        sys.exit(0 if all_ok else 1)

    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found"); sys.exit(1)

    CONFIGS = make_configs(SEED)
    _pin = (DEVICE.type == "cuda")
    tr = make_subset_loader(str(data_path), fraction=0.5, batch_size=BATCH,
                            seed=SEED, pin_memory=_pin)
    _, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED, pin_memory=_pin)

    print(f"\n{'='*70}")
    print(f"step993 — Additive dynamic connectivity T1 (75ep, 50% data, Imagenette, N=512)")
    print(f"  T0 results: Ref={T0_REF:.4f}  A={T0_A:.4f}(+0.67)  B={T0_B:.4f}(+1.51)  C={T0_C:.4f}(+1.15)")
    print(f"  All 3 T0 configs advance. T1 confirms real signal.")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}  batch={BATCH}")
    print(f"  Advance: any config >= Ref +0.5pp → defaults update / T2")
    print(f"{'='*70}\n")

    results = {"step": "step993", "seed": SEED, "N": N, "D": D, "configs": {}}
    for name, builder in CONFIGS.items():
        results["configs"][name] = train_config(name, builder, tr, va)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = results["configs"]["Ref"]["best"]
    print(f"\n{'='*70}")
    print(f"step993 SUMMARY  (Ref={ref_best:.4f}  N={N}  75ep)")
    for name, r in results["configs"].items():
        delta   = r["best"] - ref_best
        verdict = ("Ref" if name == "Ref"
                   else "ADVANCE→T2/defaults" if delta >= 0.005
                   else "NEUTRAL/KILL")
        print(f"  {name:<30}  {r['best']:.4f}  ({delta:+.4f})  {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
