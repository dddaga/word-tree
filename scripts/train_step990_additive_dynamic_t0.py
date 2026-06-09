"""Step 990: Additive r*-threshold dynamic connectivity — T0 scout (20ep, 50% Imagenette).

Tests brief §3.5 additive dynamic connectivity at modern base (N=512, D=16, ΔW-proj removed).
    Z = normalize(Z_static + alpha_dyn * Z_dyn)
Key distinction: ADDITIVE (not multiplicative gate). Gate-death theorem covers multiplicative
gating only (g^K→0 for K_iter≥4). This additive form was never tested at modern base.

If all configs KILLED → core idea 2 of original brief CONFIRMED dead at N=512 D=16 modern base.
If any config advances → proceed to T1 N=512 75ep to confirm.

Scale: N=512 (O(N²) cdist tractable at N=512; N=2048 would be ~16× slower per step).

Configs:
  Ref              — SGNNET_SmallWorld N=512 D=16 K_iter=5 (static baseline at N=512)
  A_additive_10    — AdditiveDynamic alpha_dyn=1.0 (full additive weight)
  B_additive_05    — AdditiveDynamic alpha_dyn=0.5 (half weight)
  C_learned_alpha  — AdditiveDynamic learn_alpha=True (learned scalar, init=1.0)

Advance criterion: any config >= Ref + 0.5pp → T1.

CUDA AUDIT:
  Model: SGNNET_AdditiveDynamic wraps SmallWorld — does NOT use SGNNET_Resonant_CUDA
  or SGNNET_AntiHebbian_CUDA (mechanism isolation). pin_memory=True. AMP via trainer.

Usage:
    python scripts/train_step990_additive_dynamic_t0.py [--device cuda] [--smoke_test]
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
parser.add_argument("--epochs",     type=int, default=20)
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
OUT_PATH = ROOT / "results" / f"train_step990_additive_dynamic_t0_seed{SEED}__{SLOT}.json"


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
    # v2: SGNNET_AH_AdditiveDynamic — full canonical chain semantics (theta +
    # reflection + AH wpos suppression) with additive dyn term inside the loop.
    # v1 used bare SmallWorld: Ref=11.6%, invalid (AH prerequisite, step218).
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
    print(f"step990 — Additive dynamic connectivity T0 (20ep, 50% data, Imagenette)")
    print(f"  N={N} D={D} K_iter={K_ITER} K_in={K_IN}  (O(N²) cdist; N=512 tractable)")
    print(f"  Additive form: Z = normalize(Z_static + alpha * Z_dyn)")
    print(f"  Gate-death theorem covers multiplicative only — this form untested.")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}  batch={BATCH}")
    print(f"  Advance: any config >= Ref +0.5pp")
    print(f"{'='*70}\n")

    results = {"step": "step990", "seed": SEED, "N": N, "D": D, "configs": {}}
    for name, builder in CONFIGS.items():
        results["configs"][name] = train_config(name, builder, tr, va)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = results["configs"]["Ref"]["best"]
    print(f"\n{'='*70}")
    print(f"step990 SUMMARY  (Ref={ref_best:.4f}  N={N})")
    for name, r in results["configs"].items():
        delta   = r["best"] - ref_best
        verdict = ("Ref" if name == "Ref"
                   else "ADVANCE→T1" if delta >= 0.005
                   else "NEUTRAL/KILL")
        print(f"  {name:<30}  {r['best']:.4f}  ({delta:+.4f})  {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
